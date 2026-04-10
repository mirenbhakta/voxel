# Render Pipeline V2 Architecture

Cubic voxels throughout. Sub-chunk DDA is the sole rendering primitive at every
distance. This document supersedes `gpu_memory_architecture.md` for rendering
architecture. `scaffold_rewrite_principles.md` still governs CPU/GPU boundary
discipline and applies unchanged.

The document is split into **Core** — the minimum viable path that validates or
invalidates the sub-chunk DDA thesis on real workloads — and **Extensions** —
features that extend the range of what the primitive handles, documented for
direction but explicitly out of scope until the core is measured and working.
Anything that is not in §Core is not in V1.

---

## Core

### Primitive Model

Rasterize the 8×8×8 sub-chunk as a bounding cube. The fragment shader
DDA-marches the occupancy bitmap to find the surface inside the volume, samples
material at the hit point, writes depth at the hit face.

At level 0 there is one primitive: the sub-chunk cube. No dual rendering paths.
No greedy surface quads. No per-voxel billboards. No impostors. The sub-chunk
is the renderable.

Why a single primitive: every hybrid previously considered collapses into a
degenerate case of sub-chunk DDA. Surface quads are a 0-step DDA against a
solid cell. Per-voxel billboards are the 1³ sub-chunk case. Impostors are a
sub-pixel statistical fallback that the LOD hierarchy (§Extensions) handles.
Collapsing them eliminates the per-path bookkeeping, the transitions between
paths, and the supporting machinery (layer storage, greedy meshing, impostor
atlases) that existed only to make the eliminated paths efficient.

**Near-field cost.** At 1 voxel ≫ 1 pixel, DDA terminates within a handful of
steps on the first opaque hit. The per-fragment cost is bounded by the
sub-chunk diameter (~14 steps worst case for an 8³ grid) and in practice much
less because the march stops at the first solid. Whether this is competitive
with or cheaper than the current greedy-quad path is workload-dependent and
must be measured — see §Prototype Milestone.

### Control Plane

CPU-authoritative allocation. GPU allocator and free-list rolled back entirely.

**Root cause of the old complexity:** chunk arrivals triggered neighbor re-meshes
which cascaded across frames. The fix was to suppress the cascade rather than
make it live, accepting one frame of latency on the whole batch.

- Chunks and all neighbor updates coalesce into a single batch off the render path
- Batch commits atomically: all present or none
- One frame of latency on chunk arrival is accepted
- CPU owns slot allocation; shadow ledger validates GPU invariants
- Scaffold rewrite principles (per-frame-in-flight messaging, backpressure) apply unchanged

**Residency is distance-based.** Every sub-chunk within R of the camera is
resident. The CPU updates the residency set each frame as the camera moves.
Batches are coalesced and committed through the control-plane channel with
the one-frame latency described above.

This is a deliberate simplification over earlier drafts that proposed
GPU-feedback-driven residency (occlusion cull emits "want finer" requests, CPU
acts on them). That pattern works for brickmap-style renderers because global
ray marching naturally requests missing data — a ray that steps into an
unloaded brick produces a miss signal that drives a load. Sub-chunk DDA is
rasterization, not global ray marching, and has no equivalent feedback signal.
A cull pass can only filter what is already resident; it cannot request what
is not. Trying to drive residency off occlusion feedback only works when the
player is stationary or moving slowly — exactly the opposite of the fast,
long-distance first-person motion this project is built around. Distance is
the only signal that is always available, always correct, and always fast
enough to preempt the camera.

Fast camera motion is the steady state, not a special case. Continuous small
residency deltas are exactly what the control-plane batching absorbs.
Worst-case spikes (view distance changes, world load) remain bounded by the
scaffold rewrite principles.

### Cull Pass

The cull pass takes the resident sub-chunk set and emits an MDI indirect draw
buffer for the sub-chunk cubes that can contribute to the image. It is the
only place Hi-Z appears in this design — Hi-Z filters the draw list, it does
not drive residency.

Three rejection stages, cheapest first:

**1. Frustum cull.** Standard per-sub-chunk AABB-vs-frustum test. No
surprises.

**2. Directional exposure mask.** Each sub-chunk carries a 6-bit mask — one
bit per face direction — set if any voxel in the sub-chunk has an exposed face
in that direction (outward neighbor is empty, resolved across sub-chunk
boundaries). The cull pass computes the three camera-visible face directions
from the view direction signs and rejects sub-chunks whose mask has no
overlap with the visible set.

This is the DDA-specific rejection that has no analog in triangle rendering. A
sub-chunk whose occupied voxels all face away from the camera would still
rasterize its bounding cube and still pay fragment DDA work, but every ray
would terminate on a voxel whose camera-facing neighbor is solid — meaning the
adjacent sub-chunk (if drawn) paints the same or closer pixel. Hi-Z catches
some of these through depth, but the mask rejects them before the cull pass
even consults Hi-Z, at the cost of one byte per sub-chunk.

The mask is computed by the existing build pipeline's first pass — the same
shader that already evaluates `occupied(v) && !occupied(neighbor(v, d))` for
face counting. Reducing per-layer per-direction counts to a per-sub-chunk
OR-reduction is an additional ~10 lines of shader code and produces the mask
as a byproduct. No new pass, no new readback, no new data path.

A hierarchical variant stores the same mask at chunk granularity (OR of its
sub-chunk masks, 1 byte per chunk). The cull pass tests the chunk mask first
and early-outs on all 64 sub-chunks in one check when the camera's visible
direction set doesn't overlap.

**3. Hi-Z occlusion cull.** Standard Hi-Z pyramid from the previous frame's
depth buffer, tested against each surviving sub-chunk's AABB. Well-validated
in the meshlet literature; more essential for sub-chunk DDA than for triangle
meshes because each surviving sub-chunk pays fragment DDA cost per covered
pixel, not textured-quad cost. Project Ascendent handles 400k chunks without
Hi-Z because it has no underground features; once the occluded volume
fraction grows (caves, built interiors, tunnels), Hi-Z goes from "nice gain"
to "required for parity."

What the cull pass does not catch:

- **Sparse sub-chunks with camera-facing exposed faces that are still mostly
  empty.** The mask keeps these (some face is exposed in a visible direction)
  and Hi-Z won't reject a sub-chunk that is actually the closest thing at its
  screen position. The DDA pays to march through the empty volume before
  hitting the thin surface. This is a fragment-shader optimization concern,
  not a cull-pass concern.
- **Sub-chunks whose exposed faces are self-occluded by the sub-chunk's own
  geometry from the current view angle.** The mask is conservative and keeps
  them. Hi-Z may or may not catch through depth.

These are understood limitations, not defects in the approach.

### Material Storage

Sub-block packing (`sub_mask` popcount addressing) preserved from the current
pipeline at level 0. The DDA hits an occupied voxel at level 0, the material
address comes from the sub-block packing at level 0. No change from the
existing scaffold — V1 reuses the data structure as-is.

Material handling at coarser levels is in §Extensions.

### Prototype Milestone

Before any of the above is committed to code, one measurable question must be
answered: *is sub-chunk DDA in the ballpark of the current greedy-quad path on
the workloads the scaffold already handles?*

The smallest prototype that answers the question:

- Use the current scaffold's L0 chunk format and sub-block packing as-is.
- Treat each 8³ sub-block region inside a 32³ chunk as a sub-chunk.
- Rasterize each sub-chunk cube via MDI; fragment shader DDAs the sub-block
  occupancy.
- Add a cull compute shader doing frustum + directional exposure mask + Hi-Z.
- Measure frame time and GPU timings against the current build/filter/render
  path on representative scenes: dense terrain, sparse asteroid field,
  modified structures.

This touches no allocator code, requires no control-plane rewrite, and
produces a single answer. If the measurement clears, the rest of this document
becomes a roadmap worth executing. If it doesn't clear, the "one primitive at
every distance" thesis is wrong in the regime we actually care about, and we
know before deleting the working pipeline.

A 2M-voxel integrated-GPU figure cited in early drafts came from a
demonstration video using a depth-map overdraw rejection technique that is
not the Hi-Z + MDI approach described here. It is a directional existence
proof that sub-chunk DDA can rasterize millions of voxels on modest hardware,
not a validation of this design's specific performance profile. The prototype
milestone is the actual validation.

---

## Extensions

The features in this section extend sub-chunk DDA from "level 0 renderer" to
"full range renderer." Each is a coherent addition on top of the core, but
each is also independently optional — the core must be able to ship without
any of them, and none may be scoped into the core pass.

### LOD Hierarchy

The 8³ sub-chunk is recursive. At each level the previous level's sub-chunks
become voxels in a new 8³ grid — same occupancy bitmap, same DDA shader, same
material addressing. The LOD ladder and the coarsening ladder are the same
ladder.

| Level | Sub-chunk span | Voxel size |
|-------|----------------|------------|
| 0     | 8 m            | 1 m        |
| 1     | 64 m           | 8 m        |
| 2     | 512 m          | 64 m       |
| 3     | 4 km           | 512 m      |
| 4     | 32 km          | 4 km       |

**Capped at 5 levels by design.** The largest body the cubic hierarchy
represents is 8³ level-4 sub-chunks ≈ 256 km across — a "mega planet," large
enough to feel immense while staying gameplay-coherent. Normal planetoids sit
in the 40–80 km range (reference: Space Engineers) and top out at level 3.
Anything larger than 256 km is intentionally not rendered as voxels; see
§Beyond Cubic Range.

Each coarser level is an OR-reduction of the one below: any voxel occupied at
the finer level marks its parent cell occupied. Topology-preserving by
construction — any feature present at fine resolution is present at coarse.
Features grow 2× per level in world space; at the correct transition distance
they remain ≈1 pixel wide in screen space, so the pop is bounded.

**Sparse in memory.** At any camera position only the 2–3 levels that
straddle the visible distance range are materialized within the camera's local
region. A 1 m voxel next to the player and a 4 km voxel on the horizon can
coexist; the rest are not allocated. Pyramid storage overhead for the levels
actually in view: 8/7 ≈ 14% over the finest resident level alone.

#### Within-level coarsening

Inside a single level, further OR-reduction of the 8³ bitmap extends the
usable distance of that level before the next recursion kicks in:

| Within-level LOD | Entry primitive       | Max DDA steps | Material          |
|------------------|-----------------------|---------------|-------------------|
| L1               | OR-reduced 2×2×2 AABB | ~8            | Full res          |
| L2               | OR-reduced 4×4×4 AABB | ~12           | Face-weighted avg |
| Tail             | Stochastic prune      | —             | Averaged          |

#### Transition threshold

Perceptual acuity, not pixel count or distance heuristic. Resolution cancels
out; only physical screen size, viewing distance, and FOV matter:

```
d = voxel_size * screen_phys_width / (2 * acuity * tan(fov/2) * viewing_distance)
```

For a 27" monitor at 60 cm, 90° FOV, 1 m voxels, 2–3 arcminute gaming acuity:
~570–856 m before the first within-level coarsening kicks in. The next level's
transition is 8× further out, and so on recursively.

The 2–3 arcminute constant is a "good enough for surface texture" threshold
chosen as a UX knob, not a visibility limit. Voxel silhouettes on high-contrast
edges are visible at finer angular resolution (Vernier acuity, ~5 arcsec);
aliasing at the transition zone may still be a thing at the default constant,
and tuning is expected once the hierarchy is built.

#### Material at coarse levels

Face-weighted average — each child material is weighted by its exposed face
count in the level below. Only surface faces are visible, so weighting by
exposed face area is correct and matches how the eye integrates distant
surfaces. The data structure is the existing sub-block packing system; only
its contents differ per level. The DDA hits an occupied cell, the material
address is resolved through the same popcount-indexed packing at whatever
level is being rendered.

OR-reduction preserves topology but not density. A sparsely occupied coarse
cell still rasterizes as solid and integrates its material over a region that
is mostly empty at level 0. Face-weighted averaging compensates by weighting
on the visible area rather than the occupied volume. At perceptual acuity
this is typically fine; close to a transition distance it may produce a
visible color shift and may need further tuning.

### Stochastic Pruning

Where even the coarsest within-level LOD primitive is sub-pixel, Cook et al.
stable hash-based stochastic culling handles fade-out. Surviving primitives
enlarge their projected footprint proportionally to preserve average coverage.
This is the Majercik 2018 Section 3 technique.

Why it is still worth having under sub-chunk DDA: OR-reduction preserves
topology but not density. A sparsely occupied coarse level still rasterizes
the whole cube and pays overdraw at the DDA entry. Stochastic pruning
addresses the tail where that overdraw stops being worth it — statistically
correct fade-out at the limit where the next level up is still too coarse for
the content but the current level is sub-pixel in screen space.

The interaction between footprint enlargement and the bitmap DDA inside an
enlarged cube needs further design work. Deferred.

### Transparency

DDA visits voxels in depth order naturally. Transparent voxels accumulate
front-to-back via over-compositing inside the fragment shader; the first opaque
hit terminates the march and writes depth.

No sort, no OIT, no layer storage, no per-axis arrays. The entire transparency
machinery from the prior design was there to serve the greedy-quad path and
is unnecessary under sub-chunk DDA.

**Inter-sub-chunk ordering.** Transparent sub-chunks render in a second pass
after opaque with depth test on, depth write off. Ordering errors can occur
only when two transparent sub-chunks overlap in screen space and neither
terminates on an opaque voxel. In typical gameplay (scattered glass, water
surfaces) this is rare enough to handle with painter's-order sub-chunk
sorting when it matters. For denser transparent content (heavy foliage,
layered glass structures) a fallback plan will be needed; scoped for the
extension pass.

**Foliage:** separate pass with non-cubic meshes. Same approach as Project
Ascendent.

### Procgen Fast Path

For unmodified chunks the procgen function IS the scalar field. It is not
stored. Coarse levels are sampled coarsely; features too small for the
sampling resolution are sub-pixel at the distance that level is used, so
undersampling is correct by construction.

For smooth far-field surfaces, ray-march the procgen function directly
instead of DDA through a coarse bitmap. Analytical gradient gives the normal.
Switches to bitmap DDA for modified chunks.

The relative cost depends on the procgen function's complexity. For
multi-octave noise with domain warps, the analytic path may be *slower* per
sample than bitmap DDA; the right framing is "save memory by not storing
unmodified terrain" rather than "unconditionally faster path." The analytical
gradient is a genuine win for far-field smooth surfaces where finite
differencing would produce visible normal aliasing.

### Sphere Alignment

Cube sphere: 6 faces, each with its own local coordinate frame where Y is the
radial direction. Each face has locally flat-aligned voxels. Seams exist at
the 12 edges and 8 corners; procgen function consistency prevents geometric
seams (the same scalar field is evaluated on both sides), only the coordinate
frame orientation differs at the crossing.

The traversal side of this problem is the real open question. A DDA crossing
a face-edge seam must rotate its step vector mid-march because the local
coordinate frame changes orientation. This is not solved here. The
recommendation is to defer cube-sphere geometry until the core is stable on
flat-world layouts; the core does not depend on this being solved.

### Beyond Cubic Range

Cubic sub-chunk DDA intentionally caps at body scale. It does not scale to
astronomical distances, and that is a design choice.

Features at astronomical scale are not voxel-shaped. The nearest approach
between two bodies is thousands of kilometers at minimum; between planets,
millions. No voxel feature spans that gap, and any that did — a 1 m pillar to
the moon — would be sub-pixel across its entire length at any plausible
viewing distance. The right primitive for a line is a line, for a sphere is a
sphere, for a distant body is an impostor. Not a coarser voxel.

The renderer above the voxel system handles everything past the cubic cap:

- **Unmodified body surface at any distance.** The procgen function is the
  scalar field, ray-marched analytically with gradient-derived normal. No
  bitmap, no hierarchy. Used for the distant portions of the body the camera
  is inside, and for the near portions of any body the camera approaches
  before the cubic bubble engages.
- **Distant bodies.** Sphere impostor with surface color or silhouette. A
  body that is a handful of pixels does not need a hierarchy.
- **Stars, spacecraft at extreme range, astronomical markers.** Point
  sprites.
- **Multi-body coordinate space.** Per-body cubic hierarchies positioned in a
  scene-graph frame above the voxel system. Origin rebasing follows the
  camera as it moves between bodies (f32 loses mm precision past ~10 km
  regardless of voxel levels, so rebasing is mandatory anyway). At any
  instant, at most one body is in the cubic bubble; the rest are impostors
  or sprites.

None of these is a level of the voxel hierarchy. They are sibling renderers
above it. The voxel system remains bounded at ~256 km per body regardless of
how large the world-space scene becomes.

---

## Rejected Directions

**Near-field surface quads / greedy meshing.** Dropped. Sub-chunk DDA subsumes
it. At near distance the DDA terminates in a handful of steps on the first
opaque hit. Eliminating the dual path removes the layer storage, greedy build
shader, cull-cascade machinery, and per-direction draw structure that existed
to make the quad path efficient. The §Prototype Milestone verifies this is
actually competitive on real workloads before the quad path is deleted.

**Layer storage / OIT.** Dropped with the quad path. Transparency is handled
inside the DDA shader in natural depth order.

**Per-voxel billboards as primary path.** Dropped. Remains the 1³ degenerate
case of sub-chunk DDA but has no separate implementation — sparse sub-chunks
just terminate their DDA quickly.

**Impostors.** Dropped. The within-level OR-reduction ladder plus stochastic
pruning at the tail covers every distance that impostors were intended to
handle. Impostor generation cost, cache invalidation, and parallax error were
solving a problem the LOD system already solves with no auxiliary data.

**Life-scale planetary hierarchy.** An earlier draft extended the cubic
hierarchy to 8 levels covering Earth radius (16,000 km spans). Dropped.
Levels 5–7 represent content that is either empty (interplanetary space) or
not voxel-shaped at that scale (a planet as 6 voxels is a crude icon, not a
rendering). The cubic hierarchy caps at level 4 (~256 km body diameter).
Everything beyond is handled by non-voxel primitives documented in §Beyond
Cubic Range. The target is gameplay-coherent mega-planets, not real-planet
simulation.

**GPU-authoritative allocation.** Rolled back. Root cause: false requirement
that chunk updates be live the frame they arrive. Suppressing the cascade was
always the fix.

**GPU-feedback-driven residency / streaming.** Rejected. Brickmap streaming
works because global ray marching naturally requests missing data; a ray that
steps into an unloaded brick produces a miss signal that drives a load. A
rasterizer has no equivalent — the cull pass can only filter the resident
set, it cannot request what is not resident. Trying to drive streaming off
occlusion feedback only serves the stationary-player case and fails on the
fast long-distance motion that defines this project. Residency is
distance-based (§Core/Control Plane), with no GPU feedback loop.

**Isosurface as default format.** Documented as an alternative per-sub-chunk
format in `isosurface_dda.md`. Compatible with the DDA shader via a
compile-time branch. Not the default because binary occupancy is 8× cheaper
to store and the representation choice can be made per sub-chunk.

**Global ray marching.** Rejected prior to this document. See
`design_rejections.md`.
