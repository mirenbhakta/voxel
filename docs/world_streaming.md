# World Streaming Architecture

Design for multi-resolution voxel world representation, residency management,
CPU/GPU data flow, persistence, and procedural generation. Covers the data
side of the renderer's operating envelope — complements `render_pipeline_v2.md`
which covers the rendering side.

**Relationship to render_pipeline_v2.md.** This document supersedes
Extensions §LOD Hierarchy in v2. V2's "sub-chunks become voxels in the next
level's grid" (8× branching per step) produces 8× perceptual jumps between
levels, which is too coarse. The revised model below uses 2× branching
(standard octree) with clipmap-style residency, and reframes v2's
"recursive 8³ OR-reduction hierarchy" as the *acceleration* structure it
actually is — separate from LOD.

V2's Core (rasterized sub-chunk DDA, control plane, cull pass, material
storage) applies unchanged.

The document is split into **Core** (the streaming and LOD model required
for the renderer to operate on anything beyond a fixed-size demo world)
and **Extensions** (structure generation, persistence, and GPU-side
worldgen, which are orthogonal to streaming correctness and can be added
incrementally).

---

## Core

### LOD Hierarchy: 2× Branching

The 8³ sub-chunk is the storage and DDA primitive at every level — this does
not change. What changes is the *hierarchy branching factor* over sub-chunks.

| Level | Voxel size | Sub-chunk extent | Children in next level |
|-------|-----------|------------------|------------------------|
| L0    | 1 m       | 8 m              | —                      |
| L1    | 2 m       | 16 m             | 2×2×2 L0 sub-chunks    |
| L2    | 4 m       | 32 m             | 2×2×2 L1 sub-chunks    |
| L3    | 8 m       | 64 m             | 2×2×2 L2 sub-chunks    |
| L4    | 16 m      | 128 m            | 2×2×2 L3 sub-chunks    |
| L5    | 32 m      | 256 m            | 2×2×2 L4 sub-chunks    |
| L6    | 64 m      | 512 m            | 2×2×2 L5 sub-chunks    |

Each level's voxel size doubles. Each parent sub-chunk covers 2× the linear
extent of each child, so one parent holds 2³ = 8 children.

**Why 2× branching.** V2's 8× branching (sub-chunk = voxel in parent)
produces perceptual jumps from 1 m to 8 m to 64 m voxels between levels. At
any LOD transition, the on-screen change is violent. 2× branching produces
gradual 1 m → 2 m → 4 m → 8 m progression — same storage primitive, same DDA
shader, same OR-reduction mechanics, but the perceptual transitions are
mild because the voxel-size step is mild.

**OR-reduction per level.** Each parent voxel is a 2×2×2 OR of the child
voxels in its spatial footprint. Because each child sub-chunk maps to exactly
one octant of its parent (children are 2× smaller in each dim), a given child
only writes to 4³ = 64 of its parent's 8³ = 512 voxels. Each child OR-reduces
independently into its own octant. Total work per parent voxel: 8 bit reads +
1 bit write. Trivial on both CPU and GPU.

### Acceleration Cascade

The LOD pyramid's OR-reduction doubles as the ray-march acceleration
structure. No separate pyramid is needed.

**Key identity.** In a 2× OR-reduction pyramid, three levels up is equivalent
to one 8× OR-reduction. An L3 voxel is `L0 OR-reduced 3× at 2× each = L0
OR-reduced 8× total`, which is exactly what an 8³-branching acceleration cell
would encode. So the 8× acceleration pyramid v2 described is *embedded* in
the 2× LOD pyramid as every-third-level.

**Level selection at ray-march time.** The acceleration cascade a ray uses
depends on its terminal LOD:

| Terminal LOD | Acceleration cascade (every 3 levels up) |
|--------------|------------------------------------------|
| L0 (1 m)     | L3, L6, L9…                             |
| L1 (2 m)     | L4, L7, L10…                            |
| L2 (4 m)     | L5, L8, L11…                            |
| L3 (8 m)     | L6, L9, L12…                            |

Each descent is 8× cleanly because two levels in the 2× pyramid apart means
8× in scale. The DDA primitive is identical at every level: 8³ bitmap
traversal. A ray terminating at L1 descends through L10 → L7 → L4 → L1, each
hop the same shader code against a different sub-chunk.

**The only consequence for residency**: the LOD levels used by a given
camera's ray-march cascade must all be resident in that direction. Section
§Clipmap Residency describes how this falls out automatically from shell
nesting.

### Clipmap Residency

Residency is a clipmap shell per level, centered on the camera's position in
that level's sub-chunk grid. At each level the camera sits inside a small
(typically 2×2×2 or 3×3×3) block of sub-chunks; that block is resident. As
the camera moves, a strip rolls in on the leading face and rolls out on the
trailing face.

**Shells are nested.** Finer-level shells sit inside coarser-level shells:
a point at 5 m from the camera is inside L0, L1, L2, … shells simultaneously;
a point at 500 m is inside L5, L6, … but outside L0's narrow shell.

```
            L6 shell (512 m sub-chunks, ~1024 m coverage)
         ┌────────────────────────────────────────────┐
         │     L5 shell (256 m sub-chunks)            │
         │   ┌────────────────────────────────────┐   │
         │   │   L4 shell (128 m sub-chunks)      │   │
         │   │ ┌──────────────────────────────┐   │   │
         │   │ │ L3 shell (64 m sub-chunks)   │   │   │
         │   │ │ ┌──────────────────────────┐ │   │   │
         │   │ │ │ L2, L1, L0 shells...     │ │   │   │
         │   │ │ │        📷 camera         │ │   │   │
         │   │ │ └──────────────────────────┘ │   │   │
         │   │ └──────────────────────────────┘   │   │
         │   └────────────────────────────────────┘   │
         └────────────────────────────────────────────┘
```

**Variable shell radii per level.** Each level declares its own residency
radius as a parameter. L0 can have a tight shell (8–16 m) for edit
interactivity; coarser levels can have wider shells sized by perceptual
criteria (screen-space voxel pixel target) or memory budget. The only
constraint is that each level's shell cover at least the inner levels'
shells (otherwise the acceleration cascade breaks).

**Residency is content-agnostic by design.** A sub-chunk's resident LOD is
a pure function of the camera's position and the per-level shell radii.
There is no mechanism for "focus detail on interesting content" —
a far-distant chunk containing a player-built structure or a distinctive
landmark receives the same coarse LOD as a far-distant chunk of empty
terrain. This purity is what bounds residency-management cost at O(hundreds
of ops per frame) regardless of world size; any content-awareness reintroduces
the scoring/budget/queue-rebuild complexity that the clipmap exists to avoid.
See §Content-Aware Residency under Out of Scope.

**Storage is bounded by level count, not view distance.** Each level holds a
constant number of sub-chunks regardless of its shell radius: a 3×3×3 shell
is 27 sub-chunks, each 8³ = 512 voxels. 10 LOD levels × 27 sub-chunks ×
512 voxels ≈ 140 K voxels total — on the order of hundreds of KB of
occupancy data covering a 1 km+ view distance. Adding another LOD level adds
constant cost; it does not scale with view volume.

**Contrast with the scaffold's single-level flat residency:**

| View distance | Scaffold chunk count | Clipmap sub-chunk count |
|---------------|----------------------|-------------------------|
| 32 chunks     | ~137 K               | ~270 (10 levels)        |
| 64 chunks     | ~1.1 M               | ~300 (11 levels)        |
| 200 chunks    | ~33 M                | ~300 (11 levels)        |
| 2000 chunks   | ~270 B (impossible)  | ~380 (14 levels)        |

The scaffold's O(R³) scaling is why view_distance = 200 hangs for a minute
on queue enumeration alone. Clipmap residency has no equivalent failure mode
— the enumeration cost is O(levels × shell_face) per frame, which is
hundreds of operations regardless of view distance.

### Residency Update Cadence

Per-level residency changes as the camera crosses level-N sub-chunk
boundaries. The rate scales inversely with level's voxel size:

| Level | Sub-chunk extent | Crossings/sec at 60 m/s | Sub-chunks rolled/sec |
|-------|------------------|-------------------------|-----------------------|
| L0    | 8 m              | 7.5                     | ~68                   |
| L1    | 16 m             | 3.75                    | ~34                   |
| L3    | 64 m             | 0.94                    | ~8                    |
| L6    | 512 m            | 0.12                    | ~1                    |

At 60 m/s (216 km/h) camera motion, the total roll rate across all levels is
~120 sub-chunks/sec ≈ 2 per frame. Per-frame residency diffing, slot
allocation, and index-table updates at that rate are effectively free.

The cadence is inherently self-throttling: finer levels change often but
have smaller sub-chunks to prepare; coarser levels have bigger sub-chunks
but change rarely. No frame-budget knob is needed at the residency-management
layer — the shape of the data structure limits work intrinsically.

### Toroidal Per-Level Storage

Each level owns a fixed slot pool sized for its shell footprint plus
hysteresis. Sub-chunk positions within the shell map to pool slots via a
toroidal (wrap-around) index function:

```
slot_index = (coord.x mod pool_width.x, coord.y mod pool_width.y, coord.z mod pool_width.z)
```

When the camera crosses a boundary at level N, its center in that level's
sub-chunk grid shifts by 1 along one axis. Most slots keep their contents
because the same positions are still inside the shell — only a "strip" on
the trailing face becomes stale and a matching strip on the leading face
needs new content. Strip size: O(shell_face × 1) = O(shell_radius²), not
O(shell_radius³).

Strip entries are enqueued for preparation in the order they roll in. There
is no queue-rebuild pass, no full-shell enumeration, no sort.

### Edit Propagation

Per-voxel edits propagate up the LOD pyramid only through the *ancestor
chain* and terminate as soon as a parent bit doesn't change state. Typical
costs:

- **Place in a region that already had occupancy nearby.** L0 bit set, L1
  parent bit was already 1 (other L0 bits in that 2×2×2 region set), cascade
  terminates after one level. Total: 1 write + 1 check.
- **Break any voxel in a sub-chunk that still has other voxels.** L0 bit
  cleared, L1 parent bit stays 1, cascade terminates. Total: 1 write + 1
  check.
- **Worst case: break the last voxel in an isolated feature.** Cascades all
  the way up. Still O(log N) = ~10 bit flips for a 10-level pyramid.

**Volumetric edits** (explosions, large placements) use a batch OR-merge
cascade instead:

1. Apply all voxel edits into L0 sub-chunks.
2. Mark the set of *dirty L0 sub-chunks* — not individual voxels.
3. Per dirty L0 sub-chunk, OR-reduce its 8³ bitmap into the 4³ = 64 L1
   voxels of its spatial octant within the parent. Mark the L1 sub-chunk
   dirty.
4. Repeat level by level until the dirty set stops propagating.

Cost is independent of edit voxel count: a 10 000-voxel explosion localized
in 5 L0 sub-chunks costs the same as a 100-voxel explosion in those 5
sub-chunks. The OR-reduction kernel is the same one used during initial
generation — generation and edit propagation share the same compute path.
The dirty tracking must be per-sub-chunk (`DirtySet<SubChunkId>`), not
per-voxel, to capture the batch benefit.

### CPU/GPU Data Flow

Two logically distinct GPU pipelines separated by a CPU readback handshake,
implementing the single-writer-to-live-world and CPU-authoritative-allocation
invariants that v2 Control Plane establishes.

```
                      staging area                 live world
                     (GPU-owned)                  (GPU-owned)
                    ─────────────                 ───────────
CPU ──[prep req]─▶  prep shader ─▶[readback]─▶ CPU ──[commit cmd]─▶ commit shader ─▶ live world
                    writes here                    reads + dispatches  writes here
```

**Prep pipeline (GPU-authoritative computation).** CPU issues a prep request
identifying a coord, level, and a request ID. GPU runs worldgen /
OR-reduction / packing and writes the result into a staging buffer. Staging
is read by nothing except the commit shader and the readback.

**Readback.** A compact per-request report lands on CPU 1–2 frames later:
`{request_id, staging_offset, occupancy_summary (empty/full/mixed)}`. CPU
tracks outstanding requests by fence ID; readback reconciles which requests
completed.

**Commit pipeline (CPU-authored, GPU-executed).** CPU consumes the report,
decides slot assignments, and issues a commit dispatch specifying
`(staging_offset → live_slot, coord → live_slot)` tuples. The commit shader
copies staging into the assigned live-world slots *and* updates the
indirection table in the same dispatch, atomically.

**Invariants that avoid the scaffold's failure modes:**

1. **Single writer to live world.** Only the commit shader writes there.
   No `queue.write_buffer` to live data, no direct CPU writes, no secondary
   GPU path mutates it. Live data is read-only to rendering and to prep.
2. **Atomic-per-subchunk commit.** Commit dispatches either complete a slot
   write or haven't started it. Since render is ordered after commit by
   intra-frame barrier (the render graph handles this), rendering reads
   consistent slot data — never torn. Kills the scaffold's
   neighbor-remesh-flicker case by construction (plus the v2 DDA primitive
   has no neighbor re-mesh anyway; only the 6-bit exposure mask depends on
   neighbors and tolerates one frame of staleness).
3. **CPU owns slot allocation; GPU owns slot data.** CPU's shadow ledger is
   the single source of truth for *which slots hold what*. GPU is the
   source of truth for *what's in each slot*. Neither side mutates the
   other's authority. No GPU-side allocation, no GPU-side freeing, no
   GPU-side discovery. Kills the scaffold's GPU-bump-allocator desync by
   construction.
4. **Indirection table and slot data update in one dispatch.** There is no
   frame where the indirection points at a slot that hasn't been populated
   yet, or vice versa.

**Barriers.** Intra-frame (commit → render) is handled by the render graph.
Inter-frame (staging visibility across frames) uses transient buffers; the
specific scheme is a separate concern and is tracked elsewhere.

### Three Prep Input Modes

The prep shader accepts three kinds of input that produce uniformly-shaped
staging output:

| Mode            | Input                              | Use case                                         |
|-----------------|------------------------------------|--------------------------------------------------|
| Generate        | coord, seed, level                 | First load, no edits, purely procedural content  |
| Generate + diff | coord, seed, level, diff buffer    | Chunk has persisted edits on top of procgen      |
| Pure data       | coord, level, raw bitmap+materials | Imports, save loads, edge cases                  |

Generate and generate+diff are the common modes. Pure data is a fallback for
paths that don't fit the procedural model (asset imports, scripted content).
All three produce a staged sub-chunk in the same format; commit doesn't
distinguish.

**Generate+diff degenerate case.** When the diff is empty, the output is
identical to pure generate. The mode distinction is a performance
optimization (skip diff bandwidth), not a correctness requirement.

---

## Extensions

### Persistence Model

Only player edits persist. Everything procedural regenerates deterministically
from `(coord, seed)`. This is the storage invariant that keeps the system
stateless and scalable.

**Per-chunk diff files.** Each chunk that has ever been edited has a
diff file on disk: a list of `(voxel_offset, material)` updates applied on
top of its procgen base. On chunk load, the prep shader runs generate+diff;
on eviction, any dirty diff is flushed to disk.

**Heuristic: diff vs. full replacement.** When a chunk's diff exceeds ~25%
of a full sub-chunk's raw size, collapse it to a pure-data record (full
bitmap + materials). Below that threshold, diffs are pure win; above it,
full data is both smaller and faster to apply.

**What this does not persist.** Procedurally-generated content is never
written to disk. Trees, rocks, villages, dungeons — all re-derived at load
time from the generation pipeline. The seed and coord are sufficient inputs;
every call to the generator produces identical output.

**Implication: generation must be fully deterministic.** Any non-determinism
(time-based seeds, floating-point cross-GPU variance in critical paths,
random atomics) breaks the invariant. Generation shaders are designed as
pure functions of `(coord, seed)`.

### Procedural Generation: Prefab Scatter with Hash-Derived Edges

**Scope decision.** V1 procedural content uses hash-derived prefab
placement exclusively. This is the only approach compatible with
GPU-side generation at scale, and it avoids the persistence complexity that
fully-authored structures (wang tiles, skeleton+expansion) would introduce.

**Why GPU scaling forces this.** Complex generation patterns
(wang tiles, L-systems, WFC) are path-dependent — computing "what voxels
are in chunk C" requires replaying a generation chain from some root. This
is hostile to GPU execution (serial, branchy, allocation-heavy) and forces
either CPU-side generation + GPU upload (throughput-limited by the
CPU→GPU boundary) or per-structure voxel persistence on disk (defeats the
"regenerate deterministically" invariant).

Hash-derived placement inverts the direction of evaluation: every chunk's
content is a pure function of its coordinates. The GPU evaluates it
independently per chunk, in parallel, at streaming bandwidth.

#### Hash-Derived Edges

Features are placed at multiple *cell scales*, independent from the 8³
sub-chunk. A cell at scale S has 6 faces; each face's edge type is
determined by hashing the face's world coordinates:

```
edge_type(face_coord, direction) = hash(face_coord, direction, seed) mod num_edge_types
```

Cells sharing a face hash the same face coordinate, so they independently
agree on the edge type without communication. Each cell then picks a
prefab from the subset of the library compatible with its 6 edges:

```
compatible    = filter(library, cell.edges)
prefab        = compatible[hash(cell_coord, seed, "pick") mod len(compatible)]
```

Within a prefab, further hash-derived parameters (rotation, height,
palette index) yield additional variation from a single template.

The library must provide at least one valid prefab for every possible edge
configuration. This is a content-authoring constraint, not a runtime
concern.

#### Multi-Scale Hierarchy

Multiple cell scales coexist. Coarser scales constrain finer scales by
participating in the finer hash input:

| Scale  | Feature family                      |
|--------|-------------------------------------|
| 64 m   | District / region type              |
| 16 m   | Building / structure placement      |
| 4 m    | Room / sub-structure detail         |
| 8 m    | Tree / rock / small feature scatter |

Each scale has its own prefab library, edge type space, and hash namespace.
Finer scales hash `(cell_coord, seed, parent_type)` where `parent_type` is
the coarser scale's pick at the enclosing cell. A commercial district's
16 m cells pick shop prefabs; a residential district's 16 m cells pick
house prefabs, constrained by the parent.

**Aligned grids.** Cell grids at coarser scales must align with grids at
finer scales (64 m boundaries coincide with 16 m boundaries coincide with
4 m boundaries). This keeps hierarchy composition straightforward. Breaking
alignment is a later refinement, not core.

#### Composition Patterns

Within a single chunk load, multiple structures compose:

- **Main + edge-triggered sub-prefabs per cell.** A cell stamps a primary
  prefab (room) plus conditional sub-prefabs based on each face's edge
  type (door on "door" face, window on "window" face).
- **Chained prefabs across cells.** Roads, walls, pipes — linear features
  where each cell picks a segment prefab based on which faces are "road."
  Neighbors agree on shared edges, so the chain is coherent globally
  without any global planning.
- **Hierarchical stacked scales.** District sets regional context, 16 m
  cells place buildings within that context, 4 m cells furnish rooms.

All evaluation is chunk-local. All storage is zero per-instance.

#### Chunk Evaluation Order

When chunk C loads, its voxels are determined by:

1. Identify cells at each active scale that overlap C.
2. For each cell, hash-derive edges, filter library, hash-pick prefab,
   hash-derive prefab parameters.
3. Stamp each prefab's voxels into C's volume, coarsest scale first
   (district-level geometry), then progressively finer (buildings, then
   rooms, then furniture).
4. Apply C's persisted diff on top (player edits are the top layer).

The order is fixed: coarse procedural → fine procedural → player edits.
This ensures player edits always win over procgen and that finer procgen
overrides coarser where they overlap.

#### Content Authoring

The prefab library is the aesthetic content of the world. The procedural
system is scaffolding; the look-and-feel lives in the library's prefabs and
their edge type declarations.

Per-scale libraries:

- **Small scales (4–16 m).** Authored individually. Hundreds of prefabs
  across houses, rooms, props. Each prefab declares edge types per face.
- **Large scales (64 m+).** Typically parametric/implicit rather than
  authored voxel sets — a "district type" is a rule set that configures
  finer scales rather than a solid voxel shape.

Edge type design is load-bearing. Standardized interface heights, widths,
and alignment conventions ensure prefabs from different authors connect
cleanly. Poor edge design produces visible seams at cell boundaries.

### GPU Worldgen

The prep shader's "generate" mode runs worldgen on GPU. For noise-based
terrain this is straightforward: every layer of the noise stack evaluates
as a pure function of world coord, parallelizes trivially, and runs at
compute-shader bandwidth.

For prefab-based structures, the shader does the following per sub-chunk
at the requested LOD level:

1. Enumerate candidate cells at each scale whose footprint intersects this
   sub-chunk. For each scale, this is `(2 × max_cell_radius / cell_size)³`
   cells — a small fixed number.
2. For each cell, compute edge hash, filter library, pick prefab, compute
   prefab parameters.
3. For each picked prefab, stamp the intersection of the prefab volume with
   this sub-chunk's volume.
4. Composite coarse-to-fine; OR-reduce occupancy and sub-block-pack
   materials for the output staging entry.

The prefab library lives in a GPU buffer (template voxel data + edge type
metadata). The hash functions are a few ALU ops each. Per-sub-chunk cost is
dominated by the stamp step, which is bounded by the total prefab voxel
count intersecting the sub-chunk — small in practice because most prefabs
don't fill most sub-chunks.

### LOD Coarse-Level Generation

A coarse LOD sub-chunk at level N is produced by evaluating worldgen at
level N-1 in its footprint and applying one OR-reduction pass to level N.
Level N-1 is generated directly via scale-aware worldgen, not by recursively
descending to L0.

**Cost is constant per coarse sub-chunk regardless of level.** An L3 target
evaluates 8 L2 sub-chunks = 4096 L2 voxel evaluations. An L6 target
evaluates 8 L5 sub-chunks = 4096 L5 voxel evaluations. Because coarser
voxels evaluate in the same O(1) ALU cost as finer voxels (noise at any
scale is still one noise lookup per voxel; prefabs use precomputed mips),
the per-sub-chunk cost is flat across the LOD pyramid.

**Why not always OR-reduce from L0.** An earlier draft proposed generating
L0 transiently in the prep shader for any coarse target and OR-reducing up
through all intermediate levels. This fails at deep LODs: an L6 sub-chunk
covers 512 m ≈ 134 M L0 voxels, which is hundreds of milliseconds of GPU
work per sub-chunk — infeasible. The "generate at N-1, reduce once" model
caps cost by generating at a scale commensurate with the target rather than
always bottoming out at L0.

**Trade: per-level generation is independent, not a strict OR-reduction of
the level below.** Noise evaluated at L2 scale is not the exact
OR-reduction of noise evaluated at L1 scale — each level's noise stack is
a pure function of coord + seed at that level's voxel size. This introduces
an implicit seam at LOD transition boundaries. The seam sits at the distance
where voxels are already near perceptual acuity (see v2 §Transition
Threshold: ~570–856 m for 1 m voxels transitioning upward), so the
mismatch is at or below what the eye resolves at that angular resolution.
Prefab-based content (which must remain consistent across LODs) handles
this by using precomputed mip chains rather than per-level authoring.

**Prefab LOD mips.** Each prefab in the library is authored at L0 and its
coarse-level representations are baked at asset build time via OR-reduction
from the L0 source. A prefab library entry carries a mip chain from L0
through the deepest usable LOD level. Runtime prefab sampling at level N
is an O(1) lookup into the N-th mip — no runtime OR-reduction of prefab
voxels, no scale-specific authoring. This gives prefabs the "exact across
levels" property that noise cannot have for free.

**Diff projection to coarse LODs.** Diffs are L0-native
(`voxel_offset, material` at 1 m granularity). Projecting an L0 edit to
level N requires OR-reducing L0 voxels in the edit's ancestor chain up to
N-1, then setting the corresponding N-1 voxel accordingly, then proceeding
through the final OR-reduce to N.

Per-edit projection cost scales with the scale ratio from L0 to the target:

| Target | L0 voxels evaluated per edit |
|--------|------------------------------|
| L1     | 8                            |
| L2     | 64                           |
| L3     | 512                          |
| L4     | 4 096                        |
| L5     | 32 768                       |
| L6     | 262 144                      |

This is bounded by edit *count* rather than coarse-sub-chunk footprint.
Localized edits (a broken wall, a placed block) project cheaply.
Dense editing of a 512 m region into a deep coarse sub-chunk is expensive
by construction — but so is any form of fine-detail preservation at that
scale.

**Subvoxel edits disappear by construction.** A 1 m-voxel edit projected to
L5 is 1/32 768 of an L5 voxel — invisible regardless of projection
accuracy. The practical edit-visibility ceiling is "edit footprint ≥ target
LOD voxel size," which is the natural behavior of LOD aggregation, not a
shortcoming of the projection path.

**The prep shader input-mode table in §Three Prep Input Modes remains
accurate**: at coarse LODs, the "generate+diff" mode means "generate at
N-1 in footprint + project diffs to N-1 + OR-reduce to N." The staging
output is at the target level; the N-1 evaluation and diff projection are
implementation details of the prep shader.

---

## Out of Scope

### Wang Tile Structures

Rejected for V1. Wang tile generation is path-dependent — chunk-local
evaluation is impossible without replaying the full generation chain from
a root. Accommodating this requires either CPU-side whole-structure
generation with per-chunk voxel diff persistence (breaks the "only player
edits persist" invariant) or skeleton-format structure data with
per-version forward-compat machinery (reintroduces serialization complexity
that adds no gameplay value).

Hash-derived prefab placement (§Procedural Generation) covers the use cases
wang tiles are typically used for (villages, dungeons, cities) with
equivalent aesthetic coherence, at the cost of giving up designer-authored
connectivity in favor of hash-derived connectivity.

### Skeleton + Expansion

Rejected for the same reasons as wang tiles — any persistence of structural
metadata separate from voxel diffs reintroduces the "assembly instructions"
complexity that's expensive to maintain and yields no meaningful storage
savings over just writing voxel diffs. If structures need to be authored
(not hash-derived), they go through the per-chunk voxel diff path like
player edits.

### GPU-Side Slot Allocation / Immediate Patch

Rejected per v2 Control Plane. The scaffold's GPU bump allocator caused an
unfixable desync; CPU-authoritative allocation eliminates the failure mode
by construction.

A later experiment in immediate-patch allocation is possible — v2's DDA
primitive has no neighbor re-mesh requirement (unlike the quad pipeline),
so the flicker problem that motivated GPU allocation in the scaffold does
not exist under DDA. Fixed-size 8³ sub-chunks also make slot management
much simpler than it was with variable-size quad emissions. If future
profiling shows the one-frame commit latency is objectionable for some
workload, GPU-immediate-patch can be added as an optimization layer over
the current design without changing the invariants.

### Complex Procedural Structures (Wang-Tile-Caliber)

Any structure type that genuinely requires authored designer-controlled
connectivity (narrative-critical dungeons, signature landmarks, story
locations) is handled as a one-off via per-chunk voxel diff persistence,
equivalent to a player edit. These are expected to be rare enough that
the one-off cost is acceptable.

### Content-Aware Residency

The clipmap residency model assigns LOD purely by camera distance. It has no
mechanism for targeting detail at chunks containing interesting content —
large player-built structures, unique landmarks, or gameplay-relevant
features remain at distance-appropriate coarse LOD even if they would be
visually or gameplay-significant at finer detail.

This is a deliberate trade. Content-awareness requires scoring, budgeting,
and eviction logic that reintroduces the per-frame complexity (queue
enumeration, importance ranking, slot competition) that distance-based
clipmap explicitly exists to eliminate.

Three extension paths are plausible if the limitation becomes a real
problem in practice:

1. **Pinned sub-chunks.** Sub-chunks with a persisted diff are added to each
   level's residency set as additional entries beyond the shell. Additive
   over the clipmap — per-level residency becomes `shell ∪ pinned`. Handles
   the common "far-away player build should stay visible" case. The pinned
   set accumulates unboundedly with edits; a cap-and-evict policy is
   required and partially contradicts the "pinned" premise. Simplest of
   the three.

2. **Visibility-driven upgrade.** On-screen chunks in the frustum get
   upgraded to finer LOD than their distance alone would dictate. Couples
   residency to camera orientation, not just position; cadence becomes
   spikier (a head-turn triggers mass residency changes). Catches the "far
   silhouette should be detailed" case but is still content-agnostic.

3. **Importance-weighted residency.** Fixed total residency budget;
   chunks compete for slots via a score (has-diff, in-frustum, is-landmark,
   etc.). Requires a scorer, a budget allocator, and eviction logic that
   weighs score against distance. Substantially more complex; subtle bugs
   in the allocator produce hard-to-diagnose LOD flicker.

Option 1 is the cheapest path if content-aware detail is ever needed and
can be added without restructuring the clipmap core. Options 2 and 3 are
probably never needed and would justify a separate design pass if they
turn out to be.

---

## Relationship to render_pipeline_v2.md

**Supersedes.** Extensions §LOD Hierarchy (8× branching paragraph; within-level
coarsening and transition-threshold sections reconcile — they describe
continuous detail reduction within a level, orthogonal to the inter-level
branching factor).

**Extends.** Core §Control Plane (adds the prep/staging/readback layer that
operates through the batched commit channel), Core §Cull Pass (6-bit
exposure mask survives unchanged; now also serves the acceleration cascade
at every LOD level).

**Unchanged.** Core §Primitive Model (8³ DDA), §Occupancy Format (10³
ghost layer per sub-chunk), §Material Storage (sub-block packing), §Prototype
Milestone (the prototype gates V2 as a whole, including this document's
architecture).
