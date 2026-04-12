# Rejected: Rendering Approaches

Rendering pipelines, resource management strategies, and visual feature approaches
evaluated and rejected, with reasoning. Organized by concern.


## Primary Visibility

### Ray Marching for Primary Visibility

**What:** Use ray marching (via octree, brickmap, or similar) as the primary rendering
method. Every pixel casts a ray and marches through a spatial acceleration structure to
find the first surface hit.

**Why rejected:** Multiple compounding pathological cases for the most common gameplay
camera (player on terrain, looking at the horizon):

- **Fragmented surface shells.** The octree subdivides space uniformly. A geometrically
  simple surface (flat terrain) shatters into thousands of leaf nodes. A horizontal ray
  traverses hundreds of these.
- **Warp divergence.** Rays in the same GPU warp take wildly different step counts. Short
  rays (hit ground nearby) and long rays (travel to distant mountains) share a warp. The
  warp runs at the speed of the slowest ray, wasting 31/63 threads for potentially
  hundreds of steps.
- **Thin feature tunneling.** A 1-block-thick wall at a grazing angle can be skipped
  entirely if the step size is based on the octree node size. Fixing this requires
  conservative step sizes near surfaces, but detecting "near a surface" requires having
  already found it.
- **Register pressure.** Hierarchical traversal maintains ray state, node pointers, and a
  traversal stack. High register usage reduces warp occupancy, amplifying memory stall
  costs.
- **Random memory access.** Octree traversal is pointer chasing through scattered global
  memory. Divergent rays chase different tree paths, thrashing GPU caches.
- **Anti-aliasing cost.** MSAA is near-free with rasterization (dedicated hardware). Ray
  marching requires supersampling (multiple rays per pixel), multiplying cost linearly.

These problems compound. Rasterization has none of them. A triangle at 500 meters costs
the same as a triangle at 5 meters.

**What to use instead:** Indirect rasterization for primary visibility. Ray marching
through the bitmask structure for secondary effects (shadows, AO, GI) where rays are
short-range, structurally coherent, or both.


### Brickmap Demand Streaming for Primary Rendering

**What:** Stream volumetric bricks on demand to the GPU based on a feedback loop (GPU
rays miss → CPU uploads brick → next frame GPU hits it). Fixed memory budget with LRU
eviction.

**Why rejected:** The CPU readback latency is at minimum one frame (more realistically 2–3
with async readback). Fast camera movement causes visible pop-in as bricks load.
Mitigations (conservative preloading, hierarchical fallback, predictive loading) add
complexity and don't fully solve the problem.

More fundamentally, brickmaps store volume data and ray march against it, inheriting all
the ray marching pathologies above.

**What to use instead:** Face bitmasks derived from occupancy. The data is always resident
(~60 KB/chunk, including volumetric material) and trivially small, so there is no demand
streaming problem to solve.


### Near-Field Surface Quads as V2 Primary Path

**What:** Use the greedy-quad pipeline (V1 architecture) as the primary rendering path in
V2, with sub-chunk DDA as a complement or fallback at distance.

**Why rejected:** Sub-chunk DDA subsumes the greedy-quad path as a general primitive: at
near distance the DDA terminates in a handful of steps on the first opaque hit. Adding the
quad pipeline as a primary path would require maintaining two full rendering paths, two
quad-extraction systems, and transitions between them — supporting machinery that exists
only to make the eliminated path efficient.

The quad pipeline remains a benchmark baseline and compatibility target. The Prototype
Milestone in `render_pipeline_v2.md` validates sub-chunk DDA against the greedy-quad path
directly before any decisions are made about the quad path's long-term role.


### Per-Voxel Billboards as Primary Path

**What:** Render each occupied voxel as a camera-facing billboard or depth-offset quad.

**Why rejected:** The 1-voxel case is the 1³ degenerate case of sub-chunk DDA — sparse
sub-chunks terminate their DDA in one step. A dedicated billboard path adds machinery that
the DDA already subsumes, with no separate implementation needed.


## Pipeline Architecture

### Meshlet Pool Architecture

**What:** Virtual-texture-style pool allocator for meshlets. Each chunk allocates
variable-length runs of meshlet slots from a global pool. Re-meshing a dirty chunk frees
old slots and allocates new ones.

**Why rejected:** Requires a mesh generation step (the thing we want to eliminate). Pool
management (allocation, fragmentation, compaction) adds complexity. Meshlet fill rate
varies wildly with surface density, leading to wasted GPU lanes on underfilled meshlets.

The directional face bitmask approach removes pool management entirely. A build compute
shader (running on edit, not per frame) produces a flat, chunk-local quad buffer via
greedy merge on the bitmasks. A filter compute shader runs per frame for visibility and
emits draw commands via MultiDrawIndirect. No pool, no fragmentation, no compaction.

**What to use instead:** Indirect rasterization from GPU-computed greedy quads via
MultiDrawIndirect. One draw call for the entire world.


### Layer Storage / OIT

**What:** Per-pixel layer lists or order-independent transparency for handling transparent
voxels in the primary render path.

**Why rejected:** Intra-sub-chunk transparency is handled inside the DDA shader in natural
depth order. Inter-sub-chunk transparent face rendering uses the quad extraction path with
GPU back-to-front sort — sorted geometry, not per-pixel OIT. OIT adds memory and bandwidth
overhead for a problem the DDA and GPU sort already solve cleanly.


## Scale and LOD

### Impostors

**What:** Cache distant chunks as camera-facing textured billboards to avoid rendering
voxel geometry at extreme distance.

**Why rejected:** The within-level OR-reduction ladder plus stochastic pruning at the tail
covers every distance that impostors were intended to handle. Impostor generation cost,
cache invalidation on chunk edits, and parallax error at oblique angles were solving a
problem the LOD hierarchy already solves with no auxiliary data.


### Life-Scale Planetary Hierarchy

**What:** Extend the cubic voxel hierarchy to 8 levels covering Earth-scale body radius
(~16,000 km spans per level-7 sub-chunk).

**Why rejected:** Levels 5–7 represent content that is either empty (interplanetary space)
or not voxel-shaped at that scale (a planet as 6 voxels is a crude icon, not a
rendering). The cubic hierarchy caps at level 4 (~256 km body diameter). Everything beyond
is handled by non-voxel primitives: procgen function ray-marched analytically for distant
body surfaces, sphere impostors for distant bodies, point sprites for stars and markers.

The target is gameplay-coherent mega-planets, not real-planet simulation.


## Resource Management

### GPU-Authoritative Allocation

**What:** The GPU manages its own quad and material buffer allocation during the build
pass — atomic bump pointer advances, overflow detection, and free-list management entirely
on the GPU without CPU involvement.

**Why rejected:** The root cause of the complexity this was meant to address was a false
requirement: that chunk updates must be live the frame they arrive. Suppressing the cascade
(coalescing chunk arrivals into a single batch commit with one-frame latency, per the V2
control plane) was always the correct fix. Once the cascade is suppressed, CPU-
authoritative allocation is straightforward and avoids the complexity of GPU-side
allocator state.


### GPU-Feedback-Driven Residency / Streaming

**What:** Drive chunk streaming and residency decisions off GPU-side visibility feedback —
the cull pass emits "want finer detail" signals for visible chunks, and the CPU loads
higher-LOD data in response.

**Why rejected:** This pattern works for brickmap-style renderers because global ray
marching naturally requests missing data: a ray that steps into an unloaded brick produces
a miss signal that drives a load. A rasterizer has no equivalent — the cull pass can only
filter the resident set, it cannot request what is not resident.

Trying to drive streaming off occlusion feedback only serves the stationary-player case
and fails on the fast long-distance motion that defines this project. Residency is
distance-based (see `render_pipeline_v2.md` §Control Plane), with no GPU feedback loop.
