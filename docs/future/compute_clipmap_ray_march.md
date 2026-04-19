# Compute Clipmap Ray March

> **Future investigation.** The current MDI + fragment-DDA renderer uses non-overlapping
> clipmap shells (`dim = 2r`) and renders correctly. Overlapping clipmaps — where per-level
> ring radii can be chosen independently rather than constrained to integer multiples of
> coarser sub-chunk extents — would enable perceptually-tuned LOD transitions, but cannot
> be rendered correctly under MDI + fragment-DDA. This document is the output of a design
> session exploring whether a compute world-space ray march replacing MDI is the right
> primitive for overlapping clipmaps.


## Motivation

Non-overlapping clipmaps place level transitions at boundaries fixed by the coarser level's
sub-chunk extent. With `dim = 2r` and power-of-two extent ratios between levels, L0 ring
transitions happen at multiples of the L1 sub-chunk extent, L1 at multiples of L2, and so
on. Ring radii are not independently tunable — the integer-r constraint ties them together.

Independent per-level ring radii (e.g. L0 at 40 m, L1 at 100 m, L2 at 300 m, chosen for
perceptual acuity rather than sub-chunk alignment) require overlapping clipmaps. Under
MDI + fragment-DDA this produces the pathology documented in `failure-lod-ps-discard-holes`:
rasterizing a sub-chunk cube covers pixels the inner DDA does not hit (lateral exits under
grazing rays), so depth-test arbitration between overlapping finer and coarser fragments
lacks a correct rule — PS-level discard either creates holes or per-pixel speckle.

Ring-based level selection as a per-ray function of `t` sidesteps this: the ray consults
exactly one level's data per `t`-range, and the level selection is deterministic, not a
per-sample decision. This requires the primitive to be a world-space ray march, not
rasterization of cube hulls.


## Architecture

Four passes:

    1. Cull.               Producer: per-level 3D integer grid + survivor list.
    2. Tile classifier.    Producer: per-tile bitmap "any candidate covers this tile?"
    3. Primary march.      Consumer: cull grid. Producer: vis buffer + depth.
    4. Deferred shading.   Consumer: vis buffer. Producer: final color.


### 1. Cull

The existing cull pass (frustum + directional exposure mask + Hi-Z) runs unchanged, with
one addition: each surviving sub-chunk writes its occupancy-slot pointer into a per-level
3D grid indexed by clipmap-local coords. Rejected sub-chunks leave a sentinel value.

The grid is a flat structured buffer of `u32`s per level, dimension `(2r)³`. At `r = 2`
with 3 levels, total storage is 192 bytes. Indexing:

    idx = k * (2r)² + j * (2r) + i

where `(i, j, k)` is the sub-chunk's coord in clipmap-local space, derived from the
sub-chunk's world coord relative to the level's corner.

Each sub-chunk writes to its own unique cell — no atomic required. Sentinel cells (empty,
interior, wrong-facing, frustum-rejected, Hi-Z-rejected) are probed cheaply by the outer
DDA in pass 3.

The cull's quad-count predicate is load-bearing. The outer grid's sparseness is inherited
from cull tightness: fully-interior mixed-material sub-chunks (which occupancy-only SVOs
and DAGs keep, because their leaves disagree on material and cannot be merged) are
sentinels here, because they have no exposed faces. This is strictly tighter than an SVO's
"same material" merge rule and is the main voxel-specific optimization over a textbook
clipmap ray marcher.


### 2. Tile Classifier

A coarse compute pass bins each cull-surviving sub-chunk's screen-space AABB into tiles
(e.g., 16×16 px). Output: one bit per tile. Tiles with no candidate receive no primary-
march dispatch. Tiles with at least one candidate are marked active.

Empty-air regions above the horizon dispatch zero work here, but this is not a
differentiator over MDI — the quad-count cull already produces zero MDI instances for air
sub-chunks, and raster naturally dispatches no fragments where no cube covers. The tile
mask's role is to give the compute primary-march path the same property raster naturally
has: no work where no ray could hit anything.


### 3. Primary March

One thread per screen pixel within active tiles; one workgroup per tile. Indirect dispatch
driven by the tile mask.

Per pixel, the ray starts at the camera and walks world space using a two-level nested DDA
per clipmap level:

    outer DDA (across the clipmap's sub-chunk grid):
      at each step, probe the per-level 3D integer grid at clipmap-local (i, j, k)
      if sentinel: step to next cell
      if pointer:  descend to inner DDA

    inner DDA (across the sub-chunk's 8³ voxel bitmap):
      standard DDA through the occupancy bitmap referenced by the pointer
      on opaque voxel hit: write vis buffer entry, terminate ray

Level selection is a function of ray-`t`, not per-cell fallback. For `t ∈ [0, R_L0)` the
ray consults only L0's grid; for `t ∈ [R_L0, R_L1)` only L1's; for `t ∈ [R_L1, R_L2)` only
L2's. Per-step cost: one `u32` load + sentinel compare. Total outer-DDA cost per ray is
linear in levels and bounded by the per-level grid diameter, not by cull-survivor count.

Vis buffer entry: `(sub_chunk_id, local_voxel_index, face)`, ~32 bits. Depth is written to
the regular depth buffer from the voxel hit. Schema rationale in
`note-dda-refactor-design-brief`.


### 4. Deferred Shading

Compute pass over active tiles (reusing the tile mask). Each thread reads its pixel's vis
buffer entry, reconstructs hit position + normal from `(sub_chunk_id, local_voxel_index,
face)`, fetches material through the palette, evaluates lighting, writes final color.

Inactive tiles and sentinel vis entries pay no shading cost.

Material binning (grouping pixels by material id and dispatching one compute per material
for coherent texture fetches) is a future throughput optimization. The vis buffer schema
is neutral to it.


## Ring-Based vs Per-Cell Level Selection

Overlapping clipmaps admit two sampling semantics:

**Per-cell priority fallback.** At each world position, check "is there L0 data here? If
yes, use it. Else L1? Else L2?" Each outer-DDA step pays up to `N_levels` lookups. A ray
traversing an overlap region pays this on every step — O(N_levels²) total per ray. This is
also what produced the `failure-lod-ps-discard-holes` bug class: finer-vs-coarser
arbitration as a per-sample decision assumes cross-level consistency that does not hold
under OR-reduction when viewed through a DDA that can laterally miss.

**Ring-based per-ray t-level selection.** The ray switches levels at `t`-boundaries derived
from the ring radii. For each `t`-range the ray consults exactly one level's grid. Total
cost per ray: sum of per-level grid-walks across the ray's `t`-range = O(N_levels ×
grid_diameter). Linear in levels.

This architecture uses ring-based selection. Functionally equivalent to classical
overlapping-clipmap sampling (finer data wins where present) as long as finer and coarser
data are consistent at the ring boundary, which OR-reduction from finer to coarser (planned
as part of worldgen) enforces.


## Cost Analysis

Three scenarios from the motivating test case (camera at ground in flat plains, mountains
at the horizon):

**Sky pixels (above horizon).** Not covered by any cull-surviving sub-chunk. Tile mask
marks them inactive. Zero dispatch. Equivalent to MDI.

**Ground pixels (camera-adjacent).** Tile mask active (ground sub-chunks touch). Outer DDA
enters the first non-sentinel cell within a few steps; inner DDA hits ground immediately.
~30 total instructions per pixel. Equivalent to MDI in magnitude.

**Horizon pixels (grazing rays).** Tile mask active. Outer DDA walks the clipmap grid,
probing sentinels cheaply. At `r = 2` with 3 levels, the worst-case outer walk is ~36 steps
per ray (sum of per-level grid diameters). First non-sentinel cell triggers inner DDA. In
flat-plains, the first hit is typically the topmost-ground sub-chunk; interior and empty-
air cells are sentinels.

Compare to MDI: the horizon pixel may be covered by many cubes due to LOD overlap or
grazing rays. Each spawns a fragment that runs a full ~24-step inner DDA. Depth-test
arbitrates. Aggregate work is bounded by `cube_count × 24`, parallelized across fragment
invocations.

The compute march trades parallelism across cube instances for per-pixel first-hit
termination. Compute wins on total work in stacked-LOD overlap regions (where multiple
cubes cover the same pixel); MDI wins on per-pixel latency hiding (parallelism across
fragment invocations). Which dominates on this GPU and this workload is empirical and not
settled by this document.


## What Compute Gives Up

1. **Hardware scanline conversion.** Raster binds work to covered pixels via fixed-function
   tile dispatch. Compute replaces this with a tile classifier pass.
2. **Early-Z.** Raster's depth test rejects occluded fragments before shading. Compute
   achieves the equivalent via first-opaque-hit termination inside the ray march — earlier
   in the pipeline, but only for this-pass rays.
3. **Hardware depth write.** Raster writes depth as part of the pipeline. Compute writes
   depth explicitly alongside the vis buffer.
4. **MSAA on hull silhouettes.** Raster antialiases cube edges for free. The voxel
   silhouette is already blocky, so the value is limited.


## What Compute Gains

1. **Correct overlapping clipmap rendering.** The primary motivation. Ring-based per-ray
   `t`-level selection is deterministic; depth-test arbitration under MDI is not.
2. **No hull-false-positive overdraw.** MDI's cube covers pixels the inner DDA does not
   hit; the fragment still marches. Compute's first-hit termination eliminates wasted
   marches on pixels the nearest cube would have laterally missed.
3. **Front-to-back transparency accumulation.** Natural in a march, awkward in MDI (needs
   A-buffer / k-buffer).
4. **Shared primitive with secondary rays.** Shadow / GI / reflection rays use compute
   march regardless. Primary-ray compute march unifies the primitive; the inner-DDA
   function is the shared piece.
5. **Deterministic level selection.** No per-pixel arbitration bugs.


## Honest Positioning

The architecture converges on compute world-space clipmap ray marching. This is a
well-understood technique (Teardown, various compute voxel engines, brickmap-style
renderers). The design-session journey pressure-tested several supposed differentiators
and retracted those that did not survive:

- **Sky cost is not a win over MDI.** Cull's quad-count predicate handles this in both
  architectures.
- **Horizon ray cost scales with grid diameter, not candidate count.** True for any
  clipmap world-space march — not specific to this engine.
- **Tile mask for dispatch bounding.** Standard compute-march optimization.

What is specific to this engine: the outer grid's sparsity is driven by the cull's
quad-count predicate (bitmask-vs-neighbors face-count), which removes fully-interior
mixed-material sub-chunks. Occupancy-only SVOs and DAGs keep these; this engine's cull
drops them. That is the one place the outer grid is tighter than a textbook clipmap ray
marcher's would be.

Everything else is standard. The decision to replace MDI is driven by whether overlapping
clipmaps are required, not by performance claims over MDI on this-project-specific
workloads.


## Open Questions

- **Measurement.** The architecture is only justified if the MDI + nested-shell path is
  insufficient for the LOD transition tuning we want. Prototype both and measure on the
  target workloads (ground-plains, horizon-mountains, dense-forest, cave-system).
- **Hi-Z feedback loop.** Current Hi-Z source is the MDI depth buffer. Under compute march
  the depth is still written (from vis buffer hits), so Hi-Z rebuild is mechanical, but
  the one-frame latency shape should be verified against the new pass ordering.
- **Tile classifier cost.** The classifier pass is proportional to `cull_survivors ×
  avg_tiles_per_AABB`. Needs cost validation vs. the saving it provides — especially for
  very-near sub-chunks whose screen AABB covers the entire screen.
- **Cross-level consistency at ring boundaries.** Ring-based level selection assumes L0
  and L1 data agree at `R_L0`. OR-reduction from finer to coarser (planned in worldgen)
  enforces this. Ensure the rendering-side ring transition matches the propagation
  direction.
- **Workgroup-shared sub-chunk prefetch.** Threads in a tile have near-parallel rays that
  visit approximately the same sequence of sub-chunks. Prefetching the first sub-chunk's
  inner bitmap into LDS per workgroup should be cheap and reduce redundant global loads.
  Implementation detail to measure.
- **Transparent voxel handling.** Vis buffer is designed for opaque first-hit. Transparency
  accumulation requires either a second vis buffer (transparent chain) or in-march
  accumulation that skips the vis buffer entirely for transparent pixels. Out of scope for
  V1 of this primitive; flagged here because compute march is what makes it tractable.
- **Secondary ray sharing.** The inner-DDA function factors cleanly into a reusable HLSL
  primitive. The outer-DDA factors less cleanly because secondary rays do not start from
  the camera; they start from arbitrary world positions and may need the full `N_levels`
  grid walk rather than a ray-`t` ring partition. Design brief in
  `note-dda-refactor-design-brief` covers this.


## Relationship to Existing Work

- `decision-render-pipeline-v2` — V2 pipeline; sub-chunk DDA as single primitive.
- `decision-lod-nested-shells-hierarchical-occupancy` — current non-overlapping clipmap
  design.
- `failure-lod-ps-discard-holes` — the bug class that motivates moving off PS-level LOD
  arbitration.
- `note-dda-refactor-design-brief` — vis buffer schema + general compute DDA
  generalization.
- `note-lighting-architecture` — deferred shading via vis buffer.
- `docs/future/scene_acceleration.md` — two-phase bitmask + hardware RT traversal;
  related traversal shape for dynamic objects.
