# GPU Memory Architecture

Design for a unified GPU memory system that resolves the ownership conflicts,
pre-allocation waste, and sync stalls in the current rendering pipeline.

> **Partial supersession (2026-04-10):** The GPU-side allocator, quad storage,
> cull cascade, far-field rendering as a distinct path, occlusion-driven streaming,
> LOD decimation, layered generation, and chunk lifecycle sections have been
> superseded by `render_pipeline_v2.md` and `renderer_rewrite_principles.md` —
> marked **[SUPERSEDED]** below. The material storage model (§Material Storage),
> async feedback principles, and occupancy discussion remain in effect and are
> used directly by V2.


## Problem Statement

The current pipeline splits memory management authority between the CPU and GPU.
The CPU allocates blocks from the quad pool before the build shader runs, but
only the GPU knows how many blocks are needed (the greedy merge output is
non-deterministic). This forces a round-trip:

```
CPU allocates blocks (guess)
  -> GPU runs build shader, fills some blocks
    -> CPU stalls, reads back quad count
      -> CPU trims excess, zeros stale page table entries
        -> if overflow: CPU re-allocates, re-dispatches, stalls again
```

Every structural bug that had to be patched traces back to this split:

- **Stale page table entries:** CPU trims blocks, GPU sees entries pointing to
  blocks now owned by another chunk.
- **Sync stalls:** CPU needs GPU's quad count to manage blocks it allocated.
  `device.poll(wait_indefinitely)` per chunk per rebuild.
- **Overflow-retry:** CPU guessed wrong, must detect and re-dispatch.
- **Load queue stalls:** CPU rate-limits enumeration based on block budget it
  controls.

Beyond the ownership split, per-chunk GPU resources compound the problem. Each
loaded chunk creates 3 dedicated buffers (occupancy, quad count, count staging)
and a bind group. That is per-chunk device allocation on the hot path.

Fixed slot counts force worst-case pre-allocation. `MAX_CHUNKS = 4096` means
128 MB of material volume even though most chunks are empty air (no material
needed) or solid interior (one material ID for the whole volume, never sampled
by any fragment).


## Design Principles

### 1. Producer owns allocation

The entity that produces data owns its allocation lifecycle.

- **CPU produces** source data (occupancy, material edits, world offsets).
  Fixed-size per chunk. CPU manages these.
- **GPU produces** derived data (quads, material packing, draw args). Variable-
  size, known only after compute. GPU manages these.

### 2. Async feedback, never sync readback

The GPU writes utilization metadata (bump pointer levels, overflow flags,
per-chunk quad counts) into a shared metadata buffer. The CPU reads it
asynchronously with one-frame latency via `map_async` callback. The CPU never
stalls waiting for GPU output.

### 3. Segmented growth, never reallocation

GPU buffers grow by appending new segments (64 MB each). When the GPU reports
low free capacity, the CPU allocates a new segment and rebuilds the bind group.
No data moves. No rebuild. No stall.

### 4. Allocate only what rendering touches

Quad and material storage is driven by visible face output. Interior sub-blocks
that contribute no visible faces cost zero storage, even if they contain
occupied voxels.


## [SUPERSEDED] Contiguous Allocation Model

*GPU-side bump allocator and free list rolled back. CPU-authoritative allocation per `renderer_rewrite_principles.md`.*

Both quad and material storage use the same principle: chunks own contiguous
ranges in shared buffers, not scattered blocks. The page table and block pool
indirection that characterized the previous design are eliminated.

### Why contiguity matters

The previous block pool design stored quads in fixed-size 256-quad blocks
scattered across a shared pool. A page table mapped virtual block indices to
physical locations. The vertex shader performed `page_table[instance_index / 256]`
on every invocation -- a dependent memory read on the hottest shader path.

The block pool existed because output size is unpredictable (greedy merge
produces a variable number of quads). But this is a sizing problem, not an
addressing problem. Solving it at the allocator level (bump allocation with
periodic compaction) eliminates the per-vertex indirection entirely.

The same logic applies to material storage. The previous sparse sub-block model
used a block table with pool pointers, requiring a dependent memory read in the
fragment shader -- an even more critical path since fragments vastly outnumber
vertices.

### The allocator: hybrid bump + free list

**Bump allocation (hot path):** New chunks advance a GPU-side atomic bump
pointer. Contiguous by construction. Zero fragmentation on initial load.

**Block-granular free list (cold path):** When a chunk is unloaded or rebuilt
with a different size, its range becomes a hole. Free space is tracked in
block-sized units (e.g., 256 quads = 1 KB for quads, 512 bytes for material).
New allocations that cannot be satisfied by the bump pointer search the free
list for a contiguous run of blocks.

**Compaction:** Periodically (or when fragmentation exceeds a threshold), a
GPU compute pass slides chunks down to fill holes. A chunk with 300 quads is
1.2 KB. Moving a thousand chunks is moving megabytes, which a GPU does in
microseconds. After compaction, the CPU updates `base_offset` for moved chunks
and the cull shader rebuilds MDI entries (which it does every frame anyway).

### Segmented growth

Each buffer is an array of equal-size segments:

```
Segment 0:  [64 MB, allocated at startup]
Segment 1:  [64 MB, allocated on demand]
Segment 2:  ...
```

Shader access uses a per-chunk `buffer_index` to select the segment. Segments
are bound via `BUFFER_BINDING_ARRAY` (wgpu feature, widely supported on
desktop) or a fixed number of binding slots with lazy allocation for
portability.

When the CPU allocates a new segment:
1. Create a new wgpu buffer.
2. Rebuild the bind group to include the new segment.
3. New chunk allocations target the new segment's bump pointer.

No data moves. Existing segments are untouched.


## [SUPERSEDED] Quad Storage

*Quads replaced by sub-chunk DDA. Quad buffers, build shader passes, and the cull cascade below are no longer part of the architecture.*

Each chunk owns a contiguous range in the quad buffer. Within that range, quads
are ordered by face direction.

### Layout

```
Quad buffer (contiguous per chunk):
[---- chunk A ----][---- chunk B ----][---- chunk C ----] ...

Within chunk A:
[+X quads][−X quads][+Y quads][−Y quads][+Z quads][−Z quads]
```

### Per-chunk metadata

```
buffer_index:      u32              // which segment in the multi-buffer array
base_offset:       u32              // starting quad index within that segment
dir_layer_counts:  [[u32; 32]; 6]   // quad count per (direction, layer)
```

~776 bytes per chunk. `dir_counts[d]` is `sum(dir_layer_counts[d])`. Direction
ranges and layer ranges within directions are all prefix sums from this table.
This enables layer-level culling.

Direction d spans `[base + sum(all dir_layer_counts for dirs 0..d-1), ...)`.
Layer L within direction d spans a sub-range addressable from the prefix sums.

### Vertex shader

```wgsl
let quad = quads[instance_index];  // direct indexing, no page table
```

One buffer read. No division, no indirection, no page table lookup.

### Quad descriptor format

Direction is no longer encoded in the quad. It is implicit in the memory
layout -- the cull shader knows which direction each MDI entry covers. Combined
with the 4 previously unused high bits, 7 bits are freed compared to the
previous 32-bit format.

| Bits  | Field   | Width |
|-------|---------|-------|
| 0-4   | col     | 5     |
| 5-9   | row     | 5     |
| 10-14 | layer   | 5     |
| 15-19 | width-1 | 5     |
| 20-24 | height-1| 5     |
| 25-31 | free    | 7     |

The 7 free bits are available for future use (LOD flags, material hints, etc).

### Build shader

The build shader uses a two-pass design. A single dispatch of `(32, 6, 1)`
workgroups cannot produce direction-ordered output because there are no
cross-workgroup barriers. Direction d's write region depends on directions
0..d-1 finishing.

**Pass 1 (count):** same face derivation and greedy merge, but only increments
per-(direction, layer) atomic counters. Produces `dir_layer_counts: [[u32; 32]; 6]`
per chunk.

**Between passes:** prefix-sum the counts to determine write positions for each
(direction, layer) bucket. Atomically advance the quad bump pointer by the
total count.

**Pass 2 (write):** re-derives quads and writes them to the correct positions
in the contiguous range. Quads are direction-ordered and layer-sorted within
each direction.

**In-place rebuild:** if the new total from pass 1 fits within the old
allocation, skip bump pointer advance and write in-place. Common for small
edits.

**Batched builds:** both passes naturally support batching. Pass 1 counts all
dirty chunks in one dispatch. Prefix sums are computed for all chunks. Pass 2
writes all chunks. Two dispatches total regardless of batch size.

### Cull Cascade

Four-stage pipeline. Each stage reduces the working set for the next.

**Stage 1: Direction cull.** Dot product of camera position vs chunk center
along each axis. Back-facing directions skipped. ~50% average rejection. One
instruction per direction per chunk.

**Stage 2: Layer cull (frustum).** For each direction, compute the visible
world layer range from frustum plane intersection. Intersect with each chunk's
layer range (chunk_coord * 32 .. chunk_coord * 32 + 31). Skip chunks with no
overlap. Narrow MDI entries to the visible layer sub-range. Pure arithmetic
from per-chunk metadata.

**Stage 3: Layer cull (Hi-Z).** For surviving (chunk, direction) pairs, test
individual layer planes against the Hi-Z depth buffer. Each layer is a plane
at a known world coordinate with a known screen-space extent. Reject occluded
layers.

This subsumes software backface culling entirely. The common optimization of
"camera is above chunk, skip bottom faces" is a degenerate case: the chunk's
own top surface occludes its bottom in the depth buffer. Hi-Z handles this
plus non-trivial occlusion (hills, caves, canyons) with no special-case code.

**Stage 4: Quad cull (Hi-Z).** Surviving quads tested individually. Compute
each quad's screen-space rect from its position and greedy merge size, sample
the appropriate Hi-Z mip. Stream-compact survivors into the final draw buffer.
Catches quads within a visible layer that are still occluded (cave openings,
windows, partial occlusion).

The cascade:
```
Direction cull -> Layer cull (frustum) -> Layer cull (Hi-Z) -> Quad cull (Hi-Z) -> Draw
  (6 tests/chunk)  (~32 tests/dir)       (per layer)          (per quad)          (survivors)
```

Each MDI entry after culling:
```
first_instance = base_offset + prefix_sum(dir_layer_counts up to visible start)
instance_count = sum(dir_layer_counts for visible layer range)
vertex_count   = 6
```


## Material Storage

Material storage uses the same contiguous allocation model as quads, but with
visibility-driven sub-block packing.

### Motivation

The fragment shader samples material identity per pixel. In the previous sparse
sub-block design, this required a block table lookup (dependent memory read) +
branch (null/homogeneous/pool) + possible pool read. Fragments vastly outnumber
vertices, making this indirection more costly than the quad page table it
replaces.

The previous design's optimizations (homogeneous sentinels, null entries) saved
memory for empty and solid-interior sub-blocks. But the fragment shader only
runs on fragments with visible faces. Those fragments only sample sub-blocks
that contain visible faces. Empty and solid-interior sub-blocks are never
sampled -- the optimizations existed for cases that never reach the fragment
shader.

### Visibility-driven packing

A 32x32x32 chunk is divided into a 4x4x4 grid of 8x8x8 sub-blocks (64 total).
The build shader determines which sub-blocks contain at least one visible face
and records this as a 64-bit bitmask (`sub_mask`). Only populated sub-blocks
are stored, packed contiguously:

```
chunk_material_base ->  [sub-block 3:  512 B]
                        [sub-block 12: 512 B]
                        [sub-block 28: 512 B]
                        [sub-block 45: 512 B]
                        total: 2048 B (vs 32 KB for full flat volume)
```

Each 512-byte entry is a dense 8x8x8 array of `u8` palette indices.

### Fragment shader resolution

The vertex shader passes `chunk_material_base` and `sub_mask` as flat varyings
(fetched once per quad, free for all fragments):

```wgsl
let bx = vx / 8u;  let by = vy / 8u;  let bz = vz / 8u;
let lx = vx % 8u;  let ly = vy % 8u;  let lz = vz % 8u;

let sub_idx = bz * 16u + by * 4u + bx;
let offset  = countOneBits(sub_mask & ((1u << sub_idx) - 1u));

let material = material_buf[chunk_material_base + offset * 512u + lz * 64u + ly * 8u + lx];
```

No branch. No block table read. One `countOneBits` (single ALU cycle on modern
GPUs) + one direct buffer read. The bitmask is in a register from the flat
varying.

### Why homogeneous optimization is unnecessary

The previous design needed homogeneous sentinels for sub-blocks where every
voxel had the same material. This model allocates based on visibility, not
occupancy:

- All air: no quads, no storage.
- All solid interior: no quads (faces culled by neighbors), no storage.
- Mixed but fully buried: no quads (all faces adjacent to solid neighbors),
  no storage. The previous model allocated a pool block for these.
- Mixed with visible faces: quads emitted, storage allocated.

The visibility test is strictly tighter than the occupancy test. Cases the
homogeneous optimization handled are absent from the data entirely.

### Material packing flow

1. CPU uploads the full 32 KB material volume to a transient staging buffer
   (reused across chunks).
2. Build shader runs, emits quads, marks populated sub-blocks via `atomicOr`
   on a shared `u64`.
3. Build shader atomically advances the material bump pointer by
   `popcount(sub_mask) * 512` bytes. Writes `chunk_material_base` and
   `sub_mask` to per-chunk metadata.
4. Material-pack compute pass (same command buffer, after build): reads the
   bitmask, iterates over set bits, copies each populated 8x8x8 region from
   staging into the packed range.

All GPU-side, single submission, no CPU readback.

### Memory usage

| Chunk type | Populated sub-blocks | Material cost |
|---|---|---|
| All air (empty) | 0 | 0 B |
| Solid interior (fully buried) | 0 | 0 B |
| Flat terrain surface | ~16 of 64 | ~8 KB |
| Rolling hills surface | ~24-32 of 64 | ~12-16 KB |
| Very jagged surface | ~40-48 of 64 | ~20-24 KB |
| Worst case (3D checkerboard) | 64 of 64 | 32 KB |

For 4096 loaded chunks with ~1200 surface chunks averaging ~20 populated
sub-blocks: ~12 MB. Compared to 128 MB for the current flat allocation.


## Per-Slot Data

A chunk "slot" is a lightweight index with minimal fixed-cost storage:

```
Chunk slot (index)
  |
  +-- Quad range     (~776 B: buffer_index, base_offset, dir_layer_counts[[32]; 6])
  +-- Material range (12 B: buffer_index, base_offset, sub_mask as 2x u32)
  +-- Chunk meta     (16 B: total quad count, flags)
  +-- Boundary cache (768 B: 6 x 32 words, neighbor surfaces)
  +-- World offset   (16 B: ivec4 chunk position)
```

Fixed cost per slot: ~1588 B. An empty chunk costs this and nothing else. Only
surface chunks allocate from the quad and material buffers.

### Shared buffers (slot-indexed)

| Buffer | Size (4096 slots) | Contents |
|--------|-------------------|----------|
| `quad_range_buf` | 4096 x 776 B = ~3.1 MB | Per-chunk quad base + direction-layer counts |
| `material_range_buf` | 4096 x 12 B = 48 KB | Per-chunk material base + sub-block mask |
| `chunk_meta_buf` | 4096 x 16 B = 64 KB | Quad count, flags |
| `boundary_cache_buf` | 4096 x 768 B = 3 MB | Neighbor boundary slices |
| `chunk_offset_buf` | 4096 x 16 B = 64 KB | World position |


## Occupancy

The occupancy bitmask (4 KB per chunk) is needed only during the build pass:
face derivation reads the current chunk's occupancy and neighbor boundaries.
After quad generation, occupancy is not referenced by the render or cull
passes.

Two options:

**a) Persist occupancy (current approach, slot-indexed):**
16 MB for 4096 slots. Simple. The build shader reads neighbor occupancy
directly from the shared buffer using neighbor slot indices passed via push
constants. No CPU-side boundary extraction.

**b) Transient occupancy, persistent boundary cache:**
Upload the current chunk's occupancy to a small staging buffer for each build
dispatch. Persist only the boundary slices (768 bytes per chunk, 6 x 32 words)
for neighbor reads. The build shader reads its own occupancy from the staging
buffer and neighbor boundaries from the persistent cache. Saves ~13 MB but adds
a CPU-side boundary extraction step on every chunk load/edit.

Option (a) is simpler and the 16 MB cost is minor compared to the savings from
sparse material. Option (b) is available if memory pressure demands it.

Either way, the build shader should read neighbor occupancy from GPU-resident
data (whether full occupancy or boundary cache), not from a per-chunk upload of
CPU-extracted slices. This eliminates `build_neighbor_slices` and the per-chunk
1216-word occupancy buffer.


## Feedback Loop

### GPU -> CPU metadata

The GPU writes to `chunk_meta_buf` during the build pass:

```
struct ChunkMeta {
    quad_count   : u32,   // total quads emitted by greedy merge
    flags        : u32,   // overflow bits, etc.
    _reserved    : u32,
    _reserved    : u32,
}
```

A separate global metadata region reports bump pointer levels and pool
utilization.

### CPU reads

The CPU maps a staging copy of the metadata buffer asynchronously (same
mechanism as the existing visible-count readback). One-frame latency. No stall.

**Decisions driven by feedback:**

| Signal | CPU response |
|--------|-------------|
| Bump pointer near segment end | Allocate new segment |
| Overflow flag set | Allocate new segment, mark chunk dirty for rebuild |
| Fragmentation above threshold | Schedule compaction pass |
| Chunk quad count available | Diagnostics, statistics overlay |

The CPU never needs to know which specific ranges a chunk owns. It only needs
pool-level utilization to decide when to grow or compact.

### Overflow

If the bump pointer exceeds the current segment during a build, the shader
writes quads/material up to the segment boundary and sets the overflow flag in
chunk meta. The chunk renders with missing quads for one frame. The CPU sees
the flag, grows the pool, and the chunk rebuilds next frame.

This is a graceful degradation path, not an error. The one-frame glitch is
imperceptible.


## [SUPERSEDED] Chunk Lifecycle

*Control plane design superseded. CPU-authoritative batched commits with one-frame latency per `render_pipeline_v2.md` §Core/Control Plane.*

### Load

1. CPU pops a slot index from the free slot list.
2. CPU uploads occupancy to `occupancy_buf[slot]` (or staging area).
3. CPU uploads material volume to transient staging buffer.
4. CPU writes chunk offset to `chunk_offset_buf[slot]`.
5. CPU marks slot dirty.

### Build (GPU, on dirty)

1. Pass 1: build shader reads occupancy and neighbor boundaries. Face
   derivation and greedy merge count quads per (direction, layer). Writes
   `dir_layer_counts` to metadata.
2. Between passes: prefix-sum `dir_layer_counts` to determine write positions.
   Atomically advance quad bump pointer (or reuse in-place if new total fits
   old allocation).
3. Pass 2: build shader writes quads to computed positions. Quads are
   direction-ordered and layer-sorted.
4. Build shader determines populated sub-blocks, advances material bump
   pointer, writes `sub_mask`.
5. Material-pack compute pass copies populated sub-blocks from staging.
6. Build shader writes chunk meta.

### Edit

1. CPU uploads modified occupancy and material to slot-indexed buffers (or
   staging).
2. CPU marks slot dirty.
3. Build pass runs. Old quad and material ranges are returned to the free
   list. New ranges allocated via bump pointer. If the new total fits within
   the old allocation, the build writes in-place without advancing the bump
   pointer.

### Unload

1. CPU returns the chunk's quad and material ranges to the free list.
2. CPU returns the slot index to the free slot list.
3. No GPU dispatch needed for freeing (ranges are tracked CPU-side as
   offset + size).

### Growth

1. CPU observes bump pointer near segment end via async feedback.
2. CPU creates a new buffer segment.
3. CPU rebuilds the bind group to include the new segment.
4. New allocations target the new segment.
5. No stalls, no data movement, no rebuild.


## [SUPERSEDED] LOD via Decimation

*Replaced by OR-reduction hierarchy. See `render_pipeline_v2.md` §LOD Hierarchy.*

Level of detail reduces quad count at distance without reducing grid resolution.
The occupancy grid stays 32x32x32. Only the surface geometry is simplified.

### Why not downsample

Uniform downsampling (32x32x32 to 16x16x16) destroys thin features. A 1-voxel
pillar either disappears (AND rule) or expands to double width (OR rule). This
is information-theoretic, not algorithmic: a 1-voxel feature cannot exist in a
half-resolution regular grid regardless of the downsampling method.

### Topology-preserving decimation

Instead of reducing resolution, iteratively remove surface voxels that disrupt
greedy merge efficiency while preserving structural features through topology
constraints.

**Merge disruption metric.** The energy function measures how much a voxel
prevents neighboring faces from merging into larger rectangles. A 1-voxel bump
on flat terrain has high disruption: it scatters faces across multiple layers
and breaks rectangular merging in each. A pillar voxel has low disruption: its
side faces already merge into tall thin rectangles.

**Simple point test.** A voxel can only be removed if doing so does not change
the topology of the solid: no disconnections, no holes. A pillar's voxels are
topologically protected because removing any one disconnects the column.

**GPU implementation.** Sub-iteration parallel thinning divides voxels into 8
subsets by xyz parity. Within a subset, no two voxels are 26-adjacent, so all
can be tested and removed simultaneously without race conditions. Eight
sub-iterations = one full pass. A few passes with decreasing importance
thresholds produce progressive LOD levels.

### Pipeline integration

The decimation pass is a compute shader that runs before the build shader:

```
Occupancy (32x32x32, unchanged)
  -> Decimation pass (produces smoothed occupancy)
    -> Build pass 1 (count, unchanged)
      -> Build pass 2 (write, unchanged)
        -> Quads with vertex color, no material sub-blocks
```

The build shader is unchanged. It receives a smoother input and produces fewer
quads. LOD chunks skip material packing entirely. The fragment shader reads a
per-quad averaged color instead of sampling the material volume.

### Natural behavior

Mountain slope detail: many small height variations, each with high merge
disruption and low individual importance. Decimated first, producing large
merged quads.

Single-voxel pillar: topologically protected (cannot remove without
disconnecting), low merge disruption (already merges well). Survives even
aggressive decimation budgets without explicit special-case logic.

Flat terrain: already merges perfectly. Unaffected by decimation.


## [SUPERSEDED] Far-Field Rendering

*Merged into the primary path. Sub-chunk DDA is the rendering primitive at all distances; there is no separate near/far split.*

Beyond the effective range of rasterization, chunks are rendered via bounded
ray marching through the occupancy bitmask. No build shader, no greedy merge,
no material storage.

### Transition threshold

The crossover distance where rasterization becomes inefficient is determined by
perceptual acuity, not pixel count. Display resolution cancels out of the
formula entirely. Only physical screen size, viewing distance, and field of
view matter:

```
d = voxel_size * screen_phys_width / (2 * acuity * tan(fov / 2) * viewing_distance)
```

For a 27" monitor at 60 cm, 90 degree FOV, 1 m voxels:

| Acuity | Distance | Context |
|---|---|---|
| 1 arcminute | 1713 m | Perfect vision, static target |
| 2 arcminute | 856 m | Relaxed viewing |
| 3 arcminute | 571 m | Motion, typical gaming |

Practical gaming threshold: 600-850 m. A single user-facing "detail distance"
slider. Display-independent.

### Why ray marching works at distance

The pathological cases that make full-scene ray marching unsuitable for primary
visibility (see `design_rejections.md`) do not apply to chunk-bounded marching:

- **No tree traversal.** The rasterizer handles spatial sorting. Each fragment
  knows exactly which chunk it tests.
- **No cross-chunk warp divergence.** Every fragment in a warp tests the same
  chunk's data.
- **Bounded within-chunk divergence.** At most 55 steps (32x32x32 diagonal).
  At distance, rays are nearly parallel, so adjacent fragments take similar
  step counts.
- **No random memory access.** One chunk's occupancy is 4 KB. Fits in L1.
  Adjacent fragments read adjacent columns.
- **No register pressure.** Ray position, direction, current cell. No stack,
  no node pointers.
- **MSAA works.** Bounding box depth is used instead of per-voxel
  `gl_FragDepth`, preserving early-Z and hardware multisampling. At distance,
  the 32-voxel depth range within a chunk is negligible.

### Rendering path

1. Rasterize each far chunk as its bounding cube.
2. Fragment shader: DDA through the chunk's 32x32x32 occupancy bitmask.
3. On hit: output solid averaged color. On miss: discard.
4. Depth: bounding box depth from rasterizer, not per-voxel.

No build shader dispatch. No quad buffer allocation. No material sub-blocks.
A far chunk needs only its 4 KB occupancy bitmask and an averaged color.


## [SUPERSEDED] Streaming Hierarchy

*Occlusion-driven residency rejected. Replaced by distance-based residency. See `render_pipeline_v2.md` §Core/Control Plane.*

Chunk data residency is a continuous hierarchy, not discrete LOD tiers. Every
level is a valid renderable. Coarser levels are always available as fallback.
Nothing pops into existence because something is always there to display.

### Hierarchy levels

```
Level 0:  Chunk averaged color (~4 B). Solid-colored cube.
Level 1:  8x8x8 occupancy + color (~576 B). Ray marched.
Level 2:  32x32x32 occupancy + averaged color (~4 KB). Ray marched or
          rasterized with decimation.
Level 3:  32x32x32 occupancy + full material. Full rasterization pipeline.
```

### Occlusion-driven promotion

The cull cascade determines per-chunk visibility every frame. That visibility
signal feeds back to the CPU (async, one-frame latency) and drives streaming
priority:

- Chunk becomes visible: promote to higher detail level.
- Chunk becomes occluded: evict to coarser level, free memory.
- Chunk approaches camera: promote further.
- Chunk recedes: demote.

Material storage can be freed for occluded chunks while keeping the quad range
and slot metadata resident. Material is re-packed when the chunk becomes
visible again. Chunks at higher LOD levels skip material entirely.

### Imposter caching

A chunk rendered via ray march can cache its result as a small texture mapped
onto the bounding cube. Subsequent frames draw the textured cube with no ray
march. Invalidated when the chunk's data changes or the camera angle shifts
beyond a threshold. Sub-chunk imposters (octants) improve parallax fidelity
at the cost of 8x storage and management.


## [SUPERSEDED] Layered Generation

*Streaming lifecycle depended on occlusion-driven promotion, which is superseded. GPU terrain fast path may still be relevant as an extension; see `render_pipeline_v2.md` §Procgen Fast Path.*

Terrain generation is split between GPU and CPU to avoid the dual source of
truth problem. Each layer has exactly one authoritative source.

### The problem

If both CPU and GPU independently evaluate the same noise function, floating
point non-determinism (different hardware, instruction ordering, FMA behavior,
transcendental implementations) produces slightly different occupancy at surface
boundaries. At distance the mismatch is invisible, but the transition as a
chunk promotes from GPU-generated to CPU-generated causes visible shimmer.

### Layer model

```
Layer 0 (GPU):  Terrain shape. Noise -> occupancy. GPU is sole authority.
Layer 1 (CPU):  Structure placement. Trees, rocks, buildings. Applied as
                diffs on top of GPU occupancy.
Layer 2 (CPU):  Material assignment. Per-voxel material identity. Only needed
                at near LOD when the material texture is sampled.
Layer 3 (CPU):  Player edits. Persistent modifications stored as diffs.
```

### Why structures stay on CPU

Structure placement is a coordination problem. Each placement depends on what
has already been placed nearby (spacing, collision, terrain fit). A tree needs
to find a valid surface position, check for conflicts, stamp a 3D template
that may span chunk boundaries. This is fundamentally serial and requires
global knowledge that the GPU's local-only parallel model cannot provide.

Simple scatter (rocks, small props) could theoretically run on GPU since each
placement is independent, but these are unnecessary at far LOD distance.

### CPU decision layer

The CPU maintains a coarse spatial index of features that require full
generation: large structures placed via voronoi or disk sampling, player
modifications, biome boundaries. This index is precomputed and trivially
cheap to query.

```
Per chunk load decision:
  No features nearby + unmodified  ->  GPU fast path (immediate)
  Features nearby or modified      ->  CPU generation (queued, budget-limited)
```

### GPU fast path

A single compute dispatch batch-generates occupancy for hundreds of far chunks
in parallel. No CPU-GPU roundtrip. No upload. No per-frame budget throttling.
The occupancy goes straight into the ray march path without touching CPU
memory.

### Streaming lifecycle

```
Far:          GPU generates 8x8x8 coarse terrain. Ray march. Immediate.
Approaching:  GPU generates 32x32x32 terrain. Ray march or rasterize.
Near:         CPU applies structure diffs (layer 1). Build shader.
Close:        CPU applies material (layer 2) + edits (layer 3). Full pipeline.
```

The coarse generation fills the view almost instantly. Detail refines
progressively as chunks promote through the hierarchy. No pop-in because the
coarse version is already on screen before the detailed version is needed.


## What This Eliminates

| Current | New |
|---------|-----|
| Per-chunk `occupancy_buf` (4864 B each) | Shared slot-indexed buffer |
| Per-chunk `quad_count_buf` + `count_staging` | `chunk_meta_buf` field, GPU-written |
| Per-chunk bind group | One shared bind group, slot via push constant |
| CPU `free_blocks` stack + `ChunkAlloc` | GPU-side bump allocator |
| Page table (6 MB, vertex shader reads every invocation) | Gone |
| Block table (1 MB, fragment shader reads every invocation) | Gone |
| `page_table[instance_index / 256]` per vertex | `quads[instance_index]` direct |
| Block table + branch + pool read per fragment | `countOneBits` + direct read |
| `read_quad_count` sync stall | Gone |
| Overflow detection + re-dispatch | Graceful degradation, one-frame retry |
| `clear_page_table_tail` | Gone |
| `build_neighbor_slices` CPU extraction | Build shader reads directly |
| `ChunkBuildData` type | Gone |
| 128 MB material volume (32 KB/chunk) | ~12 MB typical, visibility-driven |
| 256 block type cap | 16-bit material IDs possible |
| Direction in quad descriptor (3 bits) | Implicit in memory layout |
| Rasterizer-only backface culling | Direction + layer + quad cull cascade |
| MDI-level direction culling | Direction -> layer -> quad cull cascade with Hi-Z |
| Uniform LOD downsampling (destroys thin features) | Topology-preserving decimation on full grid |
| Single rendering path at all distances | Rasterization near + ray march far |
| CPU-only chunk generation (streaming bottleneck) | GPU terrain generation for far unmodified chunks |
| Hard LOD transitions (pop-in) | Continuous streaming hierarchy with imposter fallback |
| Full generation cost regardless of distance | Coarse 8x8x8 for far, full 32x32x32 for near |


## Open Questions

### Free list granularity

The free list tracks holes in block-sized units. Optimal block size for quads
(256 quads? 512?) and material (one sub-block = 512 bytes?) needs measurement
under real rebuild churn.

### Compaction scheduling

How often to compact, and how to prioritize which chunks to move. Options:
incremental (move N chunks per frame), threshold-triggered (compact when
fragmentation exceeds X%), or opportunistic (compact during low-load frames).

### Material upload path

The CPU uploads the full 32 KB material volume to staging, and a GPU compute
pass packs only the populated sub-blocks. This wastes some upload bandwidth.
Alternative: CPU reads back the sub-block mask (async, one-frame delay) and
uploads only the relevant sub-blocks next frame. Saves bandwidth, adds one
frame of material latency.

### Bind group mechanism

Segmented buffers require multiple buffer bindings. Options:
- `BUFFER_BINDING_ARRAY` feature (wgpu) -- natural, but feature-gated.
- Fixed binding slots (4-8) with lazy allocation -- portable, slightly verbose.

### Cross-segment chunks

A chunk's contiguous range must fit within a single segment. If a segment is
nearly full, a new chunk may not fit. The allocator must detect this and start
the chunk in the next segment, potentially wasting the tail of the current
segment. This waste is bounded by the maximum chunk size.

### Material format width

At `u8` per voxel (256 palette entries), each sub-block is 512 bytes. At `u16`
(65K entries), each is 1 KB. The popcount addressing works identically either
way -- only the multiplier changes. The right width depends on the game's
material palette size.

### Hi-Z buffer generation

The cull cascade assumes a Hi-Z depth buffer is available. Generation strategy:
render the previous frame's depth buffer into a mip chain (each level stores
the max depth of the 2x2 block below it). The Hi-Z buffer is one frame behind,
which can cause momentary pop-in for newly revealed geometry. Whether this
latency is perceptible needs testing.

### Quad cull cost

Stage 4 (per-quad Hi-Z) tests every surviving quad individually. For a dense
surface, this could be thousands of quads per chunk. The test itself is cheap
(compute screen rect, sample one mip texel) but the dispatch width matters.
Need to measure whether stage 3 (layer Hi-Z) reduces the set enough that
stage 4 is tractable, or whether stage 4 should be optional and only enabled
at close range.

### Decimation budget tuning

How many voxels to remove per LOD level. Options: fixed budget per level,
distance-to-camera mapped to a target quad count reduction, or adaptive based
on screen-space quad density. The merge disruption metric needs measurement
under real terrain to determine effective thresholds.

### Decimation averaged color computation

How to compute the averaged color for decimated chunks. Options: accumulate
material colors from removed voxels into survivors during decimation, or
precompute a per-chunk average from the material volume before discarding it.
The former is more accurate but adds complexity to the decimation pass.

### LOD boundary seams

Two adjacent chunks at different LOD levels have different surface geometry at
the shared boundary. The boundary cache stores neighbor surfaces, so the
decimation pass could read neighbor boundary state to keep edges consistent.
The severity of visual artifacts at LOD transitions needs testing.

### Ray march depth precision

Using bounding box depth instead of per-voxel depth avoids the cost of
`gl_FragDepth` output but introduces depth error up to one chunk diagonal
(~55 voxels). Whether this causes Z-fighting between adjacent far chunks
needs testing. Potential mitigation: conservative depth bias.

### Imposter invalidation policy

Camera angle threshold for invalidating cached imposters, and whether
per-octant sub-chunk imposters are worth the 8x storage cost for parallax
fidelity. Depends on how noticeable the flat-billboard artifact is at the
distances where imposters are used.

### GPU terrain noise functions

Which noise algorithms are suitable for GPU compute terrain generation.
Integer-hash-based noise (cellular, value noise) avoids floating point
entirely. Simplex noise needs careful implementation to avoid warp divergence
from conditional branches. The noise function quality at 8x8x8 coarse
resolution may differ visibly from 32x32x32 if the noise frequency is high
relative to the coarse sample rate.
