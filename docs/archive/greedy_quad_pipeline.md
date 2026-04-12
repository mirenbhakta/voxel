# Greedy-Quad Rendering Pipeline

> Archived. This was the V1 scaffold architecture, superseded by `render_pipeline_v2.md`
> (sub-chunk DDA). Preserved as a design reference and benchmark baseline.
>
> The material storage model (§Material Storage) remains active in V2 at Level 0 —
> sub-block packing with `sub_mask` popcount addressing is used directly.

---

## Design Principles

### 1. Producer owns allocation

The entity that produces data owns its allocation lifecycle.

- **CPU produces** source data (occupancy, material edits, world offsets). Fixed-size per
  chunk. CPU manages these.
- **GPU produces** derived data (quads, material packing, draw args). Variable-size, known
  only after compute. GPU manages these.

### 2. Async feedback, never sync readback

The GPU writes utilization metadata (bump pointer levels, overflow flags, per-chunk quad
counts) into a shared metadata buffer. The CPU reads it asynchronously with one-frame
latency via `map_async` callback. The CPU never stalls waiting for GPU output.

### 3. Segmented growth, never reallocation

GPU buffers grow by appending new segments (64 MB each). When the GPU reports low free
capacity, the CPU allocates a new segment and rebuilds the bind group. No data moves. No
rebuild. No stall.

### 4. Allocate only what rendering touches

Quad and material storage is driven by visible face output. Interior sub-blocks that
contribute no visible faces cost zero storage, even if they contain occupied voxels.

---

## Contiguous Allocation Model

Chunks own contiguous ranges in shared buffers, not scattered blocks. Page table and block
pool indirection are eliminated.

### Why contiguity matters

The previous block pool stored quads in fixed-size 256-quad blocks scattered across a
shared pool. A page table mapped virtual block indices to physical locations. The vertex
shader performed `page_table[instance_index / 256]` on every invocation — a dependent
memory read on the hottest shader path.

The block pool existed because output size is unpredictable (greedy merge produces
variable quads). But this is a sizing problem, not an addressing problem. Solving it at
the allocator level (bump allocation with periodic compaction) eliminates the per-vertex
indirection.

The same logic applies to material storage. The previous sparse sub-block model used a
block table with pool pointers, requiring a dependent memory read in the fragment shader —
an even more critical path since fragments vastly outnumber vertices.

### Allocator: hybrid bump + free list

**Bump allocation (hot path):** New chunks advance a GPU-side atomic bump pointer.
Contiguous by construction. Zero fragmentation on initial load.

**Block-granular free list (cold path):** When a chunk is unloaded or rebuilt with a
different size, its range becomes a hole. Free space is tracked in block-sized units. New
allocations that cannot be satisfied by the bump pointer search the free list for a
contiguous run of blocks.

**Compaction:** Periodically, a GPU compute pass slides chunks down to fill holes. A chunk
with 300 quads is 1.2 KB. Moving a thousand chunks is megabytes, which a GPU does in
microseconds. After compaction, the CPU updates `base_offset` for moved chunks and the
cull shader rebuilds MDI entries (which it does every frame anyway).

### Segmented growth

Each buffer is an array of equal-size segments:

```
Segment 0:  [64 MB, allocated at startup]
Segment 1:  [64 MB, allocated on demand]
Segment 2:  ...
```

Shader access uses a per-chunk `buffer_index` to select the segment. Segments are bound
via `BUFFER_BINDING_ARRAY` or a fixed number of binding slots for portability.

When the CPU allocates a new segment: create a new buffer, rebuild the bind group, new
chunk allocations target the new segment's bump pointer. No data moves. Existing segments
are untouched.

---

## Quad Storage

Each chunk owns a contiguous range in the quad buffer. Within that range, quads are
ordered by face direction.

### Layout

```
Quad buffer (contiguous per chunk):
[---- chunk A ----][---- chunk B ----][---- chunk C ----] ...

Within each chunk:
[+X quads][−X quads][+Y quads][−Y quads][+Z quads][−Z quads]
```

### Per-chunk metadata

```
buffer_index:      u32              // which segment
base_offset:       u32              // starting quad index within that segment
dir_layer_counts:  [[u32; 32]; 6]   // quad count per (direction, layer)
```

~776 bytes per chunk. `dir_counts[d]` is `sum(dir_layer_counts[d])`. Direction ranges and
layer ranges within directions are all prefix sums from this table, enabling layer-level
culling.

### Vertex shader

```wgsl
let quad = quads[instance_index];  // direct indexing, no page table
```

One buffer read. No division, no indirection, no page table lookup.

### Quad descriptor format

Direction is implicit in the memory layout (the cull shader knows which direction each MDI
entry covers), not encoded in the quad:

| Bits  | Field    | Width |
|-------|----------|-------|
| 0–4   | col      | 5     |
| 5–9   | row      | 5     |
| 10–14 | layer    | 5     |
| 15–19 | width-1  | 5     |
| 20–24 | height-1 | 5     |
| 25–31 | free     | 7     |

The 7 free bits are available for future use (LOD flags, material hints, etc.).

### Build shader: two-pass design

A single dispatch of `(32, 6, 1)` workgroups cannot produce direction-ordered output
because there are no cross-workgroup barriers. Direction d's write region depends on
directions 0..d-1 finishing.

**Pass 1 (count):** Same face derivation and greedy merge, but only increments
per-(direction, layer) atomic counters. Produces `dir_layer_counts: [[u32; 32]; 6]` per
chunk.

**Between passes:** Prefix-sum the counts to determine write positions for each
(direction, layer) bucket. Atomically advance the quad bump pointer by the total count.

**Pass 2 (write):** Re-derives quads and writes them to the correct positions in the
contiguous range. Quads are direction-ordered and layer-sorted within each direction.

**In-place rebuild:** If the new total from Pass 1 fits within the old allocation, skip
bump pointer advance and write in-place. Common for small edits.

**Batched builds:** Both passes naturally support batching. Pass 1 counts all dirty chunks
in one dispatch; Pass 2 writes all chunks. Two dispatches total regardless of batch size.

---

## Cull Cascade

Four-stage pipeline. Each stage reduces the working set for the next.

**Stage 1: Direction cull.** Dot product of camera position vs chunk center along each
axis. Back-facing directions skipped. ~50% average rejection. One instruction per
direction per chunk.

**Stage 2: Layer cull (frustum).** For each direction, compute the visible world layer
range from frustum plane intersection. Intersect with each chunk's layer range
(`chunk_coord * 32 .. chunk_coord * 32 + 31`). Skip chunks with no overlap. Narrow MDI
entries to the visible layer sub-range. Pure arithmetic from per-chunk metadata.

**Stage 3: Layer cull (Hi-Z).** For surviving (chunk, direction) pairs, test individual
layer planes against the Hi-Z depth buffer. Each layer is a plane at a known world
coordinate with a known screen-space extent. Reject occluded layers. This subsumes
software backface culling entirely — "camera is above chunk, skip bottom faces" is a
degenerate case: the chunk's own top surface occludes its bottom in the depth buffer. Hi-Z
handles this plus non-trivial occlusion (hills, caves, canyons) with no special-case code.

**Stage 4: Quad cull (Hi-Z).** Surviving quads tested individually. Compute each quad's
screen-space rect from its position and greedy merge size, sample the appropriate Hi-Z
mip. Stream-compact survivors into the final draw buffer. Catches quads within a visible
layer that are still occluded (cave openings, windows, partial occlusion).

```
Direction cull → Layer cull (frustum) → Layer cull (Hi-Z) → Quad cull (Hi-Z) → Draw
  (6 tests/chunk)   (~32 tests/dir)      (per layer)         (per quad)         (survivors)
```

MDI entry after culling:

```
first_instance = base_offset + prefix_sum(dir_layer_counts up to visible start)
instance_count = sum(dir_layer_counts for visible layer range)
vertex_count   = 6
```

---

## Material Storage

Material storage uses the same contiguous allocation model, but with visibility-driven
sub-block packing.

> **Still active in V2.** Sub-block packing (`sub_mask` popcount addressing) is preserved
> at Level 0. The DDA hits an occupied voxel; the material address is resolved through
> sub-block packing at whatever level is being rendered.

### Motivation

The fragment shader samples material identity per pixel. In the previous sparse sub-block
design, this required a block table lookup (dependent memory read) + branch
(null/homogeneous/pool) + possible pool read. Fragments vastly outnumber vertices, making
this indirection more costly than the quad page table it replaces.

The previous design's optimizations (homogeneous sentinels, null entries) saved memory for
empty and solid-interior sub-blocks. But the fragment shader only runs on fragments with
visible faces. Those fragments only sample sub-blocks that contain visible faces. Empty
and solid-interior sub-blocks are never sampled — the optimizations existed for cases that
never reach the fragment shader.

### Visibility-driven packing

A 32×32×32 chunk is divided into a 4×4×4 grid of 8×8×8 sub-blocks (64 total). The build
shader determines which sub-blocks contain at least one visible face and records this as a
64-bit bitmask (`sub_mask`). Only populated sub-blocks are stored, packed contiguously:

```
chunk_material_base → [sub-block 3:  512 B]
                      [sub-block 12: 512 B]
                      [sub-block 28: 512 B]
                      [sub-block 45: 512 B]
                      total: 2048 B (vs 32 KB for full flat volume)
```

Each 512-byte entry is a dense 8×8×8 array of `u8` palette indices.

### Fragment shader resolution

The vertex shader passes `chunk_material_base` and `sub_mask` as flat varyings (fetched
once per quad, free for all fragments):

```wgsl
let bx = vx / 8u;  let by = vy / 8u;  let bz = vz / 8u;
let lx = vx % 8u;  let ly = vy % 8u;  let lz = vz % 8u;

let sub_idx = bz * 16u + by * 4u + bx;
let offset  = countOneBits(sub_mask & ((1u << sub_idx) - 1u));

let material = material_buf[chunk_material_base + offset * 512u + lz * 64u + ly * 8u + lx];
```

No branch. No block table read. One `countOneBits` (single ALU cycle on modern GPUs) +
one direct buffer read. The bitmask is in a register from the flat varying.

### Why homogeneous optimization is unnecessary

The previous design needed homogeneous sentinels for sub-blocks where every voxel had the
same material. This model allocates based on visibility, not occupancy:

- All air: no quads, no storage.
- All solid interior: no quads (faces culled by neighbors), no storage.
- Mixed but fully buried: no quads (all faces adjacent to solid neighbors), no storage.
  The previous model allocated a pool block for these.
- Mixed with visible faces: quads emitted, storage allocated.

The visibility test is strictly tighter than the occupancy test. Cases the homogeneous
optimization handled are absent from the data entirely.

### Material packing flow

1. CPU uploads the full 32 KB material volume to a transient staging buffer (reused across
   chunks).
2. Build shader runs, emits quads, marks populated sub-blocks via `atomicOr` on a shared
   `u64`.
3. Build shader atomically advances the material bump pointer by `popcount(sub_mask) * 512`
   bytes. Writes `chunk_material_base` and `sub_mask` to per-chunk metadata.
4. Material-pack compute pass (same command buffer, after build): reads the bitmask,
   iterates over set bits, copies each populated 8×8×8 region from staging into the
   packed range.

All GPU-side, single submission, no CPU readback.

### Memory usage

| Chunk type              | Populated sub-blocks | Material cost |
|-------------------------|----------------------|---------------|
| All air                 | 0                    | 0 B           |
| Solid interior          | 0                    | 0 B           |
| Flat terrain surface    | ~16 of 64            | ~8 KB         |
| Rolling hills surface   | ~24–32 of 64         | ~12–16 KB     |
| Very jagged surface     | ~40–48 of 64         | ~20–24 KB     |
| Worst case (checkerboard) | 64 of 64           | 32 KB         |

For 4096 loaded chunks with ~1200 surface chunks averaging ~20 populated sub-blocks: ~12
MB. Compared to 128 MB for the previous flat allocation.

---

## Per-Slot Data

A chunk slot is a lightweight index with minimal fixed-cost storage:

```
Chunk slot (index)
  +-- Quad range     (~776 B: buffer_index, base_offset, dir_layer_counts[[32]; 6])
  +-- Material range (12 B: buffer_index, base_offset, sub_mask as 2×u32)
  +-- Chunk meta     (16 B: total quad count, flags)
  +-- Boundary cache (768 B: 6 × 32 words, neighbor surfaces)
  +-- World offset   (16 B: ivec4 chunk position)
```

Fixed cost per slot: ~1588 B. An empty chunk costs this and nothing else. Only surface
chunks allocate from the quad and material buffers.

### Shared buffers (slot-indexed)

| Buffer               | Size (4096 slots) | Contents                                  |
|----------------------|-------------------|-------------------------------------------|
| `quad_range_buf`     | ~3.1 MB           | Per-chunk quad base + direction-layer counts |
| `material_range_buf` | 48 KB             | Per-chunk material base + sub-block mask  |
| `chunk_meta_buf`     | 64 KB             | Quad count, flags                         |
| `boundary_cache_buf` | 3 MB              | Neighbor boundary slices                  |
| `chunk_offset_buf`   | 64 KB             | World position                            |

---

## What This Eliminated

Compared to the original scaffold pipeline:

| Original                                               | This design                          |
|--------------------------------------------------------|--------------------------------------|
| Per-chunk `quad_count_buf` + `count_staging`           | `chunk_meta_buf` field, GPU-written  |
| Per-chunk bind group                                   | One shared bind group, slot via push constant |
| CPU `free_blocks` stack + `ChunkAlloc`                 | GPU-side bump allocator              |
| Page table (6 MB, vertex shader reads every invocation) | Gone                                |
| Block table (1 MB, fragment shader reads every invocation) | Gone                             |
| `page_table[instance_index / 256]` per vertex          | `quads[instance_index]` direct       |
| Block table + branch + pool read per fragment          | `countOneBits` + direct read         |
| `read_quad_count` sync stall                           | Gone                                 |
| Overflow detection + re-dispatch                       | Graceful degradation, one-frame retry |
| `clear_page_table_tail`                                | Gone                                 |
| 128 MB material volume (32 KB/chunk)                   | ~12 MB typical, visibility-driven    |
| 256 block type cap                                     | 16-bit material IDs possible         |
| Direction in quad descriptor (3 bits)                  | Implicit in memory layout            |
| MDI-level direction culling                            | Direction → layer → quad cull cascade with Hi-Z |

---

## Open Questions

These remain relevant to any greedy-quad rendering implementation.

**Free list granularity.** Optimal block size for quads (256 quads? 512?) and material
(one sub-block = 512 bytes?) needs measurement under real rebuild churn.

**Compaction scheduling.** Options: incremental (move N chunks per frame),
threshold-triggered (compact when fragmentation exceeds X%), or opportunistic (compact
during low-load frames).

**Material upload path.** The CPU uploads the full 32 KB material volume to staging, and a
GPU compute pass packs only the populated sub-blocks. Alternative: CPU reads back the
sub-block mask (async, one-frame delay) and uploads only the relevant sub-blocks next
frame. Saves bandwidth, adds one frame of material latency.

**Quad cull cost (Stage 4).** Tests every surviving quad individually. Need to measure
whether Stage 3 (layer Hi-Z) reduces the set enough that Stage 4 is tractable, or whether
Stage 4 should be optional at close range only.

**Material format width.** At `u8` per voxel (256 palette entries), each sub-block is
512 bytes. At `u16` (65K entries), each is 1 KB. The popcount addressing works identically
either way — only the multiplier changes.
