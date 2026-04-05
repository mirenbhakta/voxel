# GPU Memory Architecture

Design for a unified GPU memory system that resolves the ownership conflicts,
pre-allocation waste, and sync stalls in the current rendering pipeline.


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
- **GPU produces** derived data (quads, page table mappings, material block
  allocations, draw args). Variable-size, known only after compute. GPU manages
  these.

The indirection tables (page table, block table) are the contract between the
two. Written by the producer, read by the consumer. Neither side reaches into
the other's allocation.

### 2. Async feedback, never sync readback

The GPU writes utilization metadata (free stack level, overflow flags,
per-chunk quad counts) into a shared metadata buffer. The CPU reads it
asynchronously with one-frame latency via `map_async` callback. The CPU never
stalls waiting for GPU output.

### 3. Segmented growth, never reallocation

GPU buffers are divided into large segments (64-256 MB). When the GPU reports
low free capacity, the CPU allocates a new segment, initializes its block IDs,
pushes them onto the free stack, and rebuilds the bind group. No data moves.
No rebuild. No stall.

### 4. Allocate only what rendering touches

Material storage is allocated per sub-block, driven by visible face output.
Interior sub-blocks that contribute no visible faces cost zero material
storage, even if they contain occupied voxels.


## Unified Memory Model

All variable-size GPU data uses the same pattern: a block pool with free stack
allocation and indirection table lookup. The allocator is the same code with
different block sizes.

### Block pools

| Pool | Block size | Consumer |
|------|-----------|----------|
| Quad pool | 256 quads x 4 B = 1 KB | Vertex shader (quad unpack) |
| Material pool | 8^3 voxels x 1 B = 512 B | Fragment shader (material lookup) |

Both pools:
- Are shared across all chunks (no per-chunk buffers).
- Use a GPU-side atomic free stack for allocation.
- Grow via segments without reallocation.
- Are freed via compute dispatch on chunk unload.

### Per-slot data

A chunk "slot" is a lightweight index with minimal fixed-cost storage:

```
Chunk slot (index)
  |
  +-- Block table     (64 entries, 256 B)  -> Material pool
  +-- Page table      (384 entries, ~1.5 KB) -> Quad pool
  +-- Chunk meta      (16 B: quad count, block count, flags)
  +-- Boundary cache  (768 B: 6 x 32 words, neighbor surfaces)
  +-- World offset    (16 B: ivec4 chunk position)
```

Fixed cost per slot: ~2.5 KB. An empty chunk costs this and nothing else. A
fully solid interior chunk costs this and nothing else. Only surface chunks
allocate from the pools.

Slot count is no longer tied to worst-case material allocation. With 2.5 KB per
slot instead of ~36 KB, the same memory budget supports far more loaded chunks,
or the same chunk count at a fraction of the memory.

### Shared buffers (slot-indexed, CPU-written)

| Buffer | Size (4096 slots) | Contents |
|--------|-------------------|----------|
| `block_table_buf` | 4096 x 256 B = 1 MB | Per-chunk 4x4x4 material block pointers |
| `page_table_buf` | 4096 x 384 x 4 B = 6 MB | Quad pool block ID mapping |
| `chunk_meta_buf` | 4096 x 16 B = 64 KB | Quad count, block count, flags |
| `boundary_cache_buf` | 4096 x 768 B = 3 MB | Neighbor boundary slices |
| `chunk_offset_buf` | 4096 x 16 B = 64 KB | World position |

These are all small relative to the current 128 MB material volume. The page
table and block table are written by the GPU during the build pass. The
boundary cache and chunk offsets are written by the CPU. Chunk meta is written
by the GPU, read asynchronously by the CPU.

### GPU-side free stack

One free stack per pool. The free stack stores block IDs (u32), allocated to
the maximum the pool could ever hold. Since the pool grows via segments, the
free stack must be pre-sized to the maximum segment count:

```wgsl
@group(0) @binding(N)   var<storage, read_write> free_stack : array<u32>;
@group(0) @binding(N+1) var<storage, read_write> stack_top  : atomic<u32>;

fn pop_block() -> u32 {
    let top = atomicSub(&stack_top, 1u) - 1u;
    if top >= arrayLength(&free_stack) {
        atomicAdd(&stack_top, 1u);  // undo
        return 0u;                   // null sentinel
    }
    return free_stack[top];
}

fn push_block(id: u32) {
    let top = atomicAdd(&stack_top, 1u);
    free_stack[top] = id;
}
```

Block 0 is reserved as the null sentinel. Writes to block 0 are harmless
(overflow landing zone). Reads from block 0 return default/zero material.

### Segmented pool growth

Each pool is an array of equal-size buffer segments:

```
Segment 0:  [block 1 .............. block 65536]     64 MB, allocated at startup
Segment 1:  [block 65537 .......... block 131072]    64 MB, allocated on demand
Segment 2:  ...
```

Shader access decomposes the global block ID:

```wgsl
let segment = (block_id * BLOCK_SIZE) / SEGMENT_SIZE;
let local   = (block_id * BLOCK_SIZE) % SEGMENT_SIZE;
```

When the CPU allocates a new segment:
1. Create a new wgpu buffer.
2. Initialize the new block IDs (sequential, starting after the previous
   segment's range).
3. Push those IDs onto the free stack via `queue.write_buffer`.
4. Rebuild the bind group to include the new segment.

No data moves. Existing segments are untouched. The bind group rebuild is cheap
(pointer setup, no GPU work).


## Sparse Volumetric Material Storage

### Motivation

The current system stores 32 KB of material per chunk (one byte per voxel,
32^3). This is the single largest GPU allocation (128 MB for 4096 chunks). It
has three problems:

1. **Empty chunks waste 32 KB** for a volume of air that will never be sampled.
2. **Solid interior chunks waste 32 KB** for a homogeneous volume where every
   voxel has the same material and no face is ever visible.
3. **The 256 block type cap** (`u8` per voxel) limits the material palette,
   even though `BlockId` is `u16` on the CPU.

### Design

Subdivide the 32^3 chunk into 8x8x8 sub-blocks (4x4x4 = 64 sub-blocks per
chunk). Each sub-block is allocated independently from the material pool.

**Block table entry** (one u32 per sub-block, 64 per chunk):

```
0x00000000                -> null (air, no material)
0x80000000 | material_id  -> homogeneous (all voxels same material)
otherwise                 -> pool block pointer
```

The high bit distinguishes homogeneous entries from pool pointers. Homogeneous
sub-blocks cost zero pool allocation, only a table entry.

**Fragment shader lookup:**

```wgsl
// Chunk-local voxel position (0..31 each axis).
let bx = vx / 8u;  let by = vy / 8u;  let bz = vz / 8u;
let lx = vx % 8u;  let ly = vy % 8u;  let lz = vz % 8u;

let entry = block_table[slot * 64u + bz * 16u + by * 4u + bx];

var block_id: u32;
if entry == 0u {
    block_id = 0u;  // air
}
else if (entry & 0x80000000u) != 0u {
    block_id = entry & 0xFFFFu;  // homogeneous
}
else {
    // Read from material pool block.
    let local_idx = lz * 64u + ly * 8u + lx;
    block_id = read_material_pool(entry, local_idx);
}
```

One extra indirection compared to the current flat array. The block table is
256 bytes per chunk (a few cache lines). In practice, the fragment shader hits
the same few sub-blocks repeatedly for a given quad, so the table entry is
cached after the first access.

### Allocation driven by visible faces

Only sub-blocks that contain at least one visible face need material storage.
The build shader already knows which sub-blocks have faces: the greedy merge
emits quads with positions, and each quad implies a surface voxel in a specific
sub-block.

After the merge (or as part of it), the build shader writes the block table:
- Sub-blocks with emitted quads: pop a material block from the pool.
- Sub-blocks without quads: null or homogeneous entry.

Material allocation is driven by the same pass that produces geometry. No
separate analysis pass needed.

**Typical terrain chunk** (half air, half stone, surface at one height):

| Sub-block type | Count (~) | Pool blocks | Cost |
|---------------|-----------|-------------|------|
| Air (above surface) | 32 | 0 | 0 B |
| Solid interior (below, fully surrounded) | 24 | 0 | 0 B |
| Surface (contains visible faces) | 8 | 8 | 4 KB |
| **Total** | 64 | 8 | **~4.25 KB** |

Compared to the current 32 KB per chunk flat array. The 24 solid-interior
sub-blocks contain occupied voxels with valid material IDs, but no fragment
shader ever samples them because all their faces are culled. They cost nothing.

### Block type cap

The block table entry provides 16 bits for homogeneous material IDs and the
pool blocks can store wider types. At `u16` per voxel (1 KB per 8^3 block),
the per-chunk maximum is 64 KB, but the actual cost is still proportional to
the surface shell. This removes the 256 block type limitation without doubling
the baseline memory.


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
    quad_count   : u32,   // quads emitted by greedy merge
    block_count  : u32,   // quad pool blocks consumed
    mat_blocks   : u32,   // material pool blocks consumed
    flags        : u32,   // overflow bits, etc.
}
```

A separate global metadata region (or the free stack top itself) reports pool
utilization.

### CPU reads

The CPU maps a staging copy of the metadata buffer asynchronously (same
mechanism as the existing visible-count readback). One-frame latency. No stall.

**Decisions driven by feedback:**

| Signal | CPU response |
|--------|-------------|
| Free stack below low-water mark | Allocate new pool segment |
| Overflow flag set | Allocate new segment, mark chunk dirty for rebuild |
| Chunk quad count available | Diagnostics, statistics overlay |

The CPU never needs to know which specific blocks a chunk owns. It only needs
pool-level utilization to decide when to grow.

### Pool exhaustion

If `pop_block()` returns null during a build, the shader writes quads to block
0 (harmless, null sentinel) and sets the overflow flag in chunk meta. The chunk
renders with missing quads for one frame. The CPU sees the flag, grows the
pool, and the chunk rebuilds next frame.

This is a graceful degradation path, not an error. The one-frame glitch is
imperceptible. The alternative (pre-allocating for worst case) is what the
current system does, and it is the source of most of the problems.


## Chunk Lifecycle

### Load

1. CPU pops a slot index from the free slot list.
2. CPU uploads occupancy to `occupancy_buf[slot]` (or staging area).
3. CPU writes chunk offset to `chunk_offset_buf[slot]`.
4. CPU marks slot dirty.
5. No material upload yet. Material is allocated by the build shader.

### Build (GPU, on dirty)

1. Build shader reads occupancy for current slot.
2. Build shader reads neighbor boundaries (from neighboring slots' occupancy or
   boundary cache).
3. Face derivation (AND-NOT, transpose) produces face layers.
4. Greedy merge emits quads, popping blocks from the quad pool free stack.
5. Build shader writes page table entries for consumed blocks.
6. Build shader determines which 8x8x8 sub-blocks contain visible faces.
7. For those sub-blocks, pops blocks from the material pool free stack.
8. CPU uploads material data for allocated sub-blocks (or the build shader
   reads from a staging upload and copies).
9. Build shader writes block table entries.
10. Build shader writes chunk meta (quad count, block counts, flags).

Note: step 8 is the one point where CPU-produced data (material identity) must
reach GPU-allocated storage. The simplest path: the CPU uploads the full 32 KB
material to a transient staging buffer. The build shader (or a follow-up
compute pass) copies only the sub-blocks that were allocated. The staging
buffer is reused across chunks.

### Edit

1. CPU uploads modified occupancy and material to slot-indexed buffers (or
   staging).
2. CPU marks slot dirty.
3. Build pass runs. Old quad and material blocks are freed first (build shader
   pushes them back to the free stacks before popping new ones).
4. New blocks allocated, page table and block table rewritten.

### Unload

1. CPU dispatches a small free compute shader for the slot.
2. Free shader reads page table and block table for the slot.
3. Pushes all non-null block IDs back onto the respective free stacks.
4. Zeros the page table and block table entries.
5. CPU returns the slot index to the free slot list.

No CPU shadow of block ownership needed. The GPU's own indirection tables are
the source of truth.

### Growth

1. CPU observes low free-stack level via async feedback.
2. CPU creates a new buffer segment.
3. CPU initializes block IDs for the new segment's range.
4. CPU pushes those IDs onto the free stack via `queue.write_buffer`.
5. CPU rebuilds the bind group to include the new segment.
6. No stalls, no data movement, no rebuild.


## Streaming and Occlusion

The sparse material storage naturally supports visibility-driven streaming.
The sub-block (8^3) is the streaming unit:

- A chunk behind an occluder can have its material blocks evicted. The block
  table stays resident (256 bytes, cheap). The blocks themselves are freed back
  to the pool.
- When the chunk becomes visible again, the build pass re-allocates material
  blocks and the CPU re-uploads the relevant sub-block data.
- Chunks at LOD boundaries can use coarser sub-blocks (e.g. one material per
  8^3 region = homogeneous entries only, zero pool cost).

This matches the brickmap model: the brick pointer table (block table) is the
permanent structure, bricks themselves are transient and streamed based on
demand.


## What This Eliminates

| Current | Unified |
|---------|---------|
| Per-chunk `occupancy_buf` (4864 B each) | Shared slot-indexed buffer |
| Per-chunk `quad_count_buf` + `count_staging` | `chunk_meta_buf` field, GPU-written |
| Per-chunk bind group | One shared bind group, slot via push constant |
| CPU `free_blocks` stack + `ChunkAlloc` | GPU-side free stack |
| CPU page table writes for blocks | GPU writes page table during build |
| `read_quad_count` sync stall | Gone |
| Overflow detection + re-dispatch | Graceful degradation, one-frame retry |
| `clear_page_table_tail` | GPU zeros its own entries on free |
| `build_neighbor_slices` CPU extraction | Build shader reads directly |
| `ChunkBuildData` type | Gone |
| 128 MB material volume (32 KB/chunk) | ~4 KB/chunk typical, pool-allocated |
| 256 block type cap | 16-bit material IDs, wider where needed |
| Special-case skip for empty/solid chunks | Naturally zero-cost (null block table) |


## Open Questions

### Material upload path

The build shader determines which sub-blocks need material, but the material
content comes from the CPU. The exact handoff needs design:
- Option A: CPU uploads full 32 KB to staging, GPU copies relevant sub-blocks.
  Simple, wastes some bandwidth.
- Option B: CPU reads back the block table (async), uploads only allocated
  sub-blocks next frame. Saves bandwidth, adds one frame of material latency.
- Option C: CPU uploads material to slot-indexed buffer (as today), build
  shader copies to pool blocks. Hybrid of both.

### Free stack contention

Multiple build workgroups popping from the same atomic stack. One pop per 256
quads (quad pool) or per sub-block (material pool) -- low frequency relative to
per-quad operations. Likely fine, but worth measuring under high rebuild load.

### Bind group mechanism

Segmented pools require multiple buffer bindings. Options:
- `BUFFER_BINDING_ARRAY` feature (wgpu) -- natural, but feature-gated.
- Fixed binding slots (4-8) with lazy allocation -- portable, slightly verbose.
- Single buffer + compute-copy on growth -- avoids the binding array question
  but reintroduces data movement.

### Maximum segment count

The free stack must be pre-sized to the maximum block count across all
segments. For 4 segments of 64 MB each (256 MB total quad pool), the stack is
~256K entries = 1 MB. Reasonable. The practical cap is GPU memory, not the
stack size.
