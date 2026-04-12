# Render Pipeline Analysis

## Architecture Overview

The pipeline is a three-stage **build/filter/render** system for voxel terrain:

```
CPU Chunk data                    GPU (on edit)                    GPU (per frame)
─────────────────                 ──────────────                   ─────────────────
Chunk occupancy (4 KB)    ──►     Build compute shader     ──►    Cull compute shader
Volumetric material (32 KB)       - face derivation (AND-NOT)     - frustum test (6 planes)
Neighbor boundary slices          - greedy merge                  - stream compaction
                                  - quad emit → block pool        - MDI command emit
                                         │                               │
                                         ▼                               ▼
                                  Quad pool (shared 64 MB)        multi_draw_indirect_count
                                  Page table (6 MB)               - vertex shader: unpack u32 quad
                                                                  - fragment shader: volumetric material lookup
```

### Key design decisions

1. **Material-agnostic greedy merge.** Materials are resolved per-pixel in the fragment
   shader via a volumetric 32 KB/chunk array, not per-quad. This decouples geometry from
   material entirely -- quads are maximally large regardless of material distribution. This
   is the central architectural insight.

2. **Virtual geometry block pool.** All chunks share one flat GPU buffer divided into
   fixed-size 256-quad blocks. Chunks pop blocks from an atomic free stack. Zero external
   fragmentation by construction, no compaction pass ever needed. Replaces the original
   per-chunk 384 KB worst-case buffers.

3. **Page table indirection.** The vertex shader maps
   `instance_index -> block_id -> physical offset` through a flat page table.
   `instance_index / 98304` yields the chunk slot, `instance_index / 256` yields the block
   index, `page_table[block_idx]` yields the physical pool location.

4. **GPU frustum culling with stream compaction.** A compute pass tests chunk AABBs against
   6 frustum planes, atomically appends visible draws to an output buffer.
   `multi_draw_indirect_count` consumes the GPU-written count -- no CPU readback needed for
   rendering.

5. **Neighbor boundary culling.** The occupancy buffer is extended from 1024 to 1216 words
   (192 extra for 6 neighbor boundary slices). The build shader uses these to AND-NOT away
   internal faces at chunk boundaries. Unloaded neighbors default to solid (cull boundary
   faces to prevent seams); rejected/air neighbors default to empty (expose boundary
   faces).


## Holes That Had to Be Patched

These are the structural problems that emerged from the design and required non-trivial
fixes.

### 1. Stale Page Table Entries After Block Trimming

**Severity:** major bug, fixed.

**The problem:** After `rebuild_subset` trims excess blocks, the GPU page table entries for
freed blocks aren't cleared. Those blocks get reallocated to other chunks. On a subsequent
rebuild (neighbor changes can cause greedy merge to produce MORE quads than before -- merge
is non-monotonic), the shader reads stale entries pointing to blocks now owned by another
chunk. Result: two adjacent chunks swap their rendered geometry.

**The fix required three coordinated parts:**
- Reserve block 0 as a permanent null sentinel (overflow writes become harmless)
- Zero stale page table entries after every trim (`clear_page_table_tail`)
- Detect overflow after dispatch, allocate the exact deficit, re-dispatch

**Two approaches failed first:**
- Pre-allocating MAX_CHUNK_BLOCKS before each rebuild deadlocked the pool (steady-state
  pool is near capacity after trimming)
- Capping quad count without re-dispatch caused visible holes

**Why this is structural:** The block pool's trim-then-reallocate pattern is inherently
racy with the greedy merge's non-monotonic output. Any system that trims blocks and later
re-merges with different neighbor state will hit this. The fix works but adds complexity to
every rebuild path.

### 2. Synchronous GPU Stalls on Build Readback

**Severity:** architectural limitation.

**The problem:** `read_quad_count()` calls `device.poll(PollType::wait_indefinitely())` --
a full CPU-GPU sync point. Every rebuild batch stalls the frame. The overflow-and-retry
path stalls *twice* (once for initial dispatch, once for re-dispatch after growing
allocation).

**Why it exists:** The CPU needs the quad count to know how many blocks to trim/grow before
the next frame can use the data. There's no async path because the indirect draw args must
be correct before the cull pass runs.

**Impact:** Chunk rebuild time is on the critical path, bounded by `builds_per_frame = 16`.
At scale, this becomes the frame-time bottleneck.

### 3. Greedy Merge is Single-Threaded Within a Workgroup

**The problem:** Thread 0 does the entire sequential scan for its (layer, direction) pair.
The other 31 threads write their face words to shared memory, then go idle.

**Why it exists:** The greedy merge algorithm is inherently sequential -- each row's merge
depends on what was consumed in previous rows. The face derivation step (which precedes it)
is fully parallel.

**Impact:** The merge phase serializes thread 0, but the GPU schedules other workgroups on
the same CU while those 31 threads are idle. With 192 workgroups per chunk dispatch (and
potentially many chunks in flight), occupancy hides the stall in practice. The merge
dominates build time primarily for single-chunk rebuilds with dense layers. For batch
rebuilds with many chunks, the GPU stays fed. Still worth improving for edit latency.

### 4. Load Queue Stall at Large View Distances

**Severity:** major bug, fixed.

**The problem:** `rebuild_load_queue` capped enumeration radius by `budget_r` derived from
remaining slot count. As chunks loaded, remaining shrank, which shrank `budget_r` below the
already-explored radius. The queue permanently emptied -- no new chunks could ever be
discovered.

**The fix:** Removed the `budget_r` cap entirely. Loading is rate-limited at the
consumption point (`loads_per_frame`, GPU capacity), not the enumeration point.

**Why this is structural:** Rate-limiting at the enumeration point creates a feedback loop
where progress shrinks the search space. This is a general anti-pattern when mixing
capacity-aware budget calculations with incremental exploration.

### 5. Material Volume Buffer is 128 MB

`MAX_CHUNKS * 32768 = 4096 * 32768 bytes`. This is the single largest GPU allocation,
larger than the quad pool itself. Each voxel's material is one byte, which also caps the
shader at 256 block types globally (despite `BlockId` being `u16` on CPU).
`material_block_ids()` casts to `u8`, baking this limitation into the GPU format.

### 6. Dual Implementation of Face Derivation

The library's `FaceMasks::from_occupancy()` (CPU, with Hacker's Delight bit-plane
transpose for X faces) and `build.wgsl` (GPU, with shared-memory gather transpose)
implement the same algorithm independently. The library version is currently unused in the
rendering path -- it exists for potential CPU meshing or raycasting. This means there are
two implementations to keep in sync if the occupancy layout ever changes, and the CPU
implementation has no test coverage via the actual render path.

### 7. The QuadDescriptor Format Diverges Between CPU and GPU

The library's `QuadDescriptor(u32)` uses bits 0-24 for geometry
(col/row/layer/width/height). The build shader packs direction into bits 25-27. The CPU
type leaves those bits as "unused." Trivial fix: add a `with_direction` method or make the
CPU type match the GPU format.


## Structural Issues for a New System

Based on the above, these are the architectural pain points a redesign should address.

### 1. Synchronous build readback

The CPU-GPU sync stall per rebuild batch is the most impactful limitation at scale. The
strongest solution is making the build fully GPU-autonomous: the GPU atomically grows its
own allocation from the free stack during the merge, writes its own indirect draw args, and
the CPU never needs to know the quad count. This also eliminates the overflow-and-retry path
(see patched hole #1). A weaker alternative is pipelining the readback with a one-frame
delay so the CPU never stalls, but this leaves the CPU in the allocation loop.

### 2. Non-monotonic merge vs. fixed allocation

The core tension is that greedy merge output is unpredictable, but the block pool requires
knowing how many blocks to allocate. The current overflow-detect-and-retry works but adds
latency and complexity. A system where the GPU can grow allocation atomically during the
merge (without CPU involvement) would eliminate this entirely.

### 3. Single-threaded merge

The per-layer greedy merge being sequential limits build throughput. A parallel merge
algorithm (e.g., parallel prefix sum to assign quad positions, then parallel quad emit)
would use the full workgroup. This is algorithmically harder but would reduce build time by
up to 32x for the merge phase.

### 4. Material resolution overhead

Per-pixel volumetric lookup means every fragment pays the cost, even for large merged quads
where a single material ID would suffice. However, the fragment cost is a single byte read
from an L1-cached volumetric array -- in practice very cheap. A hybrid approach (per-quad
material for uniform quads, per-pixel fallback for multi-material) would re-introduce
material-geometry coupling: the merge would need to check material identity, bringing back
the HashMap-style grouping problem the architecture was designed to eliminate. Leave this
alone unless profiling shows fragment shading is actually the bottleneck.

### 5. 128 MB material volume

Pre-allocating 32 KB per slot for 4096 chunks is wasteful when most chunks are sparsely
populated. Material data is only uploaded when a chunk is loaded, but the full allocation is
reserved. The real fix is a virtual-allocation scheme or bindless sparse texture, not a
fundamentally different data structure. The 256 block type cap from `u8` storage is the more
pressing limitation (`BlockId` is `u16` on CPU, truncated to `u8` at the GPU boundary).

### 6. LOD is designed but unimplemented

The docs describe a resolution-agnostic pipeline (face derivation + merge work at any
bitmask resolution) with four LOD levels. Seam handling at LOD boundaries (T-junctions,
gaps at chunk edges with different LOD levels) is explicitly called out as an open problem.

### 7. No occlusion culling

The cull pass does frustum testing only. The design docs describe Hi-Z occlusion culling as
a planned extension (the `dst_indirect_buf` has `STORAGE` usage specifically to enable a
second compute pass), but it's not implemented. For dense terrain with many occluded
underground chunks, this is significant.

### 8. No transparency or non-cubic blocks

Both design docs flag this as requiring a secondary rendering path. The current pipeline
assumes all geometry is opaque axis-aligned planar slabs. Water, glass, vegetation, etc.
need a different solution.

### 9. Page table vertex shader indirection at scale

Every vertex invocation reads `page_table[imm.block_base + block_idx]` -- a random-access
read into a 6 MB buffer. As chunk count grows, this table exceeds L1 cache capacity. The
cull pass improves locality by compacting draws, but at scale with thousands of visible
chunks the indirection could become a bottleneck.

### 10. Atomic contention in build and cull passes

The build shader's `quad_count` atomic is a global contention point: all workgroups across
all dispatched chunks atomically increment the same counter, serializing at the atomic unit
under high parallelism. The cull shader's `draw_count` atomic has the same pattern. Neither
is a problem at current scale, but both will become throughput limiters as chunk count grows.


## Priority Ordering

Based on the above analysis, ranked by impact:

1. **GPU-autonomous block allocation** -- eliminates the sync stall AND the overflow-retry
   path. Biggest single improvement.
2. **Parallel merge** (horizontal-only or two-phase) -- reduces build latency, especially
   for single-chunk edits.
3. **LOD** -- necessary for view distance scaling; the bitmask-mip approach from the design
   docs is clean.
4. **Hi-Z occlusion culling** -- significant for terrain with caves and overhangs.
5. **256 block type cap** -- will bite eventually but not urgent.
