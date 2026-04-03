# Directional Face Acceleration: Computational Analysis

A stage-by-stage verification of the rendering pipeline described in
`directional_face_acceleration.md`. Each stage is examined for correctness,
cost, and worst-case behavior.


## Occupancy Storage Layout

The occupancy bitmask for a 32x32x32 chunk is stored as 1024 u32 words:

    occ[z * 32 + y] = u32, bit x set if voxel (x, y, z) is occupied

This layout makes face derivation cheap for all six directions. X is the
intra-word axis (bit index), Y is the intra-layer axis (word offset within a
32-word layer), Z is the inter-layer axis (layer index).

Total: 1024 x 4 bytes = 4096 bytes = 4 KB.


## Stage 1: Face Bitmask Derivation

Face bitmasks are derived from occupancy via bitwise operations. The output
for each direction is 32 layers of 32x32 bits = 4 KB, organized with the
layer axis aligned to the face normal.

### Z-Direction (trivial)

Adjacent Z-layers are contiguous 32-word slices in the occupancy array:

    face_+z[z][y] = occ[z*32 + y] & ~occ[(z+1)*32 + y]
    face_-z[z][y] = occ[z*32 + y] & ~occ[(z-1)*32 + y]

32 AND-NOT operations per layer, 32 layers. Output layout matches occupancy.
No transformation required.

### Y-Direction (row comparison)

Adjacent Y-rows within a Z-slice are adjacent words:

    face_+y[y][z] = occ[z*32 + y] & ~occ[z*32 + (y+1)]
    face_-y[y][z] = occ[z*32 + y] & ~occ[z*32 + (y-1)]

The operation per word is trivial. The output for "layer y" gathers one row
from each of the 32 Z-slices (stride-32 reads from the occupancy array). On
a GPU, load the 4 KB occupancy into shared memory first. The stride reads
become random-access reads from shared memory.

### X-Direction (bit shift + transpose)

X is the intra-word axis. Face detection between adjacent X positions is a
shift within each word:

    all_+x[z][y] = occ[z*32 + y] & ~(occ[z*32 + y] >> 1)

This produces a u32 where bit x is set if there is a +X face at (x, y, z).
But the output must be organized as 32 layers (one per x-coordinate), where
each layer is a 32x32 bitmask indexed by (z, y). This requires a bit-plane
extraction: extract bit x from all 1024 words and pack them into a new
32x32 bitmask.

This is a 32x1024 bit transpose. In a GPU compute shader with 4 KB shared
memory:

    1. Each thread computes all_+x for its (z, y) pair and writes to shared
    2. For each x-layer, each thread extracts bit x from 32 words (one per
       y-coordinate) and packs them into a new u32

32K extract-and-set operations total per chunk. A single workgroup of 1024
threads with 32 iterations each. Runs once per chunk edit, not per frame.

The -X direction is identical but with a left shift:

    all_-x[z][y] = occ[z*32 + y] & ~(occ[z*32 + y] << 1)

### Boundary Handling

Layer 31 (for +X, +Y, +Z) and layer 0 (for -X, -Y, -Z) require the
neighboring chunk's adjacent slice. One 128-byte fetch per direction per
affected edge. At most three neighbor chunks (face neighbors) for a corner
voxel edit. Edge and corner neighbors are never needed.

For X-direction specifically: the shift at bit 31 (for +X) brings in a zero,
which means "not occupied at x+1=32." If the neighbor chunk IS occupied at
x=0, this is wrong. The boundary fix:

    all_+x[z][y] = occ[z*32 + y] & ~(occ[z*32 + y] >> 1)
    all_+x[z][y] |= (occ[z*32 + y] >> 31) & ~neighbor_occ_x0[z*32 + y]  // bit 31 fixup

Equivalent fixup for -X at bit 0.


## Stage 2: Derived Metadata

### Layer Face Counts

    layer_count[dir][layer] = sum of popcount(face_bitmask[dir][layer][row])
                              for row in 0..32

32 popcounts per layer, 32 layers per direction, 6 directions = 6144
popcounts per chunk. Output: 6 x 32 x u16 = 384 bytes.

### Layer Prefix Sums

    prefix[dir][0]  = 0
    prefix[dir][i] = prefix[dir][i-1] + layer_count[dir][i-1]

Enables binary search in the vertex shader. 6 x 32 x u32 = 768 bytes per
chunk. Computed alongside face counts.

### Layer Occupancy

    layer_occ[dir] = u32 where bit i is set if layer_count[dir][i] > 0

6 x u32 = 24 bytes.

### Memory Budget Per Chunk

    Occupancy:           4,096 B   (4 KB)
    Face bitmasks:      24,576 B  (24 KB, 6 directions)
    Layer face counts:     384 B
    Layer prefix sums:     768 B
    Layer occupancy:        24 B
    ─────────────────────────────
    Total:              29,848 B  (~29 KB)

Constant regardless of geometric complexity. For 256 loaded chunks: ~7.5 MB.


## Stage 3: Compute Shader -- Draw Command Generation

Runs once per frame. For each visible chunk, for each of 6 directions:

    1. Read layer_occ[chunk][dir]. If zero, skip.
    2. Read prefix[chunk][dir][31] + layer_count[chunk][dir][31] = total faces.
    3. If total > 0: atomically increment global draw count to get draw_index.
    4. Write VkDrawIndirectCommand at draw_index:
         vertexCount   = 6       (two triangles per quad)
         instanceCount = total_faces
         firstVertex   = 0
         firstInstance = 0
    5. Write draw metadata at draw_index:
         chunk_id, direction, buffer offset to prefix sums

The vertex shader uses gl_DrawID to index the metadata buffer and recover
its chunk and direction.

### Draw Count

Maximum: 256 chunks x 6 directions = 1536 draw commands.
Per command: 16 bytes (VkDrawIndirectCommand) + 12 bytes (metadata) = 28 bytes.
Total buffer: 43 KB. Trivially small.

### Compute Cost

1536 potential evaluations. Most are skipped by the layer_occ zero-check.
Typical: 200-400 actual draw commands. Microseconds of GPU time.


## Stage 4: Vertex Shader -- Instance ID to Face Position

This is the most expensive stage per-instance. The vertex shader maps an
instance ID to a world-space face quad.

### Inputs

- gl_DrawID -> metadata -> (chunk_id, direction, prefix_buffer_offset)
- gl_InstanceIndex -> face index within this chunk-direction pair (0..N-1)
- gl_VertexIndex -> 0..5 (which vertex of the two-triangle quad)

### Step 4a: Find the Layer (binary search)

Binary search over 32 prefix-sum entries:

    int lo = 0, hi = 31;
    for (int i = 0; i < 5; i++) {
        int mid = (lo + hi) >> 1;
        if (prefix[base + mid + 1] <= face_id)
            lo = mid + 1;
        else
            hi = mid;
    }
    uint layer    = lo;
    uint local_id = face_id - prefix[base + layer];

5 iterations. Each iteration: 1 buffer read + 2 ALU. Total: ~15 instructions.
Buffer reads are coherent across instances in the same chunk-direction
group (nearby instance IDs search similar ranges).

### Step 4b: Find the Row (linear popcount scan)

Linear scan over the 32 u32 words of the target layer's bitmask:

    uint remaining = local_id;
    uint row;
    for (row = 0; row < 32; row++) {
        uint word = face_bitmask[bitmask_base + row];
        uint count = bitCount(word);
        if (remaining < count) break;
        remaining -= count;
    }

Each iteration: 1 buffer read + 1 bitCount + 1 compare + 1 subtract.
Worst case: 32 iterations = 128 instructions.
Average case: ~16 iterations = 64 instructions.

All instances in the same chunk-direction group read the same 128-byte
layer. This fits in a single GPU cache line. Memory access is coherent even
when control flow diverges.

### Step 4c: Find the Bit (nth set bit extraction)

Extract the remaining-th set bit from the target word:

    uint word = face_bitmask[bitmask_base + row];
    for (uint i = 0; i < remaining; i++) {
        word &= word - 1;  // clear lowest set bit
    }
    uint col = findLSB(word);

Each iteration: 2 ALU (AND + subtract). No memory access.
Worst case: 31 iterations = 62 instructions.
Average case: ~4-8 iterations = 8-16 instructions.

### Step 4d: Position and Quad Generation

Direction + layer + row + col determine the face position in chunk-local
coordinates. The mapping depends on direction:

    +X: face at (layer+1, col, row)    -X: face at (layer, col, row)
    +Y: face at (col, layer+1, row)    -Y: face at (col, layer, row)
    +Z: face at (col, row, layer+1)    -Z: face at (col, row, layer)

gl_VertexIndex (0..5) selects a vertex of the two-triangle quad. The quad
normal and tangent vectors are known statically from the direction. Expand
the face position by +/-0.5 in the two tangent directions.

Cost: ~15 instructions (coordinate swizzle + chunk origin offset + MVP
multiply).

### Total Vertex Shader Cost

    Step              Worst    Average
    ─────────────────────────────────
    Layer search         15       15
    Row scan            128       64
    Bit extraction       62       16
    Position + MVP       15       15
    ─────────────────────────────────
    Total               220      110

A plain vertex shader (MVP transform only) is ~20-30 instructions. This
shader is 4-7x more expensive.

### Throughput Estimate

Modern GPUs process ~10 billion simple vertices/sec. At 4x overhead:
~2.5 billion instances/sec. At 60 fps: ~42 million instances per frame
budget. This exceeds even the pathological 3D-checkerboard worst case
(25M faces for 256 chunks).

### Divergence

Instances within the same draw command access the same chunk-direction
bitmask data but search for different bit positions. Control flow diverges
(different loop iteration counts), but memory access is coherent (same
128-byte layer). GPU wavefronts tolerate control divergence when memory
access is uniform. The cost is idle lanes during the longest iteration,
not cache thrashing.


## Stage 5: Worst-Case Face Counts

### Pathological: 3D Checkerboard

Every other voxel is solid. Each solid voxel has 6 exposed faces.

    Solid voxels per chunk:     16,384  (50%)
    Faces per direction:        16,384
    Faces per chunk (all dirs): 98,304
    Triangles per chunk:       196,608
    256 chunks:                ~25M faces = ~50M triangles

50M triangles at 60 fps requires ~3 billion triangles/sec rasterization
throughput. A mid-range GPU (RTX 3070) sustains ~10 billion triangles/sec.
The pathological case uses ~30% of rasterization budget. Tight but viable.

No real voxel world produces a 3D checkerboard. This is the theoretical
ceiling.

### Realistic Worst Case: Fragmented Terrain

Caves, overhangs, scattered blocks. ~30% solid, ~30-40% surface.

    Surface voxels per chunk:  3,000 - 4,000
    Exposed faces each:        ~3
    Faces per chunk:           9,000 - 12,000
    256 chunks:                ~2.5M - 3M faces = ~5M - 6M triangles

Well within budget at any resolution.

### Realistic Per-Direction: Terrain From Above

Looking straight down at flat terrain. Only -Y faces visible. One layer
per chunk with up to 1024 faces.

    Typical: 256 chunks x 800 faces = 205K faces = 410K triangles

Trivial.

### Comparison With Greedy Meshing

Greedy meshing merges coplanar adjacent faces into larger quads, typically
achieving 5-20x face count reduction for natural terrain. The bitmask
approach renders every 1x1 face individually.

    Bitmask approach:      ~3M faces for 256 terrain chunks
    Greedy equivalent:     ~150K - 600K faces

The bitmask approach is 5-20x more triangles. The trade-off:

    Bitmask                          Greedy
    ──────────────────────────────────────────────────────
    No mesh generation               O(n) meshing per edit
    O(1) voxel edit                  Remesh affected chunk
    Constant memory (29 KB/chunk)    Variable (depends on surface)
    No buffer reallocation           Allocator + fragmentation
    Simple compute shader            Complex mesh builder

Greedy merging can be layered on top of the bitmask representation later.
The per-direction per-layer bitmask is exactly the input that greedy
algorithms expect. This is an extension point, not a redesign.


## Stage 6: Voxel Edit Path

A single voxel flip at (x, y, z) in chunk C:

### Local Recomputation

Each direction has two affected layers: the voxel's own layer and the
adjacent layer (which compared against the voxel's position).

    +X: layers x and x-1     -X: layers x and x+1
    +Y: layers y and y-1     -Y: layers y and y+1
    +Z: layers z and z-1     -Z: layers z and z+1

Up to 12 layer recomputes. Each recompute is an AND-NOT on a 32x32 bitmask
(128 bytes), followed by popcount (32 words) and prefix sum update.

    Data touched:  12 x 128 bytes = 1,536 bytes
    Operations:    12 x 32 AND-NOTs + 12 x 32 popcounts = 768 total
    Prefix sums:   6 x 32 additions = 192 additions

### Cross-Chunk Propagation

If the voxel is on a chunk boundary (x=0/31, y=0/31, z=0/31), the
neighboring chunk's adjacent face layer also needs recomputation. A
boundary voxel affects at most 3 neighbor chunks (face neighbors, never
edge or corner neighbors). Each neighbor requires up to 2 additional layer
recomputes.

    Worst case (corner voxel): 12 local + 6 cross-chunk = 18 layer recomputes
    Data touched: 18 x 128 = 2,304 bytes

### Edit Latency

Total per edit: ~2.3 KB data touched, ~1000 arithmetic operations.
Microseconds on CPU or GPU. No buffer reallocation. No remeshing.
The next frame's compute dispatch picks up the new counts automatically.


## Potential Optimizations

Listed for reference. None are required for the base architecture to be
viable.

### Row Prefix Sums

Precompute a 32-entry prefix sum per layer to replace the linear row scan
(step 4b) with a binary search. Reduces worst-case row scan from 128 to
~15 instructions.

Cost: 6 directions x 32 layers x 32 entries x 4 bytes = 24 KB per chunk.
For 256 chunks: 6 MB. Worthwhile if vertex shader throughput becomes the
bottleneck.

### Two-Level Row Hierarchy

Split the 32 rows into 4 groups of 8. Store 4 group-level popcounts per
layer. Binary search over groups (2 iterations) then linear scan within a
group (up to 8 iterations). Total: ~10 iterations instead of up to 32.

Cost: 6 x 32 x 4 x u16 = 1.5 KB per chunk. Negligible.

### Precomputed Face Positions

The compute shader writes a buffer of explicit face positions (layer, row,
col packed into a u32) per face. The vertex shader does a single buffer
read instead of the scan.

Cost: variable per chunk. Worst case 98K faces x 4 bytes = 393 KB per
chunk. Requires a pool allocator with the fragmentation and compaction
complexity that the bitmask approach was designed to avoid. Only consider
if the vertex shader scan proves to be the dominant bottleneck and the
simpler optimizations above are insufficient.


## Open Issues

1. **Material data association.** Each face needs a voxel type for texturing.
   Simplest approach: a 32x32x32 byte array (32 KB/chunk) storing palette
   indices, indexed by the recovered (x, y, z) position. Adds 32 KB per
   chunk (8 MB for 256 chunks). Alternative: pack material into the
   occupancy bitmask using wider words.

2. **Frustum culling granularity.** Chunk-level AABB culling in the compute
   shader is the natural first pass. Per-direction culling is possible (if
   the camera faces +X, no chunk's -X faces contribute) but the layer
   occupancy zero-check already skips empty directions. Chunk-level culling
   is likely sufficient.

3. **LOD.** Coarser bitmask levels (16x16x16, 8x8x8) are natural mip levels
   of the occupancy. A 16x16x16 bitmask is 512 bytes occupancy + ~3 KB
   faces. Reduces face count by ~8x for distant chunks. The prefix-sum
   vertex shader works identically at any resolution.

4. **Transparency and non-cubic shapes.** The bitmask representation assumes
   opaque 1x1x1 cubes. Transparent blocks require separate handling (no
   face culling at transparent/opaque boundaries, draw order for blending).
   Non-cubic shapes (slabs, stairs) cannot be represented as single bits.
   Both likely require a secondary rendering path.
