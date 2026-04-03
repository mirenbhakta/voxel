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

### Layer Occupancy

    layer_occ[dir] = u32 where bit i is set if any face exists in layer i

6 x u32 = 24 bytes. Used for fast layer skipping during ray traversal and
as the first-pass cull in the filter shader (zero layer_occ means no
geometry for that direction -- skip it).

### Quad Buffer

Produced by the build step's greedy merge. Each entry is a packed u32
quad descriptor:

    col     5 bits     column position within the layer
    row     5 bits     row position within the layer
    layer   5 bits     layer index along the face normal axis
    width   5 bits     horizontal extent minus one
    height  5 bits     vertical extent minus one
    ────────────────
    total  25 bits     fits in a u32 with room to spare

Organized by chunk and direction. Direction is implicit from draw command
metadata (gl_DrawID), not stored per quad.

Variable size per chunk-direction pair. Each 32x32 layer contains at most
1024 faces pre-merge; the greedy merge reduces this substantially. A flat
wall merges to a single quad. Typical chunks produce a few KB of quad data.

### Memory Budget Per Chunk

    Occupancy:            4,096 B   (4 KB)
    Volumetric material: 32,768 B  (32 KB)
    Face bitmasks:       24,576 B  (24 KB, 6 directions)
    Layer occupancy:         24 B
    Quad buffer:          variable  (typically a few KB)
    ─────────────────────────────────
    Fixed total:        ~60 KB + quad buffer

For 256 loaded chunks: ~15 MB fixed + quad buffers.


## Stage 3: Build Shader (On Edit)

Runs when chunk occupancy changes. Derives face bitmasks, performs
material-agnostic greedy merge, writes quad buffer.

### Face Derivation

Same operations as Stage 1. Per direction: 32 layers of 32 AND-NOT
operations, plus the X-direction transpose via shared memory. Total per
chunk: 6 directions of face bitmask derivation. Cost dominated by the
X-direction transpose (32K extract-and-set operations).

### Greedy Merge

The merge operates on 32x32 bitmask layers using bitwise operations. It
is material-agnostic -- material is resolved per-pixel in the fragment
shader, so the merge considers only geometry.

Per layer (32x32 bitmask):

    1. Row-wise run detection via bit manipulation:
         starts = bits & ~(bits << 1)   // left edges of runs
         ends   = bits & ~(bits >> 1)   // right edges of runs
       Each (start, end) pair defines a horizontal span.

    2. Cross-row merging: extend spans vertically by comparing adjacent
       rows. Two spans in adjacent rows with identical start and end
       positions merge into a single rectangle.

Total work per direction: 32 layers of merge operations. Cost is
proportional to the number of set bits and transitions in the bitmask,
not the grid size. A fully solid layer has one transition per row (32
spans, merging to 1 rectangle). A checkerboard layer has 512 transitions
per row (no merging possible).

For local edits, the rebuild is scoped to 4x4x4 subchunk blocks. Only
the affected face layers need face derivation and re-merge. A single
voxel edit touches at most 12 layers (2 per direction) and re-merges
only those layers.


## Stage 3b: Filter Shader (Per Frame)

Runs each frame. For each visible chunk, for each of 6 directions:

    1. Read layer_occ[chunk][dir]. If zero, skip.
    2. Backface test: dot product of face normal with view direction.
       If non-negative, skip.
    3. Frustum test: chunk AABB against view frustum planes.
    4. Hi-Z occlusion test. All geometry is axis-aligned planar slabs.
       4 coplanar corners projected to screen space, single depth value
       compared against the Hi-Z pyramid. No conservative bounding box
       approximation -- the geometry IS its own bound.
    5. If visible: emit one MDI command pointing into the quad buffer.

### Evaluation Count

Maximum: 256 chunks x 6 directions = 1536 evaluations. Most are
eliminated early:

    - layer_occ zero-check eliminates empty directions (cheap, ~1 cycle)
    - Backface test eliminates 3 of 6 directions for any view (~1 cycle)
    - Frustum test eliminates off-screen chunks (~5 cycles)
    - Hi-Z test eliminates occluded chunk-direction pairs (~10 cycles)

Typical surviving draw commands: 200-400. The Hi-Z test adds a few
instructions per candidate that passes the frustum test, but eliminates
occluded draws that would otherwise waste vertex and rasterization
throughput.

Per command: 16 bytes (VkDrawIndirectCommand) + metadata = ~28 bytes.
Total buffer at maximum: 43 KB. Trivially small.


## Stage 4: Vertex Shader

The vertex shader reads one packed u32 quad descriptor from the quad
buffer, unpacks 5 fields, and expands to 6 vertices (two triangles).

### Inputs

- gl_DrawID -> metadata -> (chunk_id, direction, quad_buffer_offset)
- gl_InstanceIndex -> quad index within this chunk-direction pair
- gl_VertexIndex -> 0..5 (which vertex of the two-triangle quad)

### Unpack and Expand

    uint descriptor = quad_buffer[offset + gl_InstanceIndex];
    uint col    = (descriptor >>  0) & 31;
    uint row    = (descriptor >>  5) & 31;
    uint layer  = (descriptor >> 10) & 31;
    uint width  = ((descriptor >> 15) & 31) + 1;
    uint height = ((descriptor >> 20) & 31) + 1;

Direction comes from draw metadata. The direction determines the axis
mapping (which of col/row/layer corresponds to x/y/z) and the quad
orientation.

gl_VertexIndex (0..5) selects a vertex of the two-triangle quad. The
quad spans from (col, row) to (col + width, row + height) in the
layer's 2D coordinate system.

Position + chunk origin offset + MVP multiply.

### Cost

    Unpack:       ~10 instructions (shifts, masks, adds)
    Position:     ~15 instructions (axis swizzle, offset, MVP)
    ─────────────────────────────────────────────────────────
    Total:        ~20-30 instructions

Comparable to a plain MVP vertex shader. No binary search, no popcount
scan, no loop of any kind.

### Comparison With Previous Architecture

The previous architecture mapped instance IDs to face positions at
runtime in the vertex shader:

    Step              Worst    Average
    ─────────────────────────────────
    Layer search         15       15
    Row scan            128       64
    Bit extraction       62       16
    Position + MVP       15       15
    ─────────────────────────────────
    Total               220      110

The new vertex shader replaces all of this with a single buffer read
and unpack. The 4-7x overhead relative to a plain vertex shader is
eliminated entirely.


## Stage 4b: Fragment Shader

The fragment shader computes the voxel position from world-space
coordinates and reads material from the volumetric material array:

    ivec3 voxel = ivec3(floor(worldPos)) & 31;
    uint material = material_volume[voxel.z * 1024 + voxel.y * 32 + voxel.x];

The material index feeds into the texture atlas or color palette for
final shading.

### Memory Access Coherence

Adjacent pixels on the same quad read adjacent or identical voxel
entries. A 4x4 quad covers at most 4x4 = 16 distinct voxel positions.
Pixels within the same voxel (at close range) read the same entry.
Pixels spanning adjacent voxels (at medium range) read entries that
differ by 1 in one axis -- adjacent in memory for X, stride-32 for Y,
stride-1024 for Z. All patterns are cache-friendly.

### Cost

    floor + bitwise AND:          ~3 instructions
    Array index computation:      ~3 instructions
    Buffer read:                  ~4 cycles (L1 cache hit expected)
    Texture atlas lookup:         ~4 cycles
    ───────────────────────────────────────────────
    Total overhead:              ~10-15 instructions

Beyond what a plain textured fragment shader would cost. The volumetric
lookup replaces per-vertex material attributes, so there is no
interpolation cost and no per-vertex storage for material data.


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

Greedy merge provides no benefit here. No two adjacent faces share an
edge in the same layer, so no merging is possible. Raw face count =
quad count = 98K per chunk.

### Realistic Worst Case: Fragmented Terrain

Caves, overhangs, scattered blocks. ~30% solid, ~30-40% surface.

    Surface voxels per chunk:  3,000 - 4,000
    Exposed faces each:        ~3
    Faces per chunk:           9,000 - 12,000
    256 chunks:                ~2.5M - 3M faces = ~5M - 6M triangles

Well within budget at any resolution.

After greedy merge: roughly 150K-500K quads for 256 chunks. The
material-agnostic merge is more aggressive than traditional greedy
meshing because material boundaries do not fragment the mesh.

### Realistic Per-Direction: Terrain From Above

Looking straight down at flat terrain. Only -Y faces visible. One layer
per chunk with up to 1024 faces.

    Typical: 256 chunks x 800 faces = 205K faces = 410K triangles

After greedy merge: 256 chunks x ~1-5 quads per chunk (contiguous
terrain merges to very few rectangles) = 256-1280 quads. Trivial.

### Comparison With Greedy Meshing

The architecture performs greedy meshing, but material-agnostic.
Traditional greedy meshing stops at material boundaries because each
quad carries a single material ID. Material-agnostic merging considers
only geometry and produces maximally large quads regardless of material
distribution. The post-merge quad count is lower than traditional
greedy meshing for heterogeneous surfaces.

    Traditional greedy:            ~150K - 600K quads (256 terrain chunks)
    Material-agnostic greedy:      ~150K - 500K quads (same geometry)

The difference grows with material heterogeneity. A surface with 10
materials in a 32x32 layer might produce 30-50 quads with traditional
greedy (one per contiguous same-material region) but 1-5 quads with
material-agnostic merging (one per contiguous geometry region).

Estimated post-merge quad counts:

    Scenario                       Raw faces    Post-merge quads
    ─────────────────────────────────────────────────────────────
    Flat terrain (top-down)         205K         256 - 1,280
    Fragmented terrain (256 ch)    ~3M           150K - 500K
    3D checkerboard (per chunk)     98K          98K (no merge)


## Stage 6: Voxel Edit Path

A single voxel flip at (x, y, z) in chunk C:

### Local Recomputation

Each direction has two affected layers: the voxel's own layer and the
adjacent layer (which compared against the voxel's position).

    +X: layers x and x-1     -X: layers x and x+1
    +Y: layers y and y-1     -Y: layers y and y+1
    +Z: layers z and z-1     -Z: layers z and z+1

Up to 12 layer recomputes. Each recompute:

    1. AND-NOT on a 32x32 bitmask (128 bytes) to regenerate face bits
    2. Re-run greedy merge for the affected layer
    3. Patch the quad buffer (replace entries for the affected layer)

For local edits, the rebuild is scoped to 4x4x4 subchunk blocks. Only
the layers intersecting the edited region need recomputation. Patching
the quad buffer replaces entries for affected layers without full
reallocation -- the buffer region for each chunk-direction pair is
updated in place.

    Data touched:  12 x 128 bytes = 1,536 bytes (face bitmasks)
                   + affected quad buffer entries
    Operations:    12 x 32 AND-NOTs + 12 layer merges

### Cross-Chunk Propagation

If the voxel is on a chunk boundary (x=0/31, y=0/31, z=0/31), the
neighboring chunk's adjacent face layer also needs recomputation. A
boundary voxel affects at most 3 neighbor chunks (face neighbors, never
edge or corner neighbors). Each neighbor requires up to 2 additional layer
recomputes.

    Worst case (corner voxel): 12 local + 6 cross-chunk = 18 layer recomputes
    Data touched: 18 x 128 = 2,304 bytes (face bitmasks)

### Edit Latency

Total per edit: ~2.3 KB face bitmask data touched, plus the greedy
re-merge and quad buffer patch for affected layers. Microseconds on
CPU or GPU. No full buffer reallocation. No remeshing from scratch.
The next frame's filter shader picks up the changes automatically.


## Potential Optimizations

Listed for reference. None are required for the base architecture to be
viable.

### Hierarchical Greedy Merge

Multi-resolution merge that finds large rectangles first, then fills
gaps with smaller ones. Could reduce merge time for dense surfaces where
the current single-pass merge spends time extending spans that will
eventually merge into the same large rectangle. Most beneficial for
surfaces that are locally uniform (flat walls, terrain plateaus).

### Persistent Quad Buffer Pool

Pre-allocate quad buffer space per chunk-direction pair based on
historical usage. A chunk-direction pair that consistently produces
~50 quads gets a 50-entry slot. Edits that do not significantly change
the quad count reuse the same slot without reallocation. Only overflow
triggers a resize. Avoids allocator pressure on rapid edits.

### Subchunk Incremental Merge

Instead of re-merging entire affected layers on edit, patch the quad
buffer by removing quads that overlap the edited 4x4x4 region and
inserting new ones for just that region. Avoids re-merging a full
32x32 layer when only a small area changed. Only worthwhile if the
full layer merge becomes a measured bottleneck -- a 32x32 bitmask
merge is already fast.


## Open Issues

1. **LOD.** Coarser bitmask levels (16x16x16, 8x8x8) are natural mip
   levels of the occupancy. The face derivation, greedy merge, and
   quad buffer pipeline are resolution-agnostic -- they work identically
   at any bitmask resolution. A 16x16x16 level reduces face count by
   ~8x for distant chunks.

2. **Transparency and non-cubic shapes.** The bitmask representation assumes
   opaque 1x1x1 cubes. Transparent blocks require separate handling (no
   face culling at transparent/opaque boundaries, draw order for blending).
   Non-cubic shapes (slabs, stairs) cannot be represented as single bits.
   Both likely require a secondary rendering path.
