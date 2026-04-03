# Directional Face Acceleration

A surface representation and rendering architecture for voxel worlds that
exploits the axis-aligned nature of voxel faces. Surfaces are derived from
occupancy bitmasks, stored as directional face bitmasks, and rendered via
indirect rasterization with no traditional mesh generation step.


## Problem

Traditional octree-based ray marching has pathological worst cases in voxel
worlds:

- **Fragmented surface shells.** The octree subdivides space uniformly. A
  surface that is geometrically simple (flat terrain) still shatters into
  thousands of leaf nodes because the octree doesn't know it's simple. A ray
  traveling parallel to the surface (the default gameplay camera looking at
  the horizon) traverses hundreds of these fragmented nodes.

- **Warp divergence.** Rays in the same GPU warp take wildly different step
  counts. Short rays (hit the ground nearby) and long rays (travel to distant
  mountains) share a warp, and the warp runs at the speed of the slowest ray.
  Idle threads waste compute.

- **Random memory access.** Octree traversal is pointer chasing through
  scattered memory. Different rays in a warp chase different tree paths,
  thrashing the cache.

- **Register pressure.** Deep hierarchy traversal requires stack state per ray,
  consuming registers, reducing occupancy, and amplifying memory stall costs.

These problems compound: divergence wastes threads, high register count reduces
occupancy, low occupancy can't hide the latency of random memory access.


## Core Insight

Every voxel face is axis-aligned and points in exactly one of six directions:
+X, -X, +Y, -Y, +Z, -Z. A traditional octree mixes all face directions into
the same spatial structure. Separating them by direction transforms the problem.

Each directional set is a **2.5D problem**: all +Y faces are horizontal quads
that vary only in X and Z. Their Y extent is zero. The same applies to every
other direction: +X faces have zero X extent, etc. This is fundamentally
simpler than a general 3D spatial query.


## View Constraint

For any ray, at most 3 of the 6 directional structures can produce a visible
hit (backface culling: a ray only sees faces whose normal has a negative dot
product with the ray direction). Half the structures are skipped before any
traversal begins.

In a typical voxel game with a nearly-upright camera, this constraint is even
stronger. A player standing on terrain cannot see +Y faces above them or -Y
faces below them. In practice, often only 1-2 directional structures contribute
meaningfully to a given view.


## Structure: Three Levels

### Level 1: Direction Selection

Dot the ray direction with each of the six axis normals. Discard directions
where the dot product is non-negative (backface). At most 3 structures remain.

### Level 2: Layer Occupancy Skip

Within each directional structure, faces are organized by their coordinate
along the face normal axis. All +Y faces at y=64 form one **layer**. All +Y
faces at y=65 form another.

A single u32 per direction (`layer_occ`) records which of the 32 layers
contain any faces. findFirstBit on this u32 locates the next occupied layer
in one cycle. A ray traveling through empty layer ranges jumps directly to
the next occupied layer without testing each one.

An earlier design proposed a layer group hierarchy with 2D bounding extents
per group, allowing rays to skip groups whose faces were spatially distant
from the ray path. When the primary rendering strategy shifted to indirect
rasterization, the data layout was optimized for raster (prefix sums,
popcounts, vertex shader binary search) and the hierarchy was simplified to
the flat u32. The flat skip is sufficient for rasterization and for
short-range ray techniques. Per-layer 2D bounds remain a potential extension
for ray-heavy workloads where long rays pass through many occupied but
spatially irrelevant layers.

### Level 3: 2D Face Set

Within a layer or layer group, the actual face positions form a 2D set in the
two non-normal axes. This is where intersection testing happens: does the ray
hit any face in this 2D region?

The faces are axis-aligned, uniform size (1x1), at integer coordinates. This
extreme regularity means the 2D structure does not need a general-purpose BVH.
Bitmask-based representations are the leading candidate (see below).


## Data Layout

### Storage Layout

The occupancy bitmask is stored as 1024 u32 words:

    occ[z * 32 + y] = u32, bit x set if voxel (x, y, z) is occupied

X is the intra-word axis (bit index), Y is the intra-layer axis (word
offset within a 32-word layer), Z is the inter-layer axis (layer index).
This layout makes face derivation cheap for all six directions without
requiring a full transpose. See the computational analysis for details.


### Source of Truth: Occupancy Bitmask

A 32x32x32 chunk stores its voxel occupancy as a bitmask:

    32 layers x 32x32 bits = 32 x 128 bytes = 4 KB per chunk

This is the only authored data. Everything below is derived from it.

### Derived: Directional Face Bitmasks

A face exists at a solid/air boundary. For each direction, the face bitmask is
a single bitwise operation between adjacent occupancy layers:

    face_+x[x] = occupancy[x] & ~occupancy[x+1]
    face_-x[x] = occupancy[x] & ~occupancy[x-1]
    face_+y[y] = occupancy[y] & ~occupancy[y+1]
    (etc.)

For Z and Y directions, this is a direct AND-NOT between adjacent slices or
rows. For X directions, the face detection is a bit shift within each u32
word, but the output must be transposed from bit-position-indexed to
layer-indexed form. This transposition runs in the face derivation compute
shader using shared memory. See the computational analysis for the concrete
algorithm.

Per direction: 32 layers x 32x32 bits = 4 KB.
All 6 directions: ~24 KB per chunk.

Edge case: the last layer (e.g., x=31) requires the neighboring chunk's x=0
slice for the boundary comparison. One 32x32 bitmask fetch from the neighbor.

### Derived: Layer Face Counts

Precomputed popcount per layer, per direction:

    layer_face_count[layer] = popcount(face_bitmask[layer])

Per direction: 32 x u16 = 64 bytes.
All 6 directions: 384 bytes per chunk.

Recomputed only for affected layers when a voxel changes.

### Derived: Layer Occupancy

A single u32 per direction indicating which of the 32 layers have any faces:

    layer_occupancy = bitwise OR of (layer_face_count[i] != 0) for each layer

6 x u32 = 24 bytes per chunk. Used for fast layer skipping (findFirstBit).

### Derived: Layer Prefix Sums

Prefix sum of layer face counts, per direction:

    prefix[dir][0] = 0
    prefix[dir][i] = prefix[dir][i-1] + layer_face_count[dir][i-1]

Per direction: 32 x u32 = 128 bytes.
All 6 directions: 768 bytes per chunk.

Used by the vertex shader to binary-search from instance ID to layer index.
Recomputed only for affected directions when a voxel changes.


### Total GPU-Side Data Per Chunk

    Occupancy:            4 KB  (source, also needed for game logic)
    Face bitmasks:       24 KB  (6 directions)
    Layer face counts:  384 B   (6 directions)
    Layer prefix sums:  768 B   (6 directions)
    Layer occupancy:     24 B   (6 directions)
    ────────────────────────────
    Total:              ~29 KB

Constant regardless of geometric complexity.
For 256 loaded chunks: ~7.5 MB.


## Rendering Pipeline

### No Mesh Generation

The traditional voxel pipeline is:

    occupancy -> mesh generation -> vertex/index buffers -> render

This architecture eliminates mesh generation entirely:

    occupancy -> face bitmasks -> render

There are no vertex buffers, no index buffers, no mesh data. The face bitmasks
are the renderable representation. The GPU reads them directly.

### Indirect Rasterization

A compute shader prepares draw commands by summing the precomputed layer face
counts:

    Per chunk, per direction:
      total_faces = sum(layer_face_count[0..31])
      if total_faces > 0:
        emit indirect draw entry

The vertex/mesh shader maps an instance ID to a specific face by walking the
bitmask: prefix sum of popcounts identifies the layer, then bit scan within
the layer identifies the position. From direction + layer index + bit position,
the world-space quad vertices are trivially computed.

### Single Multi-Draw-Indirect

All visible chunk-direction pairs are batched into a single MultiDrawIndirect
call. The compute shader writes one draw command per non-empty chunk-direction
pair and atomically increments the draw count.

The entire renderer is two GPU commands per frame:

    1. Compute dispatch: sum layer counts, write indirect buffer
    2. MultiDrawIndirect: render everything

### Voxel Edit Path

When a voxel changes:

    1. Flip the occupancy bit (CPU or compute)
    2. AND-NOT to recompute affected face bitmask layers (1-3 per direction)
    3. popcount to update those layer_face_count entries
    4. Next frame, the compute shader picks up the new counts automatically

No remeshing. No buffer reallocation. A voxel edit touches a few hundred bytes
of derived data.


## Ray Traversal

The directional face bitmask structure also serves as a ray acceleration
structure. The traversal has three steps:

1. **Direction selection.** Dot the ray direction with each of the six axis
   normals. Discard directions where the dot product is non-negative
   (backface). At most 3 structures remain.

2. **Layer skip.** The layer occupancy u32 enables fast skipping along the
   normal axis. findFirstBit locates the next occupied layer in one cycle. A
   ray traveling through empty layer ranges jumps directly to the next
   occupied layer.

3. **2D face test.** Within an occupied layer, the ray's position maps to a
   bit index in the 32x32 bitmask. One bit test determines intersection.

**Cross-chunk:** A ray exiting a chunk looks up the next chunk in the chunk
grid. The layer occupancy u32 for the next chunk immediately indicates
whether there is anything to hit. The chunk grid acts as the coarsest
traversal level.

For applications of ray traversal through this structure, see:
- [Voxel Global Illumination](voxel_global_illumination.md) — shadows, AO,
  GI, reflections
- [Scene Acceleration](scene_acceleration.md) — coarse acceleration for
  arbitrary triangle geometry


## Relationship to Meshing

This decomposition mirrors how greedy meshing works: process one face direction
at a time, sweeping through 2D slices along the normal axis. The key difference
is that greedy meshing merges coplanar faces into larger quads to reduce
triangle count. This architecture instead keeps faces at single-voxel
granularity and relies on the GPU's ability to render large instance counts
efficiently via indirect draws.

Greedy merging could still be applied on top of this system if triangle count
becomes a bottleneck, but the bitmask representation is simpler, faster to
update, and avoids the complexity of maintaining merged quads across edits.


## Open Questions

- Material/texture data: how to associate voxel type information with face
  instances without per-vertex attributes. Leading approach: a 32x32x32
  byte array (32 KB/chunk) storing palette indices, indexed by the
  recovered face position.
- Transparency and non-cubic block shapes: how far the bitmask
  representation extends beyond simple opaque cubes. Likely requires a
  secondary rendering path.
- LOD integration: coarser bitmask levels (16x16x16, 8x8x8) as natural
  mip levels for distant chunks. The vertex shader prefix-sum mapping works
  identically at any resolution.


## Resolved Questions

- **Prefix sum mapping from instance ID to face position in the vertex
  shader.** Binary search over 32 layer prefix sums (5 iterations), linear
  popcount scan over 32 row words (16 avg iterations), nth-set-bit
  extraction within the target word (~8 avg iterations). Total: ~110
  instructions average, ~220 worst case. 4-7x a plain vertex shader.
  Within budget for realistic scenes. See the computational analysis for
  the full breakdown.
- **View frustum culling.** Chunk-level AABB test in the compute shader
  before emitting draw commands. Per-direction culling is possible but the
  layer occupancy zero-check already eliminates empty directions.
- **X-direction face derivation.** Bit shift within each u32 word, followed
  by a bit-plane transpose in the compute shader using shared memory. Same
  cost class as the other directions.
