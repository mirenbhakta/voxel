# Directional Face Acceleration

A surface representation and rendering architecture for voxel worlds that
exploits the axis-aligned nature of voxel faces. Surfaces are derived from
occupancy bitmasks, merged into greedy quads by a GPU compute shader, and
rendered via indirect rasterization with per-pixel material lookup from a
volumetric palette.


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

This is the only authored geometry data. Everything below is derived from it.

### Source of Truth: Volumetric Material

A 32x32x32 byte array storing palette indices. 32 KB per chunk. Each entry
is a local palette index (0-255) that maps to a global block ID through the
chunk's local palette.

The local palette allows the global block ID domain to be arbitrarily wide
(16-bit, 32-bit, or larger) while per-voxel storage cost stays at 1 byte.
A chunk with fewer than 256 distinct materials -- the common case -- pays
no more than 1 byte per voxel regardless of the global ID space.

The volumetric material array is persistent on GPU. The fragment shader
reads material directly from this array using world-space position. The
build compute shader also reads it during the greedy merge step.

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

These bitmasks serve dual purpose: input to the greedy merge during the
build step, and acceleration structure for ray traversal.

Edge case: the last layer (e.g., x=31) requires the neighboring chunk's x=0
slice for the boundary comparison. One 32x32 bitmask fetch from the neighbor.

### Derived: Layer Occupancy

A single u32 per direction indicating which of the 32 layers have any faces:

    layer_occupancy = bitwise OR reduction of face_bitmask words per layer

6 x u32 = 24 bytes per chunk. Used for fast layer skipping (findFirstBit)
during ray traversal.

### Derived: Quad Buffer

Produced by the build step's greedy merge. Each entry is a packed quad
descriptor encoding position and extent:

    position:  col 5 bits, row 5 bits, layer 5 bits
    extent:    width-1 5 bits, height-1 5 bits
    total:     25 bits, fits in a u32 with room to spare

Organized by chunk and direction. Direction is implicit from the draw
command metadata (gl_DrawID), not stored per quad.

Variable size per chunk-direction pair but bounded. Each 32x32 layer
contains at most 1024 faces pre-merge; the greedy merge reduces this
substantially. A flat wall merges to a single quad. Typical chunks
produce a few KB of quad data.


### Total GPU-Side Data Per Chunk

    Occupancy:            4 KB   (source, ray traversal + game logic)
    Volumetric material: 32 KB   (source, palette-indexed)
    Face bitmasks:       24 KB   (6 directions, ray traversal + build)
    Layer occupancy:     24 B    (6 directions, ray traversal)
    Quad buffer:         variable (greedy-merged, typically a few KB)
    ────────────────────────────
    Fixed total:        ~60 KB + quad buffer

For 256 loaded chunks: ~15 MB fixed + quad buffers.

Ray traversal uses a subset of this data: occupancy (4 KB), face bitmasks
(24 KB), and layer occupancy (24 B) -- roughly 28 KB per chunk.


## Rendering Pipeline

### No Mesh Generation

The traditional voxel pipeline is:

    occupancy -> mesh generation -> vertex/index buffers -> render

This architecture eliminates traditional mesh generation:

    occupancy -> face bitmasks -> greedy merge -> quad buffer -> render

There are no vertex buffers, no index buffers, no mesh data in the
conventional sense. The quad buffer is a flat array of packed descriptors
produced by a compute shader. The face bitmasks are an intermediate form
consumed by the build step, not a renderable representation.

### Build Step (On Edit)

A compute shader runs when chunk occupancy changes. It performs:

1. Read the occupancy bitmask.
2. Derive face bitmasks via AND-NOT between adjacent layers (same as before,
   including X-direction bit shift + shared memory transpose).
3. Perform material-agnostic greedy merge on the face bitmasks.
4. Write packed quad descriptors to the quad buffer.
5. Compute layer occupancy from the face bitmasks (for ray traversal).

The greedy merge operates purely on geometry -- it does not consider material
boundaries. This is possible because material is resolved per-pixel in the
fragment shader via volumetric lookup, not per-quad. The merge produces
maximally large quads regardless of material distribution. A 32x32 flat wall
of 5 different materials merges into a single quad.

For local edits, the rebuild can be scoped to 4x4x4 subchunk blocks. Only
the affected face layers and quad buffer regions need recomputation. A
single voxel edit touches a few KB of data.

### Filter Step (Per Frame)

A compute shader runs each frame to determine visibility:

1. For each chunk, test against the view frustum (AABB test).
2. For each direction, backface test (dot product with view direction).
3. Hierarchical Z-buffer occlusion culling.

Hi-Z culling is particularly effective here because all geometry is
axis-aligned quads. In a general pipeline, Hi-Z tests project bounding
boxes (8 corners) to get a conservative screen-space rectangle. With
axis-aligned quads, the geometry IS its own bound. Each chunk-direction
pair covers a planar slab -- 4 coplanar corners, exact screen-space
projection, single depth value. No conservative approximation needed.

For each visible chunk-direction pair, the filter shader emits one
MultiDrawIndirect command pointing into the quad buffer.

### Render Step (Per Frame)

The vertex shader reads a packed quad descriptor from the quad buffer,
unpacks position and extent, and expands to 6 vertices (two triangles).
Direction comes from draw command metadata (gl_DrawID). Roughly 20-30
instructions, comparable to a plain vertex shader.

The fragment shader computes the voxel position from world-space
coordinates and reads the material from the volumetric material array:

    ivec3 voxel = ivec3(floor(worldPos)) & 31;
    uint material = material_volume[voxel.z * 1024 + voxel.y * 32 + voxel.x];

The material index feeds into the texture atlas or color palette for
final shading.

The entire renderer is three GPU commands per frame:

    1. Compute dispatch: filter visibility, write MDI commands
    2. MultiDrawIndirect: render all visible quads
    3. (The build dispatch only runs on edit, not per frame)


### Voxel Edit Path

When a voxel changes:

    1. Flip the occupancy bit
    2. AND-NOT to recompute affected face bitmask layers (scoped to
       4x4x4 subchunk)
    3. Re-run greedy merge for affected layers
    4. Patch the quad buffer
    5. Next frame, the filter shader picks up the changes automatically

No remeshing. No buffer reallocation. A voxel edit touches a few KB of
derived data. Cross-chunk propagation: boundary voxels affect at most 3
face-neighbor chunks (one per axis where the voxel sits on the chunk edge).


## 2x2 Quad Technique

A single quad can render 4 discrete voxel textures using world-space UV
partitioning. A 2x2 group of coplanar faces maps to one quad whose 4
corners align with the 4 original voxel cell centers. The fragment shader
uses `floor(worldPos)` to determine which quadrant the pixel falls in and
samples the corresponding material's texture.

The greedy merge can group 2x2 blocks of faces into single quads where all
4 cells are occupied. Partially occupied 2x2 groups (at surface edges) fall
back to individual quads. At higher voxel resolution, the surface is
smoother relative to the grid, so a larger fraction of groups are fully
occupied.

Two applications:

- **Resolution doubling.** A 64x64x64 chunk produces roughly the same
  instance count as a 32x32x32 chunk with 1x1 quads. Doubled voxel
  resolution per axis at the same rendering cost.

- **LOD transitions.** A LOD 1 quad covers a 2x2 area of LOD 0 voxels
  with full material fidelity. Each quadrant displays its own discrete
  material, preserving visual sharpness without blending.


## LOD

Coarser bitmask levels are natural mip levels of the occupancy:

    LOD 0: 32x32x32 occupancy + full material palette    (36 KB CPU)
    LOD 1: 16x16x16 occupancy + averaged RGBA per voxel  (16.5 KB CPU)
    LOD 2:  8x8x8   occupancy + averaged RGBA per voxel  (2 KB CPU)
    LOD 3:  4x4x4   occupancy + averaged RGBA per voxel  (264 B CPU)

The face derivation, greedy merge, and quad buffer pipeline are
resolution-agnostic. They work identically at any bitmask resolution.

### Material Handling by LOD Level

- **LOD 0:** Discrete material palette indices. The fragment shader does a
  texture atlas lookup using the palette index.
- **LOD 1+:** Pre-baked averaged colors, computed when the chunk transitions
  from a finer LOD. The fragment shader reads flat color directly -- no atlas
  lookup.

At LOD 1, the 2x2 quad technique can render 4 discrete averaged colors per
quad with hard edges between them, preserving visual sharpness without
blending artifacts.

### Occupancy Downsampling

Rule: "any surface voxel in the 2x2x2 group is solid." This preserves the
surface shell at the cost of slight thickening. Thin features that
disappear at lower LODs are subpixel at the distances where those LODs
activate.

### LOD Transitions

Dithered cross-fade between adjacent LOD levels. Opaque geometry only -- no
alpha blending, no sort order issues.


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

Ray traversal data is a subset of the full per-chunk data: occupancy
(4 KB), face bitmasks (24 KB), and layer occupancy (24 B). Roughly 28 KB
per chunk.

For applications of ray traversal through this structure, see:
- [Voxel Global Illumination](voxel_global_illumination.md) -- shadows, AO,
  GI, reflections
- [Scene Acceleration](scene_acceleration.md) -- coarse acceleration for
  arbitrary triangle geometry


## Relationship to Meshing

This architecture performs greedy meshing, but a variant that differs from
the traditional approach in several ways:

- **Material-agnostic.** Traditional greedy meshing fragments quads at
  material boundaries because each quad carries a single material ID. Here,
  materials are resolved per-pixel in the fragment shader via volumetric
  lookup. The merge considers only geometry, producing maximally large quads
  regardless of material distribution.

- **GPU-resident.** The merge runs in a compute shader, not on the CPU. No
  data round-trips between CPU and GPU.

- **Bitmask-native.** The merge operates on 32x32 bitmask layers using
  bitwise operations (shifts, ANDs, leading/trailing zero counts), not
  polygon soup or vertex lists.

- **Edit-scoped.** The merge only reruns for affected layers on voxel edit,
  not per frame. The per-frame cost is zero for static geometry.

The combination of material-agnostic merging and per-pixel material lookup
is the central design trade-off. It moves material complexity from the
geometry pipeline (where it fragments quads and inflates instance counts)
to the fragment shader (where it becomes a single texture fetch per pixel).


## Open Questions

- Transparency and non-cubic block shapes: how far the bitmask
  representation extends beyond simple opaque cubes. Likely requires a
  secondary rendering path.
- Seam handling at LOD boundaries: adjacent chunks at different LOD levels
  may expose T-junctions or gaps at shared edges. Stitching geometry or
  screen-space solutions are both candidates.


## Resolved Questions

- **Material association.** Volumetric palette-indexed array (32 KB/chunk).
  Each chunk has a local palette (up to 256 entries) mapping local indices
  to global block IDs. The fragment shader reads material directly via
  world-space position: `floor(worldPos) & 31` indexes into the 32x32x32
  array.

- **Prefix sum mapping.** Eliminated entirely. The vertex shader reads a
  packed quad descriptor from the quad buffer and unpacks position + extent.
  No binary search, no popcount scan, no instance-ID-to-face mapping.

- **View frustum culling.** Chunk-level AABB in the filter compute shader.
  Hi-Z occlusion culling with exact bounds for axis-aligned geometry -- each
  chunk-direction pair is a planar slab with 4 coplanar corners, allowing
  exact screen-space projection instead of conservative bounding box tests.

- **X-direction face derivation.** Bit shift within each u32 word, followed
  by shared memory transpose in the compute shader. Same cost class as the
  other directions. Now feeds into greedy merge like all other directions.

- **Greedy meshing integration.** Material-agnostic greedy merge in the
  build compute shader. Operates on face bitmasks after derivation.
  Materials resolved per-pixel in the fragment shader, so the merge
  considers only geometry and produces maximally large quads.
