# Scene Acceleration via Voxel Occupancy

Using voxel occupancy grids as a general-purpose ray acceleration structure
for arbitrary scene geometry (non-voxel meshes, characters, props). The voxel
grid acts as a coarse spatial filter, and precise per-triangle intersection
is deferred to hardware RT or a fine-grained BVH only within occupied regions.


## Concept

Traditional hardware ray tracing builds a BVH (BLAS/TLAS) over triangle
geometry. This works well for static scenes but becomes expensive for dynamic
ones — moving objects require BLAS refits or rebuilds each frame.

The voxel occupancy grid provides an alternative first stage: a coarse,
trivially updatable spatial index that filters out empty space before the
expensive triangle-level intersection.

The ray traversal becomes two phases:

    Phase 1: Bitmask march (directional face structure)
      "Is there anything here?" — bit test, near zero cost.
      Skip empty space at voxel granularity.

    Phase 2: Hardware RT / triangle intersection
      "What exactly did I hit?" — BVH over actual triangles.
      Only invoked in occupied voxel cells.


## Why This Works

The key win is decoupling spatial occupancy (which changes when objects move)
from geometric detail (which is static per object).

A character walking across the world changes *which voxel cells are occupied*
but does not change *the character's triangle mesh*. The object-local BLAS is
built once and never rebuilt. Only the voxel occupancy — a handful of bit
flips — updates per frame.

For a moving object:

    1. Clear old voxel occupancy (object's previous AABB)
    2. Set new voxel occupancy (object's current AABB)
    3. Object-local BLAS: unchanged

This is O(volume of AABB in voxel cells) per object per frame, which for
typical game objects (characters, props, vehicles) is a few dozen bit flips.


## Two-Level Object Grids

Objects can carry their own voxelization grid at a resolution appropriate to
their size. A character might use an 8x8x16 grid. A vehicle might use
16x8x8.

The traversal hierarchy becomes:

    World voxel grid (coarse, dynamic, trivially updated per frame)
      -> Object voxel grid (finer, static in object space, moves with transform)
        -> Hardware BVH (triangles, static, built once)

Each level filters out more empty space before the expensive triangle test.
The first two levels are bitmask operations with no tree rebuilds or refits.

A ray entering a world voxel cell occupied by an object:
1. Transforms into the object's local coordinate space
2. Marches the object's local bitmask grid
3. Only invokes triangle intersection in occupied local cells

This is especially effective for objects with large bounding boxes but sparse
actual geometry (a tree with a wide canopy, an open vehicle frame, a character
with outstretched limbs).


## Voxelization

The voxelization step does not need to be precise. It is a conservative
occupancy estimate, not a visual representation. The requirements are:

- Every cell that contains geometry must be marked occupied (no false negatives)
- False positives (cells marked occupied that contain no geometry) are
  acceptable — they cost a wasted triangle intersection but produce no
  visual artifacts
- The voxelization must be fast enough to update per frame for moving objects

For axis-aligned bounding box voxelization (marking all cells within the
object's AABB), this is trivially computed from the object's world-space
bounds. For tighter voxelization (following the object's shape), the object's
local grid can be precomputed offline and transformed at runtime.

Precomputed local grids can be stored per mesh as a small bitmask alongside
the mesh data. An 8x8x16 grid is 128 bytes. This is static — built once when
the mesh is authored, never changes.


## Overlapping Objects

Multiple objects can occupy the same world voxel cell. The world grid cell
must store which objects are present, not just a single occupancy bit.

Options:
- **Small fixed-size set** (2-4 object IDs per cell). Sufficient for most
  cases. Overflow falls back to testing all objects in the cell's chunk.
- **Index into a per-cell object list.** More general but adds indirection.
- **Layered bitmasks.** One bitmask per object in the chunk. A ray tests each
  object's bitmask independently. Scales with the number of objects per chunk,
  not per cell.

For coarse voxel granularity, cell overlap is relatively rare. A small
fixed-size set handles the common case without indirection.


## Directional Culling

The same directional face logic that works for terrain applies to object
presence. If a ray's direction cannot intersect any face of an occupied cell
(determined by the directional bitmask), the object in that cell is skipped
entirely — no coordinate transform, no local grid march, no triangle test.

This is backface culling at the voxel-cell level, applied to arbitrary
objects.


## Dynamic vs. Static Partitioning

Static world geometry (terrain, buildings, placed objects) can use the primary
voxel chunk system directly. Dynamic objects (characters, physics objects,
projectiles) maintain their own occupancy in a separate overlay grid or by
writing into the world grid each frame.

Separating static and dynamic occupancy avoids the cost of re-deriving face
bitmasks for terrain when only a character moved. The dynamic overlay is
purely for acceleration — it has no face bitmasks or layer counts, just
raw occupancy for coarse ray filtering.


## Traversal Characteristics

Phase 1 (bitmask march) uses the directional face structure's `layer_occ` u32
for layer skipping and the 32x32 face bitmask for hit testing within each
layer.

Within a 32-layer chunk, the worst case is 32 bit tests per direction (96 for
3 active directions) when all layers are occupied. This is cheap per chunk but
adds up across many chunks for long rays. The flat `layer_occ` provides coarse
skip: completely empty layers are skipped, but layers with faces far from the
ray's path are still tested. The cost is wasted bit tests, not incorrect
results.

For the scene acceleration use case specifically, rays tend to be longer than
typical GI rays. Tracing through the world to find objects means crossing many
chunks, and within each chunk the false-positive rate (testing occupied layers
where faces are spatially irrelevant) is more visible than for short AO or
shadow rays.

The chunk grid partially compensates: entire empty chunks are skipped in one
test, providing world-scale spatial skip. Per-layer 2D bounds or a coarser
world-level occupancy grid (separate from the terrain chunk grid) are potential
extensions if traversal cost for long rays becomes a concern.


## Open Questions

- Appropriate voxel resolution for the world-level acceleration grid (matching
  the terrain chunk resolution, or coarser?)
- Latency of object voxelization for fast-moving objects (does the occupancy
  lag behind the object's visual position by a frame?)
- Integration with the terrain face bitmask structure (shared grid, separate
  overlay, or hybrid?)
- Whether the two-phase approach (bitmask then hardware RT) can be expressed
  efficiently in existing graphics APIs (Vulkan ray tracing pipeline stages)
- Cost/benefit vs. simply rebuilding the TLAS each frame, which modern
  hardware handles increasingly well
- Whether the terrain face bitmask structure is the right granularity for
  scene-level ray acceleration, or whether a separate coarser occupancy grid
  (e.g., 8x8x8 per chunk) would be more appropriate for the long-range rays
  this use case involves
