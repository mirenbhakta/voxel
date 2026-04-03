# Design Rejections

Techniques and architectures that were evaluated and rejected, with reasoning.
Documented to prevent revisiting dead ends.


## Global Morton Addressing

**What:** Encode world-space voxel coordinates directly into a single Morton
code (u32 inputs, 21 bits per axis, ~2M blocks per axis) instead of splitting
into chunk lookup + chunk-local Morton.

**Why rejected:** Dissolves chunk boundaries entirely. The Morton curve is a
volume traversal order, but most voxel operations (meshing, simulation,
streaming) need to reason about bounded spatial regions. With global Morton,
there is no concept of "where a chunk starts and stops" — it is one continuous
space-filling curve. Iteration gains from Morton ordering require knowing the
bounds of the region being iterated, which global addressing removes.

Additionally, the surface (which meshing cares about) is scattered
unpredictably through the Morton sequence. You would iterate the entire volume
to find surface voxels, negating any locality benefit.

**What to use instead:** Linear chunk lookup + u8 Morton for chunk-local
indexing. The chunk is the locality boundary. u16 Morton for chunk-coordinate-
scale queries.


## Ray Marching for Primary Visibility

**What:** Use ray marching (via octree, brickmap, or similar) as the primary
rendering method. Every pixel casts a ray and marches through a spatial
acceleration structure to find the first surface hit.

**Why rejected:** Multiple compounding pathological cases for the most common
gameplay camera (player on terrain, looking at the horizon):

- **Fragmented surface shells.** The octree subdivides space uniformly. A
  geometrically simple surface (flat terrain) shatters into thousands of leaf
  nodes. A horizontal ray traverses hundreds of these.

- **Warp divergence.** Rays in the same GPU warp take wildly different step
  counts. Short rays (hit ground nearby) and long rays (travel to distant
  mountains) share a warp. The warp runs at the speed of the slowest ray,
  wasting 31/63 threads for potentially hundreds of steps.

- **Thin feature tunneling.** A 1-block-thick wall at a grazing angle can be
  skipped entirely if the step size is based on the octree node size. Fixing
  this requires conservative step sizes near surfaces, but detecting "near a
  surface" requires having already found it.

- **Register pressure.** Hierarchical traversal maintains ray state, node
  pointers, and a traversal stack. High register usage reduces warp occupancy,
  which reduces latency hiding, which amplifies the cost of every cache miss.

- **Random memory access.** Octree traversal is pointer chasing through
  scattered global memory. Divergent rays chase different tree paths,
  thrashing GPU caches.

- **Anti-aliasing cost.** MSAA is near-free with rasterization (dedicated
  hardware). Ray marching requires supersampling (multiple rays per pixel),
  multiplying cost linearly.

These problems compound. Rasterization has none of them. A triangle at 500
meters costs the same as a triangle at 5 meters.

**What to use instead:** Indirect rasterization via GPU-computed greedy quads
from directional face bitmasks for primary visibility. Ray marching through the
same bitmask structure for secondary effects (shadows, AO, GI) where rays are
short-range, structurally coherent, or both.


## Octree as Ray Acceleration Structure

**What:** Standard sparse voxel octree for spatial queries and ray traversal.

**Why rejected:** The octree's rigid power-of-two spatial subdivision is
structurally mismatched with voxel surface geometry. The surface shell (where
all the interesting geometry lives) forces full subdivision regardless of
geometric complexity. A flat terrain and a fractal landscape produce the same
tree depth at the surface boundary.

The directional face bitmask structure solves this by decomposing the surface
into 6 axis-aligned sets, each of which has much simpler spatial distribution
than the combined surface.

**What to use instead:** Directional face bitmasks with layer-based
organization.


## Sparse Voxel DAGs for Traversal Acceleration

**What:** Kampe et al. style SVDAGs — deduplicate identical octree subtrees by
sharing pointers, turning the octree into a directed acyclic graph.

**Why rejected:** SVDAGs are a memory compression technique, not a traversal
acceleration technique. Two identical subtrees share a pointer, saving memory,
but the ray doesn't know they are shared until it has already descended into
them. The traversal topology is unchanged — the same number of nodes are
visited in the same order. The fundamental octree surface shell fragmentation
problem remains.

**What to use instead:** Directional face decomposition addresses the
structural rigidity problem that DAGs leave untouched.


## BVH over Voxel Face Soup

**What:** Build a standard bounding volume hierarchy over all voxel faces
treated as general geometry.

**Why rejected:** BVHs are designed for irregular, arbitrarily-shaped
primitives. Voxel faces are axis-aligned, uniform size, at integer
coordinates. Using a BVH ignores this extreme regularity and pays the cost of
a general-purpose structure (node storage, pointer chasing, deep traversal)
when bitmask arithmetic can answer the same queries with zero indirection.

Additionally, BVHs are expensive to update for dynamic scenes. Voxel worlds
change frequently (block placement/removal). A bitmask flip is O(1); a BVH
refit is O(depth) at best and degrades in quality over time, eventually
requiring a full rebuild.

**What to use instead:** Bitmask-based directional face structure. The
regularity of voxel geometry makes bitmasks strictly superior to tree-based
spatial indices.


## Brickmap Demand Streaming for Primary Rendering

**What:** Stream volumetric bricks on demand to the GPU based on a feedback
loop (GPU rays miss -> CPU uploads brick -> next frame GPU hits it). Fixed
memory budget with LRU eviction.

**Why rejected:** The CPU readback latency is at minimum one frame (more
realistically 2-3 with async readback). Fast camera movement causes visible
pop-in as bricks load. Mitigations (conservative preloading, hierarchical
fallback, predictive loading) add complexity and don't fully solve the
problem.

More fundamentally, brickmaps store volume data and ray march against it,
inheriting all the ray marching pathologies described above.

**What to use instead:** Face bitmasks derived from occupancy. The data is
always resident (~60 KB/chunk, including volumetric material) and trivially
small, so there is no demand streaming problem to solve.


## Meshlet Pool Architecture

**What:** Virtual-texture-style pool allocator for meshlets. Each chunk
allocates variable-length runs of meshlet slots from a global pool. Re-meshing
a dirty chunk frees old slots and allocates new ones.

**Why rejected:** Requires a mesh generation step (the thing we want to
eliminate). The pool management (allocation, fragmentation, compaction) adds
complexity. The meshlet fill rate varies wildly with surface density, leading
to wasted GPU lanes on underfilled meshlets.

The directional face bitmask approach removes pool management entirely. A
build compute shader (running on edit, not per frame) produces a flat,
chunk-local quad buffer via greedy merge on the bitmasks. A filter compute
shader runs per frame for visibility and emits draw commands via
MultiDrawIndirect. No pool, no fragmentation, no compaction.

**What to use instead:** Indirect rasterization from GPU-computed greedy quads
via MultiDrawIndirect. One draw call for the entire world.
