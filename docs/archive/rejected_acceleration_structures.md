# Rejected: Spatial Acceleration Structures

Spatial data structures and query approaches evaluated and rejected for the voxel world.
All were rejected in favor of directional face bitmasks, which exploit the axis-aligned
regularity of voxel geometry that general-purpose structures ignore.


## Global Morton Addressing

**What:** Encode world-space voxel coordinates directly into a single Morton code (u32
inputs, 21 bits per axis, ~2M blocks per axis) instead of splitting into chunk lookup +
chunk-local Morton.

**Why rejected:** Dissolves chunk boundaries entirely. The Morton curve is a volume
traversal order, but most voxel operations (meshing, simulation, streaming) need to reason
about bounded spatial regions. With global Morton, there is no concept of "where a chunk
starts and stops" — it is one continuous space-filling curve. Iteration gains from Morton
ordering require knowing the bounds of the region being iterated, which global addressing
removes.

Additionally, the surface (which meshing cares about) is scattered unpredictably through
the Morton sequence. You would iterate the entire volume to find surface voxels, negating
any locality benefit.

**What to use instead:** Linear chunk lookup + u8 Morton for chunk-local indexing. The
chunk is the locality boundary. u16 Morton for chunk-coordinate-scale queries.


## Octree as Ray Acceleration Structure

**What:** Standard sparse voxel octree for spatial queries and ray traversal.

**Why rejected:** The octree's rigid power-of-two spatial subdivision is structurally
mismatched with voxel surface geometry. The surface shell (where all the interesting
geometry lives) forces full subdivision regardless of geometric complexity. A flat terrain
and a fractal landscape produce the same tree depth at the surface boundary.

The directional face bitmask structure solves this by decomposing the surface into 6
axis-aligned sets, each of which has much simpler spatial distribution than the combined
surface.

**What to use instead:** Directional face bitmasks with layer-based organization.


## Sparse Voxel DAGs for Traversal Acceleration

**What:** Kampe et al. style SVDAGs — deduplicate identical octree subtrees by sharing
pointers, turning the octree into a directed acyclic graph.

**Why rejected:** SVDAGs are a memory compression technique, not a traversal acceleration
technique. Two identical subtrees share a pointer, saving memory, but the ray doesn't know
they are shared until it has already descended into them. The traversal topology is
unchanged — the same number of nodes are visited in the same order. The fundamental octree
surface shell fragmentation problem remains.

**What to use instead:** Directional face decomposition addresses the structural rigidity
problem that DAGs leave untouched.


## BVH over Voxel Face Soup

**What:** Build a standard bounding volume hierarchy over all voxel faces treated as
general geometry.

**Why rejected:** BVHs are designed for irregular, arbitrarily-shaped primitives. Voxel
faces are axis-aligned, uniform size, at integer coordinates. Using a BVH ignores this
extreme regularity and pays the cost of a general-purpose structure (node storage, pointer
chasing, deep traversal) when bitmask arithmetic can answer the same queries with zero
indirection.

Additionally, BVHs are expensive to update for dynamic scenes. Voxel worlds change
frequently (block placement/removal). A bitmask flip is O(1); a BVH refit is O(depth) at
best and degrades in quality over time, eventually requiring a full rebuild.

**What to use instead:** Bitmask-based directional face structure. The regularity of voxel
geometry makes bitmasks strictly superior to tree-based spatial indices.
