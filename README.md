## Voxel Research & Investigation
## Core References

* Voxely Blog: Focuses on voxel-based global illumination and engine architecture.
* Inigo Quilez (IQ) Articles: Essential for SDF (Signed Distance Field) math and ray-marching optimizations.
* Vulkan Guide: Ascendant: A modern Vulkan implementation of a voxel engine.

## Data Structures

* Sparse Voxel Octree (SVO): Medium memory usage; very easy to construct and modify for dynamic worlds. ([Fast Voxel Data Structures](https://bink.eu.org/fast-voxel-datastructures/))
* Sparse Voxel Directed Acyclic Graph (SVDAG): Low memory usage by sharing identical sub-trees, but difficult to modify once built. ([ACM Paper](https://dl.acm.org/doi/10.1145/3728301), [I3D Paper](https://publications.graphics.tudelft.nl/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBdjBUIiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--3e9c1bfab97521b420d59c348403286749dcf70d/I3D_authors_version.pdf), [JCGT Paper](https://jcgt.org/published/0006/02/01/paper.pdf), [JCGT Overview](https://jcgt.org/published/0006/02/01/))
* Symmetry/Transform-Aware SVDAG: Same as a regular SVDAG but adds mirroring and translation matching to further compress geometry (often by 20–30%). ([Transform-Aware SVDAGs](https://research.tudelft.nl/en/publications/transform-aware-sparse-voxel-directed-acyclic-graphs/))
* NanoVDB: A GPU-friendly, linearized version of [OpenVDB](https://www.openvdb.org/documentation/doxygen/NanoVDB_FAQ.html). It is static-topology (topology cannot change on GPU), but allows for high-speed voxel value modification and ray-tracing on the GPU. ([OpenVDB Discussion](https://github.com/AcademySoftwareFoundation/openvdb/discussions/1208), [NVIDIA NanoVDB](https://research.nvidia.com/labs/prl/publication/nanovdb/))

## Rendering & Masking Techniques

* Brickmaps (Voxel-as-Object Masking): Uses 64-bit occupancy masks to represent 4×4×4 "bricks" of voxels. This allows for extremely fast empty-space skipping during ray-marching. ([16-byte Node Structure for a 64-tree](https://www.reddit.com/r/VoxelGameDev/comments/1mmjbmm/my_16byte_node_structure_for_a_64tree_with_simple/))
* Relief Mapping (POM) for Imposters: Using [Parallax Occlusion Mapping](https://www.youtube.com/watch?v=K18qfcTFkNw) on distant chunk faces (quads) to simulate 3D depth without rendering thousands of actual voxels. ([Relief Map Selection](https://upcommons.upc.edu/bitstreams/4517c6aa-1a9b-47d8-aa5c-183e794ce167/download), [Relief Impostors](https://www.researchgate.net/figure/Relief-impostors-with-decreasing-texture-sizes_fig17_220507345), [Omnidirectional Relief Impostors](https://www.academia.edu/1290800/Omni_directional_Relief_Impostors))
* Shallow SVDAGs & Hi-Z Occlusion: Using multiple shallow graphs combined with Hierarchical-Z (Hi-Z) culling to prevent loading the entire world into VRAM at once.
* Parallax Ray Marching: Groups voxels into bounding boxes; the fragment shader then marches rays inside those boxes to simulate fine-grained detail. ([Voxel Ray Tracing](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/), [VoxelRT Source](https://github.com/dubiousconst282/VoxelRT))

------------------------------
## What’s Missing?
If you are building a modern voxel engine, you should also look into these "next-level" optimizations:

* Separation of Geometry & Attributes: Standard SVDAGs only store binary occupancy. To add color or materials, you need a separate pointer-based attribute array (e.g., indexed via Morton order) to keep the DAG nodes shareable.
* Two-Level Ray Tracing (TLAS/BLAS): Similar to hardware Ray Tracing (RTX), use a Top-Level Acceleration Structure for large-scale chunk culling and a Bottom-Level (like your SVO/SVDAG) for the actual voxel data.
* Hash-Based Storage: Investigate Spatial Hashing (used in Cyberpunk 2077 and [OpenVDB](https://www.openvdb.org/)) as an alternative to Octrees for infinite world coordinates without deep tree traversals.
* SDF Voxelization: Since you're looking at IQ's articles, consider Voxel SDFs. Instead of binary data, each voxel stores the distance to the nearest surface, allowing for massive "sphere-tracing" leaps in empty space.
* Hardware Acceleration: Research using Vulkan Ray Tracing (VK_KHR_ray_tracing_pipeline) to trace your voxel structures. Modern GPUs can treat AABBs (Axis-Aligned Bounding Boxes) as custom primitives, often outperforming manual DDA (Digital Differential Analyzer) loops. ([Voxel Ray Tracing Thesis](https://dspace.cuni.cz/bitstream/handle/20.500.11956/148776/120397271.pdf?sequence=1&isAllowed=y), [Smooth Terrain Docs](https://voxel-tools.readthedocs.io/en/latest/smooth_terrain/))
