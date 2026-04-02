## Voxel Research & Investigation## Core References

* Voxely Blog: Focuses on voxel-based global illumination and engine architecture.
* Inigo Quilez (IQ) Articles: Essential for SDF (Signed Distance Field) math and ray-marching optimizations.
* Vulkan Guide: Ascendant: A modern Vulkan implementation of a voxel engine.

## Data Structures

* Sparse Voxel Octree (SVO): Medium memory usage; very easy to construct and modify for dynamic worlds.
* Sparse Voxel Directed Acyclic Graph (SVDAG): Low memory usage by sharing identical sub-trees, but difficult to modify once built.
* Symmetry/Transform-Aware SVDAG: Same as a regular SVDAG but adds mirroring and translation matching to further compress geometry (often by 20–30%).
* NanoVDB: A GPU-friendly, linearized version of [OpenVDB](https://www.openvdb.org/documentation/doxygen/NanoVDB_FAQ.html). It is static-topology (topology cannot change on GPU), but allows for high-speed voxel value modification and ray-tracing on the GPU. [1, 2, 3, 4, 5, 6, 7, 8, 9] 

## Rendering & Masking Techniques

* Brickmaps (Voxel-as-Object Masking): Uses 64-bit occupancy masks to represent 4×4×4 "bricks" of voxels. This allows for extremely fast empty-space skipping during ray-marching.
* Relief Mapping (POM) for Imposters: Using [Parallax Occlusion Mapping](https://www.youtube.com/watch?v=K18qfcTFkNw) on distant chunk faces (quads) to simulate 3D depth without rendering thousands of actual voxels.
* Shallow SVDAGs & Hi-Z Occlusion: Using multiple shallow graphs combined with Hierarchical-Z (Hi-Z) culling to prevent loading the entire world into VRAM at once.
* Parallax Ray Marching: Groups voxels into bounding boxes; the fragment shader then marches rays inside those boxes to simulate fine-grained detail. [10, 11, 12, 13, 14, 15] 

------------------------------
## What’s Missing?
If you are building a modern voxel engine, you should also look into these "next-level" optimizations:

* Separation of Geometry & Attributes: Standard SVDAGs only store binary occupancy. To add color or materials, you need a separate pointer-based attribute array (e.g., indexed via Morton order) to keep the DAG nodes shareable.
* Two-Level Ray Tracing (TLAS/BLAS): Similar to hardware Ray Tracing (RTX), use a Top-Level Acceleration Structure for large-scale chunk culling and a Bottom-Level (like your SVO/SVDAG) for the actual voxel data.
* Hash-Based Storage: Investigate Spatial Hashing (used in Cyberpunk 2077 and [OpenVDB](https://www.openvdb.org/)) as an alternative to Octrees for infinite world coordinates without deep tree traversals.
* SDF Voxelization: Since you're looking at IQ's articles, consider Voxel SDFs. Instead of binary data, each voxel stores the distance to the nearest surface, allowing for massive "sphere-tracing" leaps in empty space.
* Hardware Acceleration: Research using Vulkan Ray Tracing (VK_KHR_ray_tracing_pipeline) to trace your voxel structures. Modern GPUs can treat AABBs (Axis-Aligned Bounding Boxes) as custom primitives, often outperforming manual DDA (Digital Differential Analyzer) loops. [1, 2, 3, 16, 17] 

[1] [https://dl.acm.org](https://dl.acm.org/doi/10.1145/3728301)
[2] [https://bink.eu.org](https://bink.eu.org/fast-voxel-datastructures/)
[3] [https://publications.graphics.tudelft.nl](https://publications.graphics.tudelft.nl/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBdjBUIiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--3e9c1bfab97521b420d59c348403286749dcf70d/I3D_authors_version.pdf)
[4] [https://jcgt.org](https://jcgt.org/published/0006/02/01/paper.pdf)
[5] [https://research.tudelft.nl](https://research.tudelft.nl/en/publications/transform-aware-sparse-voxel-directed-acyclic-graphs/)
[6] [https://jcgt.org](https://jcgt.org/published/0006/02/01/)
[7] [https://github.com](https://github.com/AcademySoftwareFoundation/openvdb/discussions/1208)
[8] [https://www.openvdb.org](https://www.openvdb.org/documentation/doxygen/NanoVDB_FAQ.html)
[9] [https://research.nvidia.com](https://research.nvidia.com/labs/prl/publication/nanovdb/)
[10] [https://www.reddit.com](https://www.reddit.com/r/VoxelGameDev/comments/1mmjbmm/my_16byte_node_structure_for_a_64tree_with_simple/#:~:text=The%20static%20environment%20is%20built%20on%20a,making%20it%20quick%20to%20perform%20intersection%20tests.)
[11] [https://dubiousconst282.github.io](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/#:~:text=Although%20we%20could%20pick%20any%20other%20arbitrary,larger%20groups%20of%20empty%20cells%20for%20space%2Dskipping.)
[12] [https://github.com](https://github.com/dubiousconst282/VoxelRT)
[13] [https://upcommons.upc.edu](https://upcommons.upc.edu/bitstreams/4517c6aa-1a9b-47d8-aa5c-183e794ce167/download#:~:text=Our%20relief%20map%20selection%20algorithm%20always%20ensures,space%20is%20the%20projection%20of%20such%20length.)
[14] [https://www.researchgate.net](https://www.researchgate.net/figure/Relief-impostors-with-decreasing-texture-sizes_fig17_220507345#:~:text=In%20%5BBSAP11%5D%20each%20character%20is%20encoded%20through,at%20the%20expense%20of%20some%20per%2Dfragment%20overhead.)
[15] [https://www.academia.edu](https://www.academia.edu/1290800/Omni_directional_Relief_Impostors)
[16] [https://dspace.cuni.cz](https://dspace.cuni.cz/bitstream/handle/20.500.11956/148776/120397271.pdf?sequence=1&isAllowed=y)
[17] [https://voxel-tools.readthedocs.io](https://voxel-tools.readthedocs.io/en/latest/smooth_terrain/)
