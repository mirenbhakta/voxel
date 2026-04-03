# Voxel Global Illumination

Techniques for lighting voxel worlds using the directional face bitmask
structure as the ray acceleration backend. All techniques share the same 29 KB
per chunk data that the rasterization pipeline already maintains — no secondary
spatial index required.


## Why Bitmask Traversal Fits GI

The rays involved in GI and lighting are either short-range, structurally
coherent, or both. None exhibit the pathological case that kills primary
visibility ray marching (long horizontal rays through fragmented surface
shells). The bitmask structure makes individual ray intersections nearly free —
a bit test instead of a tree traversal.

Adjacent screen pixels tend to cast similar lighting rays, so threads in a GPU
warp traverse the same bitmask data. This keeps warp divergence low and cache
utilization high.


## Techniques

### Shadows

Sun shadow rays travel in a single direction toward the light source. Only one
directional structure is queried. Screen-space coherence is high — adjacent
pixels cast nearly parallel shadow rays, traversing the same bitmask layers.

For a directional light, all shadow rays share the same direction, so the layer
skip pattern (which layers to check, in what order) is identical across the
entire screen. The compute cost is dominated by the number of occupied layers
between the surface and the light, not by distance.

Point lights and spotlights produce divergent shadow rays, but their range is
bounded, limiting traversal depth.

### Ambient Occlusion

Short-range hemisphere samples around a surface point. Rays extend only a few
blocks, crossing 1-3 layers per direction per ray. A handful of bit tests per
sample.

This is world-space AO, not screen-space. It correctly captures occlusion from
geometry that is not visible to the camera (behind the surface, around corners)
which SSAO misses entirely. The bitmask structure makes it cheap enough to do
per-pixel rather than at reduced resolution.

### Radiance Cascades

Alexander Sannikov's technique. Cascades of radiance probes at geometrically
increasing spacing. Each cascade traces rays at a fixed length to gather
incoming light, then cascades merge to propagate illumination.

Natural fit for bitmask traversal:
- Each cascade level has a fixed, bounded ray length
- Short rays at fine cascade levels = few layer crossings
- Longer rays at coarse levels can use mipped bitmasks (see LOD section)
- The work per ray is predictable, which keeps GPU utilization high

### Voxel Cone Tracing

Cast wide cones instead of thin rays. At close range, sample the face bitmask
at full 32x32 resolution. At distance, sample a coarser mip.

The mip hierarchy is derived from the same occupancy data:
- Level 0: 32x32 face bitmask (full resolution)
- Level 1: 16x16 (OR-reduce 2x2 groups)
- Level 2: 8x8
- Level 3: 4x4
- Level 4: 2x2
- Level 5: 1x1 (single bit — is there any face in this chunk-layer at all?)

As the cone widens, the bitmask gets coarser, and the work per sample stays
roughly constant. The mip levels are trivially derived and tiny (each level is
1/4 the size of the previous).

### Light Propagation

Flood-fill light values through the bitmask structure. The face bitmasks
directly encode which cells can exchange light — a face exists at a boundary
means light cannot pass freely between those two cells.

The layer structure provides a natural wavefront for propagation: sweep layer
by layer per direction. The face bitmask at each layer acts as a blocking mask
for light transfer.

### Reflections

For flat reflective surfaces (water, polished stone), reflected rays are highly
coherent — all reflection rays point in nearly the same direction. One
directional structure dominates, warp coherence is tight.

Rough reflections require multiple rays per pixel with varying directions,
but the rays are typically short-range, keeping traversal bounded.


## LOD for Ray Queries

The bitmask representation has a natural mip hierarchy. OR-reducing 2x2 blocks
of bits produces a coarser bitmask where each bit represents a 2x2 region. A
set bit means "at least one face exists somewhere in this region."

This can be applied in two ways:

- **Distance-based LOD:** Rays traveling far from the camera sample coarser
  mip levels. Reduces traversal cost for distant queries (coarse GI, far
  shadow cascades) at the expense of spatial precision.

- **Cone-width LOD:** For cone tracing, the mip level is selected based on the
  cone footprint at the current sample distance. This is analogous to how
  texture mipmapping works — sample at the resolution that matches the query's
  spatial extent.

The mip data is tiny. A full mip chain from 32x32 down to 1x1 adds ~340 bits
(~43 bytes) per layer per direction. For a full chunk across all 6 directions
and 32 layers: ~8 KB additional. Negligible compared to the base 29 KB.


## Data Layout: Raster vs Ray

All techniques in this document use the same 29 KB/chunk data that the
rasterization pipeline maintains. No secondary spatial index is required for
ray traversal.

The ray path uses `layer_occ` (a u32 per direction) for layer skipping and
the 32x32 face bitmask for per-layer hit testing. For short-range techniques
(AO, shadows, short radiance cascade rays), the flat `layer_occ` skip is
sufficient. Few layers are tested per ray regardless of how coarse the skip
is.

For longer rays (coarse radiance cascades, cone tracing at distance, long
reflections), the flat `layer_occ` can only skip completely empty layers. It
cannot skip layers where faces exist but are spatially far from the ray's
path. This means rays test every occupied layer along their direction even
when none of those faces are near the ray. The cost is wasted bit tests (false
positives), not incorrect results.

The mip hierarchy described for cone tracing helps with spatial precision
within a layer but does not help skip layers along the ray direction. It
answers "is there a face near this point in 2D" more coarsely, not "is there
a face in this layer worth testing at all."

If ray traversal performance becomes a bottleneck for longer-range techniques,
per-layer 2D bounds (from the original hierarchy proposal in the acceleration
structure design) or transposed occupancy bitmasks could be added as
supplementary metadata without changing the base 29 KB structure.


## Open Questions

- Radiance cascade probe placement relative to chunk boundaries
- Temporal stability of GI under rapid voxel edits (block placement/removal)
- Integration with emissive voxels (lava, glowstone) as primary light sources
- Whether cone tracing quality justifies its cost over radiance cascades for
  this specific data structure
- Denoising requirements for stochastic ray-based techniques at low sample
  counts
