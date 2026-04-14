# Isosurface Rendering via DDA

Future direction. Compatible with the V2 pipeline; the sub-chunk DDA path is the
natural fit for isosurface rendering.

## Why DDA and not per-voxel billboards

The 1-quad-per-voxel method isolates each primitive to a single voxel at render time.
An isosurface through a voxel is determined by the scalar field values of its neighbors,
which aren't available in that invocation. Isosurface rendering is therefore incompatible
with the per-voxel billboard path at render time.

DDA marches through multiple voxels per fragment invocation, building up neighbor context
naturally along the ray. The scalar field values needed to find the zero crossing and
compute the surface normal are either already visited by the march or a small number of
additional samples at the hit point. The neighbor problem is tractable.

## Storage

Replace the binary occupancy bitmap with a signed scalar field — negative inside solid,
positive outside, zero at the surface. Same 8³ grid per sub-chunk. The surface is the
zero level set of the field.

A 1-cell ghost layer (10³ storage instead of 8³) provides neighbor values at sub-chunk
boundaries, making gradient computation fully self-contained within each sub-chunk.

For unmodified procedural terrain the scalar field is never stored — the procgen function
IS the scalar field, evaluated analytically along the ray. No ghost layer needed.

## LOD behavior

Scalar field mip levels are averages, not OR-reductions. Thin features shrink at coarser
levels and eventually fall below the zero threshold, rather than growing 2× per level as
they do under OR-reduction. LOD degradation is perceptually correct: features too small
to resolve at a given level fade out rather than inflate.

## Compatibility

Scalar field storage is an alternative per-sub-chunk format alongside the binary bitmap.
The DDA shader handles both with a compile-time branch. The rest of the pipeline —
control plane, layer transparency, material storage, LOD hierarchy, planetary scale — is
unchanged.
