// occupancy.hlsl — shared Occupancy struct + helpers.
//
// The 64-byte occupancy representation shared across prep, exposure, and
// the DDA. One bit per voxel in an 8×8×8 grid, packed as 16 u32s grouped
// into four uint4 to avoid HLSL std140 array-element padding.
//
// `face_mask.hlsl` and `dda.hlsl` both require `Occupancy` to be in scope
// before they are included; consumers must `#include "include/occupancy.hlsl"`
// first.

#ifndef RENDERER_OCCUPANCY_HLSL
#define RENDERER_OCCUPANCY_HLSL

// Matches the 16-u32 packed layout of `SubchunkOccupancy::to_gpu_bytes`
// on the Rust side: bit `y * 8 + x` of word `z * 2 + (bit >> 5)` is set
// iff voxel (x, y, z) is occupied.
struct Occupancy {
    uint4 plane[4];
};  // 64 bytes

// Returns true when every voxel in the sub-chunk is occupied. Iff all 16
// words are 0xFFFFFFFF.
bool occupancy_is_fully_solid(Occupancy o) {
    uint a = o.plane[0].x & o.plane[0].y & o.plane[0].z & o.plane[0].w
           & o.plane[1].x & o.plane[1].y & o.plane[1].z & o.plane[1].w
           & o.plane[2].x & o.plane[2].y & o.plane[2].z & o.plane[2].w
           & o.plane[3].x & o.plane[3].y & o.plane[3].z & o.plane[3].w;
    return a == 0xFFFFFFFFu;
}

#endif // RENDERER_OCCUPANCY_HLSL
