// request.hlsl — shared PrepRequest struct (32 bytes).
//
// Used for both full-prep and exposure-only refresh dispatches. The Rust
// side (`crates/renderer/src/subchunk.rs`) uses a single `PrepRequest` type
// for both dispatches (`write_exposure_requests(&[PrepRequest])`); this
// header keeps the HLSL in sync.

#ifndef RENDERER_REQUEST_HLSL
#define RENDERER_REQUEST_HLSL

struct PrepRequest {
    int3 coord;   // sub-chunk coord at this request's LOD
    uint level;   // LOD level; voxel_size = 1 << level
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};  // 32 bytes

#endif // RENDERER_REQUEST_HLSL
