// camera.hlsl — shared Camera struct (64 bytes, std140-compatible).
//
// Declare the Camera constant buffer with a binding appropriate to the
// consuming pipeline; this header only provides the struct definition.
// Matches the Rust-side layout in `crates/renderer/src/subchunk.rs` byte
// for byte.
//
// Consumers:
//   subchunk.vs.hlsl  — [[vk::binding(0, 0)]] ConstantBuffer<Camera> g_camera
//   subchunk.ps.hlsl  — [[vk::binding(0, 0)]] ConstantBuffer<Camera> g_camera
//   subchunk_shade.cs.hlsl  — [[vk::binding(5, 0)]] ConstantBuffer<Camera> g_camera
//   subchunk_cull.cs.hlsl   — [[vk::binding(0, 0)]] ConstantBuffer<Camera> g_camera

#ifndef RENDERER_CAMERA_HLSL
#define RENDERER_CAMERA_HLSL

struct Camera {
    float3 pos;
    float  fov_y;
    float3 forward;
    float  aspect;
    float3 right;
    float  _pad0;
    float3 up;
    float  _pad1;
};

#endif // RENDERER_CAMERA_HLSL
