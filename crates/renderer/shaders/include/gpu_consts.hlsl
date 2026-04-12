// gpu_consts.hlsl — HLSL mirror of `renderer::gpu_consts::GpuConstsData`.
//
// This is the ONE place `GpuConsts` is declared for the shader side. Any
// shader that reads shared constants does so through `g_consts`, which is
// forced to descriptor set 0 binding 0 by the `BindingLayout` injection
// on the Rust side (see `pipeline/binding.rs`) and matched here with an
// explicit `[[vk::binding(0, 0)]]`.
//
// The field order, types, and count MUST match `GpuConstsData` in Rust.
// The Rust side const-asserts `size_of::<GpuConstsData>() == 32`; the
// SPIR-V reflection check in `ComputePipeline::new` asserts this struct
// reflects to the same byte size.

#ifndef RENDERER_GPU_CONSTS_HLSL
#define RENDERER_GPU_CONSTS_HLSL

struct GpuConsts {
    // --- Reserved (formerly ring slot machinery; now graph-managed) ---
    uint _reserved0;
    uint _reserved1;
    uint frame_count;
    uint _reserved2;

    // --- Validation binary only ---
    uint frame_sentinel;

    // --- std140 padding to a 16-byte multiple ---
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 0)]] ConstantBuffer<GpuConsts> g_consts;

#endif // RENDERER_GPU_CONSTS_HLSL
