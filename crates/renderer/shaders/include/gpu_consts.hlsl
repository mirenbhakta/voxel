// gpu_consts.hlsl — HLSL mirror of `renderer::gpu_consts::GpuConstsData`.
//
// This is the ONE place `GpuConsts` is declared for the shader side. Any
// shader that reads shared constants does so through `g_consts`, which is
// forced to descriptor set 0 binding 0 by the `BindingLayout` injection
// on the Rust side (see `pipeline/binding.rs`) and matched here with an
// explicit `[[vk::binding(0, 0)]]`.
//
// The field order, types, and count MUST match `GpuConstsData` in Rust.
// The Rust side const-asserts `size_of::<GpuConstsData>() == 32`; a
// SPIR-V reflection check landing in Increment 6 will assert this struct
// reflects to the same byte size. See `.local/renderer_plan.md` §5.

#ifndef RENDERER_GPU_CONSTS_HLSL
#define RENDERER_GPU_CONSTS_HLSL

struct GpuConsts {
    // --- Ring machinery ---
    uint upload_slot;
    uint readback_slot;
    uint frame_count;
    uint upload_capacity;

    // --- Validation binary only ---
    uint frame_sentinel;

    // --- std140 padding to a 16-byte multiple ---
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 0)]] ConstantBuffer<GpuConsts> g_consts;

#endif // RENDERER_GPU_CONSTS_HLSL
