// gpu_consts.hlsl — HLSL mirror of `renderer::gpu_consts::GpuConstsData`.
//
// This is the ONE place `GpuConsts` is declared for the shader side. Any
// shader that reads shared constants does so through `g_consts`, pinned
// to descriptor set 0 binding 0 here via `[[vk::binding(0, 0)]]`. Pipeline
// constructors reflect the SPIR-V and assert slot 0 is a UniformBuffer
// sized to `GpuConstsData` (see `ComputePipeline::new` in
// `pipeline/compute.rs`).
//
// The field order, types, and count MUST match `GpuConstsData` in Rust.
// The Rust side const-asserts `size_of::<GpuConstsData>() == 48 + 32 *
// MAX_LEVELS`; the SPIR-V reflection check in `ComputePipeline::new`
// asserts this struct reflects to the same byte size.

#ifndef RENDERER_GPU_CONSTS_HLSL
#define RENDERER_GPU_CONSTS_HLSL

// Must match `renderer::subchunk::MAX_LEVELS`. If this number changes,
// both sides fail to compile until the other is updated.
#define GPU_CONSTS_MAX_LEVELS 16

// Per-level static directory metadata. 32 bytes per entry; each entry is
// two 16-byte chunks so D3D cbuffer layout ( `-fvk-use-dx-layout` ) matches
// the Rust `LevelStatic` struct exactly. The array element stride in a
// `ConstantBuffer` is the struct size rounded to a 16-byte multiple (= 32).
struct LevelStatic {
    uint3 pool_dims;      // x, y, z toroidal pool dimensions
    uint  capacity;       // pool_dims.x * pool_dims.y * pool_dims.z
    uint  global_offset;  // base directory index for this level
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
};

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

    // --- Per-level sub-chunk directory metadata ---
    uint level_count;
    uint _pad_level_count0;
    uint _pad_level_count1;
    uint _pad_level_count2;

    LevelStatic levels[GPU_CONSTS_MAX_LEVELS];

    // --- Worldgen ---
    // Seed for `include/worldgen.hlsl`'s integer hash. Written once at
    // `WorldView::new` and never mutated — the `(coord, seed) → density`
    // purity invariant in `decision-world-streaming-architecture` forbids
    // runtime seed changes once any sub-chunk has been generated.
    uint world_seed;
    uint _pad_world_seed0;
    uint _pad_world_seed1;
    uint _pad_world_seed2;
};

[[vk::binding(0, 0)]] ConstantBuffer<GpuConsts> g_consts;

#endif // RENDERER_GPU_CONSTS_HLSL
