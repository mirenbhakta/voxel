// material.hlsl — shader-side types + pure accessors for the M1 material
// system (decision-material-system-m1-sparse).
//
// Two independent pieces live here:
//
//   1. `MaterialDesc` + `material_fetch`: the global descriptor table. Small,
//      dense (256 entries), published once at startup from the CPU
//      `BlockRegistry`. sRGB on the CPU → linear at fetch time.
//
//   2. `MaterialBlock` + `material_block_read_id`: one 1 KB block from the
//      per-sub-chunk material-data pool. The pool is a binding array of
//      fixed-count 64 MB segments; each slot holds 512 u16 material IDs
//      packed two-per-u32. Consumers read
//      `material_data_pool[segment_idx][local_idx].packed_ids[...]`.
//
// This file declares *no bindings*. Consumer shaders declare their own
// `[[vk::binding(N, S)]]` for `StructuredBuffer<MaterialDesc> g_materials`
// and `StructuredBuffer<MaterialBlock> material_data_pool[...]`, matching
// the project's pure-include convention (`directory.hlsl`, `dda.hlsl`).

#ifndef RENDERER_MATERIAL_HLSL
#define RENDERER_MATERIAL_HLSL

// Capacity constants (MATERIAL_DESC_CAPACITY, SLOTS_PER_SEGMENT,
// MAX_MATERIAL_POOL_SEGMENTS, MATERIAL_DATA_SLOT_INVALID) are authored in
// Rust (`crates/renderer/src/subchunk.rs`) and emitted to this header by
// `crates/renderer/build.rs`. The DXC invocation adds `${OUT_DIR}/shaders`
// to its include path so the unqualified include resolves.
#include "gpu_constants.hlsl"

// -----------------------------------------------------------------------
// MaterialDesc — 32-byte global descriptor table entry.
//
// Layout matches Rust `renderer::MaterialDesc` byte-for-byte:
//   float4 albedo     (+0)    // sRGB RGBA — decoded to linear at fetch
//   uint4  _reserved  (+16)   // M3 PBR extension (albedo_tex, normal_tex, ...)
//
// The table is indexed by `mat_id`, the 16-bit per-voxel block identifier
// written by the prep shader into `MaterialBlock.packed_ids`.
// -----------------------------------------------------------------------
struct MaterialDesc {
    float4 albedo;
    uint4  _reserved;
};

// -----------------------------------------------------------------------
// MaterialBlock — one 1 KB slot of the per-sub-chunk material-data pool.
//
// Packs 512 u16 voxel material IDs into 256 u32 words, two IDs per word:
//   - Low  16 bits: voxel `(word_idx * 2 + 0)` material ID.
//   - High 16 bits: voxel `(word_idx * 2 + 1)` material ID.
//
// `voxel_idx` is the local (x, y, z) → linear index the prep shader uses;
// the extraction math is encapsulated in `material_block_read_id`.
// -----------------------------------------------------------------------
struct MaterialBlock {
    uint packed_ids[256];
};

// -----------------------------------------------------------------------
// sRGB → linear decode on the per-channel float. Piecewise form matches
// the IEC 61966-2-1 reference. Kept branch-free via `max`/`pow` — the
// input range is `[0, 1]` so the cutoff at `0.04045` is always well-
// defined.
// -----------------------------------------------------------------------
float srgb_to_linear_channel(float c) {
    return (c <= 0.04045) ? (c / 12.92) : pow((c + 0.055) / 1.055, 2.4);
}

float3 srgb_to_linear(float3 srgb) {
    return float3(
        srgb_to_linear_channel(srgb.x),
        srgb_to_linear_channel(srgb.y),
        srgb_to_linear_channel(srgb.z)
    );
}

// -----------------------------------------------------------------------
// Fetch the linear-space albedo for `mat_id` from the descriptor table.
//
// The `materials` argument is the consumer-shader-declared
// `StructuredBuffer<MaterialDesc>`. Having the table as a parameter keeps
// this helper pure (no global reference inside the include) so a future
// material system with multiple descriptor tables (e.g. lightmapped vs.
// standard) can pass in whichever one is active.
// -----------------------------------------------------------------------
float3 material_fetch_albedo_linear(
    StructuredBuffer<MaterialDesc> materials,
    uint                           mat_id
) {
    MaterialDesc desc = materials[mat_id];
    return srgb_to_linear(desc.albedo.rgb);
}

// -----------------------------------------------------------------------
// Unpack a u16 material ID from a `MaterialBlock` at the given
// local-voxel linear index. The voxel index layout matches the prep
// shader's writer: linear = x + 8*y + 64*z for 8³ sub-chunks.
// -----------------------------------------------------------------------
uint material_block_read_id(MaterialBlock block, uint voxel_idx) {
    uint word = block.packed_ids[voxel_idx >> 1u];
    uint hi   = voxel_idx & 1u;
    return (hi != 0u) ? (word >> 16u) : (word & 0xFFFFu);
}

// -----------------------------------------------------------------------
// Convenience: read the material ID for `voxel_idx` out of the
// material-data pool at `material_data_slot`.
//
// `pool` is the consumer-shader-declared binding array of segment
// buffers (`StructuredBuffer<MaterialBlock> material_data_pool[MAX_MATERIAL_POOL_SEGMENTS]`).
// Decomposes the flat global slot into `(segment_idx, local_idx)` per
// the M1 invariant (see `decision-material-system-m1-sparse`).
//
// Caller is responsible for gating on
// `material_data_slot != MATERIAL_DATA_SLOT_INVALID` — this helper does
// no sentinel check, so passing INVALID is a shader-side programmer
// error.
// -----------------------------------------------------------------------
// NB: HLSL doesn't let us bind the outer `pool` by parameter because
// `StructuredBuffer<...>[...]` cannot be a function parameter in the
// profiles we target. Consumers inline the read — see the subchunk
// shade shader for the reference implementation.

#endif // RENDERER_MATERIAL_HLSL
