// subchunk_shade.cs.hlsl — phase-1 vis-buffer consumer.
//
// Reads the sub-chunk vis buffer written by `subchunk.ps.hlsl` (SV_Target0)
// and writes a shaded colour into a transient `shaded_color` image. The
// shaded image is consumed by the blit pass (task 1.5) that copies into
// the swapchain.
//
// Phase-1 scope: reproduces the pre-refactor debug output
// `float3(local_voxel) / 7.0` from the vis packing. No palette fetch, no
// lighting, no textures — those land in phase 2 once the vis-buffer
// substrate is verified.
//
// # Vis-buffer packing (mirror of `subchunk.ps.hlsl`)
//
//   bits  0..8  (9) : local_voxel_index (8³ = 512)
//   bits  9..11 (3) : face              (0..5)
//   bits 12..15 (4) : level_idx         (0..15)
//   bits 16..31(16) : material_slot     (65536)
//
// Sentinel 0xFFFFFFFF = miss → fragment cleared the output to black.
//
// # Bindings
//
//   set 0 binding 0 : `g_vis` — `Texture2D<uint>` containing the packed
//                     vis values produced by the previous pass.
//   set 0 binding 1 : `g_out` — `RWTexture2D<float4>` storage image the
//                     blit pass reads from.
//
// The workgroup size is 8×8 — one thread per output pixel. The dispatch
// grid is `(w + 7)/8 × (h + 7)/8` so right/bottom edge threads may read
// outside the image; the `GetDimensions` check guards the stores.
//
// The pass takes no `GpuConsts` binding: phase-1 debug shading reads
// nothing from the constants table and `GetDimensions` covers the
// resolution check. When phase-2 shading needs sub-chunk metadata
// (directory, LOD, palette), the include for `gpu_consts.hlsl` goes
// here and the Rust caller starts passing `gpu_consts` to
// `create_bind_group`.

[[vk::binding(0, 0)]] Texture2D<uint> g_vis;

// `rgba8` pins the SPIR-V ImageFormat so reflection knows the storage
// texture's texel format without reading shader-side metadata. Matches
// the `TextureFormat::Rgba8Unorm` picked on the Rust side — see
// `SHADED_COLOR_FORMAT` in `crates/renderer/src/subchunk.rs`.
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]]
RWTexture2D<float4> g_out;

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dims;
    g_vis.GetDimensions(dims.x, dims.y);
    if (tid.x >= dims.x || tid.y >= dims.y) {
        return;
    }

    uint v = g_vis.Load(int3(int2(tid.xy), 0));

    if (v == 0xFFFFFFFFu) {
        g_out[tid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Phase-1 debug shading uses only `local_idx`; `face`, `level_idx`,
    // and `material_slot` will drive palette fetches + lighting in
    // phase 2 and are unpacked on the consumer side at that point.
    uint local_idx = v & 0x1FFu;

    uint3 voxel = uint3(
        local_idx         & 7u,
        (local_idx >> 3u) & 7u,
        (local_idx >> 6u) & 7u
    );
    g_out[tid.xy] = float4(float3(voxel) / 7.0, 1.0);
}
