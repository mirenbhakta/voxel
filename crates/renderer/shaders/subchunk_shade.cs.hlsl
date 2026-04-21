// subchunk_shade.cs.hlsl — vis-buffer consumer; first lit pass.
//
// Reads the sub-chunk vis buffer written by `subchunk.ps.hlsl` (SV_Target0)
// and writes a shaded colour into the transient `shaded_color` image. The
// shaded image is consumed by the blit pass that copies into the swapchain.
//
// Shading model (first real-shading bite):
//   * face → axis-aligned normal, decoded from the 3-bit `face` field.
//   * directional sun (Lambertian) against a hardcoded direction.
//   * single sun-shadow ray per pixel, origin at the depth-reconstructed
//     world-space hit point on the voxel face.
//   * hemispherical ambient (sky above, ground below) keyed off N.y.
//   * constant grey albedo. Per-voxel material lookup is the next bite.
//
// Linear-space throughout. The blit pass writes a `Bgra8UnormSrgb`
// swapchain target; hardware does the linear→sRGB encoding on store, so no
// manual gamma here.
//
// # Vis-buffer packing (mirror of `subchunk.ps.hlsl`)
//
//   bits  0..8  (9) : local_voxel_index (8³ = 512)
//   bits  9..11 (3) : face              (0..5; matches `Direction` enum)
//   bits 12..15 (4) : level_idx         (0..15)
//   bits 16..31(16) : occ_slot          (directory pass-through today)
//
// Sentinel `0xFFFFFFFF` = miss → output gets the sky colour so the
// background reads as a real environment instead of pure black.
//
// # Bindings
//
// `g_consts` (slot 0) is implicit — `directory.hlsl` includes
// `gpu_consts.hlsl` which pins it to [[vk::binding(0, 0)]]. The Rust side
// must pass `Some(gpu_consts)` to `create_bind_group` for this pass.
//
//   set 0 binding 0 : `g_consts`           — implicit from `directory.hlsl`.
//   set 0 binding 1 : `g_vis`              — `Texture2D<uint>` vis buffer.
//   set 0 binding 2 : `g_out`              — `RWTexture2D<float4>` storage image.
//   set 0 binding 3 : `g_directory`        — `StructuredBuffer<DirEntry>`, read-only.
//   set 0 binding 4 : `g_occ_array`        — `StructuredBuffer<Occupancy>`, read-only.
//   set 0 binding 5 : `g_camera`           — `ConstantBuffer<Camera>`, per-frame camera.
//   set 0 binding 6 : `g_depth`            — `Texture2D<float>` depth buffer (Depth32Float).
//   set 0 binding 7 : `g_materials`        — `StructuredBuffer<MaterialDesc>`, descriptor table.
//   set 1 binding 0 : `material_data_pool` — `StructuredBuffer<MaterialBlock>[MAX_MATERIAL_POOL_SEGMENTS]`,
//                                            per-sub-chunk material IDs; partially bound.
//                                            Kept in set 1 because wgpu disallows mixing a binding
//                                            array with uniform buffers in one set (the set-0
//                                            `g_consts` + `g_camera` are uniforms).
//
// The workgroup size is 8×8 — one thread per output pixel. The dispatch
// grid is `(w + 7)/8 × (h + 7)/8` so right/bottom edge threads may read
// outside the image; the `GetDimensions` check guards the stores.

// `directory.hlsl` includes `gpu_consts.hlsl`, pinning `g_consts` at
// slot 0. It also brings in `DirEntry`, `direntry_*` accessors,
// `resolve_coord_to_slot`, and `resolve_and_verify` needed by `dda_world`.
#include "include/directory.hlsl"
#include "include/material.hlsl"
#include "include/occupancy.hlsl"
#include "include/projection.hlsl"

[[vk::binding(1, 0)]] Texture2D<uint> g_vis;

// `rgba8` pins the SPIR-V ImageFormat so reflection knows the storage
// texture's texel format without reading shader-side metadata. Matches
// the `TextureFormat::Rgba8Unorm` picked on the Rust side — see
// `SHADED_COLOR_FORMAT` in `crates/renderer/src/subchunk.rs`.
[[vk::binding(2, 0)]] [[vk::image_format("rgba8")]]
RWTexture2D<float4> g_out;

// `g_directory` must be declared before including dda.hlsl so that
// `dda_world` (which references it under the RENDERER_DIRECTORY_HLSL guard)
// sees the binding.
[[vk::binding(3, 0)]] StructuredBuffer<DirEntry> g_directory;

// `g_occ_array` — occupancy bitmaps indexed by occ_slot. Consumed by
// `dda_sub_chunk` inside `dda.hlsl`.
[[vk::binding(4, 0)]] StructuredBuffer<Occupancy> g_occ_array;

// dda.hlsl provides `MarchResult`, `dda_sub_chunk`, and (since directory.hlsl
// is already in scope) `dda_world` under the RENDERER_DIRECTORY_HLSL guard.
#include "include/dda.hlsl"

#include "include/camera.hlsl"

[[vk::binding(5, 0)]] ConstantBuffer<Camera> g_camera;

// Depth buffer written by `subchunk.ps.hlsl` as `SV_Depth`. Format is
// `Depth32Float` — accessed here as a plain float sampled texture.
[[vk::binding(6, 0)]] Texture2D<float> g_depth;

// Global material descriptor table (`MaterialDesc[MATERIAL_DESC_CAPACITY]`).
// Populated once at startup from the BlockRegistry. Fetched by
// `material_fetch_albedo_linear` on every shaded pixel.
[[vk::binding(7, 0)]] StructuredBuffer<MaterialDesc> g_materials;

// Per-sub-chunk material-data pool — binding array of 64 MB segments.
// Lives in set 1 (not set 0) so it can coexist with the uniform
// buffers in set 0 — wgpu rejects "binding array + uniform buffer in
// the same set" (validation error 'Bind groups may not contain both a
// binding array and a uniform buffer').
//
// Indexed as `material_data_pool[segment_idx][local_idx]` after the
// `material_data_slot` from DirEntry is decomposed into
// `(segment_idx, local_idx) = (slot / SLOTS_PER_SEGMENT,
// slot % SLOTS_PER_SEGMENT)`. See `decision-material-system-m1-sparse`
// for the reads-through-DirEntry-indirection invariant; the compute
// shader body is the only place that performs the div/mod decode.
[[vk::binding(0, 1)]]
StructuredBuffer<MaterialBlock> material_data_pool[MAX_MATERIAL_POOL_SEGMENTS];

// --- Shading constants (linear sRGB primaries). ---

// Direction *toward* the light (unit vector). Up-and-back-right of the
// scene; tweak freely. Negate for "light propagation direction".
static const float3 SUN_DIR_TO_LIGHT = normalize(float3(0.4, 0.8, -0.3));

// Sun radiance reaching the surface. LDR-scaled so a top face hit head-on
// lands near 1.0 after combining with ambient — the output is `Rgba8Unorm`,
// which clamps anything above 1.
static const float3 SUN_COLOR    = float3(1.00, 0.95, 0.85);

// Hemispherical ambient endpoints. SKY doubles as the miss-pixel
// background so surfaces and sky read out of the same environment.
static const float3 SKY_COLOR    = float3(0.50, 0.70, 0.95);
static const float3 GROUND_COLOR = float3(0.15, 0.12, 0.10);

// Sentinel colour drawn when `DirEntry.material_data_slot ==
// MATERIAL_DATA_SLOT_INVALID`. Two causes: (1) the materializer was
// allocator-exhausted this frame and the grow is deferred to next frame;
// (2) the pool ceiling has been reached and no more segments can be
// appended. Either way the user-visible magenta is the signal.
static const float3 MAGENTA_SENTINEL = float3(1.00, 0.00, 1.00);

// Maximum shadow-ray distance in world units. Well past the horizon at
// level 0 and still meaningful at coarser levels — any occluder further
// than 1024 world units is treated as non-shadowing.
static const float MAX_SHADOW_T  = 1024.0;

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dims;
    g_vis.GetDimensions(dims.x, dims.y);
    if (tid.x >= dims.x || tid.y >= dims.y) {
        return;
    }

    uint v = g_vis.Load(int3(int2(tid.xy), 0));

    if (v == 0xFFFFFFFFu) {
        g_out[tid.xy] = float4(SKY_COLOR, 1.0);
        return;
    }

    // Unpack vis-buffer fields.
    uint local_idx = v & 0x1FFu;
    uint face      = (v >> 9u)  & 0x7u;
    uint level_idx = (v >> 12u) & 0xFu;
    uint occ_slot  = (v >> 16u) & 0xFFFFu;

    float3 normal = face_to_normal(face);

    // Lambertian sun term.
    float ndotl = max(dot(normal, SUN_DIR_TO_LIGHT), 0.0);

    // Hemispherical ambient — surfaces facing up see sky, surfaces facing
    // down see ground. Cheap stand-in for a real environment lookup.
    float  hemi_t  = 0.5 + 0.5 * normal.y;
    float3 ambient = lerp(GROUND_COLOR, SKY_COLOR, hemi_t);

    // voxel_size is still needed for the shadow-ray epsilon and for
    // `dda_world`'s level parameter. `g_directory` is still needed by
    // `dda_world` internally for torus-verified slot lookups.
    float voxel_size = float(1u << level_idx);

    // Reconstruct the per-pixel world-space hit point from the depth
    // buffer so every screen pixel casts its own shadow ray.
    //
    // Depth encoding from `subchunk.ps.hlsl`:
    //   depth = A + B / vz   where A = FAR/(FAR-NEAR), B = -NEAR*FAR/(FAR-NEAR)
    // Invert to recover view-space Z:
    //   vz = B / (depth - A)
    //
    // NDC sign convention: `sv_pos.y = f * vy` (no negation in VS), and
    // wgpu maps NDC y=+1 → screen y=0, so ndc.y = 1 - 2 * uv.y.
    float depth = g_depth.Load(int3(int2(tid.xy), 0));
    float ndc_x = 2.0 * (float(tid.x) + 0.5) / float(dims.x) - 1.0;
    float ndc_y = 1.0 - 2.0 * (float(tid.y) + 0.5) / float(dims.y);

    float vz = decode_vz(depth);

    float tan_half_y = tan(g_camera.fov_y * 0.5);
    float tan_half_x = tan_half_y * g_camera.aspect;
    float vx = ndc_x * tan_half_x * vz;
    float vy = ndc_y * tan_half_y * vz;

    float3 hit_ws = g_camera.pos
                  + g_camera.right   * vx
                  + g_camera.up      * vy
                  + g_camera.forward * vz;

    // Offset by a small epsilon along the face normal so the shadow ray
    // does not self-intersect the surface voxel. Epsilon scales with
    // voxel_size so it stays meaningful across LOD levels.
    float3 shadow_origin_ws = hit_ws + normal * (voxel_size * 1e-3);

    // Cast a single sun-shadow ray. `dda_world_clipmap` demotes-and-promotes
    // along the ray so it always marches at the finest level resident at
    // each step. Passing `0u` is the simplest contract: the demote check on
    // iteration 0 is a no-op (already at L0), and if L0 isn't resident at
    // the ray origin the first step promotes to whatever level is. Whenever
    // the ray re-enters a finer shell mid-march, it demotes back down —
    // critical for a shadow ray fired from an L_n>0 receiver that crosses
    // into L0-resident territory: without demote, the L_n OR-reduced
    // occupancy reports false occluders at the coarse granularity even
    // though precise finer data is available along the ray's actual path.
    MarchResult s = dda_world_clipmap(shadow_origin_ws, SUN_DIR_TO_LIGHT, MAX_SHADOW_T, 0u);

    // Shadow attenuation: 0.0 if occluded, 1.0 if unoccluded.
    float shadow = s.hit ? 0.0 : 1.0;

    // --- Per-voxel albedo via DirEntry.material_data_slot ---
    //
    // Read the material-data slot through the authoritative indirection
    // (`direntry_get_material_data_slot`). Must never substitute
    // `occ_slot` in the pool decode — that re-introduces the
    // failure-allocator-identity-drift bug class that M1 explicitly
    // builds away from (see `decision-material-system-m1-sparse`).
    DirEntry de  = g_directory[occ_slot];
    uint     mds = direntry_get_material_data_slot(de);

    float3 albedo;
    if (mds == MATERIAL_DATA_SLOT_INVALID) {
        // Deferred-grow or ceiling-reached sub-chunk. Surface the state
        // visually so the user + RenderDoc can trace it.
        albedo = MAGENTA_SENTINEL;
    } else {
        // The flat slot decodes to `(segment, local)` at this single
        // point — div/mod on `mds` (never on `occ_slot`). `local_idx`
        // unpacked from the vis buffer above is the sub-chunk-local
        // voxel index (0..511); it parameterises `material_block_read_id`.
        uint          seg_idx   = mds / SLOTS_PER_SEGMENT;
        uint          pool_slot = mds - seg_idx * SLOTS_PER_SEGMENT;
        MaterialBlock mb        = material_data_pool[seg_idx][pool_slot];
        uint          mat_id    = material_block_read_id(mb, local_idx);
        albedo = material_fetch_albedo_linear(g_materials, mat_id);
    }

    float3 lit = albedo * (SUN_COLOR * ndotl * shadow + ambient);

    g_out[tid.xy] = float4(lit, 1.0);
}
