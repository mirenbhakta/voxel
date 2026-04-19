// subchunk_shade.cs.hlsl — vis-buffer consumer; first lit pass.
//
// Reads the sub-chunk vis buffer written by `subchunk.ps.hlsl` (SV_Target0)
// and writes a shaded colour into the transient `shaded_color` image. The
// shaded image is consumed by the blit pass that copies into the swapchain.
//
// Shading model (first real-shading bite):
//   * face → axis-aligned normal, decoded from the 3-bit `face` field.
//   * directional sun (Lambertian) against a hardcoded direction.
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
//   set 0 binding 0 : `g_vis` — `Texture2D<uint>` containing the packed
//                     vis values produced by the previous pass.
//   set 0 binding 1 : `g_out` — `RWTexture2D<float4>` storage image the
//                     blit pass reads from.
//
// The workgroup size is 8×8 — one thread per output pixel. The dispatch
// grid is `(w + 7)/8 × (h + 7)/8` so right/bottom edge threads may read
// outside the image; the `GetDimensions` check guards the stores.
//
// The pass takes no `GpuConsts` binding: shading constants are baked in
// for now. When palette fetch / per-light state lands, the include for
// `gpu_consts.hlsl` goes here and the Rust caller starts passing
// `gpu_consts` to `create_bind_group`.

[[vk::binding(0, 0)]] Texture2D<uint> g_vis;

// `rgba8` pins the SPIR-V ImageFormat so reflection knows the storage
// texture's texel format without reading shader-side metadata. Matches
// the `TextureFormat::Rgba8Unorm` picked on the Rust side — see
// `SHADED_COLOR_FORMAT` in `crates/renderer/src/subchunk.rs`.
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]]
RWTexture2D<float4> g_out;

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

// Constant grey until per-voxel material lookup lands. Picked low enough
// that `albedo * (sun + sky)` stays in the displayable range.
static const float3 ALBEDO       = float3(0.50, 0.50, 0.50);

// --- face → axis-aligned normal. ---
//
// Face codes match `Direction` enum / `dda.hlsl`:
//   0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z.
float3 face_to_normal(uint face) {
    switch (face) {
        case 0u: return float3( 1.0,  0.0,  0.0);
        case 1u: return float3(-1.0,  0.0,  0.0);
        case 2u: return float3( 0.0,  1.0,  0.0);
        case 3u: return float3( 0.0, -1.0,  0.0);
        case 4u: return float3( 0.0,  0.0,  1.0);
        default: return float3( 0.0,  0.0, -1.0);
    }
}

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

    uint   face   = (v >> 9u) & 0x7u;
    float3 normal = face_to_normal(face);

    // Lambertian sun term.
    float ndotl = max(dot(normal, SUN_DIR_TO_LIGHT), 0.0);

    // Hemispherical ambient — surfaces facing up see sky, surfaces facing
    // down see ground. Cheap stand-in for a real environment lookup.
    float  hemi_t  = 0.5 + 0.5 * normal.y;
    float3 ambient = lerp(GROUND_COLOR, SKY_COLOR, hemi_t);

    float3 lit = ALBEDO * (SUN_COLOR * ndotl + ambient);

    g_out[tid.xy] = float4(lit, 1.0);
}
