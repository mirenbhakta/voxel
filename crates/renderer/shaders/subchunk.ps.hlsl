// subchunk.ps.hlsl — voxel DDA starting from the rasterized sub-chunk hull.
//
// The vertex stage rasterizes the per-instance [origin, origin+8]^3 sub-chunk
// bounding cube. Each fragment corresponds to a point on the cube's surface
// (or, for the camera-inside-cube case, a point inside it). The DDA starts at
// that world-space point and marches along the camera → fragment direction —
// the rasterizer has already done the ray-AABB entry intersection for us.
//
// Storage: 8 XY planes, each a 64-bit occupancy bitmask (one bit per voxel in
// the 8×8 XY grid). Stored as 16 uint32s (2 per Z layer), packed in four
// uint4 to avoid HLSL std140 array-element padding.  The occupancy array holds
// one SubchunkOcc per candidate; the per-instance occ_slot selects the entry.
//
// Output: the vis buffer (R32_UINT, SV_Target0) per
// `decision-vis-buffer-deferred-shading-phase-1`. Packing layout:
//
//     bits  0..8  (9) : local_idx   (8^3 voxel index)
//     bits  9..11 (3) : face        (face-direction code 0..5)
//     bits 12..15 (4) : level_idx   (LOD level of this sub-chunk)
//     bits 16..31 (16): occ_slot    (low 16 bits of the caller's slot id)
//
// Miss pixels are `discard`ed; the vis buffer's frame clear to
// `0xFFFFFFFF` covers them. Depth stays unchanged from the pre-vis
// version: the fragment writes the analytically-projected voxel-hit depth
// via SV_Depth so the inside-cube case and the cross-sub-chunk depth-sort
// keep working exactly as before.
//
// Binding layout across VS + PS (merged into one set-0 BGL at pipeline
// construction via SPIR-V reflection):
//   0: Camera       (VS + PS)
//   1: instances    (VS only)
//   2: visible      (VS only)
//   3: occ_array    (PS only)
//
// wgpu-hal's Vulkan backend sequentially compacts BGL bindings to VK
// bindings 0, 1, 2, ... in sorted-binding order, and the SPIR-V passthrough
// path does not remap — so HLSL binding N must equal N's ordinal position
// in the merged BGL. Keeping the set contiguous from 0 guarantees this;
// gaps would silently off-by-one every shader that reads past the gap.

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

// `SubchunkOcc` is the storage type consumed by `include/dda.hlsl`. That
// header declares the primitive but leaves the binding to the consumer —
// binding-slot choice depends on the pipeline's BGL layout, which differs
// between the raster draw pass and any future compute caller.
struct SubchunkOcc {
    uint4 plane[4];
};

[[vk::binding(0, 0)]] ConstantBuffer<Camera>        g_camera;
[[vk::binding(3, 0)]] StructuredBuffer<SubchunkOcc> g_occ_array;

#include "include/dda.hlsl"

// Must match the VS projection. Kept in sync by hand — if you change these,
// update `subchunk.vs.hlsl` too. Used to project the voxel-hit position back
// to NDC depth so fragments write the actual hit depth instead of the
// rasterized hull depth. The inside-cube case rasterizes the back face (far
// from the camera), so writing hull depth would lose depth-sort battles
// against neighbouring cubes' front faces.
static const float NEAR_PLANE = 0.1;
static const float FAR_PLANE  = 1000.0;

// --- pack_vis — MarchResult → R32_UINT per the committed schema. ---
//
// Keep in sync with the table in the file header and with any consumer
// unpacking in the shade pass.
uint pack_vis(uint local_idx, uint face, uint level_idx, uint occ_slot) {
    return ((occ_slot  & 0xFFFFu) << 16)
         | ((level_idx & 0xFu)    << 12)
         | ((face      & 0x7u)    <<  9)
         |  (local_idx & 0x1FFu);
}

// --- Pixel shader entry point ---

struct PSOutput {
    // R32_UINT vis buffer — see file header for bit layout.
    uint  vis   : SV_Target0;
    // Overriding SV_Depth with the actual voxel-hit depth — rather than the
    // rasterized hull depth — is what keeps the inside-cube path correct.
    // The inside path rasterizes the cube's BACK faces (far from the camera),
    // so using hull depth would lose depth-sort battles against neighbouring
    // cubes' front faces and voxels near the camera would vanish or be
    // overdrawn by adjacent sub-chunks. The outside path benefits too: depth
    // now matches the true voxel surface instead of the front-face bias.
    float depth : SV_Depth;
};

PSOutput main(float4                   sv_pos      : SV_Position,
              float3                   world_pos   : TEXCOORD0,
              float                    inside_flag : TEXCOORD1,
              nointerpolation uint     occ_slot    : TEXCOORD2,
              nointerpolation int3     origin      : TEXCOORD3,
              nointerpolation float    voxel_size  : TEXCOORD4,
              nointerpolation uint     level_idx   : TEXCOORD5) {
    // Ray direction is the view ray through this pixel regardless of which
    // face rasterized (front for outside-camera, back for inside-camera, or
    // the near-plane-clipped edge of either) — world_pos stays colinear with
    // the camera ray through the pixel even when clipped.
    //
    // Direction vectors are scale-invariant under uniform scaling, so ray_dir
    // is valid in both world and sub-chunk local frames; only origins get
    // divided by voxel_size below.
    float3 ray_dir = normalize(world_pos - g_camera.pos);

    // Convert world-space position and camera origin to the sub-chunk's local
    // [0, 8]^3 voxel-index frame: translate by -origin, then scale by
    // 1/voxel_size so one local unit = one voxel regardless of LOD level.
    float  inv_vs       = 1.0 / voxel_size;
    float3 local_origin = (world_pos     - float3(origin)) * inv_vs;
    float3 local_cam    = (g_camera.pos  - float3(origin)) * inv_vs;

    // Ray entry selection:
    //   inside  → start at the camera's local position; world_pos is on the
    //             far face (or the near-plane clip edge) and is not a valid entry.
    //   outside → start at the rasterized front-face hull point in local space.
    //
    // The outside path's `local_origin` comes from a rasterizer-interpolated
    // world_pos and can fall a hair outside [0, 8]^3 due to FP slop at the
    // cube's outer edge. Clamp before handing off to the DDA; the primitive
    // treats its `origin` as authoritative.
    float3 ray_origin = (inside_flag > 0.5) ? local_cam : clamp(local_origin, 0.0, 8.0);

    // `1e30` acts as "no parameter bound"; the sub-chunk's own [0, 8]^3
    // extent is what bounds traversal in practice (the DDA exits as soon
    // as the voxel index leaves [0, 7]^3). Future callers with an explicit
    // shadow-ray or distance cap can pass a meaningful `max_t`.
    MarchResult h = dda_sub_chunk(ray_origin, ray_dir, 1e30, occ_slot);
    if (!h.hit) {
        // Miss: no occupied voxel along the ray inside this sub-chunk. Drop
        // the fragment so we neither write vis nor commit the cube front-face
        // depth — otherwise the hull of a near sub-chunk would depth-occlude
        // hits in the sub-chunks behind it. The vis buffer's frame clear to
        // 0xFFFFFFFF covers the uncovered-pixel case downstream.
        discard;
        // Unreachable; DXC still wants a return value.
        PSOutput o0;
        o0.vis   = 0xFFFFFFFFu;
        o0.depth = 1.0;
        return o0;
    }

    // Reconstruct the hit position from the ray parameter, then scale the
    // local [0, 8]^3 coordinate back to world space.
    float3 hit_local = ray_origin + h.t * ray_dir;
    float3 hit_world = hit_local * voxel_size + float3(origin);

    // Project the hit position back to the VS's NDC z. Must match the VS
    // formula exactly, else z-fighting. Clamp `vz` to NEAR_PLANE to cover
    // the degenerate case where the camera stands inside an occupied voxel
    // (hit at t=0 → world hit == camera.pos → vz == 0).
    float  vz = max(dot(hit_world - g_camera.pos, g_camera.forward), NEAR_PLANE);
    float  A  = FAR_PLANE / (FAR_PLANE - NEAR_PLANE);
    float  B  = -NEAR_PLANE * FAR_PLANE / (FAR_PLANE - NEAR_PLANE);

    PSOutput o;
    o.vis   = pack_vis(h.local_idx, h.face, level_idx, h.occ_slot);
    o.depth = A + B / vz;
    return o;
}
