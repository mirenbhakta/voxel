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
// Traversal: permute axes so the dominant ray direction becomes "Z"; iterate
// through Z layers, one per step, testing the bit at the current (x,y)
// position against the plane's 64-bit mask. O(8) operations, no per-voxel DDA.
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

struct SubchunkOcc {
    uint4 plane[4];
};

[[vk::binding(0, 0)]] ConstantBuffer<Camera>        g_camera;
[[vk::binding(3, 0)]] StructuredBuffer<SubchunkOcc> g_occ_array;

// Must match the VS projection. Kept in sync by hand — if you change these,
// update `subchunk.vs.hlsl` too. Used to project the voxel-hit position back
// to NDC depth so fragments write the actual hit depth instead of the
// rasterized hull depth. The inside-cube case rasterizes the back face (far
// from the camera), so writing hull depth would lose depth-sort battles
// against neighbouring cubes' front faces.
static const float NEAR_PLANE = 0.1;
static const float FAR_PLANE  = 1000.0;

// --- Occupancy lookup ---

uint pick_word(uint4 v, uint idx) {
    if (idx == 0u) return v.x;
    if (idx == 1u) return v.y;
    if (idx == 2u) return v.z;
    return v.w;
}

bool get_voxel(int x, int y, int z, uint occ_slot) {
    if (x < 0 || x > 7 || y < 0 || y > 7 || z < 0 || z > 7)
        return false;
    uint bit  = uint(y * 8 + x);              // 0..63 within the Z plane
    uint word = uint(z) * 2u + (bit >> 5u);   // which of the 16 u32s
    uint4 row = g_occ_array[occ_slot].plane[word >> 2u]; // select uint4 (0..3)
    uint  val = pick_word(row, word & 3u);    // select component (0..3)
    return (val >> (bit & 31u)) & 1u;
}

// --- 3D DDA traversal from a hull entry point ---
//
// `ro` is the entry point in the sub-chunk's local [0,8]^3 frame. The ray
// direction is the same in world and local space (translation-only change).
//
// Amanatides-Woo: track the t at which the ray crosses the next cell
// boundary along each axis; each step advances one cell along whichever
// axis has the smallest t_next. Worst case = 8 cells in each of 3 axes =
// 24 steps before exiting the sub-chunk. One voxel test per step.
//
// A dominant-axis-only planar march would miss cells the ray passes through
// sideways within a single layer, producing holes and stepped artefacts when
// viewed along a non-primary axis.

struct Hit {
    bool   hit;
    int3   voxel;
    float3 pos; // hit position in the sub-chunk's local [0, 8]^3 frame
};

Hit trace(float3 ro, float3 rd, uint occ_slot) {
    Hit miss;
    miss.hit = false;

    // Clamp the rasterized hull entry to [0, 8] — fragments at the cube's
    // outer edge can fall a hair outside due to interpolation FP slop.
    ro = clamp(ro, 0.0, 8.0);

    int3 vox = clamp(int3(floor(ro)), int3(0, 0, 0), int3(7, 7, 7));

    int3 step;
    step.x = rd.x > 0.0 ? 1 : (rd.x < 0.0 ? -1 : 0);
    step.y = rd.y > 0.0 ? 1 : (rd.y < 0.0 ? -1 : 0);
    step.z = rd.z > 0.0 ? 1 : (rd.z < 0.0 ? -1 : 0);

    // Per-axis t at the next cell boundary, and per-axis t increment per
    // full cell crossing. Axes with zero ray-direction component get HUGE
    // values so the min-select never picks them — the ray never crosses a
    // boundary on that axis.
    const float HUGE = 1e30;
    float3 t_next, t_delta;

    if (step.x != 0) {
        float boundary = step.x > 0 ? float(vox.x + 1) : float(vox.x);
        t_next.x  = (boundary - ro.x) / rd.x;
        t_delta.x = abs(1.0 / rd.x);
    } else { t_next.x = HUGE; t_delta.x = HUGE; }

    if (step.y != 0) {
        float boundary = step.y > 0 ? float(vox.y + 1) : float(vox.y);
        t_next.y  = (boundary - ro.y) / rd.y;
        t_delta.y = abs(1.0 / rd.y);
    } else { t_next.y = HUGE; t_delta.y = HUGE; }

    if (step.z != 0) {
        float boundary = step.z > 0 ? float(vox.z + 1) : float(vox.z);
        t_next.z  = (boundary - ro.z) / rd.z;
        t_delta.z = abs(1.0 / rd.z);
    } else { t_next.z = HUGE; t_delta.z = HUGE; }

    // Test the entry cell first — the ray starts inside it.
    if (get_voxel(vox.x, vox.y, vox.z, occ_slot)) {
        Hit h; h.hit = true; h.voxel = vox; h.pos = ro; return h;
    }

    // 8 cells × 3 axes = 24 step bound.
    //
    // `t_entry` is the ray parameter at which we crossed INTO the current
    // voxel — i.e., the min of `t_next` BEFORE the axis step that produced it.
    // Using it to compute the hit position (`ro + t_entry * rd`) gives the
    // true entry point of the occupied voxel along the ray, which the pixel
    // shader projects back to NDC depth.
    float t_entry = 0.0;

    [loop]
    for (int i = 0; i < 24; i++) {
        if (t_next.x < t_next.y) {
            if (t_next.x < t_next.z) {
                t_entry   = t_next.x;
                vox.x    += step.x;
                t_next.x += t_delta.x;
            } else {
                t_entry   = t_next.z;
                vox.z    += step.z;
                t_next.z += t_delta.z;
            }
        } else {
            if (t_next.y < t_next.z) {
                t_entry   = t_next.y;
                vox.y    += step.y;
                t_next.y += t_delta.y;
            } else {
                t_entry   = t_next.z;
                vox.z    += step.z;
                t_next.z += t_delta.z;
            }
        }

        if (any(vox < int3(0, 0, 0)) || any(vox > int3(7, 7, 7)))
            break;

        if (get_voxel(vox.x, vox.y, vox.z, occ_slot)) {
            Hit h; h.hit = true; h.voxel = vox; h.pos = ro + t_entry * rd; return h;
        }
    }

    return miss;
}

// --- Pixel shader entry point ---

struct PSOutput {
    float4 color : SV_Target0;
    // Overriding SV_Depth with the actual voxel-hit depth — rather than the
    // rasterized hull depth — is what keeps the inside-cube path correct.
    // The inside path rasterizes the cube's BACK faces (far from the camera),
    // so using hull depth would lose depth-sort battles against neighbouring
    // cubes' front faces and voxels near the camera would vanish or be
    // overdrawn by adjacent sub-chunks. The outside path benefits too: depth
    // now matches the true voxel surface instead of the front-face bias.
    float  depth : SV_Depth;
};

PSOutput main(float4                   sv_pos      : SV_Position,
              float3                   world_pos   : TEXCOORD0,
              float                    inside_flag : TEXCOORD1,
              nointerpolation uint     occ_slot    : TEXCOORD2,
              nointerpolation int3     origin      : TEXCOORD3) {
    // Ray direction is the view ray through this pixel regardless of which
    // face rasterized (front for outside-camera, back for inside-camera, or
    // the near-plane-clipped edge of either) — world_pos stays colinear with
    // the camera ray through the pixel even when clipped.
    float3 ray_dir = normalize(world_pos - g_camera.pos);

    // Convert world-space position and camera origin to the sub-chunk's local
    // [0, 8]^3 frame. Ray direction is unchanged (direction-only).
    float3 local_origin = world_pos - float3(origin);
    float3 local_cam    = g_camera.pos - float3(origin);

    // Ray entry selection:
    //   inside  → start at the camera's local position; world_pos is on the
    //             far face (or the near-plane clip edge) and is not a valid entry.
    //   outside → start at the rasterized front-face hull point in local space.
    float3 ray_origin = (inside_flag > 0.5) ? local_cam : local_origin;

    Hit h = trace(ray_origin, ray_dir, occ_slot);
    if (!h.hit) {
        // Miss: no occupied voxel along the ray inside this sub-chunk. Drop
        // the fragment so we neither write colour nor commit the cube
        // front-face depth — otherwise the hull of a near sub-chunk would
        // depth-occlude hits in the sub-chunks behind it.
        discard;
        // Unreachable; DXC still wants a return value.
        PSOutput o0;
        o0.color = float4(0.0, 0.0, 0.0, 0.0);
        o0.depth = 1.0;
        return o0;
    }

    // Project the hit position back to the VS's NDC z. Must match the VS
    // formula exactly, else z-fighting. Clamp `vz` to NEAR_PLANE to cover the
    // degenerate case where the camera stands inside an occupied voxel
    // (hit at t=0 → world hit == camera.pos → vz == 0 → 1/vz blowup).
    float3 hit_world = h.pos + float3(origin);
    float  vz        = max(dot(hit_world - g_camera.pos, g_camera.forward), NEAR_PLANE);
    float  A         = FAR_PLANE / (FAR_PLANE - NEAR_PLANE);
    float  B         = -NEAR_PLANE * FAR_PLANE / (FAR_PLANE - NEAR_PLANE);

    PSOutput o;
    o.color = float4(float3(h.voxel) / 7.0, 1.0);
    o.depth = A + B / vz;
    return o;
}
