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
// GpuConsts at binding 0 is unused by this test, but must be declared and
// bound. wgpu-hal's Vulkan backend sequentially compacts BGL bindings to
// VK bindings 0, 1, 2, ... regardless of the user-supplied binding numbers
// (see wgpu-hal vulkan/device.rs `create_bind_group_layout`). Since the
// SPIR-V passthrough path skips naga's binding remap, the shader's binding
// numbers MUST match the compacted VK binding numbers — i.e., bindings
// must start at 0 and be contiguous. Skipping slot 0 causes the shader's
// bindings 1 and 2 to silently read the wrong / unbound descriptors.
//
// Bindings 2 and 3 (instances and visible) are vertex-stage-only; this shader
// does not declare them. Their positions are still reflected in the BGL by the
// Rust side, which is what determines the VK binding compaction order. The PS
// only declares the bindings it actually reads (1 and 4); DXC emits only those
// descriptor references into the SPIR-V, and the VK binding numbers for those
// match because no slots are skipped in the BGL.
#include "include/gpu_consts.hlsl"

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

[[vk::binding(1, 0)]] ConstantBuffer<Camera>        g_camera;
[[vk::binding(4, 0)]] StructuredBuffer<SubchunkOcc> g_occ_array;

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
    bool hit;
    int3 voxel;
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
        Hit h; h.hit = true; h.voxel = vox; return h;
    }

    // 8 cells × 3 axes = 24 step bound.
    [loop]
    for (int i = 0; i < 24; i++) {
        if (t_next.x < t_next.y) {
            if (t_next.x < t_next.z) {
                vox.x    += step.x;
                t_next.x += t_delta.x;
            } else {
                vox.z    += step.z;
                t_next.z += t_delta.z;
            }
        } else {
            if (t_next.y < t_next.z) {
                vox.y    += step.y;
                t_next.y += t_delta.y;
            } else {
                vox.z    += step.z;
                t_next.z += t_delta.z;
            }
        }

        if (any(vox < int3(0, 0, 0)) || any(vox > int3(7, 7, 7)))
            break;

        if (get_voxel(vox.x, vox.y, vox.z, occ_slot)) {
            Hit h; h.hit = true; h.voxel = vox; return h;
        }
    }

    return miss;
}

// --- Pixel shader entry point ---

float4 main(float4                   sv_pos      : SV_Position,
            float3                   world_pos   : TEXCOORD0,
            float                    inside_flag : TEXCOORD1,
            nointerpolation uint     occ_slot    : TEXCOORD2,
            nointerpolation int3     origin      : TEXCOORD3) : SV_Target0 {
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
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    // Voxel position (0..7) mapped to RGB (0..1).
    float3 col = float3(h.voxel) / 7.0;
    return float4(col, 1.0);
}
