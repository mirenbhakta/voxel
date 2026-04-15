// subchunk.ps.hlsl — voxel DDA starting from the rasterized sub-chunk hull.
//
// The vertex stage rasterizes the [0,8]^3 sub-chunk bounding cube. Each
// fragment corresponds to a point on the cube's surface (or, once we support
// the camera-inside-cube case, a point inside it). The DDA starts at that
// world-space point and marches along the camera → fragment direction — the
// rasterizer has already done the ray-AABB entry intersection for us.
//
// Storage: 8 XY planes, each a 64-bit occupancy bitmask (one bit per voxel in
// the 8×8 XY grid). Stored as 16 uint32s (2 per Z layer), packed in four
// uint4 to avoid HLSL std140 array-element padding.
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

[[vk::binding(1, 0)]] ConstantBuffer<Camera>     g_camera;
[[vk::binding(2, 0)]] ConstantBuffer<SubchunkOcc> g_occ;

// --- Occupancy lookup ---

uint pick_word(uint4 v, uint idx) {
    if (idx == 0u) return v.x;
    if (idx == 1u) return v.y;
    if (idx == 2u) return v.z;
    return v.w;
}

bool get_voxel(int x, int y, int z) {
    if (x < 0 || x > 7 || y < 0 || y > 7 || z < 0 || z > 7)
        return false;
    uint bit  = uint(y * 8 + x);         // 0..63 within the Z plane
    uint word = uint(z) * 2u + (bit >> 5u); // which of the 16 u32s
    uint4 row = g_occ.plane[word >> 2u];    // select uint4 (0..3)
    uint  val = pick_word(row, word & 3u);  // select component (0..3)
    return (val >> (bit & 31u)) & 1u;
}

// --- Planar bitmask traversal from a hull entry point ---
//
// `ro` is the world-space entry point on the sub-chunk's surface produced
// by the rasterizer. The ray direction is computed from the camera, so the
// march is always "into" the cube from `ro`. We iterate one Z-layer per
// step and stop when the layer index leaves [0, 7].

struct Hit {
    bool hit;
    int3 voxel;
};

Hit trace(float3 ro, float3 rd) {
    Hit miss;
    miss.hit = false;

    // Permute so dominant axis becomes the new "Z".
    float3 a = abs(rd);
    int dom;
    float3 p_ro, p_rd;
    if (a.x >= a.y && a.x >= a.z) {
        dom  = 0;
        p_ro = float3(ro.y, ro.z, ro.x);
        p_rd = float3(rd.y, rd.z, rd.x);
    } else if (a.y >= a.z) {
        dom  = 1;
        p_ro = float3(ro.x, ro.z, ro.y);
        p_rd = float3(rd.x, rd.z, rd.y);
    } else {
        dom  = 2;
        p_ro = ro;
        p_rd = rd;
    }

    // We start at the hull, so t begins at 0.
    float  t       = 0.0;
    int    iz      = clamp(int(floor(p_ro.z)), 0, 7);
    int    iz_step = (p_rd.z >= 0.0) ? 1 : -1;

    // t at the Z boundary between the current layer and the next.
    float z_boundary = float(iz_step > 0 ? iz + 1 : iz);
    float t_z_next   = (z_boundary - p_ro.z) / p_rd.z;
    float t_z_delta  = abs(1.0 / p_rd.z);

    [loop]
    for (int iter = 0; iter < 9; iter++) {
        if (iz < 0 || iz > 7)
            break;

        // Ray position at entry of this Z layer (permuted frame).
        float3 p  = p_ro + p_rd * t;
        int    ix = clamp(int(floor(p.x)), 0, 7);
        int    iy = clamp(int(floor(p.y)), 0, 7);

        // Un-permute to get actual voxel coordinates.
        int3 vox;
        if      (dom == 0) vox = int3(iz, ix, iy);
        else if (dom == 1) vox = int3(ix, iz, iy);
        else               vox = int3(ix, iy, iz);

        if (get_voxel(vox.x, vox.y, vox.z)) {
            Hit h;
            h.hit   = true;
            h.voxel = vox;
            return h;
        }

        // Advance to next Z layer.
        t        = t_z_next;
        t_z_next += t_z_delta;
        iz       += iz_step;
    }

    return miss;
}

// --- Pixel shader entry point ---

float4 main(float4 sv_pos      : SV_Position,
            float3 world_pos   : TEXCOORD0,
            float  inside_flag : TEXCOORD1) : SV_Target0 {
    // Ray direction is the view ray through this pixel regardless of which
    // face rasterized (front for outside-camera, back for inside-camera, or
    // the near-plane-clipped edge of either) — world_pos stays colinear with
    // the camera ray through the pixel even when clipped.
    float3 ray_dir = normalize(world_pos - g_camera.pos);

    // Ray entry selection:
    //   inside  → start at the camera itself; world_pos is on the far face
    //             (or the near-plane clip edge) and is not a valid entry.
    //   outside → start at the rasterized front-face hull point.
    float3 ray_origin = (inside_flag > 0.5) ? g_camera.pos : world_pos;

    Hit h = trace(ray_origin, ray_dir);
    if (!h.hit)
        return float4(0.0, 0.0, 1.0, 1.0); // blue: entered hull, no occupied voxel

    // Voxel position (0..7) mapped to RGB (0..1).
    float3 col = float3(h.voxel) / 7.0;
    return float4(col, 1.0);
}
