// subchunk.ps.hlsl — isolated sub-chunk ray cast using planar bitmask traversal.
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

// --- Camera (binding 0) ---
//
// Matches Rust `TestCamera` layout (64 bytes, repr(C)):
//   float3 pos     (+0)
//   float  fov_y   (+12)
//   float3 forward (+16)
//   float  aspect  (+28)
//   float3 right   (+32)
//   float  _pad0   (+44)
//   float3 up      (+48)
//   float  _pad1   (+60)
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

// --- SubchunkOcc (binding 1) ---
//
// Matches Rust `SubchunkOccupancy` layout (64 bytes, repr(C)):
//   [u32; 16]  (stored as uint4[4] here to guarantee 16-byte element stride)
//
// Bit layout: plane[z*2 + (bit/32)] bit (bit%32) = voxel (x,y,z) occupied,
// where bit = y*8+x.  word = z*2 + bit/32 selects which of the 16 u32s;
// plane_vec[word/4][word%4] unpacks it.
struct SubchunkOcc {
    uint4 plane[4];
};

// Slot 0 holds `g_consts` via gpu_consts.hlsl above (unused here; bound to
// satisfy wgpu's sequential binding compaction).
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

// --- Ray–AABB intersection for [0,8]^3 ---

bool aabb_hit(float3 ro, float3 rd, out float t0, out float t1) {
    float3 inv = rcp(rd);
    float3 ta  = (float3(0, 0, 0) - ro) * inv;
    float3 tb  = (float3(8, 8, 8) - ro) * inv;
    float3 tn  = min(ta, tb);
    float3 tf  = max(ta, tb);
    t0 = max(max(tn.x, tn.y), tn.z);
    t1 = min(min(tf.x, tf.y), tf.z);
    return t1 >= max(t0, 0.0);
}

// --- Planar bitmask traversal ---

struct Hit {
    bool hit;
    int3 voxel;
};

Hit trace(float3 ro, float3 rd) {
    Hit miss;
    miss.hit = false;

    float t_enter, t_exit;
    if (!aabb_hit(ro, rd, t_enter, t_exit)) return miss;

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

    // Entry into sub-chunk in permuted space.
    float  t        = max(t_enter, 0.0);
    float3 entry    = p_ro + p_rd * t;
    int    iz       = clamp(int(floor(entry.z)), 0, 7);
    int    iz_step  = (p_rd.z >= 0.0) ? 1 : -1;

    // t at the Z boundary between the current layer and the next.
    float z_boundary = float(iz_step > 0 ? iz + 1 : iz);
    float t_z_next   = (z_boundary - p_ro.z) / p_rd.z;
    float t_z_delta  = abs(1.0 / p_rd.z);

    [loop]
    for (int iter = 0; iter < 9; iter++) {
        if (iz < 0 || iz > 7 || t >= t_exit)
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

float4 main(float4 sv_pos : SV_Position,
            float2 uv     : TEXCOORD0) : SV_Target0 {
    // Reconstruct world-space ray direction.
    // uv (0,0) = top-left, (1,1) = bottom-right.
    // Flip Y so that world +Y is up.
    float2 ndc      = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float  half_fov = tan(g_camera.fov_y * 0.5);
    float3 ray_dir  = normalize(
        g_camera.forward
        + ndc.x * half_fov * g_camera.aspect * g_camera.right
        + ndc.y * half_fov * g_camera.up);

    // --- Diagnostic: visualise AABB hit vs. sphere hit ---
    // Red   = ray misses the [0,8]^3 AABB entirely (camera not facing cube).
    // Blue  = ray enters AABB but hits no occupied voxel (occupancy empty?).
    // Color = voxel position / 7 when the sphere is hit.
    // Remove this block once the sphere is confirmed visible.
    float diag_t0, diag_t1;
    if (!aabb_hit(g_camera.pos, ray_dir, diag_t0, diag_t1))
        return float4(1.0, 0.0, 0.0, 1.0);

    Hit h = trace(g_camera.pos, ray_dir);
    if (!h.hit)
        return float4(0.0, 0.0, 1.0, 1.0);

    // Voxel position (0..7) mapped to RGB (0..1).
    float3 col = float3(h.voxel) / 7.0;
    return float4(col, 1.0);
}
