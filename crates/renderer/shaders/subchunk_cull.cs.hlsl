// subchunk_cull.cs.hlsl — GPU frustum cull pass for the sub-chunk prototype.
//
// One thread per candidate sub-chunk (MAX_CANDIDATES = 64). Thread 0
// initialises the indirect-args and count buffers so downstream draws start
// from a clean slate. After the barrier, every thread tests its candidate
// against the camera frustum and, if visible, atomically appends its index
// to the visible list and increments the indirect draw instance count.
//
// Binding layout (set 0) — all user bindings consecutive from 1 so the
// wgpu-hal Vulkan compaction leaves HLSL binding N == VK binding N:
//   0: GpuConsts    (injected by BindingLayout, unused here but must be present)
//   1: Camera       (uniform, 64 bytes)
//   2: instances    (StorageBuffer<Instance>, read-only)
//   3: visible      (RWStorageBuffer<uint>)
//   4: indirect     (RWStorageBuffer<uint4>, one entry = DrawIndirectArgs)
//   5: count        (RWStorageBuffer<uint>,  one entry)
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

struct Instance {
    int3 origin;
    uint occ_slot;
};

[[vk::binding(1, 0)]] ConstantBuffer<Camera>     g_camera;
[[vk::binding(2, 0)]] StructuredBuffer<Instance> g_instances;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>   g_visible;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint4>  g_indirect; // uint4 = {vertex_count, instance_count, first_vertex, first_instance}
[[vk::binding(5, 0)]] RWStructuredBuffer<uint>   g_count;

static const uint  MAX_CANDIDATES = 64u;
static const float SUB            = 8.0;
static const float NEAR_PLANE     = 0.1;
static const float FAR_PLANE      = 1000.0;

// AABB frustum test.
//
// Builds 6 frustum planes from the camera basis and tests the AABB [lo, hi]
// against each. For each plane (outward normal n, anchor point p), the
// "corner-closest" test picks the corner of the AABB most aligned with n —
// if that corner is behind the plane, the entire AABB is outside the frustum.
//
// The 6 planes are:
//   near : anchor = pos + NEAR  * forward, normal = +forward
//   far  : anchor = pos + FAR   * forward, normal = -forward
//   left : built from -(right * tan_half_x + forward)
//   right: built from  (right * tan_half_x - forward)  (negate left.x)
//   down : built from -(up    * tan_half_y + forward)
//   up   : built from  (up    * tan_half_y - forward)  (negate down.y)
//
// The side planes are derived so that any point inside the frustum satisfies
// dot(n, point - anchor) >= 0.  We set anchor = camera pos for the side
// planes (the frustum apex).
bool frustum_visible(float3 lo, float3 hi) {
    float tan_half_y = tan(g_camera.fov_y * 0.5);
    float tan_half_x = tan_half_y * g_camera.aspect;

    // Pick the AABB corner most aligned with plane normal n and test it.
    // If that corner is behind the plane, the whole box is outside.
    // dot(n, corner - anchor) < 0  →  outside.
    #define TEST_PLANE(n, anchor)                                  \
    {                                                              \
        float3 p_pos = float3(                                     \
            (n).x >= 0.0 ? (hi).x : (lo).x,                      \
            (n).y >= 0.0 ? (hi).y : (lo).y,                       \
            (n).z >= 0.0 ? (hi).z : (lo).z                        \
        );                                                         \
        if (dot((n), p_pos - (anchor)) < 0.0) return false;       \
    }

    float3 cam = g_camera.pos;

    // Near plane: anchor = cam + NEAR*forward, normal = +forward.
    TEST_PLANE(g_camera.forward, cam + NEAR_PLANE * g_camera.forward)

    // Far plane: anchor = cam + FAR*forward, normal = -forward.
    TEST_PLANE(-g_camera.forward, cam + FAR_PLANE * g_camera.forward)

    // Left side: normal = forward - right * tan_half_x  (unnormalised; safe
    // for sign test because length is > 0).
    TEST_PLANE(g_camera.forward - g_camera.right * tan_half_x, cam)

    // Right side: normal = forward + right * tan_half_x.
    TEST_PLANE(g_camera.forward + g_camera.right * tan_half_x, cam)

    // Bottom side: normal = forward - up * tan_half_y.
    TEST_PLANE(g_camera.forward - g_camera.up * tan_half_y, cam)

    // Top side: normal = forward + up * tan_half_y.
    TEST_PLANE(g_camera.forward + g_camera.up * tan_half_y, cam)

    #undef TEST_PLANE

    return true;
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint ltid : SV_GroupIndex) {
    // Thread 0 initialises the indirect-draw entry and draw count. This runs
    // before any candidate test so every atomic below starts from
    // instance_count = 0.
    //
    // g_indirect holds one DrawIndirectArgs: vertex_count=36, instance_count
    // starts at 0 and accumulates via InterlockedAdd below.
    // g_count is the number of draw entries (always 1 — the fanout is in
    // instance_count, not in additional draw entries).
    if (ltid == 0u) {
        g_indirect[0] = uint4(36u, 0u, 0u, 0u);
        g_count[0]    = 1u;
    }

    GroupMemoryBarrierWithGroupSync();

    if (tid.x >= MAX_CANDIDATES) {
        return;
    }

    Instance inst = g_instances[tid.x];
    float3   lo   = float3(inst.origin);
    float3   hi   = lo + float3(SUB, SUB, SUB);

    if (!frustum_visible(lo, hi)) {
        return;
    }

    // Atomically append this candidate's index to the visible list and
    // increment the instance count in the indirect-draw entry.
    uint slot;
    InterlockedAdd(g_indirect[0].y, 1u, slot);
    g_visible[slot] = tid.x;
}
