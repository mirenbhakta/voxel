// subchunk_cull.cs.hlsl — GPU frustum + exposure cull pass for the
// sub-chunk prototype.
//
// One thread per candidate sub-chunk (MAX_CANDIDATES = 64). Thread 0
// initialises the transient indirect-args buffer (pool hands back undefined
// contents). After the barrier, every thread runs two cheap rejections
// against its candidate:
//   1. Frustum test on the [origin, origin+8]^3 AABB.
//   2. Directional-exposure test: if none of the sub-chunk's face-exposure
//      bits overlap the camera-visible face directions (derived from the
//      camera position relative to the AABB), the sub-chunk contributes no
//      visible surface from this view and is dropped.
// Survivors atomically append their index to the visible list and
// increment the indirect draw instance count.
//
// The draw-count buffer is a persistent constant = 1 written once at
// SubchunkTest construction and never touched here; fanout lives in
// instance_count on the single indirect entry.
//
// Binding layout:
//
// Set 0 — caller-supplied bindings, contiguous from 0 (wgpu-hal's Vulkan
// backend compacts BGL entries sequentially, and the SPIR-V passthrough
// path does not remap — so HLSL binding numbers must equal their ordinal
// position within the reflected set):
//   0: Camera       (uniform, 64 bytes)
//   1: instances    (StorageBuffer<Instance>, read-only; slot_mask carries
//                   the 6-bit directional exposure mask in its high bits)
//   2: visible      (RWStorageBuffer<uint>)
//
// Set 1 — owned by the cull node; the indirect-args bind group is built
// from this pipeline's reflected set-1 layout, not supplied by the caller:
//   0: indirect     (RWStorageBuffer<uint4>, one entry = DrawIndirectArgs)

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

// slot_mask packs two fields:
//   low 26 bits: occupancy slot index (unused by the cull shader itself)
//   high  6 bits: directional exposure mask
//                 bit 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z
struct Instance {
    int3 origin;
    uint slot_mask;
};

[[vk::binding(0, 0)]] ConstantBuffer<Camera>     g_camera;
[[vk::binding(1, 0)]] StructuredBuffer<Instance> g_instances;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>   g_visible;
[[vk::binding(0, 1)]] RWStructuredBuffer<uint4>  g_indirect; // uint4 = {vertex_count, instance_count, first_vertex, first_instance}

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
    // Thread 0 initialises the indirect-draw entry. g_indirect is a graph
    // transient — pool hands back undefined contents — so we must write the
    // full DrawIndirectArgs here before any atomic increment.
    //
    // vertex_count=36 (cube mesh baked into the VS); instance_count starts
    // at 0 and accumulates via InterlockedAdd below.
    if (ltid == 0u) {
        g_indirect[0] = uint4(36u, 0u, 0u, 0u);
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

    // Directional-exposure rejection.
    //
    // A face direction d is "camera-visible" on this AABB when the camera
    // lies on the d-facing side of the box. If the camera is between lo
    // and hi on some axis (inside the box in that axis), both face
    // directions of that axis can be visible. Use <= / >= for conservative
    // inclusion at the boundary.
    //
    // Bit order matches SubchunkInstance: 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z.
    float3 cam_pos  = g_camera.pos;
    uint   relevant = 0u;
    if (cam_pos.x <= hi.x) relevant |= 1u << 0;
    if (cam_pos.x >= lo.x) relevant |= 1u << 1;
    if (cam_pos.y <= hi.y) relevant |= 1u << 2;
    if (cam_pos.y >= lo.y) relevant |= 1u << 3;
    if (cam_pos.z <= hi.z) relevant |= 1u << 4;
    if (cam_pos.z >= lo.z) relevant |= 1u << 5;

    uint exposure = inst.slot_mask >> 26u;
    if ((exposure & relevant) == 0u) {
        return;
    }

    // Atomically append this candidate's index to the visible list and
    // increment the instance count in the indirect-draw entry.
    uint slot;
    InterlockedAdd(g_indirect[0].y, 1u, slot);
    g_visible[slot] = tid.x;
}
