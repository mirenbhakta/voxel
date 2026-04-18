// subchunk_cull.cs.hlsl — GPU frustum + exposure cull pass for the
// sub-chunk pipeline.
//
// One thread per candidate sub-chunk (MAX_CANDIDATES = 256). Thread 0
// initialises the transient indirect-args buffer (pool hands back undefined
// contents). After the barrier, every thread runs three cheap rejections
// against its candidate:
//   0. Padding sentinel: slot_mask bit 31 = `SubchunkInstance::PADDING_BIT`.
//      Tail entries in the instance array carry this bit and are dropped
//      before touching any per-slot buffer (so padding never depends on
//      the contents of `g_directory[0]`).
//   1. Frustum test on the [origin, origin + 8*voxel_size]^3 AABB.
//   2. Directional-exposure test: if none of the sub-chunk's face-exposure
//      bits overlap the camera-visible face directions (derived from the
//      camera position relative to the AABB), the sub-chunk contributes no
//      visible surface from this view and is dropped. Exposure is unpacked
//      from `g_directory[slot].bits` via `direntry_get_exposure`; the
//      directory is CPU-authored and rewritten on every residency event.
// Survivors atomically append their index to the visible list and
// increment the indirect draw instance count.
//
// Exposure flow (as of Step 3):
//
// On residency insert, the CPU no longer pre-authors `exposure = 0x3F`.
// Instead, prep classifies the sub-chunk (isolated exposure for now —
// `0x3F` when any voxel is solid, `0` when fully empty) and ferries that
// classification through the widened dirty list back into the directory
// via CPU-driven `queue.write_buffer` at retirement. An empty sub-chunk
// therefore retires with exposure=0 and `material_slot=INVALID`, so the
// `direntry_has_material` / `direntry_get_exposure` early-outs below drop
// it before any further work. Step 4 replaces isolated exposure with
// neighbor-aware exposure + a real is_solid hint.
//
// `slot_mask` packs slot / level + a high-bit padding flag; see
// subchunk.vs.hlsl for the bit layout. Level is needed here for the
// per-instance cube extent; exposure is NOT stored in the instance.
//
// The draw-count buffer is a persistent constant = 1; fanout lives in
// instance_count on the single indirect entry.
//
// Binding layout:
//
// Set 0 — caller-supplied bindings, contiguous from 0 (wgpu-hal's Vulkan
// backend compacts BGL entries sequentially, and the SPIR-V passthrough
// path does not remap — so HLSL binding numbers must equal their ordinal
// position within the reflected set):
//   0: Camera     (uniform, 64 bytes)
//   1: instances  (StorageBuffer<Instance>, read-only)
//   2: visible    (RWStorageBuffer<uint>)
//   3: lod_mask   (uniform, 512 bytes) — per-level finer-shell AABB;
//                 sub-chunks whose AABB is fully inside their level's
//                 mask entry are dropped (the finer level renders them).
//   4: directory  (StructuredBuffer<DirEntry>, read-only, one entry per
//                  directory_index) — carries the exposure mask, resident
//                  bit, and material-slot index for each sub-chunk.
//
// Set 1 — owned by the cull node; the indirect-args bind group is built
// from this pipeline's reflected set-1 layout, not supplied by the caller:
//   0: indirect     (RWStorageBuffer<uint4>, one entry = DrawIndirectArgs)

// Pulls in `DirEntry` + `direntry_*` accessors. The header also includes
// `gpu_consts.hlsl`, but the cull pass doesn't perform coord resolution —
// cull is handed the directory index directly via the instance's slot
// field, so the torus-verify path in the header is unused here.
#include "include/directory.hlsl"

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

// slot_mask packs two fields + a padding sentinel:
//   bits  0-21: occupancy slot index (22 bits)  — selects `g_directory` entry
//   bits 22-25: LOD level (4 bits)              — used here for extent
//   bits 26-30: reserved (5 bits), must be zero
//   bit  31   : padding sentinel — matches `SubchunkInstance::PADDING_BIT`
struct Instance {
    int3 origin;
    uint slot_mask;
};

// Per-level finer-shell AABB. Layout mirrors Rust `LodMaskUniform`:
// `lo[N].xyz` and `hi[N].xyz` describe the world-space box that level N
// should defer to (the next-finer configured level's shell). `hi[N].w` is
// an active flag — `0` for level 0 and for any unconfigured level.
struct LodMask {
    float4 lo[16];
    float4 hi[16];
};

[[vk::binding(0, 0)]] ConstantBuffer<Camera>     g_camera;
[[vk::binding(1, 0)]] StructuredBuffer<Instance> g_instances;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>   g_visible;
[[vk::binding(3, 0)]] ConstantBuffer<LodMask>    g_lod_mask;
[[vk::binding(4, 0)]] StructuredBuffer<DirEntry> g_directory;
[[vk::binding(0, 1)]] RWStructuredBuffer<uint4>  g_indirect; // uint4 = {vertex_count, instance_count, first_vertex, first_instance}

static const uint  MAX_CANDIDATES   = 256u;
static const float SUBCHUNK_VOXELS  = 8.0;   // voxels per sub-chunk edge (level-invariant)
static const float NEAR_PLANE       = 0.1;
static const float FAR_PLANE        = 1000.0;

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

[numthreads(256, 1, 1)]
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

    // Padding sentinel: tail entries in the instance array carry bit 31
    // (`SubchunkInstance::PADDING_BIT`). Drop them before touching any
    // slot-indexed buffer — `inst.slot_mask`'s slot field on a padding
    // entry has no owner, and reading `g_directory[slot]` for it would
    // conflate padding with an unused-but-valid directory entry.
    if ((inst.slot_mask & 0x80000000u) != 0u) {
        return;
    }

    uint   slot        = inst.slot_mask & 0x3FFFFFu;
    uint   level       = (inst.slot_mask >> 22u) & 0xFu;
    float  voxel_size  = float(1u << level);
    float  cube_extent = SUBCHUNK_VOXELS * voxel_size;
    float3 lo          = float3(inst.origin);
    float3 hi          = lo + float3(cube_extent, cube_extent, cube_extent);

    if (!frustum_visible(lo, hi)) {
        return;
    }

    // LOD cascade rejection: if this sub-chunk's AABB is fully inside the
    // next-finer level's shell, a finer-level instance is rendering every
    // fragment we would produce — drop without emission. Partial overlaps
    // proceed; the pixel shader handles the partial-coverage discard.
    float4 mask_hi_v = g_lod_mask.hi[level];
    if (mask_hi_v.w > 0.5) {
        float3 mask_lo = g_lod_mask.lo[level].xyz;
        float3 mask_hi = mask_hi_v.xyz;
        if (all(lo >= mask_lo) && all(hi <= mask_hi)) {
            return;
        }
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

    DirEntry dir = g_directory[slot];

    // Future-proof early-out. Step 4 will emit a uniformly-solid hint for
    // sub-chunks whose interior voxels are all solid; those contribute no
    // visible surface from any view and can be rejected here regardless
    // of the exposure mask. The bit is always clear today, so this branch
    // costs one load and a zero-test.
    if (direntry_is_solid(dir)) {
        return;
    }

    uint exposure = direntry_get_exposure(dir);
    if ((exposure & relevant) == 0u) {
        return;
    }

    // Atomically append this candidate's index to the visible list and
    // increment the instance count in the indirect-draw entry.
    uint visible_idx;
    InterlockedAdd(g_indirect[0].y, 1u, visible_idx);
    g_visible[visible_idx] = tid.x;
}
