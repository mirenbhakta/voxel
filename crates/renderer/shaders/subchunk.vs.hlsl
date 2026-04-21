// Sub-chunk cube vertex shader.
//
// Rasterizes the per-instance sub-chunk bounding cube so that the fragment
// shader can start the voxel DDA at the rasterized hull-entry point instead
// of tracing the ray-AABB intersection per pixel from the camera.
//
// SV_VertexID 0..35 indexes into a static 36-vertex unit cube (6 faces × 2
// tris × 3 verts). Windings are CCW from outside so that
// `PrimitiveState::cull_mode = Back` leaves exactly the faces visible to the
// camera — when the camera is outside the cube, at most three faces survive.
//
// SV_InstanceID is the index into the visible list produced by the cull pass.
// The visible list maps each slot to a candidate index; the instance buffer
// carries the per-sub-chunk origin and occupancy slot.
//
// The camera uniform already carries an orthonormal basis + fov_y + aspect;
// the projection matrix is reconstructed inline rather than adding a new
// uniform.
//
// Bindings are contiguous from 0 within the set so that the HLSL-declared
// binding number equals the VK binding number produced by wgpu-hal's
// sequential BGL compaction (the SPIR-V passthrough path does not remap).
// The shared camera/instances/visible slots (0..2) are co-declared with
// `subchunk.ps.hlsl` so the two stages merge into a single set-0 BGL.

#include "include/camera.hlsl"
#include "include/instance.hlsl"
#include "include/projection.hlsl"

[[vk::binding(0, 0)]] ConstantBuffer<Camera>     g_camera;
[[vk::binding(1, 0)]] StructuredBuffer<Instance> g_instances;
[[vk::binding(2, 0)]] StructuredBuffer<uint>     g_visible;

// 36 cube-corner positions in the unit cube [0,1]^3. Windings are CCW from
// outside — the outward-normal cross product of (v1 - v0) × (v2 - v0) matches
// the face's axis-aligned outward direction for every triangle.
static const float3 CUBE_VERTS[36] = {
    // -X
    float3(0, 0, 0), float3(0, 0, 1), float3(0, 1, 1),
    float3(0, 0, 0), float3(0, 1, 1), float3(0, 1, 0),
    // +X
    float3(1, 0, 0), float3(1, 1, 0), float3(1, 1, 1),
    float3(1, 0, 0), float3(1, 1, 1), float3(1, 0, 1),
    // -Y
    float3(0, 0, 0), float3(1, 0, 0), float3(1, 0, 1),
    float3(0, 0, 0), float3(1, 0, 1), float3(0, 0, 1),
    // +Y
    float3(0, 1, 0), float3(0, 1, 1), float3(1, 1, 1),
    float3(0, 1, 0), float3(1, 1, 1), float3(1, 1, 0),
    // -Z
    float3(0, 0, 0), float3(0, 1, 0), float3(1, 1, 0),
    float3(0, 0, 0), float3(1, 1, 0), float3(1, 0, 0),
    // +Z
    float3(0, 0, 1), float3(1, 0, 1), float3(1, 1, 1),
    float3(0, 0, 1), float3(1, 1, 1), float3(0, 1, 1),
};

// Inside-cube margin: when the camera is within NEAR_PLANE of a cube face,
// the front face gets clipped by the near plane and the rasterizer falls
// back on the back face — `world_pos` then sits on the far face, which the
// DDA must not use as its entry. Treating "near-plane-close to the cube" as
// "inside" routes those fragments through the camera-origin path instead.
static const float INSIDE_MARGIN = NEAR_PLANE;

void main(uint vid : SV_VertexID,
          uint iid : SV_InstanceID,
          out float4                   sv_pos      : SV_Position,
          out float3                   world_pos   : TEXCOORD0,
          out float                    inside_flag : TEXCOORD1,
          out nointerpolation uint     occ_slot    : TEXCOORD2,
          out nointerpolation int3     origin      : TEXCOORD3,
          out nointerpolation float    voxel_size  : TEXCOORD4,
          out nointerpolation uint     level_idx   : TEXCOORD5) {
    // Look up which candidate this instance corresponds to.
    uint     slot = g_visible[iid];
    Instance inst = g_instances[slot];

    // Per-instance LOD scale: voxel edge length in world units.
    // At L0 voxel_size = 1, cube_extent = 8. Each level doubles both.
    uint  lvl         = instance_level(inst);
    float vox_size    = float(1u << lvl);
    float cube_extent = 8.0 * vox_size;

    // Uniform per-cube: is the camera inside (or near-plane-close to) this
    // sub-chunk's [origin, origin + cube_extent]^3 box? Evaluated 36 times
    // per draw rather than per-fragment to keep the fragment stage branchless
    // beyond a single select.
    float3 lo     = float3(inst.origin);
    float3 hi     = lo + float3(cube_extent, cube_extent, cube_extent);
    float3 margin = float3(INSIDE_MARGIN, INSIDE_MARGIN, INSIDE_MARGIN);
    bool inside   = all(g_camera.pos > lo - margin)
                 && all(g_camera.pos < hi + margin);

    // Winding flip for inside cubes: swap verts 1↔2 within each triangle so
    // the rasterizer sees the inside-facing side as the "front" that survives
    // back-face culling. Outside cubes use the natural outward-CCW winding
    // from CUBE_VERTS. Per-instance uniform branch — no wavefront divergence
    // beyond a handful of ALU ops.
    uint local   = vid % 3u;
    uint flipped = (local == 0u) ? 0u : (3u - local); // (0,1,2) → (0,2,1)
    uint vid_eff = inside ? ((vid / 3u) * 3u + flipped) : vid;

    float3 unit = CUBE_VERTS[vid_eff];
    float3 w    = unit * cube_extent + float3(inst.origin);

    // View-space coordinates via the camera basis. forward is the look
    // direction so view-space z is +forward for points in front of the camera.
    float3 rel = w - g_camera.pos;
    float  vx  = dot(rel, g_camera.right);
    float  vy  = dot(rel, g_camera.up);
    float  vz  = dot(rel, g_camera.forward);

    // Standard perspective projection. wgpu's clip-space Z range is [0, 1]:
    //   ndc_z = FAR/(FAR-NEAR) - (NEAR*FAR/(FAR-NEAR)) / vz
    float f = 1.0 / tan(g_camera.fov_y * 0.5);

    sv_pos      = float4((f / g_camera.aspect) * vx, f * vy, DEPTH_A * vz + DEPTH_B, vz);
    world_pos   = w;
    inside_flag = inside ? 1.0 : 0.0;
    occ_slot    = instance_slot(inst);
    origin      = inst.origin;
    voxel_size  = vox_size;
    // 4-bit LOD level passed through for vis-buffer packing. Shading
    // consumers unpack via `(vis >> 12) & 0xFu`.
    level_idx   = lvl;
}
