// projection.hlsl — shared projection constants + depth encode/decode.
//
// wgpu's clip-space Z range is [0, 1]. The VS projection formula is:
//   sv_pos.z = A * vz + B
// where sv_pos.w = vz, so after perspective divide:
//   ndc_z = (A * vz + B) / vz = A + B / vz = encode_depth(vz)
//
// Consumers:
//   subchunk.vs.hlsl      — NEAR/FAR for INSIDE_MARGIN, DEPTH_A/DEPTH_B for projection
//   subchunk.ps.hlsl      — encode_depth for voxel-hit depth write
//   subchunk_shade.cs.hlsl — decode_vz for world-space hit reconstruction
//   subchunk_cull.cs.hlsl  — NEAR/FAR for frustum planes

#ifndef RENDERER_PROJECTION_HLSL
#define RENDERER_PROJECTION_HLSL

static const float NEAR_PLANE = 0.1;
static const float FAR_PLANE  = 1000.0;

// Matches the VS projection: depth = A + B / vz.
static const float DEPTH_A = FAR_PLANE / (FAR_PLANE - NEAR_PLANE);
static const float DEPTH_B = -NEAR_PLANE * FAR_PLANE / (FAR_PLANE - NEAR_PLANE);

// Encode view-space Z to NDC depth ([0, 1] range, matching wgpu clip-space).
float encode_depth(float vz) { return DEPTH_A + DEPTH_B / vz; }

// Decode NDC depth back to view-space Z.
float decode_vz(float depth)  { return DEPTH_B / (depth - DEPTH_A); }

#endif // RENDERER_PROJECTION_HLSL
