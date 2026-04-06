// Pixel shader: material resolve + shading.
//
// Resolves the voxel identity by nudging half a voxel inward along
// the face normal, then samples the volumetric material buffer for
// block ID and applies texture + lighting.

#include "include/bindings.hlsl"
#include "include/quad.hlsl"
#include "include/material.hlsl"
#include "include/shading.hlsl"

// Bindings (render pass, pixel shader).
cbuffer CameraCB : register(b0, space0) {
    Camera camera;
};

ByteAddressBuffer  material_volume : register(t4, space0);
ByteAddressBuffer  material_table  : register(t5, space0);
ByteAddressBuffer  face_textures   : register(t6, space0);
Texture2DArray     block_textures  : register(t7, space0);
SamplerState       tex_sampler     : register(s8, space0);

// Input from vertex shader.
struct PS_Input {
    float4 clip_pos    : SV_Position;
    float3 world_pos   : TEXCOORD0;
    nointerpolation uint direction : TEXCOORD1;
    nointerpolation uint slot      : TEXCOORD2;
    float2 quad_uv     : TEXCOORD3;
    nointerpolation float2 quad_size : TEXCOORD4;
};

float4 main(PS_Input input) : SV_Target {
    // Nudge half a voxel toward the interior to identify the owning
    // voxel. Without the nudge, positive-direction faces resolve to
    // the neighbor.
    float3 normal  = NORMAL_VEC[input.direction];
    float3 nudge   = normal * 0.5;
    float3 voxel_f = floor(input.world_pos - nudge);

    // Wrap to chunk-local coordinates (0..31).
    uint vx = uint(int(voxel_f.x) & 31);
    uint vy = uint(int(voxel_f.y) & 31);
    uint vz = uint(int(voxel_f.z) & 31);

    // Read block ID from the volumetric material array.
    uint block_id = resolve_block_id(material_volume, input.slot, vx, vy, vz);

    // Look up material properties.
    float4 color = material_color(material_table, block_id);

    // Resolve texture index.
    uint tex_idx = resolve_texture_idx(
        material_table, face_textures, block_id, input.direction
    );

    // Compute UV from world position based on face direction.
    float2 uv;
    if (input.direction < 2)
        uv = frac(input.world_pos.yz);      // X faces
    else if (input.direction < 4)
        uv = frac(input.world_pos.xz);      // Y faces
    else
        uv = frac(input.world_pos.xy);      // Z faces

    // Sample the texture array.
    float4 tex_color = block_textures.Sample(tex_sampler, float3(uv, tex_idx));
    float3 base = tex_color.rgb * color.rgb;

    // Directional shading: Lambert diffuse with ambient floor.
    bool shading_on = (camera.flags & 1) != 0;
    if (shading_on) {
        base = apply_lambert(base, normal, camera.sun_dir.xyz, camera.sun_dir.w);
    }

    // Quad edge outlines.
    bool outline_on = (camera.flags & 2) != 0;
    if (outline_on) {
        base = apply_outline(base, input.quad_uv, input.quad_size);
    }

    return float4(base, 1.0);
}
