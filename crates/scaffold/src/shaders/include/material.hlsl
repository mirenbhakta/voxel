// Material sampling from the volumetric material buffer.
//
// Phase 1: flat 128 MB volume, same lookup as the WGSL shader.
// Phase 2: replaces internals with countbits + packed sub-blocks.

#ifndef MATERIAL_HLSL
#define MATERIAL_HLSL

#include "include/bindings.hlsl"

/// Resolve the block ID at a voxel position within a chunk slot.
///
/// The material volume stores one byte per voxel, packed 4 to a u32.
/// Layout: slot * 8192 words + (z * 1024 + y * 32 + x) / 4.
uint resolve_block_id(ByteAddressBuffer material_volume, uint slot,
                      uint vx, uint vy, uint vz) {
    uint voxel_idx = vz * 1024 + vy * 32 + vx;
    uint word_idx  = slot * 8192 + voxel_idx / 4;
    uint byte_off  = (voxel_idx % 4) * 8;
    uint word      = material_volume.Load(word_idx * 4);
    return (word >> byte_off) & 0xFF;
}

/// Look up material color from the material table.
/// Returns unpacked RGBA as float4.
float4 material_color(ByteAddressBuffer material_table, uint block_id) {
    // MaterialEntry is 16 bytes. color_rgba is the first u32.
    uint color_packed = material_table.Load(block_id * 16);

    float4 color;
    color.r = float((color_packed >>  0) & 0xFF) / 255.0;
    color.g = float((color_packed >>  8) & 0xFF) / 255.0;
    color.b = float((color_packed >> 16) & 0xFF) / 255.0;
    color.a = float((color_packed >> 24) & 0xFF) / 255.0;
    return color;
}

/// Resolve the texture index for a block and face direction.
/// Returns the texture array layer index.
uint resolve_texture_idx(ByteAddressBuffer material_table,
                         ByteAddressBuffer face_textures,
                         uint block_id, uint direction) {
    // MaterialEntry: [color_rgba, texture_idx, face_offset, _pad]
    uint texture_idx = material_table.Load(block_id * 16 + 4);
    uint face_offset = material_table.Load(block_id * 16 + 8);

    if (face_offset != 0) {
        texture_idx = face_textures.Load((face_offset + direction) * 4);
    }

    return texture_idx;
}

#endif // MATERIAL_HLSL
