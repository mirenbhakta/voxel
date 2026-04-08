// Material sampling from the sparse packed material buffer.
//
// Each chunk's visible sub-blocks (8x8x8 regions) are packed
// contiguously. The sub_mask bitmask indicates which of the 64
// sub-blocks are present. Resolution uses countbits to compute the
// packed offset -- one ALU instruction plus one direct buffer read.

#ifndef MATERIAL_HLSL
#define MATERIAL_HLSL

#include "include/bindings.hlsl"

/// Resolve the block ID at a voxel position within a chunk.
///
/// The material buffer stores packed sub-blocks of u16 block IDs.
/// Only sub-blocks with visible faces are present, indexed via
/// popcount on the sub_mask bitmask.
///
/// # Arguments
///
/// * `material_buf` - Packed sparse material buffer.
/// * `mat_base`     - Byte offset of this chunk's first sub-block.
/// * `sub_mask`     - 64-bit sub-block visibility mask (lo, hi).
/// * `vx, vy, vz`   - Chunk-local voxel coordinates (0..31).
uint resolve_block_id(ByteAddressBuffer material_buf,
                      uint mat_base, uint2 sub_mask,
                      uint vx, uint vy, uint vz) {
    uint bx = vx >> 3;  uint by = vy >> 3;  uint bz = vz >> 3;
    uint lx = vx & 7;   uint ly = vy & 7;   uint lz = vz & 7;

    uint sub_idx = bz * 16 + by * 4 + bx;

    // Count populated sub-blocks before this one.
    uint offset;
    if (sub_idx < 32)
        offset = countbits(sub_mask.x & ((1u << sub_idx) - 1u));
    else
        offset = countbits(sub_mask.x)
               + countbits(sub_mask.y & ((1u << (sub_idx - 32)) - 1u));

    // Linear index within the 8x8x8 sub-block.
    uint local_idx = lz * 64 + ly * 8 + lx;
    uint byte_addr = mat_base + offset * SUB_BLOCK_BYTES
                   + local_idx * 2;

    // Load u16 from ByteAddressBuffer (aligned u32 load + extract).
    uint word = material_buf.Load(byte_addr & ~3u);
    return (byte_addr & 2u) ? (word >> 16) : (word & 0xFFFF);
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
