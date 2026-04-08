// Material packing compute shader.
//
// Copies populated 8x8x8 sub-blocks from the transient staging buffer
// to contiguous positions in the packed material buffer. Only sub-blocks
// with visible faces (as indicated by the sub_mask bitmask) are copied.
//
// Dispatch: (popcount(sub_mask), 1, 1) workgroups of 256 threads.
// Each workgroup copies one 1024-byte sub-block (8x8x8 x u16).
// Thread i copies one u32 (2 voxels).

#include "include/bindings.hlsl"

// Bindings (material pack pass).
ByteAddressBuffer   material_staging   : register(t0, space0);
RWByteAddressBuffer material_buf       : register(u1, space0);
ByteAddressBuffer   material_range_buf : register(t2, space0);

// Push constants (8 bytes -- device limit).
struct MaterialPackPush {
    uint staging_index;   // index into staging buffer (0..63)
    uint slot;            // chunk slot for reading material_range_buf
};

[[vk::push_constant]] MaterialPackPush push;

[numthreads(256, 1, 1)]
void main(uint3 lid : SV_GroupThreadID,
          uint3 gid : SV_GroupID) {
    uint sub_rank = gid.x;
    uint tid      = lid.x;

    // Read material range from the per-slot buffer.
    // MaterialRange: [buffer_index(4), base_offset(4),
    //   sub_mask_lo(4), sub_mask_hi(4)].
    uint mr_offset = push.slot * 16;
    uint dst_base  = material_range_buf.Load(mr_offset + 4);
    uint2 mask     = uint2(
        material_range_buf.Load(mr_offset + 8),
        material_range_buf.Load(mr_offset + 12)
    );

    // Find the actual sub-block index for this rank by walking set
    // bits. All threads in the workgroup agree on sub_rank, so this
    // is uniform and branch-free across the warp.
    uint sub_idx = 0;
    uint count   = 0;

    [loop]
    for (uint i = 0; i < 64; i++) {
        uint word = (i < 32) ? mask.x : mask.y;
        uint bit  = (i < 32) ? i : (i - 32);

        if (word & (1u << bit)) {
            if (count == sub_rank) {
                sub_idx = i;
                break;
            }

            count++;
        }
    }

    // Sub-block 3D coordinates within the 4x4x4 grid.
    uint bx = sub_idx & 3;
    uint by = (sub_idx >> 2) & 3;
    uint bz = (sub_idx >> 4) & 3;

    // Map thread ID to local coordinates within the 8x8x8 sub-block.
    // 256 threads = 8*8*8 / 2 pairs. Each thread copies one u32
    // (two u16 voxels).
    //
    // Thread layout: lz = tid/32, ly = (tid/4)%8, lx_pair = tid%4.
    // Each lx_pair covers voxel columns lx_pair*2 and lx_pair*2+1.
    uint lz      = tid / 32;
    uint ly      = (tid / 4) % 8;
    uint lx_pair = tid % 4;

    // Global voxel coordinates in the full 32x32x32 chunk.
    uint gx = bx * 8 + lx_pair * 2;
    uint gy = by * 8 + ly;
    uint gz = bz * 8 + lz;

    // Source: staging buffer. Layout is a flat 32x32x32 array of u16,
    // stride: z*1024 + y*32 + x, in bytes: linear * 2.
    uint src_linear = gz * 1024 + gy * 32 + gx;
    uint src_byte   = push.staging_index * 65536 + src_linear * 2;

    // Destination: packed sub-block in material_buf. Within the
    // destination sub-block, layout is lz*64 + ly*8 + lx (in u16
    // units), or lz*128 + ly*16 + lx_pair*4 (in bytes).
    uint dst_local = lz * 128 + ly * 16 + lx_pair * 4;
    uint dst_byte  = dst_base + sub_rank * SUB_BLOCK_BYTES + dst_local;

    uint value = material_staging.Load(src_byte);
    material_buf.Store(dst_byte, value);
}
