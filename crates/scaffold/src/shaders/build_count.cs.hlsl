// Build stage pass 1: count quads per (direction, layer).
//
// Dispatch: (32, 6, 1) -- 32 layers x 6 directions.
// Workgroup: 32 threads, one per row in the 32x32 face layer.
//
// Reads chunk occupancy and neighbor boundaries from shared slot-indexed
// buffers. Runs face derivation and greedy merge, but only counts quads.
// Writes dir_layer_counts to quad_range_buf and total to chunk_meta_buf.

#define MERGE_MODE_COUNT
#include "include/bindings.hlsl"

// Bindings (build count pass).
ByteAddressBuffer  occupancy_buf     : register(t0, space0);
ByteAddressBuffer  boundary_cache_buf: register(t1, space0);
RWByteAddressBuffer chunk_meta_buf   : register(u2, space0);
RWByteAddressBuffer quad_range_buf   : register(u3, space0);

// Push constants.
struct BuildPush {
    uint slot_index;
    uint base_offset;  // unused in count pass, but same layout as write
};

[[vk::push_constant]] BuildPush push;

// Shared memory for face derivation and greedy merge.
// Must be declared before including face.hlsl/merge.hlsl which
// reference g_shared_face by name.
groupshared uint g_shared_face[32];

#include "include/face.hlsl"
#include "include/merge.hlsl"

// Per-direction layer count accumulator (one per workgroup).
groupshared uint layer_quad_count;

// Per-workgroup sub-block visibility mask accumulator.
// Two u32s encoding a 64-bit mask over a 4x4x4 grid of 8x8x8
// sub-blocks. Set bits indicate sub-blocks containing visible faces.
groupshared uint g_sub_mask[2];

[numthreads(32, 1, 1)]
void main(uint3 lid : SV_GroupThreadID,
          uint3 gid : SV_GroupID) {
    uint row   = lid.x;   // 0..31, this thread's row
    uint layer = gid.x;   // 0..31, layer along the normal axis
    uint dir   = gid.y;   // 0..5, direction

    uint occ_base   = push.slot_index * OCC_WORDS;
    uint bound_base = push.slot_index * 192;

    // Face derivation: produces one face word per thread.
    uint face_word = derive_face(
        occupancy_buf, occ_base,
        boundary_cache_buf, bound_base,
        dir, layer, row
    );

    // Store to shared for greedy merge.
    g_shared_face[row] = face_word;

    // Initialize accumulators.
    if (row == 0) {
        layer_quad_count = 0;
        g_sub_mask[0]    = 0;
        g_sub_mask[1]    = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Mark sub-blocks that contain visible faces. Each thread checks
    // its face word and marks the sub-blocks at (col/8, row/8, layer/8)
    // in chunk-local coordinates.
    if (face_word != 0) {
        uint sub_row   = row / 8;
        uint sub_layer = layer / 8;

        // Which 8-column sub-ranges have at least one face?
        uint col_mask = 0;
        if (face_word & 0x000000FFu) col_mask |= 1u;
        if (face_word & 0x0000FF00u) col_mask |= 2u;
        if (face_word & 0x00FF0000u) col_mask |= 4u;
        if (face_word & 0xFF000000u) col_mask |= 8u;

        for (uint sc = 0; sc < 4; sc++) {
            if (!(col_mask & (1u << sc))) continue;

            // Map canonical (col, row, layer) to chunk (x, y, z).
            uint bx, by, bz;
            if (dir < 2) {
                // X faces: layer=x, col=y, row=z
                bx = sub_layer; by = sc; bz = sub_row;
            }
            else if (dir < 4) {
                // Y faces: layer=y, col=z, row=x
                bx = sub_row; by = sub_layer; bz = sc;
            }
            else {
                // Z faces: layer=z, col=x, row=y
                bx = sc; by = sub_row; bz = sub_layer;
            }

            uint sub_idx = bz * 16 + by * 4 + bx;
            uint word    = sub_idx < 32 ? 0 : 1;
            uint bit     = sub_idx < 32 ? sub_idx : sub_idx - 32;
            uint dummy;
            InterlockedOr(g_sub_mask[word], 1u << bit, dummy);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Greedy merge (thread 0 only): count quads in this layer.
    if (row == 0) {
        layer_quad_count = greedy_merge(layer, dir);
    }
    GroupMemoryBarrierWithGroupSync();

    // Thread 0 writes the layer count and sub-mask contributions.
    if (row == 0) {
        // QuadRange layout: buffer_index(4) + base_offset(4) +
        // dir_layer_counts[6][32] (768 bytes).
        // Offset to dir_layer_counts[dir][layer]:
        uint range_offset = push.slot_index * 776 + 8
                          + dir * 128 + layer * 4;

        quad_range_buf.Store(range_offset, layer_quad_count);

        // Atomically accumulate the total quad count in chunk_meta.
        // ChunkMeta layout: quad_count(4) + flags(4) + sub_mask(8).
        uint meta_offset = push.slot_index * 16;
        uint dummy;
        chunk_meta_buf.InterlockedAdd(meta_offset, layer_quad_count, dummy);

        // Merge workgroup sub_mask into per-slot mask.
        if (g_sub_mask[0])
            chunk_meta_buf.InterlockedOr(meta_offset + 8, g_sub_mask[0], dummy);

        if (g_sub_mask[1])
            chunk_meta_buf.InterlockedOr(meta_offset + 12, g_sub_mask[1], dummy);
    }
}
