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
    GroupMemoryBarrierWithGroupSync();

    // Initialize the layer count accumulator.
    if (row == 0) {
        layer_quad_count = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Greedy merge (thread 0 only): count quads in this layer.
    if (row == 0) {
        layer_quad_count = greedy_merge(layer, dir);
    }
    GroupMemoryBarrierWithGroupSync();

    // Thread 0 writes the layer count to quad_range_buf.
    if (row == 0) {
        // QuadRange layout: buffer_index(4) + base_offset(4) +
        // dir_layer_counts[6][32] (768 bytes).
        // Offset to dir_layer_counts[dir][layer]:
        uint range_offset = push.slot_index * 776 + 8
                          + dir * 128 + layer * 4;

        quad_range_buf.Store(range_offset, layer_quad_count);

        // Atomically accumulate the total quad count in chunk_meta.
        // ChunkMeta layout: quad_count(4) + flags(4) + reserved(8).
        uint meta_offset = push.slot_index * 16;
        uint dummy;
        chunk_meta_buf.InterlockedAdd(meta_offset, layer_quad_count, dummy);
    }
}
