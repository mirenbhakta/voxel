// Build stage pass 2: write packed quad descriptors.
//
// Dispatch: (32, 6, 1) -- 32 layers x 6 directions.
// Workgroup: 32 threads, one per row in the 32x32 face layer.
//
// Re-derives faces and merge (same logic as count pass via shared
// includes), then writes packed quads to the contiguous range at
// prefix-summed positions. Direction-ordered, layer-sorted.

#define MERGE_MODE_WRITE
#include "include/bindings.hlsl"

// Bindings (build write pass).
ByteAddressBuffer   occupancy_buf     : register(t0, space0);
ByteAddressBuffer   boundary_cache_buf: register(t1, space0);
ByteAddressBuffer   quad_range_buf    : register(t2, space0);
RWByteAddressBuffer quad_buf          : register(u3, space0);

// Push constants.
struct BuildPush {
    uint slot_index;
    uint base_offset;  // starting quad index in the quad buffer
};

[[vk::push_constant]] BuildPush push;

// Shared memory for face derivation and greedy merge.
// Must be declared before including face.hlsl/merge.hlsl.
groupshared uint g_shared_face[32];

#include "include/face.hlsl"
#include "include/merge.hlsl"

[numthreads(32, 1, 1)]
void main(uint3 lid : SV_GroupThreadID,
          uint3 gid : SV_GroupID) {
    uint row   = lid.x;   // 0..31, this thread's row
    uint layer = gid.x;   // 0..31, layer along the normal axis
    uint dir   = gid.y;   // 0..5, direction

    uint occ_base   = push.slot_index * OCC_WORDS;
    uint bound_base = push.slot_index * 192;

    // Face derivation (identical to count pass).
    uint face_word = derive_face(
        occupancy_buf, occ_base,
        boundary_cache_buf, bound_base,
        dir, layer, row
    );

    // Store to shared for greedy merge.
    g_shared_face[row] = face_word;
    GroupMemoryBarrierWithGroupSync();

    // Thread 0 runs the merge and writes quads.
    if (row == 0) {
        // Compute the write offset for this (direction, layer) bucket.
        // The CPU has prefix-summed dir_layer_counts and stored the
        // result in base_offset. We need the offset within the chunk's
        // range for this specific (dir, layer).
        //
        // QuadRange layout: buffer_index(4) + base_offset(4) +
        //   dir_layer_counts[6][32].
        // We sum counts for all prior (dir, layer) pairs.
        uint range_base = push.slot_index * 776 + 8;
        uint prefix = 0;

        // Sum all directions before this one.
        for (uint d = 0; d < dir; d++) {
            for (uint l = 0; l < 32; l++) {
                prefix += quad_range_buf.Load(range_base + d * 128 + l * 4);
            }
        }

        // Sum all layers before this one within the current direction.
        for (uint l = 0; l < layer; l++) {
            prefix += quad_range_buf.Load(range_base + dir * 128 + l * 4);
        }

        uint write_offset = push.base_offset + prefix;

        greedy_merge(layer, dir, quad_buf, write_offset);
    }
}
