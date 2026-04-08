// Build stage alloc pass: GPU-side bump allocation for quads.
//
// Dispatch: (1, 1, 1) -- single thread.
// Reads quad_count per chunk from chunk_meta_buf (written by the
// preceding count pass), advances a persistent bump pointer, and
// writes base_offset to quad_range_buf for the following write pass.
//
// The bump pointer lives in bump_state_buf and persists across
// frames. The CPU syncs it before each dispatch_build call.

#include "include/bindings.hlsl"

// Bindings (build alloc pass).
RWByteAddressBuffer bump_state_buf  : register(u0, space0);
ByteAddressBuffer   build_batch_buf : register(t1, space0);
RWByteAddressBuffer chunk_meta_buf  : register(u2, space0);
RWByteAddressBuffer quad_range_buf  : register(u3, space0);

// Push constants.
struct AllocPush {
    uint batch_size;
    uint capacity;
};

[[vk::push_constant]] AllocPush push;

[numthreads(1, 1, 1)]
void main() {
    uint bump = bump_state_buf.Load(0);

    for (uint i = 0; i < push.batch_size; i++) {
        uint slot       = build_batch_buf.Load(i * 4);
        uint quad_count = chunk_meta_buf.Load(slot * 16);

        if (quad_count == 0) {
            continue;
        }

        // Signal overflow so the CPU can grow the buffer and retry.
        if (bump + quad_count > push.capacity) {
            chunk_meta_buf.Store(slot * 16 + 4, 1u);
            continue;
        }

        // Write base_offset into quad_range_buf for the write pass.
        quad_range_buf.Store(slot * 776 + 4, bump);
        bump += quad_count;
    }

    bump_state_buf.Store(0, bump);
}
