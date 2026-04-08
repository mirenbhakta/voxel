// Build stage alloc pass: GPU-side bump allocation for quads.
//
// Dispatch: (1, 1, 1) -- single thread.
// Reads quad_count per chunk from chunk_meta_buf (written by the
// preceding count pass), advances a persistent bump pointer, and
// writes base_offset to quad_range_buf for the following write pass.
//
// The bump pointer lives in bump_state_buf and persists across
// frames. The CPU seeds it before each dispatch_build call.
//
// A free list of (offset, size) pairs enables reuse of freed quad
// ranges. The CPU pushes freed ranges before dispatch; the alloc
// pass scans first-fit before falling back to the bump pointer.

#include "include/bindings.hlsl"

// Bindings (build alloc pass).
RWByteAddressBuffer bump_state_buf          : register(u0, space0);
ByteAddressBuffer   build_batch_buf         : register(t1, space0);
RWByteAddressBuffer chunk_meta_buf          : register(u2, space0);
RWByteAddressBuffer quad_range_buf          : register(u3, space0);
RWByteAddressBuffer quad_free_list_buf      : register(u4, space0);
RWByteAddressBuffer material_range_buf      : register(u5, space0);
RWByteAddressBuffer material_bump_state_buf : register(u6, space0);
RWByteAddressBuffer material_free_list_buf  : register(u7, space0);
RWByteAddressBuffer material_dispatch_buf   : register(u8, space0);

// Push constants.
struct AllocPush {
    uint batch_size;
    uint quad_capacity;
    uint material_capacity;
    uint material_segment_units;
};

[[vk::push_constant]] AllocPush push;

/// Try to allocate `count` units from a free list buffer (first-fit).
///
/// Returns the offset on success, or 0xFFFFFFFF if no fit found.
/// On success, the matched entry is shrunk or swapped with the last
/// entry and the count is decremented.
///
/// Free list layout: [count(4), entries...] where each entry is
/// (offset: u32, size: u32) = 8 bytes.
uint alloc_from_free_list(RWByteAddressBuffer buf, uint count) {
    uint free_count = buf.Load(0);

    for (uint f = 0; f < free_count; f++) {
        uint fo = buf.Load(4 + f * 8);
        uint fs = buf.Load(4 + f * 8 + 4);

        if (fs >= count) {
            if (fs > count) {
                // Shrink: advance offset, reduce size.
                buf.Store(4 + f * 8,     fo + count);
                buf.Store(4 + f * 8 + 4, fs - count);
            }
            else {
                // Exact fit: swap with last entry, decrement count.
                uint last = free_count - 1;

                if (f != last) {
                    uint lo = buf.Load(4 + last * 8);
                    uint ls = buf.Load(4 + last * 8 + 4);
                    buf.Store(4 + f * 8,     lo);
                    buf.Store(4 + f * 8 + 4, ls);
                }

                buf.Store(0, last);
            }

            return fo;
        }
    }

    return 0xFFFFFFFF;
}

[numthreads(1, 1, 1)]
void main() {
    uint quad_bump = bump_state_buf.Load(0);
    uint mat_bump  = material_bump_state_buf.Load(0);

    uint seg_units = push.material_segment_units;
    uint seg_mask  = seg_units - 1;

    for (uint i = 0; i < push.batch_size; i++) {
        uint slot       = build_batch_buf.Load(i * 4);
        uint quad_count = chunk_meta_buf.Load(slot * 16);

        // Default: no material work for this batch entry.
        material_dispatch_buf.Store(i * 12 + 0, 0u);
        material_dispatch_buf.Store(i * 12 + 4, 1u);
        material_dispatch_buf.Store(i * 12 + 8, 1u);

        if (quad_count == 0) {
            continue;
        }

        // -- Quad allocation --

        // Try the free list first.
        uint base_offset = alloc_from_free_list(
            quad_free_list_buf, quad_count
        );

        if (base_offset == 0xFFFFFFFF) {
            // Fall back to bump allocation.
            if (quad_bump + quad_count > push.quad_capacity) {
                // Signal quad overflow (bit 0) and zero quad_count so
                // the cull shader skips this chunk (no isolation writes
                // to restore old values).
                chunk_meta_buf.Store(slot * 16,     0u);
                chunk_meta_buf.Store(slot * 16 + 4, 1u);
                continue;
            }

            base_offset = quad_bump;
            quad_bump  += quad_count;
        }

        quad_range_buf.Store(slot * QUAD_RANGE_BYTES + 4, base_offset);

        // Compute prefix sums from the raw dir_layer_counts the count
        // pass wrote into dir_layer_pfx[d][0..32]. Convert in-place to
        // exclusive prefix sums and write dir_prefix[d].
        uint range_base = slot * QUAD_RANGE_BYTES;

        for (uint d = 0; d < 6; d++) {
            uint pfx_base = range_base + QUAD_RANGE_COUNTS_OFFSET
                          + d * 132;
            uint running = 0;

            for (uint l = 0; l < 32; l++) {
                uint count = quad_range_buf.Load(pfx_base + l * 4);
                // Overwrite [l] with the exclusive prefix sum, then
                // shift everything up by 1 position. We do this by
                // writing pfx[l+1] = running + count and pfx[0] = 0
                // after the loop. But to avoid a second pass, write
                // pfx[l] = running (exclusive) and accumulate.
                quad_range_buf.Store(pfx_base + l * 4, running);
                running += count;
            }

            // pfx[32] = total quads for this direction.
            quad_range_buf.Store(pfx_base + 32 * 4, running);

            // dir_prefix[d] = total quads for direction d.
            quad_range_buf.Store(
                range_base + 8 + d * 4, running
            );
        }

        // -- Material allocation --

        uint sub_mask_lo = chunk_meta_buf.Load(slot * 16 + 8);
        uint sub_mask_hi = chunk_meta_buf.Load(slot * 16 + 12);
        uint popcount    = countbits(sub_mask_lo) + countbits(sub_mask_hi);

        if (popcount > 0) {
            // Try material free list first.
            uint mat_offset = alloc_from_free_list(
                material_free_list_buf, popcount
            );

            if (mat_offset == 0xFFFFFFFF) {
                // Bump allocation with segment boundary enforcement.
                uint local = mat_bump & seg_mask;

                if (local + popcount > seg_units) {
                    mat_bump = (mat_bump + seg_mask) & ~seg_mask;
                }

                if (mat_bump + popcount > push.material_capacity) {
                    // Signal material overflow (bit 1) and zero
                    // quad_count so the cull shader skips this chunk.
                    chunk_meta_buf.Store(slot * 16,     0u);
                    chunk_meta_buf.Store(slot * 16 + 4, 2u);

                    // Return the quad allocation to the free list --
                    // it was already consumed but the chunk will be
                    // retried, so the range is dead.
                    uint fc = quad_free_list_buf.Load(0);
                    quad_free_list_buf.Store(4 + fc * 8,     base_offset);
                    quad_free_list_buf.Store(4 + fc * 8 + 4, quad_count);
                    quad_free_list_buf.Store(0, fc + 1);
                    continue;
                }

                mat_offset = mat_bump;
                mat_bump  += popcount;
            }

            uint buf_idx  = mat_offset / seg_units;
            uint base_off = (mat_offset & seg_mask) * SUB_BLOCK_BYTES;

            material_range_buf.Store(slot * 16 + 0,  buf_idx);
            material_range_buf.Store(slot * 16 + 4,  base_off);
            material_range_buf.Store(slot * 16 + 8,  sub_mask_lo);
            material_range_buf.Store(slot * 16 + 12, sub_mask_hi);

            // Write indirect dispatch args for material pack.
            material_dispatch_buf.Store(i * 12 + 0, popcount);
            material_dispatch_buf.Store(i * 12 + 4, 1u);
            material_dispatch_buf.Store(i * 12 + 8, 1u);
        }
    }

    bump_state_buf.Store(0, quad_bump);
    material_bump_state_buf.Store(0, mat_bump);
}
