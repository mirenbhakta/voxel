// Build stage alloc pass: GPU-authoritative allocation for quads and materials.
//
// Dispatch: (1, 1, 1) -- single thread.
//
// Two-phase structure:
//   Phase 1: Scan all slots for dead allocations. A slot whose
//            chunk_meta quad_count is zero but chunk_alloc_buf holds
//            non-zero ranges is dead. The scan frees those ranges to
//            the GPU-owned free lists and zeros the page table entry.
//            No CPU upload required -- the GPU detects stale state
//            autonomously.
//   Phase 2: Allocate for builds. Frees old ranges (rebuilds), allocates
//            new ranges via free list + bump, writes the page table.
//
// The CPU never writes to the free lists. They are GPU-internal state
// initialized to count=0 and managed entirely here.

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
RWByteAddressBuffer chunk_alloc_buf         : register(u9, space0);

// Push constants.
struct AllocPush {
    uint batch_size;
    uint quad_capacity;
    uint material_capacity;
    uint material_segment_units;
};

[[vk::push_constant]] AllocPush push;

/// Push a freed range onto a free list buffer.
///
/// Appends an (offset, size) entry at the end of the list. The caller
/// is responsible for ensuring the list does not overflow (sized to
/// 2 * MAX_CHUNKS entries, which handles worst-case accumulation).
void push_to_free_list(RWByteAddressBuffer buf, uint offset, uint size) {
    uint fc = buf.Load(0);
    buf.Store(4 + fc * 8,     offset);
    buf.Store(4 + fc * 8 + 4, size);
    buf.Store(0, fc + 1);
}

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

    // -- Phase 1: Scan for dead allocations --
    //
    // A slot is dead when the CPU has zeroed its chunk_meta quad_count
    // (on removal) but chunk_alloc_buf still holds a non-zero allocation.
    // Scan all slots, free stale ranges, and zero the page table entry.

    for (uint s = 0; s < MAX_CHUNKS; s++) {
        uint meta_qc = chunk_meta_buf.Load(s * 16);

        if (meta_qc != 0) {
            continue;
        }

        uint old_qoff = chunk_alloc_buf.Load(s * CHUNK_ALLOC_BYTES);
        uint old_qcnt = chunk_alloc_buf.Load(s * CHUNK_ALLOC_BYTES + 4);
        uint old_moff = chunk_alloc_buf.Load(s * CHUNK_ALLOC_BYTES + 8);
        uint old_mcnt = chunk_alloc_buf.Load(s * CHUNK_ALLOC_BYTES + 12);

        if (old_qcnt == 0 && old_mcnt == 0) {
            continue;
        }

        if (old_qcnt > 0) {
            push_to_free_list(quad_free_list_buf, old_qoff, old_qcnt);
        }

        if (old_mcnt > 0) {
            push_to_free_list(material_free_list_buf, old_moff, old_mcnt);
        }

        // Zero the page table entry.
        chunk_alloc_buf.Store4(s * CHUNK_ALLOC_BYTES, uint4(0, 0, 0, 0));
    }

    // -- Phase 2: Allocate for builds --
    //
    // Process the build batch. Free old allocations from the page table
    // (handles rebuilds), then allocate fresh ranges.

    for (uint i = 0; i < push.batch_size; i++) {
        uint slot       = build_batch_buf.Load(i * 4);
        uint quad_count = chunk_meta_buf.Load(slot * 16);

        // Default: no material work for this batch entry.
        material_dispatch_buf.Store(i * 12 + 0, 0u);
        material_dispatch_buf.Store(i * 12 + 4, 1u);
        material_dispatch_buf.Store(i * 12 + 8, 1u);

        // Free old allocation from the page table (handles rebuilds).
        uint old_qoff = chunk_alloc_buf.Load(slot * CHUNK_ALLOC_BYTES);
        uint old_qcnt = chunk_alloc_buf.Load(slot * CHUNK_ALLOC_BYTES + 4);
        uint old_moff = chunk_alloc_buf.Load(slot * CHUNK_ALLOC_BYTES + 8);
        uint old_mcnt = chunk_alloc_buf.Load(slot * CHUNK_ALLOC_BYTES + 12);

        if (old_qcnt > 0) {
            push_to_free_list(quad_free_list_buf, old_qoff, old_qcnt);
        }

        if (old_mcnt > 0) {
            push_to_free_list(material_free_list_buf, old_moff, old_mcnt);
        }

        if (quad_count == 0) {
            // Zero the page table entry and skip allocation.
            chunk_alloc_buf.Store4(
                slot * CHUNK_ALLOC_BYTES, uint4(0, 0, 0, 0)
            );
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

                // Zero the page table -- old ranges already freed above.
                chunk_alloc_buf.Store4(
                    slot * CHUNK_ALLOC_BYTES, uint4(0, 0, 0, 0)
                );
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

        for (uint dir = 0; dir < 6; dir++) {
            uint pfx_base = range_base + QUAD_RANGE_COUNTS_OFFSET
                          + dir * 132;
            uint running = 0;

            for (uint l = 0; l < 32; l++) {
                uint cnt = quad_range_buf.Load(pfx_base + l * 4);
                quad_range_buf.Store(pfx_base + l * 4, running);
                running += cnt;
            }

            // pfx[32] = total quads for this direction.
            quad_range_buf.Store(pfx_base + 32 * 4, running);

            // dir_prefix[d] = total quads for direction d.
            quad_range_buf.Store(
                range_base + 8 + dir * 4, running
            );
        }

        // -- Material allocation --

        uint sub_mask_lo = chunk_meta_buf.Load(slot * 16 + 8);
        uint sub_mask_hi = chunk_meta_buf.Load(slot * 16 + 12);
        uint popcount    = countbits(sub_mask_lo) + countbits(sub_mask_hi);

        uint mat_offset_flat = 0;

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
                    push_to_free_list(
                        quad_free_list_buf, base_offset, quad_count
                    );

                    // Zero the page table -- old ranges already freed.
                    chunk_alloc_buf.Store4(
                        slot * CHUNK_ALLOC_BYTES, uint4(0, 0, 0, 0)
                    );
                    continue;
                }

                mat_offset = mat_bump;
                mat_bump  += popcount;
            }

            mat_offset_flat = mat_offset;

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

        // Write page table entry.
        chunk_alloc_buf.Store(slot * CHUNK_ALLOC_BYTES,      base_offset);
        chunk_alloc_buf.Store(slot * CHUNK_ALLOC_BYTES + 4,  quad_count);
        chunk_alloc_buf.Store(slot * CHUNK_ALLOC_BYTES + 8,  mat_offset_flat);
        chunk_alloc_buf.Store(slot * CHUNK_ALLOC_BYTES + 12, popcount);
    }

    bump_state_buf.Store(0, quad_bump);
    material_bump_state_buf.Store(0, mat_bump);
}
