// dirty_list.hlsl — shared dirty-list append helper.
//
// Dirty-list buffer layout (RWByteAddressBuffer):
//
//   offset  0: uint count       (monotonic InterlockedAdd counter)
//   offset  4: uint _pad0
//   offset  8: uint overflow    (set to 1 if count reaches DIRTY_LIST_CAPACITY)
//   offset 12: uint _pad2
//   offset 16 + i * 16: DirtyEntry i for i in [0, min(count, DIRTY_LIST_CAPACITY))
//     +0:  directory_index
//     +4:  new_bits_partial  ([0..5] exposure, [6] is_solid, [7] resident)
//     +8:  staging_request_idx
//     +12: _pad
//
// Mirror of Rust `renderer::DirtyEntry` (16 bytes; see `subchunk.rs`).
//
// `EXPOSURE_STAGING_REQUEST_IDX_SENTINEL` (0xFFFFFFFFu) in the
// `staging_request_idx` field signals to CPU retirement that this entry
// has no staging payload to copy (exposure-only refresh path).

#ifndef RENDERER_DIRTY_LIST_HLSL
#define RENDERER_DIRTY_LIST_HLSL

static const uint DIRTY_LIST_CAPACITY                    = 256u;
static const uint EXPOSURE_STAGING_REQUEST_IDX_SENTINEL  = 0xFFFFFFFFu;

// Append a DirtyEntry (16 bytes) to the dirty-list buffer, or raise the
// overflow flag if capacity is reached. See subchunk_prep.cs.hlsl header
// for the full layout (count at 0, overflow at 8, entries from 16).
void dirty_list_append(
    RWByteAddressBuffer buf,
    uint directory_index,
    uint exposure,
    uint is_solid_bit,
    uint staging_request_idx
) {
    uint idx;
    buf.InterlockedAdd(0u, 1u, idx);
    if (idx >= DIRTY_LIST_CAPACITY) {
        uint _prev;
        buf.InterlockedMax(8u, 1u, _prev);
        return;
    }
    uint entry_off = 16u + idx * 16u;
    uint new_bits_partial = exposure | (is_solid_bit << 6) | (1u << 7);
    buf.Store(entry_off +  0u, directory_index);
    buf.Store(entry_off +  4u, new_bits_partial);
    buf.Store(entry_off +  8u, staging_request_idx);
    buf.Store(entry_off + 12u, 0u);
}

#endif // RENDERER_DIRTY_LIST_HLSL
