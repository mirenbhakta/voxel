// subchunk_exposure.cs.hlsl — exposure-only refresh for neighbour-driven
// sub-chunks.
//
// Companion dispatch to `subchunk_prep.cs.hlsl`. Full-prep voxelizes
// terrain, computes exposure with neighbour reads, and writes staging so
// the CPU patch step can copy the new occupancy into `material_pool`.
// Exposure-only refresh does *not* re-voxelize: it reads the sub-chunk's
// existing occupancy out of `g_material_pool[self.material_slot]`,
// recomputes the neighbour-aware 6-bit exposure mask, and emits a dirty
// entry. No staging write, no patch copy — only the directory `bits`
// field changes on retirement.
//
// Why this dispatch exists:
//   * Full-prep at frame F dispatches all newly-resident coords together.
//     Intra-batch mutually-adjacent sub-chunks both voxelize against each
//     other's *old* live content, so they settle with conservative
//     exposure bits for one patch cycle. See
//     `decision-world-streaming-architecture` §Intra-batch neighbour
//     conservatism.
//   * Any sub-chunk whose *neighbour* changes without *itself* changing
//     has out-of-date exposure bits after that neighbour's patch lands.
//     Full-prep would repeat the 512-density-eval voxelization for no
//     payoff; exposure-only needs zero density evals.
//
// # Dispatch shape
//
// One workgroup per `ExposureRequest`. Workgroup size `[1, 1, 1]` — each
// request does O(1) reads (self DirEntry + self Occupancy + 6 neighbour
// DirEntry + up-to-6 neighbour Occupancy face extracts) and a single
// dirty-list append. The 64-thread shape of `subchunk_prep` (designed for
// 512 parallel density evaluations) does not apply here.
//
// # Dirty-list layout
//
// Identical to the full-prep dirty list (16 B per entry; see
// `renderer::DirtyEntry`). The `staging_request_idx` field is set to
// `EXPOSURE_STAGING_REQUEST_IDX_SENTINEL` (= `0xFFFFFFFFu`) — a signal to
// the retirement logic that this entry has no staging payload to copy.
// Exposure-only entries flow through a *separate* dirty-list buffer +
// readback channel; retirement handles them on a parallel path that
// updates the directory `bits` field in-place.
//
// # Binding layout
//
// Set 0 — slot 0 is the implicit `g_consts` (the shader calls
// `resolve_coord_to_slot` which reads per-level `pool_dims` /
// `global_offset` from it). Explicit bindings follow from slot 1:
//
//   0: g_consts            (ConstantBuffer<GpuConsts>,      read-only)    — implicit
//   1: g_requests          (StructuredBuffer<ExposureRequest>, read-only)
//   2: g_material_pool     (StructuredBuffer<Occupancy>,    read-only — self + neighbour)
//   3: g_dirty_list        (RWByteAddressBuffer, exposure-only-dirty-list)
//   4: g_directory         (StructuredBuffer<DirEntry>,     read-only)

#include "include/directory.hlsl"

// Mirrors `PrepRequest` in Rust — same 32-byte shape so the CPU can reuse
// its `PrepRequest` type for exposure-only request construction without
// introducing a new Pod struct. The shader-visible fields are identical
// (coord + level); `_pad0..3` round out the stride.
struct ExposureRequest {
    int3 coord;
    uint level;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};

// Matches the 16-u32 packed layout of `SubchunkOccupancy::to_gpu_bytes`
// on the Rust side — identical to the prep shader's `Occupancy`.
struct Occupancy {
    uint4 plane[4];
};

#include "include/face_mask.hlsl"

[[vk::binding(1, 0)]] StructuredBuffer<ExposureRequest> g_requests;
[[vk::binding(2, 0)]] StructuredBuffer<Occupancy>       g_material_pool;
[[vk::binding(3, 0)]] RWByteAddressBuffer               g_dirty_list;
[[vk::binding(4, 0)]] StructuredBuffer<DirEntry>        g_directory;

#include "include/exposure.hlsl"

// Matches the Rust-side sentinel in `renderer::DirtyEntry` — a dirty entry
// carrying this value in `staging_request_idx` has no staging payload.
static const uint EXPOSURE_STAGING_REQUEST_IDX_SENTINEL = 0xFFFFFFFFu;

// Mirror of `renderer::subchunk::MAX_CANDIDATES` — ceiling on dirty-list
// entries per dispatch. See `subchunk_prep.cs.hlsl` for the overflow
// semantics; the Step-7 shadow ledger surfaces the flag as
// `dirty_exposure_overflow` in `WorldRendererStats`.
static const uint DIRTY_LIST_CAPACITY = 256u;

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID) {
    ExposureRequest req = g_requests[gid.x];

    // Self-resolve directory index from coord + level, same mapping as
    // prep's `resolve_coord_to_slot` (see `include/directory.hlsl`).
    uint self_slot = resolve_coord_to_slot(req.coord, req.level);
    DirEntry self  = g_directory[self_slot];

    // Defensive guards — the CPU filter drops coords whose directory
    // entry is non-resident or has no material slot, so these branches
    // should never fire under correct wiring. If they do (torus-collision
    // drift, request dedup miss), bail silently rather than dereference a
    // stale or zero material slot.
    if (!direntry_is_resident(self)) {
        return;
    }
    if (!direntry_has_material(self)) {
        return;
    }
    if (!direntry_coord_matches(self, req.coord)) {
        // Torus-collision: some other coord currently owns this slot. We
        // were filed as a refresh for `req.coord` but the slot doesn't
        // belong to us anymore. Drop silently — when our coord re-enters
        // residency, full-prep will re-seed it.
        return;
    }

    uint mslot = direntry_get_material_slot(self);
    Occupancy me = g_material_pool[mslot];

    // Recompute exposure from the live occupancy + current neighbours.
    // Shared helper with subchunk_prep so the two dispatches produce
    // bit-identical mask values for the same inputs.
    uint exposure = compute_exposure_mask(me, req.coord, req.level);

    // Recompute is_solid from the live occupancy. Same AND-fold as
    // subchunk_prep: all 16 words must be 0xFFFFFFFF.
    uint and_accum = me.plane[0].x & me.plane[0].y & me.plane[0].z & me.plane[0].w
                   & me.plane[1].x & me.plane[1].y & me.plane[1].z & me.plane[1].w
                   & me.plane[2].x & me.plane[2].y & me.plane[2].z & me.plane[2].w
                   & me.plane[3].x & me.plane[3].y & me.plane[3].z & me.plane[3].w;
    uint is_solid_bit = (and_accum == 0xFFFFFFFFu) ? 1u : 0u;

    // Append the dirty entry. `new_bits_partial` mirrors subchunk_prep's
    // format so retirement can share the decoding logic — the only
    // distinguishing field is `staging_request_idx`, which carries the
    // sentinel here.
    //
    // Note: we intentionally do NOT gate on "exposure/is_solid differ
    // from self.bits". The dirty list is already short (only neighbours
    // of recently-retired sub-chunks are scheduled) and an unchanged-bits
    // retirement is a no-op directory write. Filtering would add
    // per-request branches without measurable savings on the current
    // workload.
    uint idx;
    g_dirty_list.InterlockedAdd(0u, 1u, idx);
    // Bounds-check. See `subchunk_prep.cs.hlsl` for the overflow rationale
    // — same invariant here: writing at `idx >= MAX_CANDIDATES` would
    // overrun the DirtyReport buffer.
    if (idx >= DIRTY_LIST_CAPACITY) {
        uint _prev;
        g_dirty_list.InterlockedMax(8u, 1u, _prev);
    } else {
        uint entry_off = 16u + idx * 16u;
        uint new_bits_partial = exposure
                              | (is_solid_bit << 6)
                              | (1u << 7);                 // resident
        g_dirty_list.Store(entry_off +  0u, self_slot);    // directory_index
        g_dirty_list.Store(entry_off +  4u, new_bits_partial);
        g_dirty_list.Store(entry_off +  8u, EXPOSURE_STAGING_REQUEST_IDX_SENTINEL);
        g_dirty_list.Store(entry_off + 12u, 0u);
    }
}
