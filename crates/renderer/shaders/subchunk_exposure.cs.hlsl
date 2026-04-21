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
// One workgroup per `PrepRequest`. Workgroup size `[1, 1, 1]` — each
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
//   1: g_requests          (StructuredBuffer<PrepRequest>, read-only)
//   2: g_material_pool     (StructuredBuffer<Occupancy>,    read-only — self + neighbour)
//   3: g_dirty_list        (RWByteAddressBuffer, exposure-only-dirty-list)
//   4: g_directory         (StructuredBuffer<DirEntry>,     read-only)

#include "include/directory.hlsl"
#include "include/occupancy.hlsl"
#include "include/request.hlsl"
#include "include/dirty_list.hlsl"

// Face-mask extractors depend on `Occupancy` being in scope; include
// after occupancy.hlsl.
#include "include/face_mask.hlsl"

[[vk::binding(1, 0)]] StructuredBuffer<PrepRequest> g_requests;
[[vk::binding(2, 0)]] StructuredBuffer<Occupancy>   g_material_pool;
[[vk::binding(3, 0)]] RWByteAddressBuffer           g_dirty_list;
[[vk::binding(4, 0)]] StructuredBuffer<DirEntry>    g_directory;

#include "include/exposure.hlsl"

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID) {
    PrepRequest req = g_requests[gid.x];

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
    uint is_solid_bit = occupancy_is_fully_solid(me) ? 1u : 0u;

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
    //
    // Bounds-check handled inside `dirty_list_append`. See
    // `subchunk_prep.cs.hlsl` for the overflow rationale — same invariant:
    // writing past MAX_CANDIDATES would overrun the DirtyReport buffer.
    dirty_list_append(
        g_dirty_list, self_slot, exposure, is_solid_bit,
        EXPOSURE_STAGING_REQUEST_IDX_SENTINEL
    );
}
