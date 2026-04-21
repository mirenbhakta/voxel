// subchunk_prep.cs.hlsl — GPU-side sub-chunk occupancy synthesis with
// diff-vs-pool dirty-list emission and neighbour-aware exposure.
//
// HLSL port of `game::world_view::synthesize_occupancy` +
// `terrain_occupied`, extended so the CPU never re-synthesizes content on
// cache hits: after the new occupancy is voxelized into a staging entry,
// the shader compares it against the previously-committed material-pool
// payload (if any) and, when any plane word differs — or when no live
// entry exists yet — appends an entry to a compact dirty list.
//
// The CPU reads the dirty list via a `ReadbackChannel` and, for every
// entry the retirement logic classifies as *sparse* (non-zero exposure),
// issues a `staging_occ → material_pool` copy. The uniform-empty case
// (exposure == 0) skips the copy entirely — the entry has no material
// pool storage.
//
// # Dispatch shape
//
// One workgroup per `PrepRequest`. Workgroup size `[4, 4, 4]` (= 64
// threads). Each thread handles a 2×2×2 cube of voxels (8 density
// evaluations per thread → 512 per sub-chunk, one per voxel).
//
// Occupancy is accumulated into `gs_planes[16]` using `InterlockedOr`.
// After the barrier, thread 0 writes the 16 words into
// `g_staging_occ[gid.x]`, resolves its own directory index from
// `req.coord`/`req.level` via `resolve_coord_to_slot`, diffs the staged
// occupancy against `g_material_pool[direntry_get_material_slot(...)]`
// (when the directory entry currently has a material slot), and — if the
// occupancy differs or no live entry exists — computes a neighbour-aware
// 6-bit exposure mask and an is_solid hint and appends a widened
// `DirtyEntry` to `g_dirty_list`.
//
// # Terrain field
//
// Voxel world-space centre:
//   voxel_size = 1 << level;
//   base       = coord * 8 * voxel_size;
//   wp         = base + (xyz + 0.5) * voxel_size;
// Voxel is solid iff `wp.y < h`, where `h` is a layered-sinusoid surface
// matching the CPU reference (`terrain_occupied` in `world_view.rs`).
//
// # Neighbour-aware exposure
//
// A direction bit is set iff any voxel on this sub-chunk's face at that
// direction is solid AND the corresponding neighbour voxel is empty. For
// each of the 6 directions we:
//   1. Compute `nbr_coord = req.coord + dir_offset`.
//   2. Resolve it via `resolve_and_verify` to a directory entry.
//   3. Classify the neighbour by its DirEntry bits:
//      - non-resident OR coord mismatch  → treat as empty (conservative,
//                                          per `decision-world-streaming-
//                                          architecture` intra-batch rule)
//      - resident + is_solid             → all-ones (neighbour face full)
//      - resident + !has_material        → empty (uniform-empty entry)
//      - resident + has_material         → read material_pool[mslot] and
//                                          extract the opposite face
//   4. Test `my_face & ~nbr_face != 0` via `face_exposed`.
//
// Intra-batch neighbours that were dispatched this frame still see OLD
// live (the patch hasn't landed yet); their conservative "empty"
// interpretation produces an over-conservative exposure bit for one
// cycle. Step 5 (exposure-only refresh) reconciles the mismatch.
//
// # Dirty-list layout (RWByteAddressBuffer)
//
//   offset  0: uint count       (monotonic InterlockedAdd counter)
//   offset  4: uint _pad0
//   offset  8: uint overflow    (set to 1 if count reaches MAX_CANDIDATES)
//   offset 12: uint _pad2
//   offset 16 + i * 16: DirtyEntry i for i in [0, min(count, MAX_CANDIDATES))
//     +0:  directory_index
//     +4:  new_bits_partial  ([0..5] exposure, [6] is_solid, [7] resident)
//     +8:  staging_request_idx  (= gid.x)
//     +12: _pad
//
// Mirror of Rust `renderer::DirtyEntry` (16 bytes; see `subchunk.rs`).
//
// The append is bounds-checked: `InterlockedAdd` returns the pre-add
// count, and if that is >= MAX_CANDIDATES the entry store is skipped and
// the `overflow` flag is raised via `InterlockedMax`. CPU reads the flag
// at retirement (Step-7 shadow ledger) and surfaces it in
// `WorldRendererStats`. Without this check the shader silently overruns
// the `DirtyReport` buffer — the Step-7 task found the path and
// installed the guard.
//
// # Binding layout
//
// Set 0 — contiguous slots 0..6. Slot 0 is implicitly `g_consts` (pinned
// via `include/gpu_consts.hlsl`); the explicit bindings follow from slot
// 1 onward. The CPU side threads the consts buffer in through
// `RenderGraph::create_bind_group(..., Some(&gpu_consts), ...)`.
//   0: g_consts        (ConstantBuffer<GpuConsts>,      read-only)   — implicit
//   1: g_requests      (StructuredBuffer<PrepRequest>,  read-only)
//   2: g_material_pool (StructuredBuffer<Occupancy>,    read-only — diff source + neighbour)
//   3: g_staging_occ   (RWStructuredBuffer<Occupancy>,  write target)
//   4: g_dirty_list    (RWByteAddressBuffer)
//   5: g_directory     (StructuredBuffer<DirEntry>,     read-only)

// `directory.hlsl` defines `DirEntry` / `Occupancy` consumers, the
// `direntry_*` accessors, and the `resolve_coord_to_slot` +
// `resolve_and_verify` helpers used below. It includes
// `gpu_consts.hlsl`, which pins `g_consts` to slot 0 — the resolver reads
// `g_consts.levels[level_idx]`, so `g_consts` is actually referenced and
// is not elided from SPIR-V reflection.
#include "include/directory.hlsl"
// `worldgen.hlsl` owns the `(coord, seed) → density` contract used by
// `terrain_occupied`. Seed flows in via `g_consts.world_seed`, set once
// at `WorldView::new`.
#include "include/worldgen.hlsl"
#include "include/occupancy.hlsl"
#include "include/request.hlsl"
#include "include/dirty_list.hlsl"

// Face-mask extractors depend on `Occupancy` being in scope; include
// after occupancy.hlsl.
#include "include/face_mask.hlsl"

[[vk::binding(1, 0)]] StructuredBuffer<PrepRequest>   g_requests;
[[vk::binding(2, 0)]] StructuredBuffer<Occupancy>     g_material_pool;
[[vk::binding(3, 0)]] RWStructuredBuffer<Occupancy>   g_staging_occ;
[[vk::binding(4, 0)]] RWByteAddressBuffer             g_dirty_list;
[[vk::binding(5, 0)]] StructuredBuffer<DirEntry>      g_directory;

// Parallel staging buffer for per-voxel material IDs. One 1 KB
// `MaterialBlock` per request (= 256 u32 packing 2 u16 IDs each). Indexed
// by `gid.x`, byte-symmetric with `g_staging_occ`. See `material.hlsl`
// for the on-disk layout.
//
// Declared at slot 6 so reflection picks it up deterministically.
#include "include/material.hlsl"
[[vk::binding(6, 0)]] RWStructuredBuffer<MaterialBlock> g_staging_material_ids;

// Multi-source neighbour read (step-2 commit-2 of the directory-pipeline
// cleanup). The prep shader consults two occupancy sources for neighbour
// resolution, in priority order:
//
//   1. g_staging_occ_prev — previous frame's staging ring slot. Contains
//      the freshly-voxelized occupancy for any sub-chunk that was dispatched
//      in frame F-1 but whose patch hasn't landed yet. The lookup
//      g_inflight_request_idx[nbr_slot] maps a neighbour's directory slot to
//      its request index in g_staging_occ_prev, or INFLIGHT_INVALID if it
//      was not in-flight last frame.
//
//   2. g_material_pool — committed occupancy (patched through frame F-2 and
//      earlier). Fallback when the neighbour was not in-flight last frame.
//
// This closes the 3-5 frame "cross-boundary conservatism window" that
// existed because prep@F previously only read from material_pool and missed
// the 1-frame-old in-flight batch. See `log-multi-source-prep-neighbour-read`.
//
// Both buffers are StructuredBuffer<Occupancy> (read-only). The lookup is
// StructuredBuffer<uint> (read-only), CPU-authored via queue.write_buffer.
//
// Define the feature macro before including exposure.hlsl so
// `neighbour_opposing_face` there picks up the conditional path.
#define PREP_INFLIGHT_STAGING
[[vk::binding(7, 0)]] StructuredBuffer<uint>      g_inflight_request_idx;
[[vk::binding(8, 0)]] StructuredBuffer<Occupancy> g_staging_occ_prev;

// The shared neighbour-aware-exposure helper depends on `g_directory`,
// `g_material_pool`, and (when PREP_INFLIGHT_STAGING is defined)
// `g_inflight_request_idx` + `g_staging_occ_prev` being in scope as
// globals, so include it after their declarations. Used by both this
// shader and `subchunk_exposure.cs.hlsl` to keep the 6-bit exposure mask
// bit-for-bit consistent between the full-prep dispatch and the
// exposure-only refresh dispatch (the exposure dispatch does not define
// PREP_INFLIGHT_STAGING and uses the material_pool path only).
#include "include/exposure.hlsl"

static const float SUBCHUNK_VOXELS = 8.0;
static const uint  OCC_WORDS       = 16u;

// Shared 16-u32 occupancy accumulator for this workgroup. `InterlockedOr`
// from per-thread bit contributions — every thread writes up to 8 bits
// across at most 4 distinct words.
groupshared uint gs_planes[OCC_WORDS];

// Terrain density: voxel centre is solid iff `wp.y < terrain_height(xz, seed)`.
// The worldgen signature itself lives in `include/worldgen.hlsl` — this
// function is the per-voxel decision that the prep shader calls 512
// times per sub-chunk. Pure heightfield: a column is either "all above
// surface = air" or "all below = solid", which is what the exposure /
// is_solid / sparsity logic downstream assumes.
bool terrain_occupied(float3 wp) {
    return wp.y < terrain_height(wp.xz, g_consts.world_seed);
}

// Hierarchical OR-reduction of `terrain_occupied` at level `lvl` over the
// 2^(3*lvl) L_0 cells covered by this L-voxel. `coarse_base_wp` is the
// world-space lower corner of the L-voxel.
//
// Implements step 2 of `decision-subchunk-visibility-storage-here-and-
// there` directly at the worldgen level: instead of point-sampling
// `terrain_occupied` once at the L-voxel's centre (which would let thin
// features vanish as LOD coarsens — a 2×1×2 L_0 column whose top sits
// below the L_1 cube centre evaluates empty at L_1 while being solid at
// L_0), we collapse the recursive `L_n cell = OR(8 covered L_{n-1}
// cells)` reduction down to its base case: an L-voxel is solid iff any
// L_0 cell it covers is solid.
//
// This fused variant *additionally* returns a representative material
// ID so the caller doesn't need a second pass. The picker is:
//
//   1. Plurality of **surface-exposed** L_0 cells — cells whose own L_0
//      neighbour in at least one of the 6 axis directions is empty.
//   2. Plurality of all solid L_0 cells, as a fallback when no cell is
//      surface-exposed (fully-interior coarse voxels, which are always
//      culled as renderables because every axis-aligned coarse neighbour
//      is also solid; the material only ever surfaces if the view-time
//      coarse-vs-finer LOD mix draws a face we didn't predict would be
//      exposed, and any bulk material is acceptable there).
//
// Why surface-exposed and not plain plurality:
//   A coarse voxel is rendered as a cube; what the viewer sees is its
//   outer face into air. The L_0 cells whose material that face shows
//   are exactly the ones on the boundary between solid and empty at
//   L_0. In a heightfield column, that's 1 grass cell vs 2 dirt vs K
//   stone — plain plurality over all solid cells structurally picks
//   dirt or stone (thicker layer) and the visible surface never shows
//   grass at coarse LODs. Selecting from surface-exposed cells only
//   inverts that: grass dominates because every surface cell *is*
//   grass, while dirt/stone are never surface-exposed under the grass.
//   Position-agnostic (no privileged axis) — the same rule handles
//   asteroid outer shells, player builds, and caves/overhangs without
//   content-specific heuristics.
//
// L_0 cells inside the L-voxel sit at integer world-space lower corners
// in `[0, 2^lvl)^3` relative to `coarse_base_wp`. Centres are offset by
// `+ 0.5` along each axis (matching `terrain_occupied`'s "voxel centre"
// contract on the L_0 grid). Neighbour probes cross the footprint
// boundary freely: `terrain_occupied` is a pure `(coord, seed) → bool`
// function, so evaluating outside the footprint is correct, not a
// conservative guess.
//
// Property (load-bearing for step 3 — cross-LOD sub/super-sample):
//   coarse_occupied(lvl, p) == false ⇒ every L_0 cell inside the L-voxel
//   at `p` evaluates `terrain_occupied` == false.
// Equivalently, the predicate is monotone-conservative: an L-voxel is
// guaranteed to be solid whenever any covered finer cell is solid.
//
// Cost: ~7× the pure footprint walk. Each solid L_0 cell does 1 own
// density eval + up to 6 neighbour density evals (short-circuited via
// `||`, so most surface-edge cells exit after 1-2 probes). At L=0 the
// 6-neighbour check still runs on a single-cell footprint — it produces
// the same single-cell material the prior implementation returned, and
// the cost is ~7 evals vs the previous 1. At the deepest LOD the
// material-data pool covers (L=4, extent=16), the worst-case bound is
// ~28k evals per coarse voxel vs 4k previously; still well inside the
// per-dispatch cost budget (prep runs on residency events, not per
// frame). When a future LOD ceiling makes this painful, cache the
// 2^(3·lvl) footprint occupancies and reuse the cache for neighbour
// probes of interior cells — internal cells amortise to zero extra
// evals, the outer shell retains up to 3 outside-footprint probes each.
bool coarse_occupied(uint lvl, float3 coarse_base_wp, out uint out_mat) {
    uint extent = 1u << lvl;

    // Counts per block id. The registry has up to 4 entries in M1
    // (air / grass / dirt / stone); resize if the registry grows. Index
    // 0 (AIR) is never incremented — air cells fall out of
    // `terrain_occupied`.
    //
    // Two parallel counters:
    //   surface_counts — solid cells with at least one empty L_0 neighbour
    //   all_counts     — all solid cells (fallback for fully-interior)
    uint surface_counts[4] = { 0u, 0u, 0u, 0u };
    uint all_counts[4]     = { 0u, 0u, 0u, 0u };

    [loop] for (uint dz = 0u; dz < extent; ++dz) {
        [loop] for (uint dy = 0u; dy < extent; ++dy) {
            [loop] for (uint dx = 0u; dx < extent; ++dx) {
                float3 wp = coarse_base_wp + float3(dx, dy, dz) + 0.5;
                if (!terrain_occupied(wp)) {
                    continue;
                }
                uint mat = terrain_material(wp, g_consts.world_seed);
                if (mat >= 4u) {
                    continue;
                }
                all_counts[mat] += 1u;

                // L_0-spacing neighbour probes. `||` short-circuits, so
                // most surface cells exit after the first or second hit.
                // +Y leads because the heightfield's surface grass layer
                // has its empty neighbour directly above; other terrains
                // will hit a different axis first but the ordering is a
                // perf hint, not a correctness dependency.
                bool exposed =
                       !terrain_occupied(wp + float3( 0.0,  1.0,  0.0))
                    || !terrain_occupied(wp + float3(-1.0,  0.0,  0.0))
                    || !terrain_occupied(wp + float3( 1.0,  0.0,  0.0))
                    || !terrain_occupied(wp + float3( 0.0,  0.0, -1.0))
                    || !terrain_occupied(wp + float3( 0.0,  0.0,  1.0))
                    || !terrain_occupied(wp + float3( 0.0, -1.0,  0.0));
                if (exposed) {
                    surface_counts[mat] += 1u;
                }
            }
        }
    }

    // Plurality pick, skipping AIR. Try surface-exposed first; a non-
    // empty surface count means some L_0 cell in the footprint actually
    // faces air and is what the viewer will see. Fall back to all
    // solid cells only when nothing is exposed (fully-interior coarse
    // voxel — its face only draws under a cross-LOD mismatch at render
    // time, and any material choice is acceptable there).
    uint best_mat   = BLOCK_ID_AIR;
    uint best_count = 0u;
    [unroll] for (uint i = 1u; i < 4u; ++i) {
        if (surface_counts[i] > best_count) {
            best_count = surface_counts[i];
            best_mat   = i;
        }
    }
    if (best_count == 0u) {
        [unroll] for (uint j = 1u; j < 4u; ++j) {
            if (all_counts[j] > best_count) {
                best_count = all_counts[j];
                best_mat   = j;
            }
        }
    }

    out_mat = best_mat;
    return best_mat != BLOCK_ID_AIR;
}

[numthreads(4, 4, 4)]
void main(uint3 ltid : SV_GroupThreadID, uint3 gid : SV_GroupID, uint lidx : SV_GroupIndex) {
    // Zero the shared accumulator. 16 words / 64 threads → first 16 threads
    // each clear one slot; the rest idle until the barrier.
    if (lidx < OCC_WORDS) {
        gs_planes[lidx] = 0u;
    }

    GroupMemoryBarrierWithGroupSync();

    PrepRequest req        = g_requests[gid.x];
    float       voxel_size = float(1u << req.level);
    float3      base       = float3(req.coord) * SUBCHUNK_VOXELS * voxel_size;

    // Each thread covers a 2×2×2 voxel cube starting at (ltid * 2).
    uint3 origin = ltid * 2u;

    // Per-thread pack of the 4 u32 words this thread contributes to the
    // material-id staging block. Initialised to 0 (all-air); each voxel
    // that resolves to solid overwrites its 16-bit lane with the block
    // id chosen by `terrain_material`.
    //
    // Thread coverage within the 2×2×2 cube at `origin = ltid * 2`:
    //   - Word layout in `MaterialBlock.packed_ids` is
    //       word[(voxel_idx) / 2]: low = voxel 2k, high = voxel 2k+1,
    //     and `voxel_idx = x + 8y + 64z`. Pairs (dx=0, dx=1) share a word.
    //   - Each thread owns exactly 4 words: (dy, dz) ∈ {0,1}² at the
    //     even dx. Precompute their indices so the final writes are
    //     loop-free.
    uint pack_word_idx[4];
    [unroll] for (uint pw = 0u; pw < 4u; ++pw) {
        uint dy_p = pw & 1u;
        uint dz_p = pw >> 1u;
        uint even_voxel_idx = origin.x
                            + 8u  * (origin.y + dy_p)
                            + 64u * (origin.z + dz_p);
        pack_word_idx[pw] = even_voxel_idx >> 1u;
    }
    uint pack_word_val[4] = { 0u, 0u, 0u, 0u };

    [unroll] for (uint dz = 0u; dz < 2u; ++dz) {
        [unroll] for (uint dy = 0u; dy < 2u; ++dy) {
            [unroll] for (uint dx = 0u; dx < 2u; ++dx) {
                uint x = origin.x + dx;
                uint y = origin.y + dy;
                uint z = origin.z + dz;

                // World-space lower corner of this L-voxel. The hierarchical
                // OR-reduction (`coarse_occupied`) walks the 2^(3*level)
                // L_0 cells inside the cube `[corner, corner + voxel_size)`
                // and OR-folds their density samples — at L=0, a single
                // eval at `corner + 0.5` matches the previous path
                // bit-for-bit. See the helper's docstring for the
                // load-bearing monotonicity property this gives step 3.
                //
                // The fused helper also returns the material ID of the
                // first solid L_0 cell it encounters, so the coarse
                // L-voxel is guaranteed to take the material of an
                // actual solid cell in its footprint (a 2×2×2 L_1 voxel
                // whose only solid cell is grass renders as grass, not
                // stone, not air).
                float3 coarse_base = base + float3(x, y, z) * voxel_size;
                uint   coarse_mat_id;
                if (!coarse_occupied(req.level, coarse_base, coarse_mat_id)) {
                    continue;
                }

                uint bit         = y * 8u + x;
                uint word        = z * 2u + (bit >> 5u);
                uint bit_in_word = bit & 31u;

                uint unused;
                InterlockedOr(gs_planes[word], 1u << bit_in_word, unused);

                // Pack the per-voxel material ID into the appropriate
                // 16-bit lane of this thread's cache. `coarse_mat_id`
                // is always a solid-cell material for a solid L-voxel
                // (see `coarse_occupied` docstring).
                uint   mat_id    = coarse_mat_id & 0xFFFFu;
                uint   pw        = dy + 2u * dz;
                uint   hi        = dx;   // dx==0 → low 16, dx==1 → high 16
                pack_word_val[pw] |= mat_id << (hi * 16u);
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Each thread stores its 4 u32 words into the material-id staging
    // block. Every `pack_word_idx` is unique per thread (no collision
    // across threads — the even_voxel_idx pairing guarantees this), so
    // a direct `[]` store is safe without InterlockedOr.
    [unroll] for (uint pw2 = 0u; pw2 < 4u; ++pw2) {
        g_staging_material_ids[gid.x].packed_ids[pack_word_idx[pw2]] = pack_word_val[pw2];
    }

    // Thread 0: write staging, classify, diff, and append to the dirty list.
    // The staging write is serialized here (16 stores) rather than fanned
    // out across 16 threads — trivial cost relative to the 512 density
    // evaluations that preceded it, and it keeps the classify + diff +
    // append in one linear branch with no extra barriers.
    if (lidx == 0u) {
        // Pack staged occupancy into four uint4s.
        uint4 staged0 = uint4(gs_planes[ 0], gs_planes[ 1], gs_planes[ 2], gs_planes[ 3]);
        uint4 staged1 = uint4(gs_planes[ 4], gs_planes[ 5], gs_planes[ 6], gs_planes[ 7]);
        uint4 staged2 = uint4(gs_planes[ 8], gs_planes[ 9], gs_planes[10], gs_planes[11]);
        uint4 staged3 = uint4(gs_planes[12], gs_planes[13], gs_planes[14], gs_planes[15]);

        // Staging is indexed by request (gid.x), not by directory slot —
        // so the staging buffer can be sized to the prep dispatch width,
        // independent of the live pool's capacity.
        Occupancy staged;
        staged.plane[0] = staged0;
        staged.plane[1] = staged1;
        staged.plane[2] = staged2;
        staged.plane[3] = staged3;
        g_staging_occ[gid.x] = staged;

        // Self-resolve the directory index from coord + level. The slot
        // field used to ride on the prep request; Step 4 removed it and
        // made the shader authoritative. The mapping is a pure function of
        // `coord` and per-level `pool_dims` (from `g_consts`) — no shell
        // corner / pool_origin involvement. See `resolve_coord_to_slot`
        // docstring for why `pool_origin` was removed.
        uint self_slot = resolve_coord_to_slot(req.coord, req.level);

        // Any-bit / all-bits checks on the staged occupancy. Fed back into
        // both the exposure computation (an empty sub-chunk has no face to
        // expose, bypass the neighbour loop) and the is_solid hint.
        uint  or_accum  = staged0.x | staged0.y | staged0.z | staged0.w
                        | staged1.x | staged1.y | staged1.z | staged1.w
                        | staged2.x | staged2.y | staged2.z | staged2.w
                        | staged3.x | staged3.y | staged3.z | staged3.w;
        bool  any_bit   = or_accum != 0u;

        bool  is_fully_solid = occupancy_is_fully_solid(staged);

        // Build the neighbour-aware exposure mask via the shared helper in
        // `include/exposure.hlsl` (bit-for-bit shared with
        // `subchunk_exposure.cs.hlsl` so the full-prep and exposure-only
        // dispatches never drift). Skip the lookup when the sub-chunk is
        // fully empty — no voxel faces the neighbour, so every bit is
        // zero regardless of neighbour occupancy.
        uint exposure = 0u;
        if (any_bit) {
            Occupancy me;
            me.plane[0] = staged0;
            me.plane[1] = staged1;
            me.plane[2] = staged2;
            me.plane[3] = staged3;
            exposure = compute_exposure_mask(me, req.coord, req.level);
        }

        uint is_solid_bit = is_fully_solid ? 1u : 0u;

        // Diff against the currently-live material pool entry, if one
        // exists. The directory's `resident` bit is the authoritative gate:
        //   resident  ⇒ material_slot points at a real material-pool entry
        //   !resident ⇒ no diff source (first-time or post-eviction)
        //
        // Relying on `direntry_has_material` would be incorrect here: the
        // buffer's zero-init state has material_slot == 0, which is a real
        // pool index — reading `g_material_pool[0]` would alias an
        // unrelated slot and silently corrupt the diff. A non-resident
        // entry always forces `dirty = true` so the CPU retirement sees
        // this prep and can either allocate a slot (sparse-first-time) or
        // record a uniform-empty directory transition.
        DirEntry d     = g_directory[self_slot];
        bool     dirty;
        if (direntry_is_resident(d)) {
            uint mslot = direntry_get_material_slot(d);
            Occupancy live = g_material_pool[mslot];
            dirty = any(staged0 != live.plane[0])
                 || any(staged1 != live.plane[1])
                 || any(staged2 != live.plane[2])
                 || any(staged3 != live.plane[3]);
        } else {
            dirty = true;
        }

        if (dirty) {
            // Bounds-check. If the pre-add count already filled the
            // buffer, `dirty_list_append` raises the overflow flag and
            // skips the store — writing past MAX_CANDIDATES would overrun
            // the DirtyReport buffer. The count header keeps incrementing
            // so CPU can see how far past capacity the dispatch tried to
            // go; CPU clamps reads to min(count, DIRTY_LIST_CAPACITY).
            // new_bits_partial packs the shader-authored fields of
            // DirEntry::bits — exposure [0..5], is_solid [6], resident
            // (= 1). The material-slot field is authored CPU-side at
            // retirement and must be zero in this shader-emitted word.
            dirty_list_append(g_dirty_list, self_slot, exposure, is_solid_bit, gid.x);
        }
    }
}
