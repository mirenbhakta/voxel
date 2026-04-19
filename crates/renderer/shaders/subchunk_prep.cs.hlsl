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

struct PrepRequest {
    int3 coord;   // sub-chunk coord at this request's LOD
    uint level;   // LOD level; voxel_size = 1 << level
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};  // 32 bytes

// Matches the 16-u32 packed layout of `SubchunkOccupancy::to_gpu_bytes`
// on the Rust side: bit `y * 8 + x` of word `z * 2 + (bit >> 5)` is set
// iff voxel (x, y, z) is occupied.
struct Occupancy {
    uint4 plane[4];
};  // 64 bytes

// Face-mask extractors depend on `Occupancy` being in scope; include
// here rather than ahead of the struct definition.
#include "include/face_mask.hlsl"

[[vk::binding(1, 0)]] StructuredBuffer<PrepRequest>   g_requests;
[[vk::binding(2, 0)]] StructuredBuffer<Occupancy>     g_material_pool;
[[vk::binding(3, 0)]] RWStructuredBuffer<Occupancy>   g_staging_occ;
[[vk::binding(4, 0)]] RWByteAddressBuffer             g_dirty_list;
[[vk::binding(5, 0)]] StructuredBuffer<DirEntry>      g_directory;

// The shared neighbour-aware-exposure helper depends on `g_directory` +
// `g_material_pool` being in scope as globals, so include it after their
// declarations. Used by both this shader and `subchunk_exposure.cs.hlsl`
// to keep the 6-bit exposure mask bit-for-bit consistent between the
// full-prep dispatch and the exposure-only refresh dispatch.
#include "include/exposure.hlsl"

static const float SUBCHUNK_VOXELS = 8.0;
static const uint  OCC_WORDS       = 16u;

// Mirror of `renderer::subchunk::MAX_CANDIDATES` — ceiling on dirty-list
// entries per dispatch. The dirty-list append guards against exceeding
// this ceiling and raises the overflow flag instead of overrunning the
// `DirtyReport` buffer.
static const uint DIRTY_LIST_CAPACITY = 256u;

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
// L_0 cells inside the L-voxel sit at integer world-space lower corners
// in `[0, 2^lvl)^3` relative to `coarse_base_wp`. Centres are offset by
// `+ 0.5` along each axis (matching `terrain_occupied`'s "voxel centre"
// contract on the L_0 grid).
//
// Property (load-bearing for step 3 — cross-LOD sub/super-sample):
//   coarse_occupied(lvl, p) == false ⇒ every L_0 cell inside the L-voxel
//   at `p` evaluates `terrain_occupied` == false.
// Equivalently, the predicate is monotone-conservative: an L-voxel is
// guaranteed to be solid whenever any covered finer cell is solid.
//
// Cost: O(8^lvl) density evals per L-voxel. At lvl=0 this collapses to
// the old single-sample path (extent=1, exactly one iteration). Early-out
// on first solid keeps heightfield columns cheap — typical solid-bottom
// L_2 voxels return after one eval.
bool coarse_occupied(uint lvl, float3 coarse_base_wp) {
    uint extent = 1u << lvl;
    [loop] for (uint dz = 0u; dz < extent; ++dz) {
        [loop] for (uint dy = 0u; dy < extent; ++dy) {
            [loop] for (uint dx = 0u; dx < extent; ++dx) {
                float3 wp = coarse_base_wp + float3(dx, dy, dz) + 0.5;
                if (terrain_occupied(wp)) {
                    return true;
                }
            }
        }
    }
    return false;
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
                float3 coarse_base = base + float3(x, y, z) * voxel_size;
                if (!coarse_occupied(req.level, coarse_base)) {
                    continue;
                }

                uint bit         = y * 8u + x;
                uint word        = z * 2u + (bit >> 5u);
                uint bit_in_word = bit & 31u;

                uint unused;
                InterlockedOr(gs_planes[word], 1u << bit_in_word, unused);
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

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

        uint  and_accum = staged0.x & staged0.y & staged0.z & staged0.w
                        & staged1.x & staged1.y & staged1.z & staged1.w
                        & staged2.x & staged2.y & staged2.z & staged2.w
                        & staged3.x & staged3.y & staged3.z & staged3.w;
        bool  is_fully_solid = and_accum == 0xFFFFFFFFu;

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
            uint idx;
            g_dirty_list.InterlockedAdd(0u, 1u, idx);
            // Bounds-check. If the pre-add count already filled the
            // buffer, raise the overflow flag and skip the store —
            // writing at `16 + idx * 16` with `idx >= 256` would overrun
            // the DirtyReport buffer (a silent corruption of whatever
            // follows it in the allocator). The count header keeps
            // incrementing so CPU can see how far past capacity the
            // dispatch tried to go; CPU clamps reads to min(count,
            // MAX_CANDIDATES).
            if (idx >= DIRTY_LIST_CAPACITY) {
                uint _prev;
                g_dirty_list.InterlockedMax(8u, 1u, _prev);
            } else {
                uint entry_off = 16u + idx * 16u;
                // new_bits_partial packs the shader-authored fields of
                // DirEntry::bits — exposure [0..5], is_solid [6], resident
                // (= 1). The material-slot field is authored CPU-side at
                // retirement and must be zero in this shader-emitted word.
                uint new_bits_partial = exposure
                                      | (is_solid_bit << 6)
                                      | (1u << 7);
                g_dirty_list.Store(entry_off +  0u, self_slot);  // directory_index
                g_dirty_list.Store(entry_off +  4u, new_bits_partial);
                g_dirty_list.Store(entry_off +  8u, gid.x);      // staging_request_idx
                g_dirty_list.Store(entry_off + 12u, 0u);
            }
        }
    }
}
