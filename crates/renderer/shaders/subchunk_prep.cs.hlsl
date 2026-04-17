// subchunk_prep.cs.hlsl — GPU-side sub-chunk occupancy synthesis with
// diff-vs-live dirty-list emission and isolated-exposure emission.
//
// HLSL port of `game::world_view::synthesize_occupancy` +
// `terrain_occupied`, extended so the CPU never re-synthesizes content on
// cache hits: after the new occupancy is voxelized into a staging slot, the
// shader compares it against the previously-committed live occupancy and,
// when any plane word differs, appends the slot to a compact dirty list.
// The CPU reads the dirty list via a `ReadbackChannel` and patches only
// those slots into residency.
//
// # Dispatch shape
//
// One workgroup per `PrepRequest`. Workgroup size `[4, 4, 4]` (= 64
// threads). Each thread handles a 2×2×2 cube of voxels (8 density
// evaluations per thread → 512 per sub-chunk, one per voxel).
//
// Occupancy is accumulated into `gs_planes[16]` using `InterlockedOr`; on
// sync, thread 0 writes the 16 words out to `g_staging_occ[slot]`,
// writes the slot's exposure into `g_staging_exposure[slot]`, and runs
// the diff against `g_live_occ[slot]`. If any occupancy word differs,
// thread 0 atomically appends `(slot, staging_offset = slot)` to
// `g_dirty_list`.
//
// `g_staging_occ[slot]` and `g_staging_exposure[slot]` are both populated
// unconditionally every dispatch — the live pool may diff against the
// previous staging payload on a subsequent frame, so the writes must land
// even when the content matches. The patch pass copies staging → live
// for both buffers on every dirty slot, so emitting exposure alongside
// occupancy is free in dirty-list bookkeeping and keeps the copy pair
// consistent with the occupancy payload it accompanies.
//
// The diff check remains driven by occupancy alone. Exposure is a pure
// function of the same occupancy bits, so "occupancy did not change"
// implies "exposure did not change"; no separate exposure diff is needed
// and the dirty-list entry still carries just `{slot, staging_offset}`.
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
// # Dirty-list layout (RWByteAddressBuffer)
//
//   offset  0:  uint count
//   offset  4:  uint _pad0
//   offset  8:  uint _pad1
//   offset 12:  uint _pad2
//   offset 16 + i * 8: { uint slot, uint staging_offset } for i in [0, count)
//
// Staging and live share a 1:1 slot→payload mapping today (one live slot
// per request). `staging_offset` is reported as `slot` for that reason;
// the field exists so that decoupling staging from live (batched compaction
// etc.) stays additive.
//
// # Binding layout
//
// Set 0 — contiguous slots 0..5 (wgpu-hal's Vulkan backend compacts BGL
// entries sequentially, and the SPIR-V passthrough path does not remap, so
// HLSL binding numbers must equal their ordinal position within the
// reflected set):
//   0: g_requests         (StructuredBuffer<PrepRequest>, read-only)
//   1: g_live_occ         (StructuredBuffer<Occupancy>,   read-only — diff source)
//   2: g_staging_occ      (RWStructuredBuffer<Occupancy>, write target)
//   3: g_staging_exposure (RWStructuredBuffer<uint>,      write target)
//   4: g_dirty_list       (RWByteAddressBuffer)

struct PrepRequest {
    int3 coord;   // sub-chunk coord at this request's LOD
    uint level;   // LOD level; voxel_size = 1 << level
    uint slot;    // target live slot index (and 1:1 staging slot)
    uint _pad0;
    uint _pad1;
    uint _pad2;
};  // 32 bytes

// Matches the 16-u32 packed layout of `SubchunkOccupancy::to_gpu_bytes`
// on the Rust side: bit `y * 8 + x` of word `z * 2 + (bit >> 5)` is set
// iff voxel (x, y, z) is occupied.
struct Occupancy {
    uint4 plane[4];
};  // 64 bytes

[[vk::binding(0, 0)]] StructuredBuffer<PrepRequest>   g_requests;
[[vk::binding(1, 0)]] StructuredBuffer<Occupancy>     g_live_occ;
[[vk::binding(2, 0)]] RWStructuredBuffer<Occupancy>   g_staging_occ;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>        g_staging_exposure;
[[vk::binding(4, 0)]] RWByteAddressBuffer             g_dirty_list;

static const float SUBCHUNK_VOXELS = 8.0;
static const uint  OCC_WORDS       = 16u;

// Shared 16-u32 occupancy accumulator for this workgroup. `InterlockedOr`
// from per-thread bit contributions — every thread writes up to 8 bits
// across at most 4 distinct words.
groupshared uint gs_planes[OCC_WORDS];

// Terrain density: voxel centre is solid iff `wp.y < h`. Layered sinusoids
// matching `world_view::terrain_occupied` on the CPU side.
bool terrain_occupied(float3 wp) {
    float h = sin(wp.x * 0.05) * 4.0
            + cos(wp.z * 0.05) * 4.0
            + sin(wp.x * 0.20) * 1.0
            + cos(wp.z * 0.20) * 1.0
            - 5.0;
    return wp.y < h;
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

                float3 wp = base + (float3(x, y, z) + 0.5) * voxel_size;
                if (!terrain_occupied(wp)) {
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

    // Thread 0: write staging, run the diff, and append to the dirty list.
    // The staging write is serialized here (16 stores) rather than fanned
    // out across 16 threads — trivial cost relative to the 512 density
    // evaluations that preceded it, and it keeps the diff + append in one
    // linear branch with no extra barriers.
    if (lidx == 0u) {
        Occupancy live    = g_live_occ[req.slot];
        uint4     staged0 = uint4(gs_planes[ 0], gs_planes[ 1], gs_planes[ 2], gs_planes[ 3]);
        uint4     staged1 = uint4(gs_planes[ 4], gs_planes[ 5], gs_planes[ 6], gs_planes[ 7]);
        uint4     staged2 = uint4(gs_planes[ 8], gs_planes[ 9], gs_planes[10], gs_planes[11]);
        uint4     staged3 = uint4(gs_planes[12], gs_planes[13], gs_planes[14], gs_planes[15]);

        Occupancy staged;
        staged.plane[0] = staged0;
        staged.plane[1] = staged1;
        staged.plane[2] = staged2;
        staged.plane[3] = staged3;
        g_staging_occ[req.slot] = staged;

        // Isolated exposure: 0 if the staged occupancy is empty, else 0x3F.
        // Matches the CPU `SubchunkOccupancy::isolated_exposure_mask`
        // semantics — cross-boundary refinement (neighbor-aware culling)
        // is a later concern. `any(uint4)` returns true when any lane is
        // nonzero (HLSL convention).
        bool any_bit = any(staged0) || any(staged1)
                    || any(staged2) || any(staged3);
        uint exposure = any_bit ? 0x3Fu : 0u;
        g_staging_exposure[req.slot] = exposure;

        bool dirty = any(staged0 != live.plane[0])
                  || any(staged1 != live.plane[1])
                  || any(staged2 != live.plane[2])
                  || any(staged3 != live.plane[3]);

        if (dirty) {
            uint idx;
            g_dirty_list.InterlockedAdd(0u, 1u, idx);
            uint entry_off = 16u + idx * 8u;
            g_dirty_list.Store(entry_off + 0u, req.slot);
            g_dirty_list.Store(entry_off + 4u, req.slot);
        }
    }
}
