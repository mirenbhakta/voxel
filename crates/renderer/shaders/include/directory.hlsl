// directory.hlsl — GPU-side types + accessors for the sub-chunk directory.
//
// Mirrors the Rust `DirEntry` layout in `crates/renderer/src/subchunk.rs`
// and the per-level static metadata in `GpuConsts::levels` (see
// `gpu_consts.hlsl`). The directory is a dense
// `StructuredBuffer<DirEntry>` indexed by a stable `directory_index`
// (= `level.global_offset + pool_slot_within_level`). It is CPU-authored,
// GPU-read-only — Principle 2: no store/pack accessors here.
//
// # Purpose
//
// Downstream shaders (prep + cull) resolve a world-space sub-chunk coord
// at a given LOD into a directory entry via `resolve_coord_to_slot`, then
// consult `direntry_*` accessors to read the per-entry contract:
//
// - `is_resident`     — entry points at real occupancy; neighbour lookups
//                       must early-out when this is clear.
// - `exposure`        — 6-bit directional face-exposure mask; the cull
//                       pass ANDs it against the camera-visible face set.
// - `is_solid`        — future hint for "uniformly solid, skip DDA".
// - `material_slot`   — 24-bit index into the material-storage pool; only
//                       meaningful under a resident entry.
// - `coord_matches`   — torus-verification check. Because
//                       `resolve_coord_to_slot` reduces the query coord
//                       modulo `pool_dims`, two different world coords can
//                       collide at the same slot. The entry's own `coord`
//                       field is the authoritative occupant; callers that
//                       might read under a torus-colliding coord should
//                       use `resolve_and_verify` instead of the raw slot.
//
// # Binding discipline
//
// This header defines types + pure functions only. The consuming shader
// is responsible for declaring its own
//   StructuredBuffer<DirEntry> g_directory;
// with a binding location appropriate to that pipeline. Keeping the header
// free of `[[vk::binding(...)]]` lets it be included into multiple shaders
// without binding-slot collisions.
//
// `g_consts` (from `gpu_consts.hlsl`) is a single fixed-slot binding
// (Principle 5) and is included directly below.

#ifndef RENDERER_DIRECTORY_HLSL
#define RENDERER_DIRECTORY_HLSL

#include "gpu_consts.hlsl"
#include "material.hlsl"

// -----------------------------------------------------------------------
// Bit layout of `DirEntry::bits`. Must stay byte-for-byte in sync with the
// Rust constants in `crates/renderer/src/subchunk.rs`:
//
//   BITS_EXPOSURE_MASK         = 0x3F         (bits 0..5 inclusive)
//   BITS_IS_SOLID              = 1 << 6       (bit 6)
//   BITS_RESIDENT              = 1 << 7       (bit 7)
//   BITS_MATERIAL_SLOT_SHIFT   = 8            (bits 8..31 — 24-bit field)
//   MATERIAL_SLOT_INVALID      = 0x00FF_FFFF  (sentinel: non-resident)
//   MATERIAL_DATA_SLOT_INVALID = 0xFFFF_FFFF  (sentinel: material-data pool
//                                              entry unallocated / deferred /
//                                              ceiling-reached — see
//                                              material.hlsl)
// -----------------------------------------------------------------------
#define DIRENTRY_BITS_EXPOSURE_MASK        0x3Fu
#define DIRENTRY_BITS_IS_SOLID             (1u << 6)
#define DIRENTRY_BITS_RESIDENT             (1u << 7)
#define DIRENTRY_BITS_MATERIAL_SLOT_SHIFT  8u
#define DIRENTRY_MATERIAL_SLOT_INVALID     0x00FFFFFFu
// MATERIAL_DATA_SLOT_INVALID (0xFFFFFFFFu) is defined in material.hlsl
// (included above). Use that name directly — a local alias would be a
// redundant definition of the same sentinel value.

// -----------------------------------------------------------------------
// DirEntry — 28-byte CPU-authored directory entry.
//
// Layout matches Rust `renderer::subchunk::DirEntry`:
//   int3 coord              (+0)
//   uint bits               (+12)  // [0..5] exposure, [6] is_solid,
//                                  // [7] resident, [8..31] material_slot
//                                  //                       | INVALID
//                                  // (material_slot is the OCCUPANCY pool
//                                  //  slot; NOT the material-data pool.)
//   uint content_version    (+16)
//   uint last_synth_version (+20)
//   uint material_data_slot (+24)  // flat-global MaterialDataPool slot, or
//                                  // MATERIAL_DATA_SLOT_INVALID (material.hlsl).
// -----------------------------------------------------------------------
struct DirEntry {
    int3 coord;
    uint bits;
    uint content_version;
    uint last_synth_version;
    uint material_data_slot;
};

// -----------------------------------------------------------------------
// Accessors on DirEntry. Pure reads — the GPU never writes directory
// entries (Principle 2: directory is CPU-authored).
// -----------------------------------------------------------------------

// Six-bit directional exposure mask (`-X, +X, -Y, +Y, -Z, +Z`, matching
// the cull shader's existing face-bit convention).
uint direntry_get_exposure(DirEntry e) {
    return e.bits & DIRENTRY_BITS_EXPOSURE_MASK;
}

// `true` when the uniformly-solid hint bit is set. Reserved for a future
// DDA fast-path; not consumed by any shader today.
bool direntry_is_solid(DirEntry e) {
    return (e.bits & DIRENTRY_BITS_IS_SOLID) != 0u;
}

// `true` when the entry points at real occupancy. Always check this before
// trusting `material_slot` — a non-resident entry carries zero in the slot
// field. The resident bit is the authoritative gate.
bool direntry_is_resident(DirEntry e) {
    return (e.bits & DIRENTRY_BITS_RESIDENT) != 0u;
}

// Unpacked 24-bit material-slot field — an index into the CPU-authoritative
// `MaterialAllocator` pool, or `DIRENTRY_MATERIAL_SLOT_INVALID` for
// non-resident entries. Use `direntry_has_material` for the "this slot is
// a real pool index" predicate.
uint direntry_get_material_slot(DirEntry e) {
    return (e.bits >> DIRENTRY_BITS_MATERIAL_SLOT_SHIFT)
         & DIRENTRY_MATERIAL_SLOT_INVALID;
}

// `true` when the material-slot field holds a real pool index (not the
// INVALID sentinel). Does NOT imply `is_resident`: a buffer-zero entry has
// slot 0 and is not resident. Callers that want "resident ∧ has a real
// slot" should AND both predicates, or rely on the CPU invariant that
// `resident ⇒ material_slot != INVALID` (enforced in `DirEntry::resident`).
bool direntry_has_material(DirEntry e) {
    return direntry_get_material_slot(e) != DIRENTRY_MATERIAL_SLOT_INVALID;
}

// Flat-global slot index into the material-data pool, or
// `MATERIAL_DATA_SLOT_INVALID` (from material.hlsl) when the pool has no
// allocation for this sub-chunk. The shade shader reads this (never `occ_slot`) to
// locate per-voxel material IDs — see `decision-material-system-m1-sparse`.
uint direntry_get_material_data_slot(DirEntry e) {
    return e.material_data_slot;
}

// `true` when the material-data slot field holds a real pool index (not
// the INVALID sentinel). Callers that observe this false should draw
// the magenta sentinel pixel rather than dereference the pool.
bool direntry_has_material_data(DirEntry e) {
    return e.material_data_slot != MATERIAL_DATA_SLOT_INVALID;
}

// Torus-verification check. `resolve_coord_to_slot` reduces the query
// coord modulo `pool_dims`; two world coords separated by `pool_dims`
// collide at the same slot. The entry's stored `coord` is the
// authoritative occupant — if it does not match `c`, the slot is holding
// a different sub-chunk and the caller's lookup is a stale alias.
bool direntry_coord_matches(DirEntry e, int3 c) {
    return all(e.coord == c);
}

// -----------------------------------------------------------------------
// resolve_coord_to_slot — world-space sub-chunk coord → directory index.
//
// Byte-for-byte mirror of `SlotPool::slot_id` in
// `crates/game/src/world/pool.rs`:
//
//     p = coord.rem_euclid(pool_dims)
//     pool_slot = p.z * pool_dims.y * pool_dims.x
//               + p.y * pool_dims.x
//               + p.x
//
// The final `directory_index` adds the level's `global_offset` so entries
// from different levels coexist in one flat `StructuredBuffer<DirEntry>`.
//
// HLSL's `%` on signed `int` is a C-style truncated remainder (can be
// negative for negative dividends). The double-mod trick — `((a % n) + n)
// % n` — produces the Euclidean remainder and matches Rust's `rem_euclid`.
//
// # Why this formula does not use `pool_origin`
//
// The per-level toroidal pool is keyed purely by `coord mod pool_dims`.
// Two coords differing by exactly `pool_dims[i]` along axis `i` collide
// at the same slot regardless of where the shell corner is. `pool_origin`
// describes where the residency shell is anchored for iteration purposes;
// it does not participate in the slot-id derivation.
//
// An earlier version of this function subtracted `pool_origin` before the
// `rem_euclid`. That makes the shader and CPU slot formulas agree only
// when `pool_origin % pool_dims == 0` — true in aligned cold-start but
// false once the shell recenters across a non-pool-dim boundary, which
// happens every coarsest-level shift. The divergence manifested as a
// cross-wired voxel render: CPU-authored instances pointed at their own
// `coord.rem_euclid(dims)` slot while the shader wrote staging into the
// `(coord - pool_origin).rem_euclid(dims)` slot. See
// `failure-resolve-coord-to-slot-diverges-from-cpu-pool` in Agentic Memory.
//
// Parameters:
//   coord      — world-space sub-chunk coord at this level's grid
//   level_idx  — index into `g_consts.levels[]`; caller's responsibility
//                to stay within `g_consts.level_count`
//
// Returns the flat `directory_index` for this coord at `level_idx`. The
// caller reads `g_directory[directory_index]` to obtain the `DirEntry` and
// should verify occupancy via `direntry_is_resident` and, under potential
// torus collisions, `direntry_coord_matches`.
// -----------------------------------------------------------------------
uint resolve_coord_to_slot(int3 coord, uint level_idx) {
    LevelStatic level       = g_consts.levels[level_idx];
    int3        pool_dims_i = int3(level.pool_dims);

    // rem_euclid via double-mod; `pool_dims_i` is strictly positive
    // (enforced CPU-side; zeroed-out unused level entries never see a
    // lookup because `level_count` gates the caller).
    int3 p = ((coord % pool_dims_i) + pool_dims_i) % pool_dims_i;

    uint pool_slot = uint(p.z) * level.pool_dims.y * level.pool_dims.x
                   + uint(p.y) * level.pool_dims.x
                   + uint(p.x);

    return level.global_offset + pool_slot;
}

// -----------------------------------------------------------------------
// resolve_and_verify — resolve + torus-collision check in one call.
//
// Reads `g_directory[resolve_coord_to_slot(coord, level_idx)]` and
// returns `true` only when the resulting `DirEntry` actually belongs to
// `coord` (`direntry_coord_matches`). Use this whenever a shader queries
// a coord it did not itself produce — neighbour lookups in particular —
// so that a stale entry living at the same torus slot under a different
// coord is not silently consumed.
//
// The resolved slot is always written to `out_slot`, regardless of the
// return value, so callers that want to surface the alias (e.g. to a
// shadow-ledger counter) can inspect the entry anyway.
// -----------------------------------------------------------------------
bool resolve_and_verify(
    int3 coord,
    uint level_idx,
    StructuredBuffer<DirEntry> directory,
    out uint out_slot
) {
    out_slot = resolve_coord_to_slot(coord, level_idx);
    DirEntry e = directory[out_slot];
    return direntry_coord_matches(e, coord);
}

#endif // RENDERER_DIRECTORY_HLSL
