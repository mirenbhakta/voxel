// dda.hlsl — sub-chunk voxel DDA primitive.
//
// Amanatides-Woo 3D DDA across an 8^3 sub-chunk's occupancy grid. Exposes
// one pure function, `dda_sub_chunk`, returning a compact `MarchResult`
// whose fields match the vis-buffer packing contract
// (`decision-vis-buffer-deferred-shading-phase-1`): `local_idx` is the
// 9-bit packed voxel index, `face` is a 3-bit axis-aligned face direction,
// and `occ_slot` is an opaque pass-through of whatever index the caller
// used to locate this sub-chunk's occupancy record. Today the sub-chunk
// raster path routes the instance's 22-bit directory field through here;
// the `occ_slot` name leaves room for that to become a material-pool slot
// or any other packed identifier without the primitive's contract changing.
// The primitive is behaviour-identical to the prior
// `trace(ro, rd, occ_slot)` in `subchunk.ps.hlsl`; it is factored out here
// so the future compute secondary-ray pass can consume the same
// implementation.
//
// # Binding discipline
//
// This header defines types + pure functions only. The consuming shader
// must declare
//
//   struct SubchunkOcc { uint4 plane[4]; };
//   [[vk::binding(N, 0)]] StructuredBuffer<SubchunkOcc> g_occ_array;
//
// at a binding slot appropriate to its pipeline (matches the pattern used
// by `directory.hlsl`, which likewise leaves `g_directory` to the
// consumer).
//
// # Face-code convention (3 bits, 0..5)
//
// Matches `crates/voxel/src/render/direction.rs::Direction`:
//
//   0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z
//
// The hit face is the face of the occupied voxel that the ray entered
// through. A ray stepping `+X` into a new voxel enters through that
// voxel's `-X` face, so the emitted face direction is opposite the step
// sign on the crossing axis.
//
// Entry-cell hit (`t == 0`): when the ray origin already lies inside an
// occupied voxel — typically the rasterized hull entry point landing on
// the sub-chunk's outer face — the primitive infers the entry face from
// origin's distance to each face of the entry cell. For the common case
// where origin sits exactly on the sub-chunk's AABB face, this resolves
// to the matching axis-aligned face (e.g. ray entering the +Y face of
// the hull on an occupied top-row voxel reports `face = +Y`). Origins
// strictly interior to the entry cell (camera inside an occupied voxel,
// or a secondary ray starting on a surface) fall back to the closest
// face — a sensible heuristic, but the "which AABB face did the ray
// cross" semantics have decayed. Callers can still detect entry-cell
// hits via `t == 0`.

#ifndef RENDERER_DDA_HLSL
#define RENDERER_DDA_HLSL

// --- MarchResult — vis-buffer-aligned DDA output. ---
//
// `local_idx` : 9-bit packed voxel index, `x | (y << 3) | (z << 6)`;
//               unpacks with `& 7`, `(>> 3) & 7`, `(>> 6) & 7`.
// `face`      : 3-bit face direction (see header comment).
// `occ_slot`  : opaque pass-through of the caller's occupancy-record
//               identifier. The primitive never reads or reinterprets it
//               beyond indexing `g_occ_array`. The raster draw path
//               supplies the instance's 22-bit directory-style field here;
//               other callers are free to supply a material-pool slot or
//               any other packed identifier, so long as it indexes the
//               same occupancy buffer.
// `t`         : ray parameter at the hit; `origin + t * dir` is the hit
//               position in the caller's frame.
// `hit`       : `false` iff the ray exited the sub-chunk (or exceeded
//               `max_t`) without finding an occupied voxel. On miss, all
//               other fields are unspecified.
struct MarchResult {
    uint  occ_slot;
    uint  local_idx;
    uint  face;
    float t;
    bool  hit;
};

// --- Face-direction codes (match `Direction` enum discriminants). ---
static const uint DDA_FACE_POS_X = 0u;
static const uint DDA_FACE_NEG_X = 1u;
static const uint DDA_FACE_POS_Y = 2u;
static const uint DDA_FACE_NEG_Y = 3u;
static const uint DDA_FACE_POS_Z = 4u;
static const uint DDA_FACE_NEG_Z = 5u;

// --- Private: pick one uint component from a uint4 by index. ---
uint dda_pick_word(uint4 v, uint idx) {
    if (idx == 0u) return v.x;
    if (idx == 1u) return v.y;
    if (idx == 2u) return v.z;
    return v.w;
}

// --- Private: occupancy lookup for (x, y, z) inside `occ_slot`. ---
//
// Out-of-bounds coordinates return `false`. Storage mirrors the `Occupancy`
// layout documented in `face_mask.hlsl`: 8 XY planes packed as 16 u32s,
// grouped into four uint4s to dodge std140 element padding.
bool dda_occupied(int x, int y, int z, uint occ_slot) {
    if (x < 0 || x > 7 || y < 0 || y > 7 || z < 0 || z > 7)
        return false;
    uint bit  = uint(y * 8 + x);                   // 0..63 within the Z plane
    uint word = uint(z) * 2u + (bit >> 5u);        // which of the 16 u32s
    uint4 row = g_occ_array[occ_slot].plane[word >> 2u];
    uint  val = dda_pick_word(row, word & 3u);
    return (val >> (bit & 31u)) & 1u;
}

// --- Sub-chunk DDA entry point. ---
//
// `origin`   : ray start in the sub-chunk's local [0, 8]^3 frame.
// `dir`      : ray direction (world or local — same up to translation;
//              does not need to be unit-length for correctness, but `t`
//              is then in units of `|dir|`).
// `max_t`    : upper bound on the ray parameter; the primitive returns
//              `hit = false` if no occupied voxel is reached by
//              `t <= max_t`. Pass a large constant (e.g. `1e30`) for
//              "no bound".
// `occ_slot` : opaque index the primitive uses to locate this sub-chunk's
//              occupancy record in `g_occ_array`; copied into
//              `MarchResult::occ_slot` untouched.
//
// Worst-case traversal is 24 steps (8 cells along each of 3 axes) before
// the ray exits the sub-chunk.
MarchResult dda_sub_chunk(float3 origin, float3 dir, float max_t, uint occ_slot) {
    MarchResult miss;
    miss.occ_slot  = occ_slot;
    miss.local_idx = 0u;
    miss.face      = DDA_FACE_POS_X;
    miss.t         = 0.0;
    miss.hit       = false;

    // `origin` is treated as canonical — callers are responsible for
    // supplying an in-range start point. The raster hull path clamps
    // interpolated edge fragments at the call site; compute callers must
    // guarantee the ray has already been intersected against the sub-chunk
    // AABB. The only defensive step here is clamping the starting voxel
    // index so a slightly-out-of-range `origin` does not read past the
    // occupancy grid's first cell before the main-loop exit check can
    // catch it.
    int3 vox = clamp(int3(floor(origin)), int3(0, 0, 0), int3(7, 7, 7));

    int3 step;
    step.x = dir.x > 0.0 ? 1 : (dir.x < 0.0 ? -1 : 0);
    step.y = dir.y > 0.0 ? 1 : (dir.y < 0.0 ? -1 : 0);
    step.z = dir.z > 0.0 ? 1 : (dir.z < 0.0 ? -1 : 0);

    const float HUGE = 1e30;

    // Entry-face inference for the entry-cell hit. For each active axis
    // measure how far origin sits from that axis's "entering" face of
    // the entry cell; the axis with the minimum distance is the one the
    // ray crossed to land here. Inactive axes (step == 0) get HUGE so
    // they never win. Stepping +X enters a cell through its -X face,
    // matching the main loop's face emission below.
    float dx = (step.x > 0) ? abs(origin.x - float(vox.x))     :
               (step.x < 0) ? abs(origin.x - float(vox.x + 1)) : HUGE;
    float dy = (step.y > 0) ? abs(origin.y - float(vox.y))     :
               (step.y < 0) ? abs(origin.y - float(vox.y + 1)) : HUGE;
    float dz = (step.z > 0) ? abs(origin.z - float(vox.z))     :
               (step.z < 0) ? abs(origin.z - float(vox.z + 1)) : HUGE;

    uint entry_face;
    if (dx <= dy && dx <= dz) {
        entry_face = (step.x > 0) ? DDA_FACE_NEG_X : DDA_FACE_POS_X;
    } else if (dy <= dz) {
        entry_face = (step.y > 0) ? DDA_FACE_NEG_Y : DDA_FACE_POS_Y;
    } else {
        entry_face = (step.z > 0) ? DDA_FACE_NEG_Z : DDA_FACE_POS_Z;
    }

    // Per-axis t at the next cell boundary, and per-axis t increment per
    // full cell crossing. Axes with zero ray-direction component get HUGE
    // values so the min-select never picks them.
    float3 t_next, t_delta;

    if (step.x != 0) {
        float boundary = step.x > 0 ? float(vox.x + 1) : float(vox.x);
        t_next.x  = (boundary - origin.x) / dir.x;
        t_delta.x = abs(1.0 / dir.x);
    } else { t_next.x = HUGE; t_delta.x = HUGE; }

    if (step.y != 0) {
        float boundary = step.y > 0 ? float(vox.y + 1) : float(vox.y);
        t_next.y  = (boundary - origin.y) / dir.y;
        t_delta.y = abs(1.0 / dir.y);
    } else { t_next.y = HUGE; t_delta.y = HUGE; }

    if (step.z != 0) {
        float boundary = step.z > 0 ? float(vox.z + 1) : float(vox.z);
        t_next.z  = (boundary - origin.z) / dir.z;
        t_delta.z = abs(1.0 / dir.z);
    } else { t_next.z = HUGE; t_delta.z = HUGE; }

    // Test the entry cell first — the ray starts inside it.
    if (dda_occupied(vox.x, vox.y, vox.z, occ_slot)) {
        MarchResult h;
        h.occ_slot  = occ_slot;
        h.local_idx = uint(vox.x) | (uint(vox.y) << 3u) | (uint(vox.z) << 6u);
        h.face      = entry_face;
        h.t         = 0.0;
        h.hit       = true;
        return h;
    }

    // 8 cells × 3 axes = 24 step bound.
    //
    // `t_entry` is the ray parameter at which we crossed INTO the current
    // voxel — i.e., the min of `t_next` BEFORE the axis step that produced
    // it. Callers can reconstruct the hit position as `origin + t * dir`.
    float t_entry = 0.0;
    uint  face    = DDA_FACE_POS_X;

    [loop]
    for (int i = 0; i < 24; i++) {
        if (t_next.x < t_next.y) {
            if (t_next.x < t_next.z) {
                t_entry   = t_next.x;
                vox.x    += step.x;
                t_next.x += t_delta.x;
                // Stepped along +X (step.x > 0) → entered new voxel
                // through its -X face, and vice versa.
                face = (step.x > 0) ? DDA_FACE_NEG_X : DDA_FACE_POS_X;
            } else {
                t_entry   = t_next.z;
                vox.z    += step.z;
                t_next.z += t_delta.z;
                face = (step.z > 0) ? DDA_FACE_NEG_Z : DDA_FACE_POS_Z;
            }
        } else {
            if (t_next.y < t_next.z) {
                t_entry   = t_next.y;
                vox.y    += step.y;
                t_next.y += t_delta.y;
                face = (step.y > 0) ? DDA_FACE_NEG_Y : DDA_FACE_POS_Y;
            } else {
                t_entry   = t_next.z;
                vox.z    += step.z;
                t_next.z += t_delta.z;
                face = (step.z > 0) ? DDA_FACE_NEG_Z : DDA_FACE_POS_Z;
            }
        }

        // Exit if the ray has left the sub-chunk or exceeded the caller's
        // parameter bound.
        if (any(vox < int3(0, 0, 0)) || any(vox > int3(7, 7, 7)))
            break;
        if (t_entry > max_t)
            break;

        if (dda_occupied(vox.x, vox.y, vox.z, occ_slot)) {
            MarchResult h;
            h.occ_slot  = occ_slot;
            h.local_idx = uint(vox.x) | (uint(vox.y) << 3u) | (uint(vox.z) << 6u);
            h.face      = face;
            h.t         = t_entry;
            h.hit       = true;
            return h;
        }
    }

    return miss;
}

// --- World-space DDA entry point (compute-callable). ---
//
// Only available when `directory.hlsl` has been included before this file —
// `dda_world` depends on `DirEntry`, `g_directory`, `direntry_is_resident`,
// and `resolve_and_verify`, all of which live in `directory.hlsl`.
// `subchunk.ps.hlsl` includes `dda.hlsl` without `directory.hlsl` (to avoid
// pulling g_consts into its pipeline layout), so the guard keeps it clean.
// Consumers that need `dda_world` (e.g. `subchunk_shade.cs.hlsl`) must
// `#include "directory.hlsl"` before `#include "dda.hlsl"`.
#ifdef RENDERER_DIRECTORY_HLSL

// Walks a world-space ray across the sub-chunk grid at a single LOD level
// using a two-level Amanatides-Woo DDA: an *outer* step that advances
// through integer sub-chunk cells, and — for each resident cell — an *inner*
// descent into `dda_sub_chunk` to test the voxel occupancy bitmap.
//
// # Purpose
//
// `dda_sub_chunk` is limited to a ray that already starts inside one
// sub-chunk. Secondary rays (shadow, GI, reflections) start at an arbitrary
// world-space surface point and must cross multiple sub-chunk cells before
// finding an occluder. `dda_world` provides that outer traversal.
//
// # Outer+inner composition and scaling invariant
//
// Each outer cell spans `sc_size = 8 * voxel_size` world units, where
// `voxel_size = 1 << level_idx`. On entry to a resident cell we compute:
//
//   local_origin = (origin_ws + t_outer * dir_ws - cell_corner_ws) / voxel_size
//   local_dir    = dir_ws / voxel_size
//
// With `|dir_ws| = 1` (required — see caller contract below):
//   `|local_dir| = 1 / voxel_size`
//
// Inside `dda_sub_chunk`, `t_delta.x = abs(1 / local_dir.x) =
// voxel_size / abs(dir_ws.x)`. For an axis-aligned ray (+X only),
// one full voxel crossing adds `t = voxel_size`, which equals the world
// distance traveled (one voxel = `voxel_size` world units). Therefore
// the `t` returned by `dda_sub_chunk` is directly in world units, and
// the hit world-position is `origin_ws + (t_outer + t_inner) * dir_ws`.
//
// # Torus-collision check
//
// `resolve_coord_to_slot` reduces the query coord modulo `pool_dims`. A
// secondary ray that wanders past the toroidal ring can alias onto a
// resident sub-chunk that belongs to a completely different world position.
// `resolve_and_verify` (from `directory.hlsl`) guards against this by
// comparing the stored `coord` against the query — a mismatching entry is
// treated as non-resident. This check is *load-bearing* for secondary rays
// but would be a redundant re-validation for primary hits (which come from
// a cull-survived instance carrying its own verified slot).
//
// # Binding requirement
//
// The consumer must also `#include "directory.hlsl"` (which brings in
// `g_consts` at slot 0 via `gpu_consts.hlsl` and the directory accessor
// functions) and declare:
//
//   [[vk::binding(N, 0)]] StructuredBuffer<DirEntry> g_directory;
//
// `dda.hlsl` cannot include `directory.hlsl` itself because today's
// fragment-shader callers (`subchunk.ps.hlsl`) do not bind the directory —
// it would pull in a binding slot those pipelines never fill, breaking
// SPIR-V reflection. The include discipline stays at the consumer.
//
// `g_occ_array` (for the inner DDA) must also be declared by the consumer
// per the existing binding discipline documented in the file header above.
//
// # Caller contract
//
// `dir_ws` must be unit-normalised. The scaling invariant above relies on
// `|dir_ws| = 1`; a non-unit direction produces `t` values that do not
// equal world distance. No defensive re-normalise is performed here — the
// caller is responsible for enforcing this invariant.
//
// Parameters:
//   origin_ws  — ray start in world space.
//   dir_ws     — ray direction, unit-normalised.
//   max_t_ws   — world-space parameter cap; `hit = false` if no occluder
//                is reached within this distance.
//   level_idx  — LOD level; selects voxel_size and the directory ring.
//
// Returns a `MarchResult` in world units. On miss, `hit = false` and all
// other fields are unspecified. The caller interprets a miss as "ray
// reached max_t_ws without finding an occupied voxel."
MarchResult dda_world(float3 origin_ws, float3 dir_ws, float max_t_ws, uint level_idx) {
    MarchResult miss;
    miss.occ_slot  = 0u;
    miss.local_idx = 0u;
    miss.face      = DDA_FACE_POS_X;
    miss.t         = 0.0;
    miss.hit       = false;

    float voxel_size = float(1u << level_idx);
    float sc_size    = 8.0 * voxel_size;   // world units per sub-chunk cell

    // Starting outer cell (integer sub-chunk coords at this LOD).
    int3 cell = int3(floor(origin_ws / sc_size));

    // Outer DDA step signs.
    int3 step;
    step.x = dir_ws.x > 0.0 ? 1 : (dir_ws.x < 0.0 ? -1 : 0);
    step.y = dir_ws.y > 0.0 ? 1 : (dir_ws.y < 0.0 ? -1 : 0);
    step.z = dir_ws.z > 0.0 ? 1 : (dir_ws.z < 0.0 ? -1 : 0);

    const float HUGE = 1e30;

    // Per-axis t at the next outer cell boundary, and per-axis increment
    // per full outer-cell crossing (= sc_size / |dir_ws[axis]|).
    // Inactive axes get HUGE so they never win the min-select.
    float3 t_next, t_delta;

    if (step.x != 0) {
        float boundary = step.x > 0 ? float(cell.x + 1) * sc_size
                                     : float(cell.x)     * sc_size;
        t_next.x  = (boundary - origin_ws.x) / dir_ws.x;
        t_delta.x = abs(sc_size / dir_ws.x);
    } else { t_next.x = HUGE; t_delta.x = HUGE; }

    if (step.y != 0) {
        float boundary = step.y > 0 ? float(cell.y + 1) * sc_size
                                     : float(cell.y)     * sc_size;
        t_next.y  = (boundary - origin_ws.y) / dir_ws.y;
        t_delta.y = abs(sc_size / dir_ws.y);
    } else { t_next.y = HUGE; t_delta.y = HUGE; }

    if (step.z != 0) {
        float boundary = step.z > 0 ? float(cell.z + 1) * sc_size
                                     : float(cell.z)     * sc_size;
        t_next.z  = (boundary - origin_ws.z) / dir_ws.z;
        t_delta.z = abs(sc_size / dir_ws.z);
    } else { t_next.z = HUGE; t_delta.z = HUGE; }

    // Accumulated outer-DDA t (world units to the current cell's entry).
    float t_outer = 0.0;

    // Cap at 128 outer steps to prevent runaway walks on long empty
    // stretches (e.g. a ray fired into open sky). Same [loop] pattern as
    // `dda_sub_chunk` to hint the driver this is a non-trivial loop.
    [loop]
    for (int i = 0; i < 128; i++) {
        if (t_outer > max_t_ws) {
            break;
        }

        // Resolve the current cell. `resolve_and_verify` returns true only
        // when the entry's stored coord matches `cell` — guards against
        // torus aliasing for secondary rays.
        uint slot;
        bool resident = resolve_and_verify(cell, level_idx, g_directory, slot);

        if (resident) {
            DirEntry e = g_directory[slot];
            if (direntry_is_resident(e)) {
                // Descend into inner DDA.
                float3 cell_corner_ws = float3(cell) * sc_size;

                // Convert origin to sub-chunk local [0,8]^3 frame.
                // The starting point is the world entry of this outer cell
                // (origin_ws + t_outer * dir_ws); we translate to the
                // cell corner and scale by 1/voxel_size.
                float3 local_origin = (origin_ws + t_outer * dir_ws - cell_corner_ws)
                                    / voxel_size;

                // Scale dir so t from dda_sub_chunk is in world units
                // (see scaling invariant in header comment).
                float3 local_dir    = dir_ws / voxel_size;

                float max_t_local   = max_t_ws - t_outer;

                MarchResult inner = dda_sub_chunk(local_origin, local_dir, max_t_local, slot);
                if (inner.hit) {
                    inner.t += t_outer;   // convert to world-space t
                    return inner;
                }
            }
        }

        // Advance to the next outer cell on the axis with the smallest t_next.
        if (t_next.x < t_next.y) {
            if (t_next.x < t_next.z) {
                t_outer   = t_next.x;
                cell.x   += step.x;
                t_next.x += t_delta.x;
            } else {
                t_outer   = t_next.z;
                cell.z   += step.z;
                t_next.z += t_delta.z;
            }
        } else {
            if (t_next.y < t_next.z) {
                t_outer   = t_next.y;
                cell.y   += step.y;
                t_next.y += t_delta.y;
            } else {
                t_outer   = t_next.z;
                cell.z   += step.z;
                t_next.z += t_delta.z;
            }
        }
    }

    return miss;
}


#endif // RENDERER_DIRECTORY_HLSL

#endif // RENDERER_DDA_HLSL
