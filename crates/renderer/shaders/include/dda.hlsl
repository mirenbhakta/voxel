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

#endif // RENDERER_DDA_HLSL
