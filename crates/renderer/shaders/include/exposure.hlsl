// exposure.hlsl — shared neighbour-aware 6-bit exposure-mask computation.
//
// Both `subchunk_prep.cs.hlsl` (full-prep dispatch) and
// `subchunk_exposure.cs.hlsl` (exposure-only refresh dispatch) compute an
// identical 6-bit exposure mask from the same 4-case neighbour classification.
// Hoisting the loop into a shared header keeps the two dispatches
// bit-for-bit consistent — a per-face mask divergence between them would
// oscillate exposure bits across frames and produce visible flicker on
// slot-retired sub-chunks.
//
// The helpers here depend on the consumer having already included
// `directory.hlsl` (for `DirEntry`, `direntry_*` accessors, and
// `resolve_and_verify`) and `face_mask.hlsl` (for `face_*` extractors and
// `face_exposed`), and on the consumer declaring `Occupancy` + the two
// buffers the helper reads through as globals:
//
//   StructuredBuffer<DirEntry>  g_directory;
//   StructuredBuffer<Occupancy> g_material_pool;
//
// Kept header-only (no bindings) so each shader owns its own
// [[vk::binding(...)]] slots.

#ifndef RENDERER_EXPOSURE_HLSL
#define RENDERER_EXPOSURE_HLSL

// Result of a neighbour-face lookup. `resolved` distinguishes "the
// neighbour does not exist at this level" (caller may fall back to a
// different LOD via super/sub-sample) from "the neighbour exists and is
// uniformly empty" (face is authoritatively zero). The previous shape —
// returning `uint2(0)` for both — collapsed the two cases together,
// which was correct under single-LOD but is wrong as soon as cross-LOD
// fallback enters the picture: a non-resident same-level neighbour
// would suppress the search for a coarser one and fall through to
// "treat as empty", which over-stores material at LOD seams (the void-
// adjacent symptom step 3 of `decision-subchunk-visibility-storage-
// here-and-there` is meant to retire).
struct NeighbourFace {
    uint2 face;
    bool  resolved;
};

// Extract the neighbour's face plane that faces this sub-chunk from the
// direction `dir_idx` (0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z — matching the
// cull shader's bit order).
//
// Classification (priority order, matches
// `decision-world-streaming-architecture` §Intra-batch neighbour
// conservatism):
//   * Not resident OR coord mismatch → unresolved. Caller decides
//     whether to fall back to coarser/finer LOD (see
//     `synthesize_coarser_face`); a step-3 cross-LOD lookup is the
//     correct treatment for "neighbour position falls outside this
//     level's shell" (the L_n outer-edge case under nested shells).
//   * is_solid                        → all-ones (every face bit set).
//   * has no material slot            → uniform-empty, all zero —
//                                       resolved (no fallback).
//   * sparse                          → read from g_material_pool and
//                                       extract the opposing face.
NeighbourFace neighbour_opposing_face(
    int3 nbr_coord,
    uint level_idx,
    uint dir_idx
) {
    NeighbourFace result;
    result.face     = uint2(0u, 0u);
    result.resolved = false;

    uint nbr_slot;
    bool coord_ok = resolve_and_verify(
        nbr_coord, level_idx, g_directory, nbr_slot
    );
    if (!coord_ok) {
        return result;
    }

    result.resolved = true;
    DirEntry d = g_directory[nbr_slot];
    if (direntry_is_solid(d)) {
        result.face = uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
        return result;
    }
    if (!direntry_has_material(d)) {
        // Resident, uniform-empty: face is authoritatively zero.
        return result;
    }

    uint mslot = direntry_get_material_slot(d);
    Occupancy nbr = g_material_pool[mslot];

    // We are testing exposure in direction `dir_idx` from *our* side; the
    // neighbour's face that meets ours is the opposite direction.
    //   dir 0 (-X) → nbr's +X face
    //   dir 1 (+X) → nbr's -X face
    //   dir 2 (-Y) → nbr's +Y face
    //   dir 3 (+Y) → nbr's -Y face
    //   dir 4 (-Z) → nbr's +Z face
    //   dir 5 (+Z) → nbr's -Z face
    switch (dir_idx) {
        case 0: result.face = face_pX(nbr); break;
        case 1: result.face = face_nX(nbr); break;
        case 2: result.face = face_pY(nbr); break;
        case 3: result.face = face_nY(nbr); break;
        case 4: result.face = face_pZ(nbr); break;
        default: result.face = face_nZ(nbr); break;  // case 5
    }
    return result;
}

// Read the bit at within-subchunk cell (x, y, z) of an `Occupancy`. Mirror
// of the layout doc in `face_mask.hlsl`: bit `y*8 + x` of word
// `z*2 + (y >= 4 ? 1 : 0)` — equivalently, plane[z>>1] is a uint4 with
// .x/.y holding z=2k word 0/1 and .z/.w holding z=2k+1 word 0/1, and
// within each word the y-row at offset `(y & 3) * 8` carries the 8 x-bits.
//
// Used by the cross-LOD super-sample path to read individual L_{n+1}
// cells out of a coarser sub-chunk's Occupancy without having to
// extract a full face plane.
uint read_cell_bit(Occupancy occ, uint x, uint y, uint z) {
    uint4 p = occ.plane[z >> 1u];
    uint  w = (z & 1u) == 0u
            ? ((y < 4u) ? p.x : p.y)
            : ((y < 4u) ? p.z : p.w);
    uint  bit_pos = (y & 3u) * 8u + x;
    return (w >> bit_pos) & 1u;
}

// Step 3 of `decision-subchunk-visibility-storage-here-and-there`:
// super-sample fallback when the same-level neighbour does not exist
// (typically the outer edge of an L_n shell, where the L_n nbr coord
// falls outside the L_n shell into L_{n+1} territory).
//
// Replicates each L_{n+1} cell's solidity bit across the corresponding
// 2×2 L_n cells on my-level face, emitting a canonical-packed `uint2`
// face that can be passed to `face_exposed` exactly like a same-level
// neighbour face. Conservative under the OR-reductive hierarchy: by the
// load-bearing invariant from step 2 (an L_{n+1} cell empty ⇒ all 8
// covered L_n cells empty), super-sampled bits never mark an actually-
// empty cell as solid in a way that produces voids. The reverse
// (replicating a coarser-solid bit when only some of the underlying L_n
// cells are solid) over-stores at the LOD seam — that is the explicit
// trade the design accepts.
//
// Returns `uint2(0, 0)` if the coarser sub-chunk is non-resident or
// torus-mismatched; the caller treats that as "no neighbour", which
// degrades to the same conservative path the single-LOD code took
// before this step.
//
// # Coordinate mapping
//
// Each L_{n+1} sub-chunk in world space covers 2×2×2 L_n sub-chunks.
// The L_{n+1} sub-chunk containing my nbr position is at coarser
// sub-chunk coord `nbr_coord_Ln >> 1` per axis (arithmetic shift, so
// floor-div semantics for negatives). Within that L_{n+1} sub-chunk's
// 8×8×8 L_{n+1} cell grid, my nbr's L_n sub-chunk occupies a 4×4×4 sub-
// region offset by `(nbr_coord_Ln & 1) * 4` per axis (the parity of the
// L_n coord picks the low or high half on each axis — the bitwise `& 1`
// gives the right parity for negatives under two's-complement).
//
// On nbr's opposing face along axis d, the d-axis L_n within-cell coord
// is pinned to 0 (when nbr's face is `-d`, my dir is `+d`) or 7 (when
// nbr's face is `+d`, my dir is `-d`); after the parity offset and the
// `>> 1` (L_n → L_{n+1}) those land at coarser within-d-axis
// `parity*4 + 0` or `parity*4 + 3` respectively.
//
// The remaining two free axes vary over `[0, 8)` on the nbr face; each
// L_n free-axis cell maps to L_{n+1} within-cell `parity_axis*4 + (cell
// >> 1)`. Reading the resulting bit out of `Occupancy` and packing it
// into the canonical face position `(b & 3) * 8 + a` reconstructs the
// super-sampled face directly in the layout `face_exposed` consumes.
uint2 synthesize_coarser_face(
    int3 nbr_coord_Ln,
    uint my_level,
    uint dir_idx
) {
    int3 coarser_coord = int3(
        nbr_coord_Ln.x >> 1,
        nbr_coord_Ln.y >> 1,
        nbr_coord_Ln.z >> 1
    );
    uint coarser_level = my_level + 1u;
    uint coarser_slot;
    if (!resolve_and_verify(
            coarser_coord, coarser_level, g_directory, coarser_slot)) {
        return uint2(0u, 0u);
    }

    DirEntry e = g_directory[coarser_slot];
    if (direntry_is_solid(e)) {
        return uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
    }
    if (!direntry_has_material(e)) {
        return uint2(0u, 0u);
    }
    uint mslot = direntry_get_material_slot(e);
    Occupancy nbr = g_material_pool[mslot];

    // Parity offsets — see the coordinate-mapping section in the
    // function docstring. `& 1` on a signed int gives the correct
    // two's-complement parity for negative coords (-3 & 1 = 1, ✓).
    uint px = uint(nbr_coord_Ln.x & 1) * 4u;
    uint py = uint(nbr_coord_Ln.y & 1) * 4u;
    uint pz = uint(nbr_coord_Ln.z & 1) * 4u;

    // d-axis position on the nbr's opposing face in L_n within-coords.
    // Direction parity: even dir_idx (0,2,4) means I'm looking in -axis,
    // so nbr's facing side is +axis (cell 7); odd dir_idx (1,3,5) means
    // I'm looking in +axis, so nbr's facing side is -axis (cell 0).
    uint d_within_nbr =
        (dir_idx == 0u || dir_idx == 2u || dir_idx == 4u) ? 7u : 0u;

    uint2 face = uint2(0u, 0u);
    [loop] for (uint b = 0u; b < 8u; ++b) {
        [loop] for (uint a = 0u; a < 8u; ++a) {
            // Map (a, b) on nbr's opposing face to its L_n within-coords.
            // Free-axis assignments match `face_mask.hlsl`'s canonical
            // layout: ±X → (a=y, b=z); ±Y → (a=x, b=z); ±Z → (a=x, b=y).
            uint nx, ny, nz;
            if (dir_idx == 0u || dir_idx == 1u) {
                nx = d_within_nbr; ny = a; nz = b;
            } else if (dir_idx == 2u || dir_idx == 3u) {
                nx = a; ny = d_within_nbr; nz = b;
            } else {
                nx = a; ny = b; nz = d_within_nbr;
            }
            // L_n within-coord → L_{n+1} within-coord (within the coarser
            // sub-chunk).
            uint cx = px + (nx >> 1u);
            uint cy = py + (ny >> 1u);
            uint cz = pz + (nz >> 1u);
            uint bit = read_cell_bit(nbr, cx, cy, cz);
            if (bit != 0u) {
                uint pos = (b & 3u) * 8u + a;
                if (b < 4u) { face.x |= (1u << pos); }
                else        { face.y |= (1u << pos); }
            }
        }
    }
    return face;
}

// Resolve neighbour face at the appropriate LOD with same-level →
// coarser super-sample fallback. This is the cross-LOD frontend used by
// `compute_exposure_mask`; encapsulates the priority order so the loop
// over directions stays tight.
//
// Sub-sample (finer-LOD fallback) is intentionally NOT implemented here.
// Under the current nested-shell topology (every L_{n-1} shell is fully
// contained in its L_n shell — see `decision-lod-nested-shells-
// hierarchical-occupancy`), the same-level neighbour always exists at
// an L_n sub-chunk's *inner* boundary against L_{n-1} (the L_n entry is
// resident, just culled from rendering), so the finer-LOD fallback case
// never fires. The decision doc lists sub-sample for completeness;
// adding it without a topology change would be dead code. Revisit if
// the shell layout ever becomes annular or per-level non-nested.
uint2 resolve_neighbour_face(
    int3 nbr_coord_Ln,
    uint my_level,
    uint dir_idx
) {
    NeighbourFace same = neighbour_opposing_face(nbr_coord_Ln, my_level, dir_idx);
    if (same.resolved) {
        return same.face;
    }
    if (my_level + 1u < g_consts.level_count) {
        return synthesize_coarser_face(nbr_coord_Ln, my_level, dir_idx);
    }
    // Coarsest level — no fallback. Conservative empty (matches the
    // pre-step-3 single-LOD behaviour at the outer-most shell edge).
    return uint2(0u, 0u);
}

// Compute the 6-bit neighbour-aware exposure mask for a sub-chunk with
// occupancy `me` at world-space sub-chunk coord `self_coord` and LOD level
// `level_idx`. Bit order matches the cull shader: 0=-X, 1=+X, 2=-Y, 3=+Y,
// 4=-Z, 5=+Z.
//
// Callers that know `me` is fully empty should skip the call — every bit
// is zero regardless of neighbour occupancy. Full-prep already filters
// this case; exposure-only refresh never gets here for non-resident
// sub-chunks.
uint compute_exposure_mask(Occupancy me, int3 self_coord, uint level_idx) {
    int3 offsets[6] = {
        int3(-1,  0,  0),
        int3( 1,  0,  0),
        int3( 0, -1,  0),
        int3( 0,  1,  0),
        int3( 0,  0, -1),
        int3( 0,  0,  1),
    };

    // Interior cell-pair contribution. For each direction d, OR-fold
    // `here & ~there` across the 7 internal interfaces along axis d
    // (`include/face_mask.hlsl`). Catches sub-chunks whose only exposed
    // faces are interior — e.g. a heightfield surface that crests inside
    // the sub-chunk and never reaches the y=7 plane. Pure self-occupancy:
    // no neighbour reads, no level dependency.
    uint interior[6];
    interior[0] = interior_exposed_nX(me);
    interior[1] = interior_exposed_pX(me);
    interior[2] = interior_exposed_nY(me);
    interior[3] = interior_exposed_pY(me);
    interior[4] = interior_exposed_nZ(me);
    interior[5] = interior_exposed_pZ(me);

    // Boundary contribution: my outer face plane vs. the neighbour's
    // opposing face plane. Only fetched when interior didn't already
    // fire — saves up to 6 neighbour reads per sub-chunk on geometry
    // that has any interior surface.
    uint2 my_faces[6];
    my_faces[0] = face_nX(me);
    my_faces[1] = face_pX(me);
    my_faces[2] = face_nY(me);
    my_faces[3] = face_pY(me);
    my_faces[4] = face_nZ(me);
    my_faces[5] = face_pZ(me);

    uint exposure = 0u;
    [unroll] for (uint d = 0u; d < 6u; ++d) {
        bool exposed = (interior[d] != 0u);
        if (!exposed) {
            int3 nbr_coord = self_coord + offsets[d];
            // Step 3 frontend: same-level → coarser super-sample. Returns
            // a canonical-packed face usable directly with `face_exposed`,
            // regardless of which LOD the neighbour was found at.
            uint2 nbr_face = resolve_neighbour_face(nbr_coord, level_idx, d);
            exposed = face_exposed(my_faces[d], nbr_face);
        }
        if (exposed) {
            exposure |= 1u << d;
        }
    }
    return exposure;
}

#endif // RENDERER_EXPOSURE_HLSL
