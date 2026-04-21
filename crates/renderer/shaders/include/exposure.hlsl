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

// Sub-sample fallback for the *inner* boundary of the L_n shell: where
// the L_{n-1} shell covers the neighbour cell, L_{n-1} sub-chunks are the
// ones actually rasterising the visible surface, not the L_n OR-reduction
// that `neighbour_opposing_face` would surface. Reading the L_n face there
// under-reports exposure — a coarse face bit is 1 whenever *any* fine
// child is solid, yet the fine rasterisation may still have gaps through
// which my L_n face is visible, and those gaps should re-expose my face.
//
// Builds a synthesised L_n-resolution face by AND-reducing each 2×2 block
// of L_{n-1} face cells. The polarity is inverse to `synthesize_coarser_
// face`: the coarse OR-reduction says "solid if any fine is solid" (right
// for primary visibility), but for my-face exposure we need "coarse face
// blocks me iff *every* fine cover is solid" — the AND. A single empty
// fine cell in the 2×2 leaves a gap, which is what the exposure test must
// see.
//
// Returns `false` (and leaves `face` zeroed) when `my_level == 0` (no
// finer level) or when any of the four face-plane L_{n-1} sub-chunks
// fails `resolve_and_verify`. The all-four-resident requirement matches
// the nested-shell invariant: inside the L_{n-1} shell, all four are
// resident; outside it, none are. The partial-coverage case (shell edge
// crossing an L_n face) is not reachable under axis-aligned nested shells,
// so the all-or-nothing check is complete for the current topology.
//
// # Coordinate mapping
//
// Each L_n sub-chunk covers 2×2×2 L_{n-1} sub-chunks. The four L_{n-1}
// sub-chunks on the face plane of my neighbour (the ones whose own face
// meets mine) are at coord `nbr_coord_Ln * 2 + axis_offset_{along_axis}
// + (i, j)_{free_axes}` for i, j ∈ {0, 1}. `axis_offset = 1 - (dir_idx &
// 1)` pins the L_{n-1} sub-chunk to the side of the L_n neighbour that
// meets me. Within each L_{n-1} sub-chunk, the row/column of face cells
// meeting me is at within-axis coord `0` or `7` (same polarity).
//
// Each L_n face cell (a, b) ∈ [0, 8)² covers L_{n-1} face cells at
// (2a..2a+1, 2b..2b+1). Those four fine cells all belong to the single
// L_{n-1} sub-chunk at (i = a >> 2, j = b >> 2) — since each L_{n-1}
// contributes 8 L_{n-1} face cells = 4 L_n face cells along each free
// axis. Within that sub-chunk, the four cells are at within-free-axis
// coords ((a & 3) << 1, (a & 3) << 1 + 1) × ((b & 3) << 1, (b & 3) << 1
// + 1).
bool sub_sample_finer_face(
    int3 nbr_coord_Ln,
    uint my_level,
    uint dir_idx,
    out uint2 face
) {
    face = uint2(0u, 0u);
    if (my_level == 0u) {
        return false;
    }

    uint finer_level = my_level - 1u;
    int3 base        = nbr_coord_Ln * 2;
    uint axis        = dir_idx >> 1u;            // 0=X, 1=Y, 2=Z
    uint axis_offset = 1u - (dir_idx & 1u);      // 1 on -dir (nbr's +axis face), 0 on +dir
    uint within_axis = axis_offset == 1u ? 7u : 0u;

    // Resolve all four face-plane finer sub-chunks, keyed by (i, j) ∈
    // {0, 1}² mapping onto the two free axes of this direction. The
    // `i * 2 + j` flattening avoids HLSL's patchy support for
    // multi-dimensional arrays of user-defined structs.
    DirEntry fe[4];
    [unroll] for (uint i = 0u; i < 2u; ++i) {
        [unroll] for (uint j = 0u; j < 2u; ++j) {
            int3 off;
            if (axis == 0u) {
                off = int3(int(axis_offset), int(i), int(j));
            } else if (axis == 1u) {
                off = int3(int(i), int(axis_offset), int(j));
            } else {
                off = int3(int(i), int(j), int(axis_offset));
            }
            int3 finer_coord = base + off;
            uint finer_slot;
            if (!resolve_and_verify(
                    finer_coord, finer_level, g_directory, finer_slot)) {
                return false;
            }
            fe[i * 2u + j] = g_directory[finer_slot];
        }
    }

    // Build the 8×8 L_n face by AND-reducing 2×2 L_{n-1} face cells into
    // each L_n face bit.
    [loop] for (uint b = 0u; b < 8u; ++b) {
        [loop] for (uint a = 0u; a < 8u; ++a) {
            uint     i = a >> 2u;
            uint     j = b >> 2u;
            DirEntry e = fe[i * 2u + j];

            uint all_solid;
            if (direntry_is_solid(e)) {
                all_solid = 1u;
            } else if (!direntry_has_material(e)) {
                all_solid = 0u;
            } else {
                uint      mslot = direntry_get_material_slot(e);
                Occupancy o     = g_material_pool[mslot];
                uint      wa0   = (a & 3u) << 1u;
                uint      wb0   = (b & 3u) << 1u;
                uint      acc   = 1u;
                [unroll] for (uint db = 0u; db < 2u; ++db) {
                    [unroll] for (uint da = 0u; da < 2u; ++da) {
                        uint wa = wa0 + da;
                        uint wb = wb0 + db;
                        uint cx, cy, cz;
                        if (axis == 0u) {
                            cx = within_axis; cy = wa; cz = wb;
                        } else if (axis == 1u) {
                            cx = wa; cy = within_axis; cz = wb;
                        } else {
                            cx = wa; cy = wb; cz = within_axis;
                        }
                        if (read_cell_bit(o, cx, cy, cz) == 0u) {
                            acc = 0u;
                        }
                    }
                }
                all_solid = acc;
            }

            if (all_solid != 0u) {
                uint pos = (b & 3u) * 8u + a;
                if (b < 4u) { face.x |= (1u << pos); }
                else        { face.y |= (1u << pos); }
            }
        }
    }
    return true;
}

// Resolve the neighbour face at the resolution actually rendered there.
// Priority:
//
//   1. Finer (sub-sample, `my_level > 0`) — the *inner* L_n boundary
//      case. When the L_{n-1} shell covers this neighbour cell, L_{n-1}
//      is rasterising it, and the L_n entry (resident for OR-reduction)
//      is not what's on screen. Read L_{n-1} directly. See
//      `sub_sample_finer_face` for the AND-reduction rationale.
//   2. Same-level — the typical case: L_n neighbour is in the L_n shell
//      but outside the L_{n-1} shell, so L_n is what draws there.
//   3. Coarser (super-sample) — the *outer* L_n boundary case, where the
//      L_n neighbour is outside the L_n shell and L_{n+1} takes over.
//
// Why finer is checked first and not as a fallback: at the inner
// boundary the L_n neighbour is resident, so `neighbour_opposing_face`
// returns `resolved=true` with the stale OR-reduced face. A fallback
// ordering would never reach the finer path. The cost of the leading
// check away from the inner boundary is a single `resolve_and_verify` on
// an L_{n-1} coord that's outside the L_{n-1} shell — it fails and we
// fall straight through to same-level.
uint2 resolve_neighbour_face(
    int3 nbr_coord_Ln,
    uint my_level,
    uint dir_idx
) {
    // Inner-boundary path: prefer L_{n-1} when all four face-plane finer
    // sub-chunks are resident.
    if (my_level > 0u) {
        uint2 finer_face;
        if (sub_sample_finer_face(nbr_coord_Ln, my_level, dir_idx, finer_face)) {
            return finer_face;
        }
    }

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
