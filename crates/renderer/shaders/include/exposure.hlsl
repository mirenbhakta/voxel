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

// Extract the neighbour's face plane that faces this sub-chunk from the
// direction `dir_idx` (0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z — matching the
// cull shader's bit order).
//
// Classification (priority order, matches
// `decision-world-streaming-architecture` §Intra-batch neighbour
// conservatism):
//   * Not resident OR coord mismatch → empty (conservative). An
//     exposure-only dispatch of *this* sub-chunk filed by a neighbour
//     that has since been evicted still emits a dirty entry; the
//     coord-mismatch early-out keeps us from dereferencing a stale
//     material slot the evicted neighbour no longer owns.
//   * is_solid                        → all-ones (every face bit set).
//   * has no material slot            → uniform-empty, all zero.
//   * sparse                          → read from g_material_pool and
//                                       extract the opposing face.
uint2 neighbour_opposing_face(
    int3 nbr_coord,
    uint level_idx,
    uint dir_idx
) {
    uint nbr_slot;
    bool coord_ok = resolve_and_verify(
        nbr_coord, level_idx, g_directory, nbr_slot
    );
    if (!coord_ok) {
        return uint2(0u, 0u);
    }

    DirEntry d = g_directory[nbr_slot];
    if (direntry_is_solid(d)) {
        return uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
    }
    if (!direntry_has_material(d)) {
        return uint2(0u, 0u);
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
        case 0: return face_pX(nbr);
        case 1: return face_nX(nbr);
        case 2: return face_pY(nbr);
        case 3: return face_nY(nbr);
        case 4: return face_pZ(nbr);
        default: return face_nZ(nbr);  // case 5
    }
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

    uint2 my_faces[6];
    my_faces[0] = face_nX(me);
    my_faces[1] = face_pX(me);
    my_faces[2] = face_nY(me);
    my_faces[3] = face_pY(me);
    my_faces[4] = face_nZ(me);
    my_faces[5] = face_pZ(me);

    uint exposure = 0u;
    [unroll] for (uint d = 0u; d < 6u; ++d) {
        int3 nbr_coord = self_coord + offsets[d];
        uint2 nbr_face = neighbour_opposing_face(nbr_coord, level_idx, d);
        if (face_exposed(my_faces[d], nbr_face)) {
            exposure |= 1u << d;
        }
    }
    return exposure;
}

#endif // RENDERER_EXPOSURE_HLSL
