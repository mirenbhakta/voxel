// face_mask.hlsl — per-face bit extraction from an 8×8×8 Occupancy.
//
// Step 4 uses these helpers to compute a 6-bit neighbour-aware exposure
// mask: a direction bit is set iff any voxel on this sub-chunk's face at
// that direction is solid AND the corresponding neighbour voxel across
// that face is empty. The per-face extractors produce a canonical view of
// the 64-bit boundary plane, packed into a `uint2`, so that
// `my_face & ~nbr_face` is a correct bit-for-bit comparison of the voxel
// pairs that meet across the face.
//
// # Occupancy layout
//
// Occupancy.plane[] is a `uint4[4]` covering 16 × u32 for the 8×8×8 voxel
// grid. Bit `y * 8 + x` of word `z * 2 + (bit >> 5)` is set when voxel
// (x, y, z) is occupied. Concretely:
//
//   plane[0].x → z=0, word 0 (bits y*8+x, y∈[0,4), x∈[0,8))
//   plane[0].y → z=0, word 1 (bits (y-4)*8+x, y∈[4,8), x∈[0,8))
//   plane[0].z → z=1, word 0
//   plane[0].w → z=1, word 1
//   plane[1].x → z=2, word 0   ...
//   plane[3].z → z=7, word 0
//   plane[3].w → z=7, word 1
//
// # Canonical face layout (CRITICAL — the correctness contract)
//
// Every extractor returns a `uint2` carrying 64 bits of face-plane
// occupancy packed identically. A face is 8×8, indexed by two "free"
// axes `(a, b)` — the two axes that vary over the face. Which axes are
// free depends on the face normal:
//
//   ±X face: free axes (a = y, b = z)
//   ±Y face: free axes (a = x, b = z)
//   ±Z face: free axes (a = x, b = y)
//
// The canonical packing of the 64 bits is:
//
//   uint2.x  — 32 bits for b ∈ [0, 4)
//   uint2.y  — 32 bits for b ∈ [4, 8)
//
// and within each half-word the 8 bits for a fixed b are contiguous at
// bit position `(b % 4) * 8`, with `a` indexing the 8 bits within that
// byte. Formally, the bit for face-coordinate (a, b) lives at
//
//   half = (b < 4) ? uint2.x : uint2.y
//   bit_pos = (b % 4) * 8 + a
//
// Because opposite-face extractors target voxel pairs that meet across
// the face (e.g. my_pX's (y, z) pairs with nbr_nX's (y, z) — same free
// axes), the canonical packing lines those pairs up bit-for-bit across
// both sides. `face_exposed(my, nbr) = any((my & ~nbr) != 0)` is then a
// pointwise per-face-cell test, with no spurious OR-folding across other
// axes.
//
// # How each extractor reaches the canonical layout
//
// - ±X: the x=7 (or x=0) column in each z-plane word has its 4 y-bits at
//   non-contiguous positions {7,15,23,31} (for x=7) or {0,8,16,24} (for
//   x=0). Compact those into a 4-bit nybble, OR the word 0 (y∈[0,4)) and
//   word 1 (y∈[4,8)) nybbles into an 8-bit byte, and pack at bit position
//   `(z % 4) * 8` in the appropriate `uint2` half.
//
// - ±Y: the y=7 row lives entirely in word 1, bits 24..31; the y=0 row
//   lives in word 0, bits 0..7. The 8 bits are already contiguous — just
//   shift and mask, then pack at `(z % 4) * 8` per z-plane.
//
// - ±Z: a whole 8×8 plane at a single z-value is already laid out with
//   contiguous x-bytes per y-row: word 0 holds y∈[0,4) at `y*8+x`, word 1
//   holds y∈[4,8) at `(y-4)*8+x`. That matches the canonical layout
//   verbatim for free axes (a = x, b = y), so the two plane words map
//   directly into the returned `uint2` with no bit-twiddling.
//
// All six extractors therefore return a canonical (uint2) that pairs
// correctly with its opposite-face counterpart.

#ifndef RENDERER_FACE_MASK_HLSL
#define RENDERER_FACE_MASK_HLSL

// --- Helpers: compact a column's 4 sparse bits into a 4-bit nybble. ---
//
// The ±X columns have their y-bits at fixed sparse positions in each
// word; these helpers collapse them onto contiguous bits 0..3. No
// branches, no loops — four masked shifts ORed together.

// x = 7 column: bits at 7, 15, 23, 31 → bits 0, 1, 2, 3.
uint compact_col_x7(uint w) {
    return ((w >>  7) & 0x1u)
         | ((w >> 14) & 0x2u)
         | ((w >> 21) & 0x4u)
         | ((w >> 28) & 0x8u);
}

// x = 0 column: bits at 0, 8, 16, 24 → bits 0, 1, 2, 3.
uint compact_col_x0(uint w) {
    return ((w >>  0) & 0x1u)
         | ((w >>  7) & 0x2u)
         | ((w >> 14) & 0x4u)
         | ((w >> 21) & 0x8u);
}

// --- Per z-plane word extraction for ±Y rows. ---
//
// The y-row bits are already contiguous within one word; just mask off
// the other 24 bits. Kept as helpers for symmetry with the ±X column
// compacters.

// y = 7 row: word 1, bits 24..31 → byte at 0..7.
uint row_y7(uint word_1) { return (word_1 >> 24) & 0xFFu; }

// y = 0 row: word 0, bits 0..7 → byte at 0..7.
uint row_y0(uint word_0) { return word_0 & 0xFFu; }

// --- Occupancy per-z-plane accessors. ---
//
// The occupancy buffer packs z-planes two at a time into each `uint4`.
// Extracting a specific z-plane's word 0 / word 1 once at the top of a
// loop makes the per-direction extractors a readable sequence of eight
// "pack byte for z=k" lines.
uint occ_word0_at_z(Occupancy occ, uint z) {
    // z=0 → plane[0].x, z=1 → plane[0].z, z=2 → plane[1].x, ...
    uint4 p = occ.plane[z >> 1];
    return (z & 1u) == 0u ? p.x : p.z;
}
uint occ_word1_at_z(Occupancy occ, uint z) {
    uint4 p = occ.plane[z >> 1];
    return (z & 1u) == 0u ? p.y : p.w;
}

// --- Helper: pack an 8-bit byte into the canonical uint2 at slot b. ---
//
// `byte8` carries 8 bits (the compacted row/column) in positions 0..7.
// `b` is the "outer" free axis coordinate (z for ±X/±Y, y for ±Z). The
// byte lands at bit `(b % 4) * 8` of `uint2.x` (b < 4) or `uint2.y`
// (b ≥ 4).
void pack_into_face(inout uint2 face, uint b, uint byte8) {
    uint shifted = byte8 << ((b & 3u) * 8u);
    if (b < 4u) {
        face.x |= shifted;
    } else {
        face.y |= shifted;
    }
}

// --- +X face (x = 7). Free axes (a = y, b = z). ---
uint2 face_pX(Occupancy occ) {
    uint2 out_face = uint2(0u, 0u);
    [unroll] for (uint z = 0u; z < 8u; ++z) {
        uint w0  = occ_word0_at_z(occ, z);
        uint w1  = occ_word1_at_z(occ, z);
        // Low nybble: y=0..3 from word 0. High nybble: y=4..7 from word 1.
        uint byte8 = compact_col_x7(w0) | (compact_col_x7(w1) << 4u);
        pack_into_face(out_face, z, byte8);
    }
    return out_face;
}

// --- -X face (x = 0). Free axes (a = y, b = z). ---
uint2 face_nX(Occupancy occ) {
    uint2 out_face = uint2(0u, 0u);
    [unroll] for (uint z = 0u; z < 8u; ++z) {
        uint w0  = occ_word0_at_z(occ, z);
        uint w1  = occ_word1_at_z(occ, z);
        uint byte8 = compact_col_x0(w0) | (compact_col_x0(w1) << 4u);
        pack_into_face(out_face, z, byte8);
    }
    return out_face;
}

// --- +Y face (y = 7). Free axes (a = x, b = z). ---
uint2 face_pY(Occupancy occ) {
    uint2 out_face = uint2(0u, 0u);
    [unroll] for (uint z = 0u; z < 8u; ++z) {
        uint w1    = occ_word1_at_z(occ, z);
        uint byte8 = row_y7(w1);
        pack_into_face(out_face, z, byte8);
    }
    return out_face;
}

// --- -Y face (y = 0). Free axes (a = x, b = z). ---
uint2 face_nY(Occupancy occ) {
    uint2 out_face = uint2(0u, 0u);
    [unroll] for (uint z = 0u; z < 8u; ++z) {
        uint w0    = occ_word0_at_z(occ, z);
        uint byte8 = row_y0(w0);
        pack_into_face(out_face, z, byte8);
    }
    return out_face;
}

// --- +Z face (z = 7). Free axes (a = x, b = y). ---
//
// At fixed z, word 0 and word 1 already lay out rows y∈[0,4) and y∈[4,8)
// with 8 bits of x per row at `(y % 4) * 8`. That is exactly the
// canonical form for this face — no bit-twiddling required.
uint2 face_pZ(Occupancy occ) {
    return uint2(occ_word0_at_z(occ, 7u), occ_word1_at_z(occ, 7u));
}

// --- -Z face (z = 0). Free axes (a = x, b = y). ---
uint2 face_nZ(Occupancy occ) {
    return uint2(occ_word0_at_z(occ, 0u), occ_word1_at_z(occ, 0u));
}

// Return `true` iff `my_face` has any set bit in a position where
// `nbr_face` is clear. Because both operands are in the canonical
// (uint2) face layout, `(my & ~nbr)` is a pointwise per-face-cell
// exposure test across the 64 voxel pairs that meet across the face.
bool face_exposed(uint2 my_face, uint2 nbr_face) {
    uint2 exposed = my_face & ~nbr_face;
    return (exposed.x | exposed.y) != 0u;
}

#endif // RENDERER_FACE_MASK_HLSL
