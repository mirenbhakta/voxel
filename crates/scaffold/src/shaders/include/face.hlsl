// Face derivation via AND-NOT on chunk occupancy bitmasks.
//
// Produces one u32 face word per (direction, layer, row) where set bits
// indicate visible faces at the corresponding column position. This is
// the canonical input to the greedy merge.
//
// Occupancy layout: words[z * 32 + y], bit x.
// Boundary cache layout: 6 slices of 32 words each, per slot.
//
// Direction indices: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.

#ifndef FACE_HLSL
#define FACE_HLSL

// -----------------------------------------------------------------------
// Occupancy access helpers
// -----------------------------------------------------------------------

/// Read an occupancy word from the shared slot-indexed buffer.
/// occ_base = slot * OCC_WORDS.
uint read_occ(ByteAddressBuffer occupancy_buf, uint occ_base, uint z, uint y) {
    uint idx = occ_base + z * 32 + y;
    return occupancy_buf.Load(idx * 4);
}

/// Read a boundary cache word.
/// bound_base = slot * 192. Direction offsets: 0,32,64,96,128,160.
uint read_boundary(ByteAddressBuffer boundary_buf, uint bound_base,
                   uint dir, uint word_idx) {
    uint idx = bound_base + dir * 32 + word_idx;
    return boundary_buf.Load(idx * 4);
}

// -----------------------------------------------------------------------
// Z-direction face derivation
// -----------------------------------------------------------------------

// Layer = z, row = y, col = x (bits). No transpose needed.
uint derive_z(ByteAddressBuffer occ_buf, uint occ_base,
              ByteAddressBuffer bound_buf, uint bound_base,
              uint layer, uint row, bool positive) {
    uint here = read_occ(occ_buf, occ_base, layer, row);

    uint there;
    if (positive) {
        // +Z: compare with z+1, or +Z neighbor boundary (dir=4).
        if (layer < 31)
            there = read_occ(occ_buf, occ_base, layer + 1, row);
        else
            there = read_boundary(bound_buf, bound_base, 4, row);
    }
    else {
        // -Z: compare with z-1, or -Z neighbor boundary (dir=5).
        if (layer > 0)
            there = read_occ(occ_buf, occ_base, layer - 1, row);
        else
            there = read_boundary(bound_buf, bound_base, 5, row);
    }

    return here & ~there;
}

// -----------------------------------------------------------------------
// Y-direction face derivation (pre-transpose)
// -----------------------------------------------------------------------

// Pre-transpose: row = z, col = x (bits). Caller transposes to
// canonical (row=x, col=z) via shared memory.
uint derive_y(ByteAddressBuffer occ_buf, uint occ_base,
              ByteAddressBuffer bound_buf, uint bound_base,
              uint layer, uint z, bool positive) {
    uint here = read_occ(occ_buf, occ_base, z, layer);

    uint there;
    if (positive) {
        // +Y: compare with y+1, or +Y neighbor boundary (dir=2).
        if (layer < 31)
            there = read_occ(occ_buf, occ_base, z, layer + 1);
        else
            there = read_boundary(bound_buf, bound_base, 2, z);
    }
    else {
        // -Y: compare with y-1, or -Y neighbor boundary (dir=3).
        if (layer > 0)
            there = read_occ(occ_buf, occ_base, z, layer - 1);
        else
            there = read_boundary(bound_buf, bound_base, 3, z);
    }

    return here & ~there;
}

// -----------------------------------------------------------------------
// X-direction face derivation (with inline transpose)
// -----------------------------------------------------------------------

// Layer = x, row = z, col = y (bits). Bit shift detects x-direction
// boundaries. Extraction loop transposes inline.
uint derive_x(ByteAddressBuffer occ_buf, uint occ_base,
              ByteAddressBuffer bound_buf, uint bound_base,
              uint layer, uint z, bool positive) {
    // Load neighbor boundary word once. At the boundary layer (x=31
    // for +X, x=0 for -X), bit y indicates whether the adjacent
    // chunk's boundary voxel is occupied.
    uint neighbor = 0;
    if (positive && layer == 31)
        neighbor = read_boundary(bound_buf, bound_base, 0, z);
    else if (!positive && layer == 0)
        neighbor = read_boundary(bound_buf, bound_base, 1, z);

    uint result = 0;

    for (uint y = 0; y < 32; y++) {
        uint word = read_occ(occ_buf, occ_base, z, y);
        uint nb   = (neighbor >> y) & 1;

        uint face;
        if (positive) {
            // +X: face at bit x if occupied here and x+1 is empty.
            face = word & ~(word >> 1);
            face &= ~(nb << 31);
        }
        else {
            // -X: face at bit x if occupied here and x-1 is empty.
            face = word & ~(word << 1);
            face &= ~nb;
        }

        // Extract bit `layer` (= x position) and pack as bit y.
        result |= ((face >> layer) & 1) << y;
    }

    return result;
}

// -----------------------------------------------------------------------
// Unified face derivation entry point
// -----------------------------------------------------------------------

// Shared memory for the 32-word face layer. All compute shaders that
// include face.hlsl or merge.hlsl must declare this before including:
//
//   groupshared uint g_shared_face[32];
//
// Both derive_face() and greedy_merge() access it by name.

/// Derive a face word for the given direction, layer, and row.
/// Handles all six directions including the Y-direction transpose
/// via g_shared_face.
///
/// Call pattern from the compute shader:
///   1. Each thread calls derive_face() with its row
///   2. workgroup barrier
///   3. g_shared_face[row] is ready for the greedy merge
uint derive_face(ByteAddressBuffer occ_buf, uint occ_base,
                 ByteAddressBuffer bound_buf, uint bound_base,
                 uint dir, uint layer, uint row) {
    uint face_word = 0;

    if (dir == 4 || dir == 5) {
        // Z faces: direct AND-NOT, no transpose.
        face_word = derive_z(occ_buf, occ_base, bound_buf, bound_base,
                             layer, row, dir == 4);
    }
    else if (dir == 2 || dir == 3) {
        // Y faces: AND-NOT produces (row=z, col=x). Transpose to
        // canonical (row=x, col=z) via shared memory.
        uint pre = derive_y(occ_buf, occ_base, bound_buf, bound_base,
                            layer, row, dir == 2);

        g_shared_face[row] = pre;
        GroupMemoryBarrierWithGroupSync();

        // Gather: extract bit `row` from each of the 32 words.
        uint transposed = 0;
        for (uint i = 0; i < 32; i++) {
            transposed |= ((g_shared_face[i] >> row) & 1) << i;
        }
        GroupMemoryBarrierWithGroupSync();

        face_word = transposed;
    }
    else {
        // X faces: inline transpose via extraction loop.
        face_word = derive_x(occ_buf, occ_base, bound_buf, bound_base,
                             layer, row, dir == 0);
    }

    return face_word;
}

#endif // FACE_HLSL
