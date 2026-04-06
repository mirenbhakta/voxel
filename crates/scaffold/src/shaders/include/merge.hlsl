// Greedy merge algorithm on a 32x32 face layer.
//
// Finds maximal axis-aligned rectangles in the face bitmask. Used by
// both the count pass (to count quads) and the write pass (to emit
// packed descriptors).
//
// The merge runs on thread 0 only (sequential row-extension). Future
// work: parallel prefix-sum + emit.
//
// Requires: shared_face[32] populated with face words (one per row),
// and a workgroup barrier before calling.

#ifndef MERGE_HLSL
#define MERGE_HLSL

#include "include/quad.hlsl"

// -----------------------------------------------------------------------
// Merge result callback interface
// -----------------------------------------------------------------------
//
// The merge is parameterized by what happens with each discovered quad.
// The count pass increments a counter. The write pass packs and stores.
// This is controlled by preprocessor defines set by the including file:
//
//   MERGE_MODE_COUNT  - define to count only (no quad output)
//   MERGE_MODE_WRITE  - define to write packed quads

// -----------------------------------------------------------------------
// Greedy merge
// -----------------------------------------------------------------------

/// Run the greedy merge on shared_face[32]. Thread 0 only.
///
/// For MERGE_MODE_COUNT: increments `out_count` for each quad found.
/// For MERGE_MODE_WRITE: writes packed quads to `out_quads` at
///   `out_base + out_count`, incrementing `out_count`.
///
/// Parameters:
///   layer     - the layer index (0-31) for this face plane
///   dir       - the direction index (0-5)
/// Accesses g_shared_face[32] declared by the including shader.
///
/// Returns the number of quads emitted from this layer.
uint greedy_merge(uint layer, uint dir
#ifdef MERGE_MODE_WRITE
                  , RWByteAddressBuffer quad_buf
                  , uint write_offset
#endif
                  ) {
    uint count = 0;

    for (uint r = 0; r < 32; r++) {
        uint bits = g_shared_face[r];

        while (bits != 0) {
            uint col = firstbitlow(bits);

            // Horizontal run: contiguous set bits starting at col.
            uint shifted  = bits >> col;
            uint run      = shifted ^ (shifted & (shifted + 1));
            uint run_mask = run << col;
            uint width    = firstbitlow(~shifted);

            // Extend downward.
            uint height = 1;
            for (uint r2 = r + 1; r2 < 32; r2++) {
                if ((g_shared_face[r2] & run_mask) == run_mask) {
                    height++;
                    g_shared_face[r2] &= ~run_mask;
                }
                else {
                    break;
                }
            }

            // Clear consumed bits in the current row.
            bits &= ~run_mask;
            g_shared_face[r] = bits;

#ifdef MERGE_MODE_WRITE
            uint packed = pack_quad(col, r, layer, width, height, dir);
            quad_buf.Store((write_offset + count) * 4, packed);
#endif

            count++;
        }
    }

    return count;
}

#endif // MERGE_HLSL
