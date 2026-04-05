// Build stage compute shader.
//
// Derives face bitmasks from chunk occupancy and emits packed quad
// descriptors via greedy merge. One dispatch per chunk.
//
// Dispatch: (32, 6, 1) -- 32 layers x 6 directions = 192 workgroups.
// Workgroup: 32 threads, one per row in the 32x32 face layer.
//
// Direction indices match the Direction enum discriminants:
//   0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z

// Chunk occupancy (1024 words, layout occ[z*32 + y], bit x) followed
// by 6 neighbor boundary slices (32 words each, 192 total).
@group(0) @binding(0) var<storage, read> occupancy : array<u32, 1216>;

// Neighbor boundary slice offsets. Each slice is 32 words appended
// after the 1024-word occupancy region. Direction order matches the
// Direction enum discriminants.
const NEIGH_POS_X = 1024u;  // +X neighbor x=0:  word[z], bit y
const NEIGH_NEG_X = 1056u;  // -X neighbor x=31: word[z], bit y
const NEIGH_POS_Y = 1088u;  // +Y neighbor y=0:  word[z], bit x
const NEIGH_NEG_Y = 1120u;  // -Y neighbor y=31: word[z], bit x
const NEIGH_POS_Z = 1152u;  // +Z neighbor z=0:  word[y], bit x
const NEIGH_NEG_Z = 1184u;  // -Z neighbor z=31: word[y], bit x

// Atomic quad count (initialized to 0 before dispatch).
@group(0) @binding(1) var<storage, read_write> quad_count : atomic<u32>;

// Shared quad pool (all chunks write into this).
@group(0) @binding(2) var<storage, read_write> quads : array<u32>;

// Page table mapping logical block indices to physical block IDs.
@group(0) @binding(3) var<storage, read> page_table : array<u32>;

// Per-dispatch chunk slot offset into the page table.
struct Immediates { block_base : u32, }
var<immediate> imm : Immediates;

// Shared memory for the 32-word face layer. Used by the Y-direction
// transpose and by the greedy merge.
var<workgroup> shared_face : array<u32, 32>;

// -----------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------

@compute @workgroup_size(32, 1, 1)
fn build(
    @builtin(local_invocation_id) lid : vec3<u32>,
    @builtin(workgroup_id)        gid : vec3<u32>,
) {
    let row   = lid.x;   // 0..31, this thread's row
    let layer = gid.x;   // 0..31, layer along the normal axis
    let dir   = gid.y;   // 0..5, direction

    // --- Face derivation ---
    //
    // Each thread produces one face word: a u32 where set bits indicate
    // visible faces at the corresponding column position.

    var face_word = 0u;

    if dir == 4u || dir == 5u {
        // Z faces: direct AND-NOT, no transpose needed.
        // Canonical order: layer=z, row=y, col=x.
        face_word = derive_z(layer, row, dir == 4u);
    }
    else if dir == 2u || dir == 3u {
        // Y faces: AND-NOT produces (row=z, col=x). Transpose to
        // canonical order (row=x, col=z) via shared memory.
        let pre = derive_y(layer, row, dir == 2u);

        shared_face[row] = pre;
        workgroupBarrier();

        // Gather: extract bit `row` from each of the 32 words.
        var transposed = 0u;
        for (var i = 0u; i < 32u; i++) {
            transposed |= ((shared_face[i] >> row) & 1u) << i;
        }
        workgroupBarrier();

        face_word = transposed;
    }
    else {
        // X faces: bit shift + inline transpose via extraction loop.
        // Output is already in canonical order (row=z, col=y).
        face_word = derive_x(layer, row, dir == 0u);
    }

    // --- Store to shared for greedy merge ---

    shared_face[row] = face_word;
    workgroupBarrier();

    // --- Greedy merge (thread 0 only) ---
    //
    // Sequential row-extension merge on the 32x32 face layer. Finds
    // maximal horizontal runs and extends them downward. Consumed bits
    // are cleared, preventing duplicate emission.

    if row == 0u {
        for (var r = 0u; r < 32u; r++) {
            var bits = shared_face[r];

            while bits != 0u {
                let col = countTrailingZeros(bits);

                // Horizontal run: contiguous set bits starting at col.
                let shifted = bits >> col;
                let run     = shifted ^ (shifted & (shifted + 1u));
                let run_mask = run << col;
                let width   = countTrailingZeros(~shifted);

                // Extend downward.
                var height = 1u;
                for (var r2 = r + 1u; r2 < 32u; r2++) {
                    if (shared_face[r2] & run_mask) == run_mask {
                        height += 1u;
                        shared_face[r2] &= ~run_mask;
                    }
                    else {
                        break;
                    }
                }

                // Clear consumed bits in the current row.
                bits &= ~run_mask;
                shared_face[r] = bits;

                // Emit packed quad descriptor.
                let packed = col
                    | (r << 5u)
                    | (layer << 10u)
                    | ((width - 1u) << 15u)
                    | ((height - 1u) << 20u)
                    | (dir << 25u);

                // Map the local quad index through the page table to a
                // physical slot in the shared quad pool.
                let local_idx = atomicAdd(&quad_count, 1u);
                let block_idx = local_idx / 256u;
                let block_off = local_idx % 256u;
                let block_id  = page_table[imm.block_base + block_idx];
                let phys_idx  = block_id * 256u + block_off;

                if phys_idx < arrayLength(&quads) {
                    quads[phys_idx] = packed;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------
// Z-direction face derivation
// -----------------------------------------------------------------------

// Layer = z, row = y, col = x (bits). No transpose needed.
fn derive_z(layer: u32, row: u32, positive: bool) -> u32 {
    let here = occupancy[layer * 32u + row];

    var there: u32;
    if positive {
        // +Z: compare with z+1, or the +Z neighbor's z=0 slice.
        if layer < 31u {
            there = occupancy[(layer + 1u) * 32u + row];
        }
        else {
            there = occupancy[NEIGH_POS_Z + row];
        }
    }
    else {
        // -Z: compare with z-1, or the -Z neighbor's z=31 slice.
        if layer > 0u {
            there = occupancy[(layer - 1u) * 32u + row];
        }
        else {
            there = occupancy[NEIGH_NEG_Z + row];
        }
    }

    return here & ~there;
}

// -----------------------------------------------------------------------
// Y-direction face derivation (pre-transpose)
// -----------------------------------------------------------------------

// Pre-transpose: row = z, col = x (bits). Caller transposes to (row=x, col=z).
fn derive_y(layer: u32, z: u32, positive: bool) -> u32 {
    let here = occupancy[z * 32u + layer];

    var there: u32;
    if positive {
        // +Y: compare with y+1, or the +Y neighbor's y=0 slice.
        if layer < 31u {
            there = occupancy[z * 32u + layer + 1u];
        }
        else {
            there = occupancy[NEIGH_POS_Y + z];
        }
    }
    else {
        // -Y: compare with y-1, or the -Y neighbor's y=31 slice.
        if layer > 0u {
            there = occupancy[z * 32u + layer - 1u];
        }
        else {
            there = occupancy[NEIGH_NEG_Y + z];
        }
    }

    return here & ~there;
}

// -----------------------------------------------------------------------
// X-direction face derivation (with inline transpose)
// -----------------------------------------------------------------------

// Layer = x, row = z, col = y (bits). The bit shift detects x-direction
// boundaries. The extraction loop transposes inline: for each y, extract
// bit `layer` from the face word and pack as bit y in the result.
fn derive_x(layer: u32, z: u32, positive: bool) -> u32 {
    // Load neighbor boundary word once outside the loop. At the
    // boundary layer (x=31 for +X, x=0 for -X) bit y indicates
    // whether the adjacent chunk's boundary voxel is occupied.
    // Non-boundary layers get 0, making the correction a no-op.
    var neighbor = 0u;
    if positive && layer == 31u {
        neighbor = occupancy[NEIGH_POS_X + z];
    }
    else if !positive && layer == 0u {
        neighbor = occupancy[NEIGH_NEG_X + z];
    }

    var result = 0u;

    for (var y = 0u; y < 32u; y++) {
        let word = occupancy[z * 32u + y];
        let nb   = (neighbor >> y) & 1u;

        var face: u32;
        if positive {
            // +X: face at bit x if occupied here and x+1 is empty.
            // At x=31, the right-shift brings in 0; the neighbor
            // correction clears the face if the adjacent chunk is solid.
            face = word & ~(word >> 1u);
            face &= ~(nb << 31u);
        }
        else {
            // -X: face at bit x if occupied here and x-1 is empty.
            // At x=0, the left-shift brings in 0; same correction.
            face = word & ~(word << 1u);
            face &= ~nb;
        }

        // Extract bit `layer` (= x position) and pack as bit y.
        result |= ((face >> layer) & 1u) << y;
    }

    return result;
}
