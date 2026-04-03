//! Directional face bitmask derivation from occupancy.
//!
//! Derives per-direction face bitmasks from a chunk's occupancy data. A face
//! exists at a solid/air boundary: `occupied(here) & ~occupied(there)` for
//! each of the six directions.
//!
//! Each direction stores 32 layers of 32-word bitmask slices in layer-first
//! order. The axis mapping per direction:
//!
//! | Direction | Layer axis | Row axis (word) | Column axis (bit) |
//! |-----------|-----------|----------------|------------------|
//! | +X / -X   | X          | Z               | Y                 |
//! | +Y / -Y   | Y          | X               | Z                 |
//! | +Z / -Z   | Z          | Y               | X                 |
//!
//! The axis mapping follows a cyclic rotation: `row = (N+2) % 3`,
//! `col = (N+1) % 3`, where N is the normal axis index. This ensures the
//! greedy merge and ray traversal operate on a consistent 2D layout
//! regardless of direction.
//!
//! Z faces derive directly from the occupancy layout. Y and X faces require
//! a 32x32 bit-plane transpose to swizzle the axes into canonical order.

#![allow(dead_code)]

use super::direction::Direction;
use super::quad::LayerOccupancy;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of layers per direction (equal to chunk side length).
const N: usize = 32;

/// Words per layer (one u32 per row, 32 rows per layer).
const WORDS_PER_LAYER: usize = 32;

/// Total words per direction (32 layers * 32 words).
const WORDS_PER_DIR: usize = N * WORDS_PER_LAYER;

// ---------------------------------------------------------------------------
// FaceNeighbors
// ---------------------------------------------------------------------------

/// Boundary occupancy data from adjacent chunks.
///
/// The face derivation needs one boundary slice from each neighbor to
/// detect faces at chunk edges. For directions where the neighbor is not
/// loaded, boundary voxels are treated as empty (faces are exposed).
pub struct FaceNeighbors<'a> {
    /// Raw occupancy words from each neighbor, if loaded.
    ///
    /// Indexed by [`Direction`] discriminant. `PosX` holds the +X
    /// neighbor's occupancy (the chunk at offset (+1, 0, 0)), etc.
    neighbors : [Option<&'a [u32]>; 6],
}

impl<'a> FaceNeighbors<'a> {
    /// No neighbors loaded. All boundary faces are exposed.
    pub fn none() -> Self {
        FaceNeighbors { neighbors: [None; 6] }
    }

    /// Set the neighbor occupancy for a direction.
    pub fn set(&mut self, dir: Direction, occ: &'a [u32]) {
        self.neighbors[dir as usize] = Some(occ);
    }

    /// Get the neighbor occupancy for a direction.
    fn get(&self, dir: Direction) -> Option<&'a [u32]> {
        self.neighbors[dir as usize]
    }
}

// ---------------------------------------------------------------------------
// FaceMasks
// ---------------------------------------------------------------------------

/// Per-chunk directional face bitmasks for all six directions.
///
/// Each direction contains 1024 `u32` words in layer-first order:
/// `data[dir][layer * 32 + row]` = u32, one bit per column.
pub struct FaceMasks {
    /// Face bitmask data indexed by direction.
    data : [[u32; WORDS_PER_DIR]; 6],
}

impl FaceMasks {
    /// Derive face bitmasks from a chunk's raw occupancy words.
    ///
    /// # Arguments
    ///
    /// * `occ` - The chunk's occupancy bitmask as 1024 `u32` words in the
    ///   standard layout: `occ[z * 32 + y]`, bit `x`.
    /// * `neighbors` - Boundary occupancy from adjacent chunks. Missing
    ///   neighbors are treated as empty (boundary faces exposed).
    pub fn from_occupancy(
        occ       : &[u32],
        neighbors : &FaceNeighbors,
    ) -> Self
    {
        debug_assert_eq!(occ.len(), WORDS_PER_DIR);

        let mut faces = FaceMasks { data: [[0u32; WORDS_PER_DIR]; 6] };

        // Z-direction faces (simplest: adjacent z-slices, no transpose).
        derive_z_faces(occ, neighbors, &mut faces);

        // Y-direction faces (adjacent y-rows within each z-slice, word
        // transpose to get layer-first order).
        derive_y_faces(occ, neighbors, &mut faces);

        // X-direction faces (bit shift within words, full bit-plane
        // transpose to get layer-first order).
        derive_x_faces(occ, neighbors, &mut faces);

        faces
    }

    /// Compute layer occupancy from the face bitmasks.
    ///
    /// For each direction, reduces 32 layers to a single `u32` where bit `i`
    /// indicates layer `i` has at least one face.
    pub fn layer_occupancy(&self) -> LayerOccupancy {
        let mut lo = LayerOccupancy::empty();

        for &dir in &Direction::ALL {
            let slice = &self.data[dir as usize];
            let mut bits = 0u32;

            for layer in 0..N {
                // OR all words in this layer. If any are nonzero, the layer
                // has faces.
                let base    = layer * WORDS_PER_LAYER;
                let any_set = slice[base..base + WORDS_PER_LAYER]
                    .iter()
                    .any(|&w| w != 0);

                if any_set {
                    bits |= 1 << layer;
                }
            }

            *lo.raw_mut(dir) = bits;
        }

        lo
    }

    /// Access the face bitmask for one direction.
    pub fn direction(&self, dir: Direction) -> &[u32; WORDS_PER_DIR] {
        &self.data[dir as usize]
    }

    /// Read a single word: `data[dir][layer * 32 + row]`.
    pub fn word(
        &self,
        dir   : Direction,
        layer : usize,
        row   : usize,
    ) -> u32
    {
        self.data[dir as usize][layer * WORDS_PER_LAYER + row]
    }
}

// ---------------------------------------------------------------------------
// Z-direction face derivation
// ---------------------------------------------------------------------------

/// Derive +Z and -Z face bitmasks.
///
/// The occupancy layout has Z as the outer axis, so adjacent z-slices are
/// contiguous 32-word blocks. Face detection is a direct AND-NOT and the
/// output is already in layer-first order (layer = z, row = y, col = x).
fn derive_z_faces(
    occ       : &[u32],
    neighbors : &FaceNeighbors,
    faces     : &mut FaceMasks,
)
{
    let pos_z = Direction::PosZ as usize;
    let neg_z = Direction::NegZ as usize;

    for z in 0..N {
        let this_base = z * WORDS_PER_LAYER;

        for y in 0..WORDS_PER_LAYER {
            let here = occ[this_base + y];

            // +Z: face if occupied here and empty at z+1.
            let above = if z < N - 1 {
                occ[(z + 1) * WORDS_PER_LAYER + y]
            }
            else {
                // Boundary: read from +Z neighbor's z=0 slice.
                neighbors
                    .get(Direction::PosZ)
                    .map(|n| n[y])
                    .unwrap_or(0)
            };
            faces.data[pos_z][this_base + y] = here & !above;

            // -Z: face if occupied here and empty at z-1.
            let below = if z > 0 {
                occ[(z - 1) * WORDS_PER_LAYER + y]
            }
            else {
                // Boundary: read from -Z neighbor's z=31 slice.
                neighbors
                    .get(Direction::NegZ)
                    .map(|n| n[(N - 1) * WORDS_PER_LAYER + y])
                    .unwrap_or(0)
            };
            faces.data[neg_z][this_base + y] = here & !below;
        }
    }
}

// ---------------------------------------------------------------------------
// Y-direction face derivation
// ---------------------------------------------------------------------------

/// Derive +Y and -Y face bitmasks.
///
/// Face detection operates on adjacent y-rows within each z-slice. The
/// natural computation produces words indexed by Z with bits indexed by X,
/// but canonical layout requires row=X, col=Z. A bit-plane transpose per
/// Y-layer swizzles the axes.
fn derive_y_faces(
    occ       : &[u32],
    neighbors : &FaceNeighbors,
    faces     : &mut FaceMasks,
)
{
    let pos_y = Direction::PosY as usize;
    let neg_y = Direction::NegY as usize;

    // Process one y-layer at a time. For each layer, collect face words
    // across all z-slices, then transpose to canonical axis order.
    for y in 0..N {
        let mut pos_layer = [0u32; 32];
        let mut neg_layer = [0u32; 32];

        for z in 0..N {
            let here = occ[z * WORDS_PER_LAYER + y];

            // +Y: face if occupied here and empty at y+1.
            let next = if y < N - 1 {
                occ[z * WORDS_PER_LAYER + y + 1]
            }
            else {
                // Boundary: +Y neighbor's y=0 row at this z.
                neighbors
                    .get(Direction::PosY)
                    .map(|n| n[z * WORDS_PER_LAYER])
                    .unwrap_or(0)
            };

            // -Y: face if occupied here and empty at y-1.
            let prev = if y > 0 {
                occ[z * WORDS_PER_LAYER + y - 1]
            }
            else {
                // Boundary: -Y neighbor's y=31 row at this z.
                neighbors
                    .get(Direction::NegY)
                    .map(|n| n[z * WORDS_PER_LAYER + N - 1])
                    .unwrap_or(0)
            };

            // Before transpose: row=z, col=x (bits represent x).
            pos_layer[z] = here & !next;
            neg_layer[z] = here & !prev;
        }

        // Transpose so that row=x, col=z (bits represent z).
        transpose_32x32(&mut pos_layer);
        transpose_32x32(&mut neg_layer);

        // Store layer y.
        let base = y * WORDS_PER_LAYER;
        faces.data[pos_y][base..base + 32].copy_from_slice(&pos_layer);
        faces.data[neg_y][base..base + 32].copy_from_slice(&neg_layer);
    }
}

// ---------------------------------------------------------------------------
// X-direction face derivation
// ---------------------------------------------------------------------------

/// Derive +X and -X face bitmasks.
///
/// X is the bit-position axis. Face detection uses bit shifts within each
/// word. The output requires a bit-plane transpose: the occupancy layout
/// has rows indexed by Y with bits indexed by X, but the layer-first
/// output needs layers indexed by X with bits indexed by Y.
fn derive_x_faces(
    occ       : &[u32],
    neighbors : &FaceNeighbors,
    faces     : &mut FaceMasks,
)
{
    let pos_x = Direction::PosX as usize;
    let neg_x = Direction::NegX as usize;

    let neighbor_pos = neighbors.get(Direction::PosX);
    let neighbor_neg = neighbors.get(Direction::NegX);

    // Process one z-slice at a time. Each z-slice is 32 words (one per y)
    // forming a 32x32 bit matrix where row = y, col = x.
    for z in 0..N {
        let base = z * WORDS_PER_LAYER;

        // Compute face bits for this slice.
        let mut pos_face = [0u32; 32];
        let mut neg_face = [0u32; 32];

        for y in 0..WORDS_PER_LAYER {
            let word = occ[base + y];

            // +X interior: bit x is a face if bit x is set and bit x+1
            // is clear. The right shift drops bit 31.
            let mut pf = word & !(word >> 1);

            // +X boundary at x=31: check neighbor's x=0 bit.
            let neighbor_bit = neighbor_pos
                .map(|n| n[base + y] & 1)
                .unwrap_or(0);
            // If the neighbor's x=0 is occupied, clear the face at x=31.
            if neighbor_bit != 0 {
                pf &= !(1 << 31);
            }

            // -X interior: bit x is a face if bit x is set and bit x-1
            // is clear. The left shift drops bit 0.
            let mut nf = word & !(word << 1);

            // -X boundary at x=0: check neighbor's x=31 bit.
            let neighbor_bit = neighbor_neg
                .map(|n| (n[base + y] >> 31) & 1)
                .unwrap_or(0);
            // If the neighbor's x=31 is occupied, clear the face at x=0.
            if neighbor_bit != 0 {
                nf &= !1;
            }

            pos_face[y] = pf;
            neg_face[y] = nf;
        }

        // Transpose the 32x32 bit matrices so that row index becomes the
        // bit position and bit position becomes the row index. After
        // transpose: pos_face[x] has bit y set if there was a +X face at
        // (x, y, z).
        transpose_32x32(&mut pos_face);
        transpose_32x32(&mut neg_face);

        // Store: layer=x, row=z, col=y.
        for x in 0..N {
            faces.data[pos_x][x * WORDS_PER_LAYER + z] = pos_face[x];
            faces.data[neg_x][x * WORDS_PER_LAYER + z] = neg_face[x];
        }
    }
}

// ---------------------------------------------------------------------------
// 32x32 bit matrix transpose
// ---------------------------------------------------------------------------

/// Transpose a 32x32 bit matrix in place.
///
/// Swaps rows and columns: after the transpose, bit `j` of `m[i]` becomes
/// bit `i` of `m[j]`. Uses the recursive block-swap algorithm from Hacker's
/// Delight (Warren), processing block sizes 16, 8, 4, 2, 1 in sequence.
fn transpose_32x32(m: &mut [u32; 32]) {
    // Each pass swaps off-diagonal blocks of size k*k between row pairs
    // separated by k. The mask selects every other k-bit group so the
    // swap only touches the bits that need to move.
    const MASKS: [u32; 5] = [
        0x0000_FFFF, // block 16
        0x00FF_00FF, // block 8
        0x0F0F_0F0F, // block 4
        0x3333_3333, // block 2
        0x5555_5555, // block 1
    ];

    let mut block: usize = 16;

    for &mask in &MASKS {
        let mut k = 0;

        while k < 32 {
            for j in 0..block {
                let a = k + j;
                let b = k + j + block;

                let t = ((m[a] >> block) ^ m[b]) & mask;
                m[a] ^= t << block;
                m[b] ^= t;
            }

            k += 2 * block;
        }

        block >>= 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkIndexer;
    use crate::storage::Bitmask;
    use eden_math::Vector3;

    /// Build occupancy from a list of filled coordinates and return the raw
    /// words.
    fn build_occ(coords: &[Vector3<u8>]) -> Vec<u32> {
        let mut bm = Bitmask::<ChunkIndexer>::new(32 * 32 * 32, false);
        for c in coords {
            bm.set(c, true);
        }
        bm.as_raw().to_vec()
    }

    /// Build a fully solid occupancy bitmask.
    fn solid_occ() -> Vec<u32> {
        vec![!0u32; 1024]
    }

    // -- transpose_identity --

    #[test]
    fn transpose_identity() {
        // The identity matrix (bit i set in row i) should remain unchanged
        // after transpose.
        let mut m = [0u32; 32];
        for i in 0..32 {
            m[i] = 1 << i;
        }

        let expected = m;
        transpose_32x32(&mut m);
        assert_eq!(m, expected);
    }

    // -- transpose_roundtrip --

    #[test]
    fn transpose_roundtrip() {
        // Transposing twice should return the original matrix.
        let mut m = [0u32; 32];
        for i in 0..32 {
            m[i] = (i as u32).wrapping_mul(0x1234_5678) ^ (i as u32 * 7);
        }

        let original = m;
        transpose_32x32(&mut m);
        transpose_32x32(&mut m);
        assert_eq!(m, original);
    }

    // -- transpose_known_value --

    #[test]
    fn transpose_known_value() {
        // Row 0 has all bits set, all other rows are zero.
        // After transpose: bit 0 should be set in every row.
        let mut m = [0u32; 32];
        m[0] = !0u32;

        transpose_32x32(&mut m);

        for i in 0..32 {
            assert_eq!(m[i], 1, "row {i} should have only bit 0 set");
        }
    }

    // -- single_voxel_six_faces --

    #[test]
    fn single_voxel_six_faces() {
        // A single voxel at (15, 15, 15) surrounded by air should produce
        // exactly one face in each of the 6 directions.
        let occ = build_occ(&[Vector3::new(15, 15, 15)]);
        let faces = FaceMasks::from_occupancy(&occ, &FaceNeighbors::none());

        for &dir in &Direction::ALL {
            let layer = 15usize;
            let slice = faces.direction(dir);

            // Count total set bits across all words.
            let total: u32 = slice.iter().map(|w| w.count_ones()).sum();
            assert_eq!(
                total, 1,
                "{dir:?} should have exactly 1 face, got {total}",
            );

            // The face should be in layer 15.
            let row_word = match dir {
                // +X/-X: layer=x=15, row=z=15, col=y=15
                Direction::PosX | Direction::NegX => {
                    slice[layer * 32 + 15]
                }
                // +Y/-Y: layer=y=15, row=z=15, col=x=15
                Direction::PosY | Direction::NegY => {
                    slice[layer * 32 + 15]
                }
                // +Z/-Z: layer=z=15, row=y=15, col=x=15
                Direction::PosZ | Direction::NegZ => {
                    slice[layer * 32 + 15]
                }
            };

            assert_eq!(
                row_word.count_ones(), 1,
                "{dir:?}: expected 1 bit set in the face word",
            );
        }
    }

    // -- enclosed_cube_no_faces --

    #[test]
    fn enclosed_cube_no_faces() {
        // A voxel completely surrounded by other voxels should produce no
        // faces. Fill a 3x3x3 cube centered at (15, 15, 15).
        let mut coords = Vec::new();
        for z in 14..=16u8 {
            for y in 14..=16u8 {
                for x in 14..=16u8 {
                    coords.push(Vector3::new(x, y, z));
                }
            }
        }

        let occ   = build_occ(&coords);
        let faces = FaceMasks::from_occupancy(&occ, &FaceNeighbors::none());

        // The center voxel (15,15,15) should have no faces.
        // But the shell voxels do have faces. Check that the total face count
        // matches: a 3x3x3 cube has 6 faces of 3x3 = 54 faces total.
        let mut total = 0u32;
        for &dir in &Direction::ALL {
            total += faces
                .direction(dir)
                .iter()
                .map(|w| w.count_ones())
                .sum::<u32>();
        }

        assert_eq!(total, 54, "3x3x3 cube should have 54 faces");
    }

    // -- flat_floor_faces --

    #[test]
    fn flat_floor_faces() {
        // Fill the entire y=0 layer (a floor). This should produce:
        // - 1024 +Y faces at y=0 (top of the floor)
        // - 1024 -Y faces at y=0 (bottom of the floor)
        // - edge faces on the X and Z boundaries (walls of the slab)
        let mut coords = Vec::new();
        for z in 0..32u8 {
            for x in 0..32u8 {
                coords.push(Vector3::new(x, 0, z));
            }
        }

        let occ   = build_occ(&coords);
        let faces = FaceMasks::from_occupancy(&occ, &FaceNeighbors::none());

        // +Y faces: layer=0, should have 1024 faces (32 rows * 32 cols).
        let pos_y_count: u32 = faces
            .direction(Direction::PosY)[0..32]
            .iter()
            .map(|w| w.count_ones())
            .sum();
        assert_eq!(pos_y_count, 1024, "+Y layer 0 should be fully set");

        // -Y faces: layer=0, same count.
        let neg_y_count: u32 = faces
            .direction(Direction::NegY)[0..32]
            .iter()
            .map(|w| w.count_ones())
            .sum();
        assert_eq!(neg_y_count, 1024, "-Y layer 0 should be fully set");

        // +Y layer 1 and above should be empty.
        for layer in 1..32 {
            let base  = layer * 32;
            let count = faces
                .direction(Direction::PosY)[base..base + 32]
                .iter()
                .any(|&w| w != 0);
            assert!(!count, "+Y layer {layer} should be empty");
        }
    }

    // -- boundary_faces_with_neighbor --

    #[test]
    fn boundary_faces_with_neighbor() {
        // Fill the z=31 layer of a chunk. With no neighbor, it should have
        // +Z faces. With a solid neighbor, the +Z faces at z=31 disappear.
        let mut coords = Vec::new();
        for y in 0..32u8 {
            for x in 0..32u8 {
                coords.push(Vector3::new(x, y, 31));
            }
        }

        let occ = build_occ(&coords);

        // Without neighbor: +Z faces at z=31.
        let no_neighbor = FaceMasks::from_occupancy(&occ, &FaceNeighbors::none());
        let pz_count: u32 = no_neighbor
            .direction(Direction::PosZ)[31 * 32..32 * 32]
            .iter()
            .map(|w| w.count_ones())
            .sum();
        assert_eq!(pz_count, 1024, "+Z at z=31 should have faces without neighbor");

        // With solid neighbor: +Z faces at z=31 should vanish.
        let solid = solid_occ();
        let mut neighbors = FaceNeighbors::none();
        neighbors.set(Direction::PosZ, &solid);

        let with_neighbor = FaceMasks::from_occupancy(&occ, &neighbors);
        let pz_count: u32 = with_neighbor
            .direction(Direction::PosZ)[31 * 32..32 * 32]
            .iter()
            .map(|w| w.count_ones())
            .sum();
        assert_eq!(pz_count, 0, "+Z at z=31 should have no faces with solid neighbor");
    }

    // -- layer_occupancy_correct --

    #[test]
    fn layer_occupancy_correct() {
        // Single voxel at (15, 15, 15): each direction should have layer 15
        // occupied.
        let occ   = build_occ(&[Vector3::new(15, 15, 15)]);
        let faces = FaceMasks::from_occupancy(&occ, &FaceNeighbors::none());
        let lo    = faces.layer_occupancy();

        for &dir in &Direction::ALL {
            assert!(
                lo.has_faces(dir, 15),
                "{dir:?} should have layer 15 occupied",
            );
            assert_eq!(
                lo.first_occupied_layer(dir),
                Some(15),
                "{dir:?} first layer should be 15",
            );
        }
    }

    // -- empty_chunk_no_layers --

    #[test]
    fn empty_chunk_no_layers() {
        let occ   = vec![0u32; 1024];
        let faces = FaceMasks::from_occupancy(&occ, &FaceNeighbors::none());
        let lo    = faces.layer_occupancy();

        for &dir in &Direction::ALL {
            assert_eq!(lo.first_occupied_layer(dir), None);
        }
    }
}
