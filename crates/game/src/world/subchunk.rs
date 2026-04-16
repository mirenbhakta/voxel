//! 8³ sub-chunk primitive: occupancy bitmap and flat material array.
//!
//! The sub-chunk is the unit of storage, residency, DDA traversal, and
//! OR-reduction throughout the world streaming system. Each LOD level
//! stores its resident content as sub-chunks; coarser levels hold an
//! OR-reduction of the level below.
//!
//! [`SubchunkOccupancy`] and [`SubchunkMaterials`] are kept as separate
//! types rather than fused into one struct because they have different
//! lifecycles: occupancy lives at every level, materials typically only
//! at L0 (plus any coarse level that wants per-voxel color).

#![allow(dead_code)]

use crate::world::coord::Octant;

/// Edge length of a sub-chunk in voxels.
pub const SUBCHUNK_EDGE: usize = 8;

/// Total voxels in a sub-chunk: `SUBCHUNK_EDGE³`.
pub const SUBCHUNK_VOLUME: usize = SUBCHUNK_EDGE * SUBCHUNK_EDGE * SUBCHUNK_EDGE;

/// Material identifier. `0` is reserved for "air" / empty.
pub type Material = u16;

// --- SubchunkOccupancy ---

/// 512-bit occupancy bitmap for an 8³ sub-chunk.
///
/// Layout: one `u64` per z-layer. Within `bits[z]`, bit `y*8 + x` is the
/// voxel at local coord `(x, y, z)`. A full z-layer fits in one word, which
/// lines up naturally with DDA stepping (each z-step loads one word).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubchunkOccupancy {
    bits: [u64; SUBCHUNK_EDGE],
}

impl SubchunkOccupancy {
    /// An empty sub-chunk (no voxels set).
    pub const fn empty() -> Self {
        Self { bits: [0; SUBCHUNK_EDGE] }
    }

    /// A fully occupied sub-chunk (every voxel set).
    pub const fn full() -> Self {
        Self { bits: [u64::MAX; SUBCHUNK_EDGE] }
    }

    /// Read the voxel at local coord `(x, y, z)`. All components must lie
    /// in `[0, SUBCHUNK_EDGE)`.
    #[inline]
    pub fn get(&self, x: u8, y: u8, z: u8) -> bool {
        debug_assert!((x as usize) < SUBCHUNK_EDGE);
        debug_assert!((y as usize) < SUBCHUNK_EDGE);
        debug_assert!((z as usize) < SUBCHUNK_EDGE);

        let bit = y * SUBCHUNK_EDGE as u8 + x;
        (self.bits[z as usize] >> bit) & 1 != 0
    }

    /// Write the voxel at local coord `(x, y, z)`. All components must lie
    /// in `[0, SUBCHUNK_EDGE)`.
    #[inline]
    pub fn set(&mut self, x: u8, y: u8, z: u8, value: bool) {
        debug_assert!((x as usize) < SUBCHUNK_EDGE);
        debug_assert!((y as usize) < SUBCHUNK_EDGE);
        debug_assert!((z as usize) < SUBCHUNK_EDGE);

        let bit  = y * SUBCHUNK_EDGE as u8 + x;
        let mask = 1u64 << bit;
        if value {
            self.bits[z as usize] |= mask;
        }
        else {
            self.bits[z as usize] &= !mask;
        }
    }

    /// Count of set voxels.
    pub fn count_ones(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    /// Whether no voxels are set.
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    /// Whether every voxel is set.
    pub fn is_full(&self) -> bool {
        self.bits.iter().all(|&w| w == u64::MAX)
    }

    /// In-place bitwise OR with another sub-chunk.
    pub fn or_with(&mut self, other: &SubchunkOccupancy) {
        for z in 0..SUBCHUNK_EDGE {
            self.bits[z] |= other.bits[z];
        }
    }

    /// OR-reduce this 8³ occupancy into the 4³ region of `parent` identified
    /// by `octant`.
    ///
    /// Each 2×2×2 block of child voxels contributes a single OR'd bit to one
    /// parent voxel. The destination region in `parent` is the 4³ block
    /// starting at `(octant.x * 4, octant.y * 4, octant.z * 4)`.
    ///
    /// This is the per-child step of the LOD pyramid OR-reduction (see
    /// `docs/world_streaming.md` §OR-reduction per level). After all 8
    /// siblings have reduced into the same parent, the parent is complete.
    pub fn or_reduce_into(&self, parent: &mut SubchunkOccupancy, octant: Octant) {
        debug_assert!(octant.x < 2 && octant.y < 2 && octant.z < 2);

        let base_x = octant.x as usize * 4;
        let base_y = octant.y as usize * 4;
        let base_z = octant.z as usize * 4;

        for cz in 0..SUBCHUNK_EDGE {
            for cy in 0..SUBCHUNK_EDGE {
                for cx in 0..SUBCHUNK_EDGE {
                    if self.get(cx as u8, cy as u8, cz as u8) {
                        let px = base_x + cx / 2;
                        let py = base_y + cy / 2;
                        let pz = base_z + cz / 2;
                        parent.set(px as u8, py as u8, pz as u8, true);
                    }
                }
            }
        }
    }

    /// 6-bit directional exposure mask for this sub-chunk in isolation
    /// (no neighbor occlusion considered).
    ///
    /// Bit layout: bit 0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z.
    ///
    /// In isolation, any non-empty sub-chunk presents a visible surface from
    /// every face — a ray from any of the six outward directions encounters
    /// at least one set voxel. An empty sub-chunk exposes nothing. Results
    /// are therefore always `0x3F` or `0`.
    ///
    /// Cross-boundary refinement (a neighbor fully occluding a face) is a
    /// residency-time operation and lives in a later slice.
    pub fn isolated_exposure_mask(&self) -> u8 {
        if self.is_empty() { 0 } else { 0x3F }
    }

    /// Raw access to the packed bit layers. `bits[z]` is the 8×8 layer at
    /// local z; bit `y*8 + x` is the voxel at `(x, y, z)`.
    pub fn raw(&self) -> &[u64; SUBCHUNK_EDGE] {
        &self.bits
    }

    /// Serialize the occupancy to a 64-byte blob in the renderer's GPU
    /// layout: 8 little-endian u64 z-layers, bit `y*8 + x` set when voxel
    /// `(x, y, z)` is occupied.
    ///
    /// The GPU shader views the same bytes as `uint4 plane[4]` — on a
    /// little-endian target the two representations are
    /// bitwise-identical, but this method does the packing explicitly so
    /// the conversion is portable and audit-friendly.
    pub fn to_gpu_bytes(self) -> [u8; SUBCHUNK_EDGE * 8] {
        let mut out = [0u8; SUBCHUNK_EDGE * 8];
        for (i, &word) in self.bits.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&word.to_le_bytes());
        }
        out
    }
}

// --- SubchunkMaterials ---

/// Flat material array for an 8³ sub-chunk: one [`Material`] per voxel.
///
/// Layout: `voxels[z*64 + y*8 + x]` holds the material at `(x, y, z)`.
/// This is the simplest viable storage; the sub-block packing described in
/// `decision-volumetric-material-system` is a later optimisation once a
/// real material budget makes it worth the machinery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubchunkMaterials {
    voxels: [Material; SUBCHUNK_VOLUME],
}

impl SubchunkMaterials {
    /// A sub-chunk filled with `material` in every voxel.
    pub fn filled(material: Material) -> Self {
        Self { voxels: [material; SUBCHUNK_VOLUME] }
    }

    /// Read the material at local coord `(x, y, z)`.
    #[inline]
    pub fn get(&self, x: u8, y: u8, z: u8) -> Material {
        self.voxels[Self::index(x, y, z)]
    }

    /// Write the material at local coord `(x, y, z)`.
    #[inline]
    pub fn set(&mut self, x: u8, y: u8, z: u8, material: Material) {
        self.voxels[Self::index(x, y, z)] = material;
    }

    /// Borrow the raw material array in flat index order.
    pub fn as_slice(&self) -> &[Material] {
        &self.voxels
    }

    /// Mutably borrow the raw material array in flat index order.
    pub fn as_mut_slice(&mut self) -> &mut [Material] {
        &mut self.voxels
    }
}

impl SubchunkMaterials {
    /// Flat index for `(x, y, z)`: `z*64 + y*8 + x`.
    #[inline]
    fn index(x: u8, y: u8, z: u8) -> usize {
        debug_assert!((x as usize) < SUBCHUNK_EDGE);
        debug_assert!((y as usize) < SUBCHUNK_EDGE);
        debug_assert!((z as usize) < SUBCHUNK_EDGE);

        (z as usize) * SUBCHUNK_EDGE * SUBCHUNK_EDGE
            + (y as usize) * SUBCHUNK_EDGE
            + (x as usize)
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    // -- SubchunkOccupancy --

    #[test]
    fn empty_and_full_constructors() {
        let e = SubchunkOccupancy::empty();
        assert!(e.is_empty());
        assert!(!e.is_full());
        assert_eq!(e.count_ones(), 0);

        let f = SubchunkOccupancy::full();
        assert!(!f.is_empty());
        assert!(f.is_full());
        assert_eq!(f.count_ones(), SUBCHUNK_VOLUME as u32);
    }

    #[test]
    fn occupancy_get_set_roundtrip() {
        let mut occ = SubchunkOccupancy::empty();
        occ.set(0, 0, 0, true);
        occ.set(7, 7, 7, true);
        occ.set(3, 5, 2, true);

        assert!(occ.get(0, 0, 0));
        assert!(occ.get(7, 7, 7));
        assert!(occ.get(3, 5, 2));
        assert!(!occ.get(0, 0, 1));
        assert_eq!(occ.count_ones(), 3);

        occ.set(0, 0, 0, false);
        assert!(!occ.get(0, 0, 0));
        assert_eq!(occ.count_ones(), 2);
    }

    #[test]
    fn occupancy_set_is_idempotent() {
        let mut occ = SubchunkOccupancy::empty();
        occ.set(4, 4, 4, true);
        occ.set(4, 4, 4, true);
        assert_eq!(occ.count_ones(), 1);
    }

    #[test]
    fn or_with_unions_bits() {
        let mut a = SubchunkOccupancy::empty();
        a.set(1, 2, 3, true);

        let mut b = SubchunkOccupancy::empty();
        b.set(4, 5, 6, true);

        a.or_with(&b);
        assert!(a.get(1, 2, 3));
        assert!(a.get(4, 5, 6));
        assert_eq!(a.count_ones(), 2);
    }

    // -- or_reduce_into --

    #[test]
    fn or_reduce_full_child_fills_octant_region() {
        let child      = SubchunkOccupancy::full();
        let mut parent = SubchunkOccupancy::empty();
        child.or_reduce_into(&mut parent, Octant { x: 1, y: 0, z: 1 });

        // Octant (1, 0, 1) maps to parent region x∈[4,8), y∈[0,4), z∈[4,8).
        for z in 0..SUBCHUNK_EDGE as u8 {
            for y in 0..SUBCHUNK_EDGE as u8 {
                for x in 0..SUBCHUNK_EDGE as u8 {
                    let inside = (4..8).contains(&x)
                              && (0..4).contains(&y)
                              && (4..8).contains(&z);
                    assert_eq!(parent.get(x, y, z), inside);
                }
            }
        }
    }

    #[test]
    fn or_reduce_same_output_voxel_for_2x2x2_block() {
        // Any voxel in the 2×2×2 child block at (0..2, 0..2, 0..2) reduces
        // to parent voxel (0, 0, 0) when placed in octant (0, 0, 0).
        for cx in 0..2 {
            for cy in 0..2 {
                for cz in 0..2 {
                    let mut child = SubchunkOccupancy::empty();
                    child.set(cx, cy, cz, true);

                    let mut parent = SubchunkOccupancy::empty();
                    child.or_reduce_into(&mut parent, Octant { x: 0, y: 0, z: 0 });

                    assert!(parent.get(0, 0, 0), "cx={cx} cy={cy} cz={cz}");
                    assert_eq!(parent.count_ones(), 1);
                }
            }
        }
    }

    #[test]
    fn or_reduce_eight_full_children_fill_parent() {
        let child = SubchunkOccupancy::full();
        let mut parent = SubchunkOccupancy::empty();

        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    child.or_reduce_into(&mut parent, Octant { x, y, z });
                }
            }
        }

        assert!(parent.is_full());
    }

    // -- exposure mask --

    #[test]
    fn isolated_exposure_is_zero_when_empty() {
        assert_eq!(SubchunkOccupancy::empty().isolated_exposure_mask(), 0);
    }

    #[test]
    fn isolated_exposure_is_0x3f_when_any_voxel_set() {
        let mut occ = SubchunkOccupancy::empty();
        occ.set(3, 3, 3, true);
        assert_eq!(occ.isolated_exposure_mask(), 0x3F);

        assert_eq!(SubchunkOccupancy::full().isolated_exposure_mask(), 0x3F);
    }

    // -- to_gpu_bytes --

    #[test]
    fn to_gpu_bytes_empty() {
        assert_eq!(SubchunkOccupancy::empty().to_gpu_bytes(), [0u8; 64]);
    }

    #[test]
    fn to_gpu_bytes_full() {
        assert_eq!(SubchunkOccupancy::full().to_gpu_bytes(), [0xFFu8; 64]);
    }

    #[test]
    fn to_gpu_bytes_single_voxel_sets_expected_bit() {
        // Voxel (3, 2, 1): z-layer 1, bit y*8+x = 2*8+3 = 19.
        // Layer 1 lives at bytes 8..16; bit 19 lies in byte 8 + 19/8 = 10,
        // bit position 19 % 8 = 3 → byte value 0b1000 = 8.
        let mut occ = SubchunkOccupancy::empty();
        occ.set(3, 2, 1, true);
        let bytes = occ.to_gpu_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            let expected = if i == 10 { 0b1000 } else { 0 };
            assert_eq!(b, expected, "byte {i}");
        }
    }

    // -- SubchunkMaterials --

    #[test]
    fn materials_roundtrip() {
        let mut mats = SubchunkMaterials::filled(0);
        mats.set(0, 0, 0, 42);
        mats.set(7, 7, 7, 1000);

        assert_eq!(mats.get(0, 0, 0), 42);
        assert_eq!(mats.get(7, 7, 7), 1000);
        assert_eq!(mats.get(1, 1, 1), 0);
    }

    #[test]
    fn materials_slice_is_flat_index_order() {
        let mut mats = SubchunkMaterials::filled(0);
        mats.set(1, 0, 0, 5);  // index 1
        mats.set(0, 1, 0, 6);  // index 8
        mats.set(0, 0, 1, 7);  // index 64

        let s = mats.as_slice();
        assert_eq!(s[0],  0);
        assert_eq!(s[1],  5);
        assert_eq!(s[8],  6);
        assert_eq!(s[64], 7);
    }
}
