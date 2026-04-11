//! Bit-packed boolean storage.

#![allow(dead_code)]

use std::marker::PhantomData;

use crate::index::VoxelIndexer;
use crate::storage::{FromVoxelStream, IntoVoxelStream};

// --- Bitmask ---

/// Bit-packed boolean storage indexed by a [`VoxelIndexer`] strategy.
///
/// Stores one bit per voxel in a contiguous array of `u32` words, using 32x
/// less memory than a `Dense<S, bool>`. The raw word array is directly
/// uploadable to GPU buffers for ray traversal and occupancy queries.
///
/// Trailing bits beyond `len()` in the last word are always clear.
pub struct Bitmask<S: VoxelIndexer> {
    /// The packed bit data, one bit per voxel.
    words   : Vec<u32>,
    /// The total number of voxels.
    ///
    /// May not be a multiple of 32, so the last word can have unused
    /// trailing bits that must remain clear.
    count   : usize,
    /// The indexing strategy.
    _marker : PhantomData<S>,
}

// --- Bitmask ---

impl<S: VoxelIndexer> Bitmask<S> {
    /// Create a new bitmask with `count` voxels, all set to `fill`.
    pub fn new(count: usize, fill: bool) -> Self {
        let word_count = count.div_ceil(32);
        let fill_word  = if fill { !0u32 } else { 0u32 };
        let mut words  = vec![fill_word; word_count];

        // Clear trailing bits in the last word so they don't inflate
        // count_ones.
        if fill && !count.is_multiple_of(32) {
            let valid_bits = count % 32;
            let last       = words.len() - 1;
            words[last]    = (1u32 << valid_bits) - 1;
        }

        Bitmask { words, count, _marker: PhantomData }
    }

    /// Returns whether the voxel at `coords` is occupied.
    #[inline]
    pub fn get(&self, coords: &S::Coords) -> bool {
        let idx  = S::voxel_enc(coords);
        let word = self.words[idx / 32];
        word & (1 << (idx % 32)) != 0
    }

    /// Set the occupancy of the voxel at `coords`.
    #[inline]
    pub fn set(&mut self, coords: &S::Coords, value: bool) {
        let idx = S::voxel_enc(coords);
        let bit = 1u32 << (idx % 32);

        if value {
            self.words[idx / 32] |= bit;
        }
        else {
            self.words[idx / 32] &= !bit;
        }
    }

    /// Returns the total number of voxels.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns whether this bitmask contains no voxels.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the number of occupied voxels.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Returns the packed bit data as a slice of `u32` words.
    pub fn as_raw(&self) -> &[u32] {
        &self.words
    }

    /// Returns the packed bit data as a mutable slice of `u32` words.
    ///
    /// Trailing bits beyond `len()` in the last word must remain clear to
    /// preserve the correctness of [`count_ones`](Self::count_ones).
    pub fn as_raw_mut(&mut self) -> &mut [u32] {
        &mut self.words
    }

    /// Build a bitmask by applying a predicate to each element of a source
    /// storage.
    ///
    /// This is a lossy projection: source values are reduced to single bits
    /// via `f`. The resulting bitmask preserves the flat index order of the
    /// source stream.
    pub fn from_map<Src>(
        src : Src,
        mut f   : impl FnMut(Src::Element) -> bool,
    ) -> Self
    where
        Src: IntoVoxelStream,
    {
        let (count, stream) = src.into_voxel_stream();
        let word_count      = count.div_ceil(32);
        let mut words       = vec![0u32; word_count];

        for (i, value) in stream.take(count).enumerate() {
            if f(value) {
                words[i / 32] |= 1 << (i % 32);
            }
        }

        Bitmask { words, count, _marker: PhantomData }
    }
}

// --- BitmaskIter ---

/// An iterator that unpacks bits into sequential `bool` values.
///
/// Yields one `bool` per voxel in flat index order, advancing through
/// packed `u32` words.
pub struct BitmaskIter {
    /// The remaining packed words.
    words     : std::vec::IntoIter<u32>,
    /// The current word being unpacked.
    current   : u32,
    /// The next bit position within the current word.
    bit_pos   : u32,
    /// The number of values left to yield.
    remaining : usize,
}

impl Iterator for BitmaskIter {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        if self.remaining == 0 {
            return None;
        }

        // Advance to the next word when the current one is exhausted.
        if self.bit_pos >= 32 {
            self.current = self.words.next().unwrap();
            self.bit_pos = 0;
        }

        let value       = self.current & (1 << self.bit_pos) != 0;
        self.bit_pos   += 1;
        self.remaining -= 1;
        Some(value)
    }
}

// --- IntoVoxelStream ---

impl<S: VoxelIndexer> IntoVoxelStream for Bitmask<S> {
    type Element = bool;
    type Iter    = BitmaskIter;

    /// Consume this bitmask and return the voxel count and an iterator
    /// yielding occupancy values in index order.
    fn into_voxel_stream(self) -> (usize, Self::Iter) {
        let count     = self.count;
        let mut words = self.words.into_iter();

        // Prime the iterator with the first word.
        let current = words.next().unwrap_or(0);

        let iter = BitmaskIter {
            words,
            current,
            bit_pos   : 0,
            remaining : count,
        };

        (count, iter)
    }
}

// --- FromVoxelStream ---

impl<S: VoxelIndexer> FromVoxelStream for Bitmask<S> {
    type Element = bool;

    /// Build a bitmask from `count` boolean values in index order.
    fn from_voxel_stream(
        count  : usize,
        stream : impl Iterator<Item = bool>,
    ) -> Self
    {
        let word_count = count.div_ceil(32);
        let mut words  = vec![0u32; word_count];

        for (i, value) in stream.take(count).enumerate() {
            if value {
                words[i / 32] |= 1 << (i % 32);
            }
        }

        Bitmask { words, count, _marker: PhantomData }
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Linear3D, Morton3D};
    use crate::storage::{Dense, convert};
    use eden_math::Vector3;

    // -- empty_bitmask --

    #[test]
    fn empty_bitmask() {
        // A bitmask filled with false should have no occupied voxels.
        let bm = Bitmask::<Morton3D>::new(32768, false);

        assert_eq!(bm.len(), 32768);
        assert_eq!(bm.count_ones(), 0);

        for &w in bm.as_raw() {
            assert_eq!(w, 0);
        }
    }

    // -- full_bitmask --

    #[test]
    fn full_bitmask() {
        // A bitmask filled with true should have all voxels occupied.
        let bm = Bitmask::<Morton3D>::new(32768, true);

        assert_eq!(bm.len(), 32768);
        assert_eq!(bm.count_ones(), 32768);
    }

    // -- set_get_roundtrip --

    #[test]
    fn set_get_roundtrip() {
        type L = Linear3D<8, 8, 8>;
        let mut bm = Bitmask::<L>::new(512, false);

        let coords: &[Vector3<u8>] = &[
            Vector3::new(0, 0, 0),
            Vector3::new(7, 7, 7),
            Vector3::new(3, 5, 2),
            Vector3::new(1, 0, 0),
        ];

        // Set the coordinates to occupied.
        for c in coords {
            bm.set(c, true);
        }

        assert_eq!(bm.count_ones(), coords.len());

        // Verify the occupied coordinates.
        for c in coords {
            assert!(bm.get(c), "expected occupied at {c:?}");
        }

        // Verify a non-occupied coordinate.
        assert!(!bm.get(&Vector3::new(6, 6, 6)));

        // Clear one and verify.
        bm.set(&Vector3::new(0, 0, 0), false);
        assert!(!bm.get(&Vector3::new(0, 0, 0)));
        assert_eq!(bm.count_ones(), coords.len() - 1);
    }

    // -- trailing_bits_clear --

    #[test]
    fn trailing_bits_clear() {
        // A non-multiple-of-32 count should have clear trailing bits.
        // 50 voxels = 2 words, second word has 18 valid bits.
        let bm = Bitmask::<Linear3D<50, 1, 1>>::new(50, true);

        assert_eq!(bm.len(), 50);
        assert_eq!(bm.count_ones(), 50);

        // The second word should only have the low 18 bits set.
        let words = bm.as_raw();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], !0u32);
        assert_eq!(words[1], (1u32 << 18) - 1);
    }

    // -- stream_roundtrip --

    #[test]
    fn stream_roundtrip() {
        // Set a pattern, stream out, stream back in, verify identical.
        type L = Linear3D<4, 4, 4>;
        let mut src = Bitmask::<L>::new(64, false);

        src.set(&Vector3::new(0, 0, 0), true);
        src.set(&Vector3::new(1, 2, 3), true);
        src.set(&Vector3::new(3, 3, 3), true);

        let expected: Vec<u32> = src.as_raw().to_vec();

        let (count, stream) = src.into_voxel_stream();
        let dst             = Bitmask::<L>::from_voxel_stream(count, stream);

        assert_eq!(dst.as_raw(), expected.as_slice());
        assert_eq!(dst.count_ones(), 3);
    }

    // -- convert_to_dense --

    #[test]
    fn convert_to_dense() {
        // Convert a bitmask to a Dense<S, bool> and verify values match.
        type L = Linear3D<4, 4, 4>;
        let mut bm = Bitmask::<L>::new(64, false);

        bm.set(&Vector3::new(0, 0, 0), true);
        bm.set(&Vector3::new(3, 3, 3), true);
        bm.set(&Vector3::new(2, 1, 0), true);

        let dense: Dense<L, bool> = convert(bm);

        assert!(*dense.get(&Vector3::new(0, 0, 0)));
        assert!(*dense.get(&Vector3::new(3, 3, 3)));
        assert!(*dense.get(&Vector3::new(2, 1, 0)));
        assert!(!*dense.get(&Vector3::new(1, 1, 1)));
    }

    // -- convert_from_dense --

    #[test]
    fn convert_from_dense() {
        // Convert a Dense<S, bool> to a bitmask and verify values match.
        type L = Linear3D<4, 4, 4>;
        let mut dense = Dense::<L, bool>::new(64, false);

        dense.set(&Vector3::new(0, 0, 0), true);
        dense.set(&Vector3::new(3, 3, 3), true);
        dense.set(&Vector3::new(2, 1, 0), true);

        let bm: Bitmask<L> = convert(dense);

        assert!(bm.get(&Vector3::new(0, 0, 0)));
        assert!(bm.get(&Vector3::new(3, 3, 3)));
        assert!(bm.get(&Vector3::new(2, 1, 0)));
        assert!(!bm.get(&Vector3::new(1, 1, 1)));
        assert_eq!(bm.count_ones(), 3);
    }

    // -- from_map --

    #[test]
    fn from_map() {
        // Build a Dense<L, u16> with some nonzero values, then derive a
        // bitmask using a "nonzero is occupied" predicate.
        type L = Linear3D<4, 4, 4>;
        let mut dense = Dense::<L, u16>::new(64, 0);

        dense.set(&Vector3::new(0, 0, 0), 10);
        dense.set(&Vector3::new(1, 2, 3), 20);
        dense.set(&Vector3::new(3, 3, 3), 30);

        let bm = Bitmask::<L>::from_map(dense, |v| v != 0);

        assert_eq!(bm.count_ones(), 3);
        assert!(bm.get(&Vector3::new(0, 0, 0)));
        assert!(bm.get(&Vector3::new(1, 2, 3)));
        assert!(bm.get(&Vector3::new(3, 3, 3)));
        assert!(!bm.get(&Vector3::new(1, 1, 1)));
    }

    // -- morton_indexing --

    #[test]
    fn morton_indexing() {
        // Verify bitmask works correctly with Morton indexing.
        let mut bm = Bitmask::<Morton3D>::new(32768, false);

        bm.set(&Vector3::new(5, 10, 15), true);
        bm.set(&Vector3::new(31, 31, 31), true);

        assert!(bm.get(&Vector3::new(5, 10, 15)));
        assert!(bm.get(&Vector3::new(31, 31, 31)));
        assert!(!bm.get(&Vector3::new(0, 0, 0)));
        assert_eq!(bm.count_ones(), 2);

        // Roundtrip through stream.
        let expected: Vec<u32> = bm.as_raw().to_vec();

        let (count, stream) = bm.into_voxel_stream();
        let dst             = Bitmask::<Morton3D>::from_voxel_stream(count, stream);

        assert_eq!(dst.as_raw(), expected.as_slice());
    }
}
