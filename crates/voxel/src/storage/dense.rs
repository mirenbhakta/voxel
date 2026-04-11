//! Dense flat array storage.

#![allow(dead_code)]

use std::marker::PhantomData;

use crate::index::VoxelIndexer;
use crate::storage::{FromVoxelStream, IntoVoxelStream};

// --- Dense ---

/// A flat array of voxel values indexed by a [`VoxelIndexer`] strategy.
///
/// Stores every voxel in a contiguous `Vec<T>`, providing O(1) random access
/// at any coordinate. The indexing strategy `S` determines the mapping from
/// spatial coordinates to flat array positions.
pub struct Dense<S: VoxelIndexer, T> {
    /// The flat array of voxel values.
    data    : Vec<T>,
    /// The indexing strategy.
    _marker : PhantomData<S>,
}

impl<S: VoxelIndexer, T: Clone> Dense<S, T> {
    /// Create a new `Dense` volume with `count` elements, each initialized to `fill`.
    pub fn new(count: usize, fill: T) -> Self {
        Dense {
            data    : vec![fill; count],
            _marker : PhantomData,
        }
    }
}

impl<S: VoxelIndexer, T> Dense<S, T> {
    /// Returns a shared reference to the voxel at `coords`.
    pub fn get(&self, coords: &S::Coords) -> &T {
        &self.data[S::voxel_enc(coords)]
    }

    /// Returns a mutable reference to the voxel at `coords`.
    pub fn get_mut(&mut self, coords: &S::Coords) -> &mut T {
        &mut self.data[S::voxel_enc(coords)]
    }

    /// Set the voxel at `coords` to `value`.
    pub fn set(&mut self, coords: &S::Coords, value: T) {
        let idx        = S::voxel_enc(coords);
        self.data[idx] = value;
    }

    /// Returns the number of voxels in this volume.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether this volume contains no voxels.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the voxel data as a flat slice in index order.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the voxel data as a mutable flat slice in index order.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

// --- IntoVoxelStream ---

impl<S: VoxelIndexer, T> IntoVoxelStream for Dense<S, T> {
    type Element = T;
    type Iter    = std::vec::IntoIter<T>;

    /// Consume this volume and return the voxel count and an iterator in index order.
    fn into_voxel_stream(self) -> (usize, Self::Iter) {
        let count = self.data.len();
        (count, self.data.into_iter())
    }
}

// --- FromVoxelStream ---

impl<S: VoxelIndexer, T: Clone> FromVoxelStream for Dense<S, T> {
    type Element = T;

    /// Build a `Dense` volume from `count` elements in index order.
    fn from_voxel_stream(
        count  : usize,
        stream : impl Iterator<Item = Self::Element>,
    ) -> Self
    {
        Dense {
            data    : stream.take(count).collect(),
            _marker : PhantomData,
        }
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Linear3D, Morton3D};
    use eden_math::Vector3;

    // -- fill_and_read --

    #[test]
    fn fill_and_read() {
        // A 32^3 Morton-indexed volume has exactly 32768 entries for 5-bit
        // coordinates. Every slot should hold the fill value after construction.
        let vol = Dense::<Morton3D, u16>::new(32768, 0);

        for &v in vol.as_slice() {
            assert_eq!(v, 0);
        }
    }

    // -- set_get_roundtrip --

    #[test]
    fn set_get_roundtrip() {
        // Use a small linear volume so indices are predictable.
        type L = Linear3D<8, 8, 8>;
        let mut vol = Dense::<L, u32>::new(512, 0);

        let cases: &[(Vector3<u8>, u32)] = &[
            (Vector3::new(0, 0, 0), 1),
            (Vector3::new(1, 0, 0), 42),
            (Vector3::new(0, 1, 0), 100),
            (Vector3::new(7, 7, 7), 999),
            (Vector3::new(3, 5, 2), 7),
        ];

        // Write all values first, then verify, so writes don't clobber reads.
        for &(ref coord, value) in cases {
            vol.set(coord, value);
        }

        for &(ref coord, expected) in cases {
            assert_eq!(*vol.get(coord), expected, "mismatch at {coord:?}");
        }
    }

    // -- stream_roundtrip --

    #[test]
    fn stream_roundtrip() {
        type L = Linear3D<4, 4, 4>;
        let mut src = Dense::<L, u16>::new(64, 0);

        // Set a handful of values so the stream is not uniformly zero.
        src.set(&Vector3::new(0, 0, 0), 10);
        src.set(&Vector3::new(1, 2, 3), 20);
        src.set(&Vector3::new(3, 3, 3), 30);

        let expected: Vec<u16> = src.as_slice().to_vec();

        // Round-trip through the stream interface.
        let (count, iter) = src.into_voxel_stream();
        let dst           = Dense::<L, u16>::from_voxel_stream(count, iter);

        assert_eq!(dst.as_slice(), expected.as_slice());
    }
}
