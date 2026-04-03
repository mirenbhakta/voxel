//! Palette-compressed storage.

#![allow(dead_code)]

use std::collections::HashMap;
use std::marker::PhantomData;

use crate::index::VoxelIndexer;
use crate::storage::{FromVoxelStream, IntoVoxelStream};

// --- Palette ---

/// Palette-compressed voxel storage.
///
/// A table of unique values paired with a per-voxel index array. Effective
/// when the number of distinct values is much smaller than the total voxel
/// count.
pub struct Palette<S: VoxelIndexer, T> {
    /// The unique values.
    entries : Vec<T>,
    /// Per-voxel index into `entries`.
    indices : Vec<u16>,
    /// The indexing strategy.
    _marker : PhantomData<S>,
}

impl<S: VoxelIndexer, T> Palette<S, T> {
    /// Returns a shared reference to the value at `coords`.
    pub fn get(&self, coords: &S::Coords) -> &T {
        &self.entries[self.indices[S::voxel_enc(coords)] as usize]
    }

    /// Returns the total number of voxels stored.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns the number of unique values in the palette.
    pub fn palette_len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the unique palette values as a slice.
    pub fn palette(&self) -> &[T] {
        &self.entries
    }
}

// --- PaletteIter ---

/// An iterator that resolves palette indices to values.
pub struct PaletteIter<T> {
    /// The shared palette entries.
    entries : Vec<T>,
    /// The remaining indices to resolve.
    indices : std::vec::IntoIter<u16>,
}

impl<T: Clone> Iterator for PaletteIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.indices.next()?;
        Some(self.entries[idx as usize].clone())
    }
}

// --- IntoVoxelStream ---

impl<S: VoxelIndexer, T: Clone> IntoVoxelStream for Palette<S, T> {
    type Element = T;
    type Iter    = PaletteIter<T>;

    /// Consume this storage and return the voxel count and an iterator in index order.
    fn into_voxel_stream(self) -> (usize, Self::Iter) {
        let count = self.indices.len();
        let iter  = PaletteIter {
            entries : self.entries,
            indices : self.indices.into_iter(),
        };
        (count, iter)
    }
}

// --- FromVoxelStream ---

impl<S: VoxelIndexer, T: Eq + std::hash::Hash + Clone> FromVoxelStream for Palette<S, T> {
    type Element = T;

    /// Build a `Palette` from `count` elements in index order.
    ///
    /// Deduplicates values into a palette table and records per-voxel indices.
    fn from_voxel_stream(
        count  : usize,
        stream : impl Iterator<Item = Self::Element>,
    ) -> Self
    {
        let mut entries = Vec::new();
        let mut map: HashMap<T, u16> = HashMap::new();
        let mut indices = Vec::with_capacity(count);

        for value in stream.take(count) {
            let idx = match map.get(&value) {
                Some(&idx) => idx,
                None => {
                    let idx = entries.len() as u16;
                    map.insert(value.clone(), idx);
                    entries.push(value);
                    idx
                }
            };
            indices.push(idx);
        }

        Self { entries, indices, _marker: PhantomData }
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Morton3D, Linear3D};
    use crate::storage::{Dense, IntoVoxelStream, FromVoxelStream};
    use eden_math::Vector3;

    // -- single_value --

    #[test]
    fn single_value() {
        // A stream of identical values must collapse to a single palette entry
        // regardless of how many voxels are in the volume.
        let stream = std::iter::repeat(42u32).take(1000);
        let palette = Palette::<Morton3D, u32>::from_voxel_stream(1000, stream);

        assert_eq!(palette.palette_len(), 1);
        assert_eq!(palette.len(), 1000);
    }

    // -- few_unique --

    #[test]
    fn few_unique() {
        // Five distinct values cycling across 100 voxels must yield exactly
        // five palette entries, and get() must resolve to the correct value.
        type L = Linear3D<10, 10, 1>;

        let values: Vec<u8> = (0..100).map(|i| (i % 5) as u8).collect();
        let palette = Palette::<L, u8>::from_voxel_stream(100, values.iter().copied());

        assert_eq!(palette.palette_len(), 5);
        assert_eq!(palette.len(), 100);

        // Verify get() returns the same value as the original stream at each position.
        for i in 0..10u8 {
            for j in 0..10u8 {
                let coords   = Vector3::new(i, j, 0u8);
                let flat_idx = L::voxel_enc(&coords);
                assert_eq!(*palette.get(&coords), (flat_idx % 5) as u8);
            }
        }
    }

    // -- roundtrip_dense_palette_dense --

    #[test]
    fn roundtrip_dense_palette_dense() {
        // Build a Dense volume with a known pattern, convert to Palette, then
        // back to Dense. The final slice must match the original.
        type L = Linear3D<4, 4, 4>;

        let mut src = Dense::<L, u16>::new(64, 0);
        src.set(&Vector3::new(0, 0, 0), 10);
        src.set(&Vector3::new(1, 0, 0), 20);
        src.set(&Vector3::new(0, 1, 0), 10);
        src.set(&Vector3::new(3, 3, 3), 30);

        let expected: Vec<u16> = src.as_slice().to_vec();

        // Convert Dense -> Palette -> Dense.
        let (count, stream)     = src.into_voxel_stream();
        let palette             = Palette::<L, u16>::from_voxel_stream(count, stream);
        let (count2, stream2)   = palette.into_voxel_stream();
        let dst                 = Dense::<L, u16>::from_voxel_stream(count2, stream2);

        assert_eq!(dst.as_slice(), expected.as_slice());
    }
}
