//! Run-length encoded storage with binary-searchable run starts.

#![allow(dead_code)]

use std::marker::PhantomData;

use crate::index::VoxelIndexer;
use crate::storage::{FromVoxelStream, IntoVoxelStream};

// --- Run ---

/// A single run of repeated values in a flat index sequence.
pub struct Run<T> {
    /// The starting index in the equivalent dense array.
    idx   : usize,
    /// The number of consecutive elements with this value.
    len   : usize,
    /// The repeated value.
    value : T,
}

impl<T> Run<T> {
    /// Returns the starting flat index of this run.
    pub fn idx(&self) -> usize {
        self.idx
    }

    /// Returns the number of elements in this run.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether this run is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a reference to the repeated value.
    pub fn value(&self) -> &T {
        &self.value
    }
}

// --- Rle ---

/// Run-length encoded storage with binary-searchable run starts.
///
/// Compresses sequences of identical values into runs. Random access is O(log n)
/// in the number of runs via binary search on run start indices. The indexing
/// strategy `S` maps spatial coordinates to flat indices for `get`.
pub struct Rle<S: VoxelIndexer, T> {
    /// The compressed run data.
    runs    : Vec<Run<T>>,
    /// The total number of logical voxels.
    count   : usize,
    /// The indexing strategy.
    _marker : PhantomData<S>,
}

impl<S: VoxelIndexer, T> Rle<S, T> {
    /// Returns a reference to the voxel at `coords`.
    ///
    /// Uses binary search on run start indices for O(log n) lookup, where n
    /// is the number of runs.
    ///
    /// # Arguments
    ///
    /// * `coords` - The spatial coordinates to look up, interpreted by the
    ///   indexing strategy `S`.
    pub fn get(&self, coords: &S::Coords) -> &T {
        let target = S::voxel_enc(coords);

        // Binary search for the last run whose start index is <= target.
        // partition_point returns the first index where the predicate is false,
        // so subtracting 1 gives the containing run.
        let run_idx = self.runs.partition_point(|r| r.idx <= target) - 1;

        &self.runs[run_idx].value
    }

    /// Returns the total number of logical voxels.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns whether this storage contains no voxels.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the number of compressed runs.
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Returns the run data as a slice.
    pub fn runs(&self) -> &[Run<T>] {
        &self.runs
    }
}

// --- RleIter ---

/// An iterator that expands runs into individual cloned values.
///
/// Yields each run's value `run.len` times before advancing to the next run.
pub struct RleIter<T> {
    /// The source run iterator.
    runs      : std::vec::IntoIter<Run<T>>,
    /// The current run being expanded, if any.
    current   : Option<Run<T>>,
    /// The number of values remaining in the current run.
    remaining : usize,
}

impl<T: Clone> Iterator for RleIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If the current run still has elements, yield one.
            if self.remaining > 0 {
                self.remaining -= 1;
                return self.current.as_ref().map(|r| r.value.clone());
            }

            // Advance to the next run. If exhausted, the iterator is done.
            match self.runs.next() {
                Some(run) => {
                    self.remaining = run.len;
                    self.current   = Some(run);
                }
                None => return None,
            }
        }
    }
}

// --- IntoVoxelStream ---

impl<S: VoxelIndexer, T: Clone> IntoVoxelStream for Rle<S, T> {
    type Element = T;
    type Iter    = RleIter<T>;

    /// Consume this storage and return the voxel count and an expanding iterator.
    fn into_voxel_stream(self) -> (usize, Self::Iter) {
        let count = self.count;
        let mut runs = self.runs.into_iter();

        // Prime the iterator with the first run so that `remaining` and
        // `current` are consistent from the first call to `next`.
        let (current, remaining) = match runs.next() {
            Some(run) => {
                let len = run.len;
                (Some(run), len)
            }
            None => (None, 0),
        };

        let iter = RleIter {
            runs,
            current,
            remaining,
        };

        (count, iter)
    }
}

// --- FromVoxelStream ---

impl<S: VoxelIndexer, T: PartialEq + Clone> FromVoxelStream for Rle<S, T> {
    type Element = T;

    /// Build an `Rle` from `count` elements in index order.
    ///
    /// Consecutive identical values are merged into a single run.
    fn from_voxel_stream(
        count  : usize,
        stream : impl Iterator<Item = Self::Element>,
    ) -> Self
    {
        let mut runs: Vec<Run<T>> = Vec::new();

        // Walk the stream, tracking the active run start index and value.
        // When the value changes, push the completed run and start a new one.
        let mut current_val:   Option<T> = None;
        let mut run_start:     usize     = 0;
        let mut run_len:       usize     = 0;

        for (i, val) in stream.take(count).enumerate() {
            match current_val.as_ref() {
                Some(cur) if *cur == val => {
                    // Extend the current run.
                    run_len += 1;
                }
                _ => {
                    // Push the completed run (if one exists) and start a new one.
                    if let Some(cur) = current_val.take() {
                        runs.push(Run {
                            idx   : run_start,
                            len   : run_len,
                            value : cur,
                        });
                    }

                    run_start   = i;
                    run_len     = 1;
                    current_val = Some(val);
                }
            }
        }

        // Push the final run.
        if let Some(cur) = current_val {
            runs.push(Run {
                idx   : run_start,
                len   : run_len,
                value : cur,
            });
        }

        Rle {
            runs,
            count,
            _marker : PhantomData,
        }
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Linear3D;
    use crate::storage::{Dense, IntoVoxelStream, FromVoxelStream};
    use eden_math::Vector3;

    // -- constant_stream --

    #[test]
    fn constant_stream() {
        // A uniform stream of identical values must compress to a single run.
        let stream = std::iter::repeat_n(42u32, 1000);
        let rle    = Rle::<Linear3D<10, 10, 10>, u32>::from_voxel_stream(1000, stream);

        assert_eq!(rle.run_count(), 1);
        assert_eq!(rle.len(), 1000);
        assert_eq!(*rle.runs()[0].value(), 42u32);
        assert_eq!(rle.runs()[0].idx(), 0);
        assert_eq!(rle.runs()[0].len(), 1000);
    }

    // -- alternating_values --

    #[test]
    fn alternating_values() {
        // An alternating stream of 0,1,0,1,... produces one run per element.
        let count  = 100usize;
        let stream = (0..count).map(|i| (i % 2) as u8);
        let rle    = Rle::<Linear3D<10, 10, 1>, u8>::from_voxel_stream(count, stream);

        assert_eq!(rle.run_count(), count);
        assert_eq!(rle.len(), count);

        // Each run has length 1 and the correct start index and value.
        for (i, run) in rle.runs().iter().enumerate() {
            assert_eq!(run.idx(), i, "run {i} start index");
            assert_eq!(run.len(), 1, "run {i} length");
            assert_eq!(*run.value(), (i % 2) as u8, "run {i} value");
        }
    }

    // -- binary_search_get --

    #[test]
    fn binary_search_get() {
        // Build an Rle from a known sequence: [0]*10, [1]*10, [2]*10.
        // Use Linear3D<30, 1, 1> so index == x coordinate.
        type L = Linear3D<30, 1, 1>;

        let stream = (0u32..3).flat_map(|v| std::iter::repeat_n(v, 10));
        let rle    = Rle::<L, u32>::from_voxel_stream(30, stream);

        assert_eq!(rle.run_count(), 3);

        // Start of each run.
        assert_eq!(*rle.get(&Vector3::new(0, 0, 0)), 0);
        assert_eq!(*rle.get(&Vector3::new(10, 0, 0)), 1);
        assert_eq!(*rle.get(&Vector3::new(20, 0, 0)), 2);

        // Middle of each run.
        assert_eq!(*rle.get(&Vector3::new(5, 0, 0)), 0);
        assert_eq!(*rle.get(&Vector3::new(15, 0, 0)), 1);
        assert_eq!(*rle.get(&Vector3::new(25, 0, 0)), 2);

        // End of each run.
        assert_eq!(*rle.get(&Vector3::new(9, 0, 0)), 0);
        assert_eq!(*rle.get(&Vector3::new(19, 0, 0)), 1);
        assert_eq!(*rle.get(&Vector3::new(29, 0, 0)), 2);
    }

    // -- roundtrip_dense_rle_dense --

    #[test]
    fn roundtrip_dense_rle_dense() {
        // Build a Dense volume with known values, convert to Rle, convert back
        // to Dense, and verify the contents are identical.
        type L = Linear3D<4, 4, 4>;

        let mut src = Dense::<L, u16>::new(64, 0);

        // Set a scattered handful of values to break uniformity.
        src.set(&Vector3::new(0, 0, 0), 10);
        src.set(&Vector3::new(1, 2, 3), 20);
        src.set(&Vector3::new(3, 3, 3), 30);
        src.set(&Vector3::new(2, 0, 1), 5);

        let expected: Vec<u16> = src.as_slice().to_vec();

        // Dense -> Rle -> Dense.
        let (count, stream) = src.into_voxel_stream();
        let rle             = Rle::<L, u16>::from_voxel_stream(count, stream);
        let (count2, iter)  = rle.into_voxel_stream();
        let dst             = Dense::<L, u16>::from_voxel_stream(count2, iter);

        assert_eq!(dst.as_slice(), expected.as_slice());
    }
}
