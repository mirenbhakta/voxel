//! Chunk storage backends and format conversion.
//!
//! Provides three storage formats for voxel data, each generic over an element
//! type `T` and parameterized by a [`VoxelIndexer`] strategy:
//!
//! - [`Dense`] -- flat array with full random access.
//! - [`Rle`] -- run-length encoded with binary-searchable run starts.
//! - [`Palette`] -- unique-value table with per-voxel index array.
//!
//! Conversion between formats uses a stream intermediate: each type emits a
//! flat sequence of values in index order via [`IntoVoxelStream`] and
//! constructs itself from one via [`FromVoxelStream`]. This reduces conversion
//! paths from N*M to 2N.

pub mod dense;
pub mod palette;
pub mod rle;

pub use dense::Dense;
pub use palette::Palette;
pub use rle::Rle;
pub use rle::Run;

// --- Stream Traits ---

/// Convert storage into a sequential stream of voxel values.
///
/// Implementors yield their elements in flat index order (0, 1, ..., N-1)
/// according to the indexing strategy they were built with.
pub trait IntoVoxelStream {
    /// The value type yielded by the stream.
    type Element;

    /// The concrete iterator type.
    type Iter: Iterator<Item = Self::Element>;

    /// Consume this storage and return the voxel count and an iterator
    /// yielding values in index order.
    fn into_voxel_stream(self) -> (usize, Self::Iter);
}

/// Construct storage from a sequential stream of voxel values.
///
/// The stream must yield exactly `count` elements in flat index order.
pub trait FromVoxelStream: Sized {
    /// The value type consumed from the stream.
    type Element;

    /// Build storage from `count` elements in index order.
    fn from_voxel_stream(
        count  : usize,
        stream : impl Iterator<Item = Self::Element>,
    ) -> Self;
}

// --- Conversion ---

/// Convert between storage formats via the voxel stream.
///
/// Consumes `src`, streams its elements, and builds a `D` from them.
pub fn convert<S, D>(src: S) -> D
where
    S: IntoVoxelStream,
    D: FromVoxelStream<Element = S::Element>,
{
    let (count, stream) = src.into_voxel_stream();
    D::from_voxel_stream(count, stream)
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Linear3D, Morton3D};
    use eden_math::Vector3;

    // -- cross_format_roundtrip --

    #[test]
    fn cross_format_roundtrip() {
        // Dense -> Rle -> Palette -> Dense must preserve the original data.
        type L = Linear3D<4, 4, 4>;

        let mut src = Dense::<L, u16>::new(64, 0);
        src.set(&Vector3::new(0, 0, 0), 10);
        src.set(&Vector3::new(1, 2, 3), 20);
        src.set(&Vector3::new(3, 3, 3), 30);
        src.set(&Vector3::new(2, 1, 0), 5);

        let expected: Vec<u16> = src.as_slice().to_vec();

        // Dense -> Rle -> Palette -> Dense.
        let rle: Rle<L, u16>     = convert(src);
        let pal: Palette<L, u16> = convert(rle);
        let dst: Dense<L, u16>   = convert(pal);

        assert_eq!(dst.as_slice(), expected.as_slice());
    }

    // -- convert_with_morton --

    #[test]
    fn convert_with_morton() {
        // Verify the stream pipeline works with Morton indexing.
        let mut src = Dense::<Morton3D, u16>::new(32768, 0);
        src.set(&Vector3::new(5, 10, 15), 42);
        src.set(&Vector3::new(31, 31, 31), 99);

        let expected: Vec<u16> = src.as_slice().to_vec();

        let rle: Rle<Morton3D, u16>   = convert(src);
        let dst: Dense<Morton3D, u16> = convert(rle);

        assert_eq!(dst.as_slice(), expected.as_slice());
    }
}
