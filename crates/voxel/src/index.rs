//! Indexing strategies for spatial coordinate to array index conversion.
//!
//! Provides a common interface for converting between N-dimensional coordinates
//! and flat array indices. Two strategies are available: Morton (Z-order curve)
//! encoding and row-major linear encoding.
//!
//! Strategies use associated types so chunk consumers only need
//! `S: VoxelIndexer` without repeating coordinate types.

#![allow(dead_code)]

use eden_math::Vector;

use crate::morton;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Convert between spatial coordinates and flat array indices.
///
/// The coordinate type and dimensionality are determined by the implementor.
/// Chunk storage parameterized over `S: VoxelIndexer` can work with any
/// strategy without naming the coordinate type directly.
pub trait VoxelIndexer {
    /// The coordinate type used for encoding and decoding.
    type Coords;

    /// The number of spatial dimensions.
    const DIMS: usize;

    /// Convert spatial coordinates to a flat array index.
    fn voxel_enc(coords: &Self::Coords) -> usize;

    /// Convert a flat array index back to spatial coordinates.
    fn voxel_dec(index: usize) -> Self::Coords;
}

// ---------------------------------------------------------------------------
// Morton 2D
// ---------------------------------------------------------------------------

/// Morton Z-order curve encoding for 2D grids with `u8` coordinates.
///
/// Interleaves coordinate bits to produce indices that preserve spatial
/// locality. The encoding is independent of grid dimensions.
pub struct Morton2D;

impl VoxelIndexer for Morton2D {
    type Coords = Vector<u8, 2>;
    const DIMS: usize = 2;

    #[inline]
    fn voxel_enc(coords: &Self::Coords) -> usize {
        morton::encode_2d_8(*coords) as usize
    }

    #[inline]
    fn voxel_dec(index: usize) -> Self::Coords {
        morton::decode_2d_8(index as u16)
    }
}

// ---------------------------------------------------------------------------
// Morton 3D
// ---------------------------------------------------------------------------

/// Morton Z-order curve encoding for 3D grids with `u8` coordinates.
///
/// Interleaves coordinate bits to produce indices that preserve spatial
/// locality. The encoding is independent of chunk dimensions.
pub struct Morton3D;

impl VoxelIndexer for Morton3D {
    type Coords = Vector<u8, 3>;
    const DIMS: usize = 3;

    #[inline]
    fn voxel_enc(coords: &Self::Coords) -> usize {
        morton::encode_3d_8(*coords) as usize
    }

    #[inline]
    fn voxel_dec(index: usize) -> Self::Coords {
        morton::decode_3d_8(index as u32)
    }
}

// ---------------------------------------------------------------------------
// Linear 2D
// ---------------------------------------------------------------------------

/// Row-major linear encoding for 2D grids with `u8` coordinates.
///
/// Index computed as `y * W + x`. Width and height are baked into the type
/// via const generics.
pub struct Linear2D<const W: usize, const H: usize>;

impl<const W: usize, const H: usize> VoxelIndexer for Linear2D<W, H> {
    type Coords = Vector<u8, 2>;
    const DIMS: usize = 2;

    #[inline]
    fn voxel_enc(coords: &Self::Coords) -> usize {
        coords.y as usize * W + coords.x as usize
    }

    #[inline]
    fn voxel_dec(index: usize) -> Self::Coords {
        Vector::with_components([
            (index % W) as u8,
            (index / W) as u8,
        ])
    }
}

// ---------------------------------------------------------------------------
// Linear 3D
// ---------------------------------------------------------------------------

/// Row-major linear encoding for 3D grids with `u8` coordinates.
///
/// Index computed as `z * W * H + y * W + x`. Dimensions are baked into
/// the type via const generics.
pub struct Linear3D<const W: usize, const H: usize, const D: usize>;

impl<const W: usize, const H: usize, const D: usize>
    VoxelIndexer for Linear3D<W, H, D>
{
    type Coords = Vector<u8, 3>;
    const DIMS: usize = 3;

    #[inline]
    fn voxel_enc(coords: &Self::Coords) -> usize {
        coords.z as usize * W * H
            + coords.y as usize * W
            + coords.x as usize
    }

    #[inline]
    fn voxel_dec(index: usize) -> Self::Coords {
        let z   = index / (W * H);
        let rem = index % (W * H);
        let y   = rem / W;
        let x   = rem % W;
        Vector::with_components([x as u8, y as u8, z as u8])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use eden_math::Vector2;
    use eden_math::Vector3;

    // -- Morton 2D --

    #[test]
    fn morton_2d_roundtrip() {
        let cases: &[Vector2<u8>] = &[
            Vector2::new(0, 0),
            Vector2::new(1, 1),
            Vector2::new(255, 255),
            Vector2::new(5, 10),
            Vector2::new(0, 255),
        ];
        for &v in cases {
            let idx = Morton2D::voxel_enc(&v);
            let dec = Morton2D::voxel_dec(idx);
            assert_eq!(dec, v, "roundtrip failed for {v:?}");
        }
    }

    // -- Morton 3D --

    #[test]
    fn morton_3d_roundtrip() {
        let cases: &[Vector3<u8>] = &[
            Vector3::new(0, 0, 0),
            Vector3::new(1, 1, 1),
            Vector3::new(255, 255, 255),
            Vector3::new(5, 10, 15),
            Vector3::new(31, 31, 31),
        ];
        for &v in cases {
            let idx = Morton3D::voxel_enc(&v);
            let dec = Morton3D::voxel_dec(idx);
            assert_eq!(dec, v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn morton_3d_known_values() {
        assert_eq!(Morton3D::voxel_enc(&Vector3::new(1, 0, 0)), 1);
        assert_eq!(Morton3D::voxel_enc(&Vector3::new(0, 1, 0)), 2);
        assert_eq!(Morton3D::voxel_enc(&Vector3::new(0, 0, 1)), 4);
        assert_eq!(Morton3D::voxel_enc(&Vector3::new(1, 1, 1)), 7);
    }

    // -- Linear 2D --

    #[test]
    fn linear_2d_roundtrip() {
        for y in 0..32u8 {
            for x in 0..32u8 {
                let v   = Vector2::new(x, y);
                let idx = Linear2D::<32, 32>::voxel_enc(&v);
                let dec = Linear2D::<32, 32>::voxel_dec(idx);
                assert_eq!(dec, v);
            }
        }
    }

    #[test]
    fn linear_2d_known_values() {
        type L = Linear2D<32, 32>;
        assert_eq!(L::voxel_enc(&Vector2::new(0, 0)), 0);
        assert_eq!(L::voxel_enc(&Vector2::new(1, 0)), 1);
        assert_eq!(L::voxel_enc(&Vector2::new(0, 1)), 32);
        assert_eq!(L::voxel_enc(&Vector2::new(31, 31)), 1023);
    }

    // -- Linear 3D --

    #[test]
    fn linear_3d_roundtrip() {
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    let v   = Vector3::new(x, y, z);
                    let idx = Linear3D::<8, 8, 8>::voxel_enc(&v);
                    let dec = Linear3D::<8, 8, 8>::voxel_dec(idx);
                    assert_eq!(dec, v);
                }
            }
        }
    }

    #[test]
    fn linear_3d_known_values() {
        type L = Linear3D<32, 32, 32>;
        assert_eq!(L::voxel_enc(&Vector3::new(0, 0, 0)), 0);
        assert_eq!(L::voxel_enc(&Vector3::new(1, 0, 0)), 1);
        assert_eq!(L::voxel_enc(&Vector3::new(0, 1, 0)), 32);
        assert_eq!(L::voxel_enc(&Vector3::new(0, 0, 1)), 1024);
        assert_eq!(L::voxel_enc(&Vector3::new(31, 31, 31)), 32 * 32 * 32 - 1);
    }

    // -- Cross-validation --

    #[test]
    fn morton_trait_matches_direct_3d() {
        for z in [0u8, 1, 15, 31, 127, 255] {
            for y in [0u8, 1, 15, 31, 127, 255] {
                for x in [0u8, 1, 15, 31, 127, 255] {
                    let v         = Vector3::new(x, y, z);
                    let direct    = morton::encode_3d_8(v) as usize;
                    let via_trait = Morton3D::voxel_enc(&v);
                    assert_eq!(direct, via_trait);
                }
            }
        }
    }

    #[test]
    fn morton_trait_matches_direct_2d() {
        for y in [0u8, 1, 17, 128, 254, 255] {
            for x in [0u8, 1, 17, 128, 254, 255] {
                let v         = Vector2::new(x, y);
                let direct    = morton::encode_2d_8(v) as usize;
                let via_trait = Morton2D::voxel_enc(&v);
                assert_eq!(direct, via_trait);
            }
        }
    }
}
