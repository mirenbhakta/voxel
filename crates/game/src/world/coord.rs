//! Typed coordinates for the voxel world.
//!
//! Three coordinate types and an octant index:
//!
//! - [`Level`] — LOD level (L0 is finest).
//! - [`SubchunkCoord`] — integer coord of a sub-chunk within one level's grid.
//! - [`VoxelCoord`] — integer coord of a voxel within one level's grid.
//! - [`Octant`] — one of the 8 child positions inside a 2×2×2 parent.
//!
//! All coordinate types are level-agnostic integer triples — the level is
//! carried by context (the containing shell / sub-chunk). This avoids the
//! verbosity of tagging every coord with a level while keeping arithmetic
//! uniform across the LOD pyramid.

#![allow(dead_code)]

use crate::world::subchunk::SUBCHUNK_EDGE;

// --- Level ---

/// LOD level index. `Level(0)` is the finest; each increment doubles voxel
/// size. `Level(N)` voxels are `2^N` meters if L0 voxels are 1 m.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Level(pub u8);

impl Level {
    /// The finest LOD level.
    pub const ZERO: Level = Level(0);

    /// Voxel edge length at this level in meters, assuming L0 is 1 m/voxel.
    pub fn voxel_size_m(self) -> f32 {
        (1u32 << self.0) as f32
    }

    /// Sub-chunk edge length at this level in meters.
    pub fn subchunk_extent_m(self) -> f32 {
        (SUBCHUNK_EDGE as f32) * self.voxel_size_m()
    }

    /// The next coarser level. `None` if `self == Level(u8::MAX)`.
    pub fn parent(self) -> Option<Level> {
        self.0.checked_add(1).map(Level)
    }

    /// The next finer level. `None` if `self == Level::ZERO`.
    pub fn child(self) -> Option<Level> {
        self.0.checked_sub(1).map(Level)
    }
}

// --- Octant ---

/// One of the 8 child positions inside a 2×2×2 parent decomposition. Each
/// component is 0 or 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Octant {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

// --- SubchunkCoord ---

/// Integer 3D coordinate of a sub-chunk within one LOD level's grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubchunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl SubchunkCoord {
    /// Construct a sub-chunk coord from integer components.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// This sub-chunk's parent in the next coarser level.
    ///
    /// A parent sub-chunk covers a 2×2×2 block of children; its coord is
    /// each child coord floor-divided by 2 (Euclidean, so negative children
    /// map correctly).
    pub fn parent_subchunk(self) -> SubchunkCoord {
        SubchunkCoord::new(
            self.x.div_euclid(2),
            self.y.div_euclid(2),
            self.z.div_euclid(2),
        )
    }

    /// The octant this child occupies within its parent.
    pub fn parent_octant(self) -> Octant {
        Octant {
            x: self.x.rem_euclid(2) as u8,
            y: self.y.rem_euclid(2) as u8,
            z: self.z.rem_euclid(2) as u8,
        }
    }
}

// --- VoxelCoord ---

/// Integer 3D coordinate of a voxel within one LOD level's grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoxelCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl VoxelCoord {
    /// Construct a voxel coord from integer components.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// The sub-chunk containing this voxel at the same level.
    pub fn subchunk(self) -> SubchunkCoord {
        let e = SUBCHUNK_EDGE as i32;
        SubchunkCoord::new(
            self.x.div_euclid(e),
            self.y.div_euclid(e),
            self.z.div_euclid(e),
        )
    }

    /// The voxel's position within its enclosing sub-chunk. Each component
    /// lies in `[0, SUBCHUNK_EDGE)`.
    pub fn subchunk_local(self) -> (u8, u8, u8) {
        let e = SUBCHUNK_EDGE as i32;
        (
            self.x.rem_euclid(e) as u8,
            self.y.rem_euclid(e) as u8,
            self.z.rem_euclid(e) as u8,
        )
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    // -- Level --

    #[test]
    fn level_voxel_size_doubles_per_level() {
        assert_eq!(Level::ZERO.voxel_size_m(),       1.0);
        assert_eq!(Level(1).voxel_size_m(),          2.0);
        assert_eq!(Level(6).voxel_size_m(),         64.0);
    }

    #[test]
    fn level_subchunk_extent_is_edge_times_voxel_size() {
        assert_eq!(Level::ZERO.subchunk_extent_m(),   8.0);
        assert_eq!(Level(3).subchunk_extent_m(),     64.0);
    }

    #[test]
    fn level_parent_child_are_inverse() {
        let l = Level(3);
        assert_eq!(l.parent().unwrap().child().unwrap(), l);
        assert!(Level::ZERO.child().is_none());
        assert!(Level(u8::MAX).parent().is_none());
    }

    // -- SubchunkCoord --

    #[test]
    fn parent_subchunk_positive() {
        let c = SubchunkCoord::new(5, 6, 7);
        assert_eq!(c.parent_subchunk(), SubchunkCoord::new(2, 3, 3));
        assert_eq!(c.parent_octant(),   Octant { x: 1, y: 0, z: 1 });
    }

    #[test]
    fn parent_subchunk_negative_uses_euclidean() {
        // (-1, -1, -1) sits in octant (1, 1, 1) of parent (-1, -1, -1).
        let c = SubchunkCoord::new(-1, -1, -1);
        assert_eq!(c.parent_subchunk(), SubchunkCoord::new(-1, -1, -1));
        assert_eq!(c.parent_octant(),   Octant { x: 1, y: 1, z: 1 });

        // (-2, -2, -2) sits in octant (0, 0, 0) of parent (-1, -1, -1).
        let c = SubchunkCoord::new(-2, -2, -2);
        assert_eq!(c.parent_subchunk(), SubchunkCoord::new(-1, -1, -1));
        assert_eq!(c.parent_octant(),   Octant { x: 0, y: 0, z: 0 });
    }

    // -- VoxelCoord --

    #[test]
    fn voxel_to_subchunk_positive() {
        let v = VoxelCoord::new(9, 16, 23);
        assert_eq!(v.subchunk(),       SubchunkCoord::new(1, 2, 2));
        assert_eq!(v.subchunk_local(), (1, 0, 7));
    }

    #[test]
    fn voxel_to_subchunk_negative_uses_euclidean() {
        let v = VoxelCoord::new(-1, -8, -9);
        assert_eq!(v.subchunk(),       SubchunkCoord::new(-1, -1, -2));
        assert_eq!(v.subchunk_local(), (7, 0, 7));
    }
}
