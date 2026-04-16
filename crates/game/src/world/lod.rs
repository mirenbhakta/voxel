//! LOD pyramid traversal helpers.
//!
//! Utilities that sit between [`Level`](crate::world::coord::Level) /
//! [`SubchunkCoord`](crate::world::coord::SubchunkCoord) arithmetic and the
//! residency / OR-reduction machinery. The per-coord parent/octant queries
//! live on `SubchunkCoord` itself; this module provides the
//! one-to-many direction (parent → its 8 children) and a flat enumeration
//! of octants for iteration.

#![allow(dead_code)]

use crate::world::coord::{Octant, SubchunkCoord};

// --- child/octant enumeration ---

/// The 8 children of `parent` at the next-finer LOD level.
///
/// Children span a 2×2×2 block starting at `(parent * 2)`. Order matches
/// [`OCTANTS`]: x varies fastest, then y, then z.
pub fn children_of(parent: SubchunkCoord) -> [SubchunkCoord; 8] {
    let base_x = parent.x * 2;
    let base_y = parent.y * 2;
    let base_z = parent.z * 2;

    let mut out = [SubchunkCoord::new(0, 0, 0); 8];
    let mut i   = 0;
    for dz in 0..2i32 {
        for dy in 0..2i32 {
            for dx in 0..2i32 {
                out[i] = SubchunkCoord::new(base_x + dx, base_y + dy, base_z + dz);
                i += 1;
            }
        }
    }
    out
}

/// All 8 octants, ordered so that `OCTANTS[i]` matches the i-th entry of
/// [`children_of`].
pub const OCTANTS: [Octant; 8] = [
    Octant { x: 0, y: 0, z: 0 },
    Octant { x: 1, y: 0, z: 0 },
    Octant { x: 0, y: 1, z: 0 },
    Octant { x: 1, y: 1, z: 0 },
    Octant { x: 0, y: 0, z: 1 },
    Octant { x: 1, y: 0, z: 1 },
    Octant { x: 0, y: 1, z: 1 },
    Octant { x: 1, y: 1, z: 1 },
];

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn children_all_report_parent() {
        let p = SubchunkCoord::new(3, -2, 5);
        for c in children_of(p) {
            assert_eq!(c.parent_subchunk(), p);
        }
    }

    #[test]
    fn children_cover_all_octants() {
        let children = children_of(SubchunkCoord::new(7, 7, 7));
        let mut seen: std::collections::HashSet<(u8, u8, u8)> =
            std::collections::HashSet::new();
        for c in children {
            let o = c.parent_octant();
            seen.insert((o.x, o.y, o.z));
        }
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn children_align_with_octants_array() {
        let children = children_of(SubchunkCoord::new(0, 0, 0));
        for (child, octant) in children.iter().zip(OCTANTS.iter()) {
            assert_eq!(child.parent_octant(), *octant);
        }
    }

    #[test]
    fn children_of_negative_parent() {
        // Parent (-1,-1,-1): base = (-2,-2,-2). Children cover x,y,z ∈ {-2,-1}.
        let children = children_of(SubchunkCoord::new(-1, -1, -1));
        assert!(children.contains(&SubchunkCoord::new(-2, -2, -2)));
        assert!(children.contains(&SubchunkCoord::new(-1, -1, -1)));
        for c in children {
            assert_eq!(c.parent_subchunk(), SubchunkCoord::new(-1, -1, -1));
        }
    }
}
