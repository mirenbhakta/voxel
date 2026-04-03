//! Axis-aligned direction enumeration.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Direction
// ---------------------------------------------------------------------------

/// One of six axis-aligned directions.
///
/// Ordered as paired positive/negative for each axis. The discriminant
/// serves as a direct index into per-direction arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Direction {
    /// Positive X (+1, 0, 0).
    PosX = 0,
    /// Negative X (-1, 0, 0).
    NegX = 1,
    /// Positive Y (0, +1, 0).
    PosY = 2,
    /// Negative Y (0, -1, 0).
    NegY = 3,
    /// Positive Z (0, 0, +1).
    PosZ = 4,
    /// Negative Z (0, 0, -1).
    NegZ = 5,
}

impl Direction {
    /// All six directions in discriminant order.
    pub const ALL: [Direction; 6] = [
        Direction::PosX, Direction::NegX,
        Direction::PosY, Direction::NegY,
        Direction::PosZ, Direction::NegZ,
    ];

    /// The axis index of this direction's normal (0 = X, 1 = Y, 2 = Z).
    pub fn normal_axis(self) -> usize {
        match self {
            Direction::PosX | Direction::NegX => 0,
            Direction::PosY | Direction::NegY => 1,
            Direction::PosZ | Direction::NegZ => 2,
        }
    }

    /// Returns `true` if this direction points along the positive axis.
    pub fn is_positive(self) -> bool {
        matches!(self, Direction::PosX | Direction::PosY | Direction::PosZ)
    }

    /// The row axis for face bitmask indexing (word index within a layer).
    ///
    /// For a face with this direction's normal, the "row" is the axis that
    /// indexes words within a layer: `face[layer * 32 + row]`.
    pub fn row_axis(self) -> usize {
        (self.normal_axis() + 2) % 3
    }

    /// The column axis for face bitmask indexing (bit position within a word).
    ///
    /// For a face with this direction's normal, the "column" is the axis
    /// packed into bit positions: bit `col` of `face[layer * 32 + row]`.
    pub fn col_axis(self) -> usize {
        (self.normal_axis() + 1) % 3
    }

    /// The direction pointing the opposite way along the same axis.
    pub fn opposite(self) -> Direction {
        match self {
            Direction::PosX => Direction::NegX,
            Direction::NegX => Direction::PosX,
            Direction::PosY => Direction::NegY,
            Direction::NegY => Direction::PosY,
            Direction::PosZ => Direction::NegZ,
            Direction::NegZ => Direction::PosZ,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- all_directions_unique --

    #[test]
    fn all_directions_unique() {
        for (i, &a) in Direction::ALL.iter().enumerate() {
            for (j, &b) in Direction::ALL.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // -- discriminant_is_index --

    #[test]
    fn discriminant_is_index() {
        for (i, &dir) in Direction::ALL.iter().enumerate() {
            assert_eq!(dir as usize, i);
        }
    }

    // -- opposite_roundtrip --

    #[test]
    fn opposite_roundtrip() {
        for &dir in &Direction::ALL {
            assert_eq!(dir.opposite().opposite(), dir);
            assert_ne!(dir, dir.opposite());
        }
    }

    // -- normal_axis_and_sign --

    #[test]
    fn normal_axis_and_sign() {
        assert_eq!(Direction::PosX.normal_axis(), 0);
        assert!(Direction::PosX.is_positive());

        assert_eq!(Direction::NegY.normal_axis(), 1);
        assert!(!Direction::NegY.is_positive());

        assert_eq!(Direction::PosZ.normal_axis(), 2);
        assert!(Direction::PosZ.is_positive());
    }

    // -- face_axis_swizzle --

    #[test]
    fn face_axis_swizzle() {
        // X faces: row=Z(2), col=Y(1)
        assert_eq!(Direction::PosX.row_axis(), 2);
        assert_eq!(Direction::PosX.col_axis(), 1);

        // Y faces: row=X(0), col=Z(2)
        assert_eq!(Direction::PosY.row_axis(), 0);
        assert_eq!(Direction::PosY.col_axis(), 2);

        // Z faces: row=Y(1), col=X(0)
        assert_eq!(Direction::PosZ.row_axis(), 1);
        assert_eq!(Direction::PosZ.col_axis(), 0);

        // All three axes accounted for in every direction.
        for &dir in &Direction::ALL {
            let n = dir.normal_axis();
            let r = dir.row_axis();
            let c = dir.col_axis();

            assert_ne!(n, r);
            assert_ne!(n, c);
            assert_ne!(r, c);
        }
    }
}
