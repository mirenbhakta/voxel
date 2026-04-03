//! Packed quad descriptor and layer occupancy.

#![allow(dead_code)]

use super::direction::Direction;

// ---------------------------------------------------------------------------
// QuadDescriptor
// ---------------------------------------------------------------------------

/// A packed quad descriptor encoding a rectangular face region.
///
/// The direction is not stored per quad -- it is implicit from the draw
/// command metadata (e.g. `gl_DrawID`).
///
/// Bit layout (25 bits used, 7 bits spare):
///
/// ```text
/// bits  0- 4: col       (5 bits, 0..31)
/// bits  5- 9: row       (5 bits, 0..31)
/// bits 10-14: layer     (5 bits, 0..31)
/// bits 15-19: width-1   (5 bits, 0..31 encodes width 1..32)
/// bits 20-24: height-1  (5 bits, 0..31 encodes height 1..32)
/// bits 25-31: unused
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuadDescriptor(u32);

impl QuadDescriptor {
    /// Pack a quad descriptor from its components.
    ///
    /// # Arguments
    ///
    /// * `col`    - Column position in the face layer (0..31).
    /// * `row`    - Row position in the face layer (0..31).
    /// * `layer`  - Layer index along the face normal axis (0..31).
    /// * `width`  - Quad width in columns (1..32).
    /// * `height` - Quad height in rows (1..32).
    pub fn new(
        col    : u8,
        row    : u8,
        layer  : u8,
        width  : u8,
        height : u8,
    ) -> Self
    {
        debug_assert!(col < 32);
        debug_assert!(row < 32);
        debug_assert!(layer < 32);
        debug_assert!((1..=32).contains(&width));
        debug_assert!((1..=32).contains(&height));

        let packed = (col as u32)
            | ((row as u32)          << 5)
            | ((layer as u32)        << 10)
            | (((width - 1) as u32)  << 15)
            | (((height - 1) as u32) << 20);

        QuadDescriptor(packed)
    }

    /// Column position (0..31).
    pub fn col(self) -> u8 {
        (self.0 & 0x1F) as u8
    }

    /// Row position (0..31).
    pub fn row(self) -> u8 {
        ((self.0 >> 5) & 0x1F) as u8
    }

    /// Layer index (0..31).
    pub fn layer(self) -> u8 {
        ((self.0 >> 10) & 0x1F) as u8
    }

    /// Quad width (1..32).
    pub fn width(self) -> u8 {
        (((self.0 >> 15) & 0x1F) + 1) as u8
    }

    /// Quad height (1..32).
    pub fn height(self) -> u8 {
        (((self.0 >> 20) & 0x1F) + 1) as u8
    }

    /// The raw packed `u32` value.
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// LayerOccupancy
// ---------------------------------------------------------------------------

/// Per-direction layer occupancy flags.
///
/// One `u32` per direction where bit `i` indicates that layer `i` contains
/// at least one face. Used for fast layer skipping during ray traversal.
pub struct LayerOccupancy {
    /// One word per direction, indexed by [`Direction`] discriminant.
    data : [u32; 6],
}

impl LayerOccupancy {
    /// Create a layer occupancy with no occupied layers in any direction.
    pub fn empty() -> Self {
        LayerOccupancy { data: [0; 6] }
    }

    /// Returns whether `layer` in `dir` contains any faces.
    pub fn has_faces(&self, dir: Direction, layer: u8) -> bool {
        self.data[dir as usize] & (1 << layer) != 0
    }

    /// Returns the index of the first occupied layer in `dir`, or `None`
    /// if no layers have faces.
    pub fn first_occupied_layer(&self, dir: Direction) -> Option<u8> {
        let word = self.data[dir as usize];

        if word == 0 {
            None
        }
        else {
            Some(word.trailing_zeros() as u8)
        }
    }

    /// Returns the raw word for a direction.
    pub fn raw(&self, dir: Direction) -> u32 {
        self.data[dir as usize]
    }

    /// Returns a mutable reference to the raw word for a direction.
    pub fn raw_mut(&mut self, dir: Direction) -> &mut u32 {
        &mut self.data[dir as usize]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- pack_unpack_roundtrip --

    #[test]
    fn pack_unpack_roundtrip() {
        let cases: &[(u8, u8, u8, u8, u8)] = &[
            (0,  0,  0,  1,  1),
            (31, 31, 31, 32, 32),
            (10, 20, 5,  8,  4),
            (0,  15, 31, 1,  32),
            (31, 0,  0,  32, 1),
        ];

        for &(col, row, layer, width, height) in cases {
            let q = QuadDescriptor::new(col, row, layer, width, height);

            assert_eq!(q.col(),    col,    "col mismatch for {cases:?}");
            assert_eq!(q.row(),    row,    "row mismatch");
            assert_eq!(q.layer(),  layer,  "layer mismatch");
            assert_eq!(q.width(),  width,  "width mismatch");
            assert_eq!(q.height(), height, "height mismatch");
        }
    }

    // -- known_values --

    #[test]
    fn known_values() {
        // col=1, row=0, layer=0, width=1, height=1
        // packed = 1 | 0 | 0 | 0 | 0 = 1
        let q = QuadDescriptor::new(1, 0, 0, 1, 1);
        assert_eq!(q.as_u32(), 1);

        // col=0, row=1, layer=0, width=1, height=1
        // packed = 0 | (1 << 5) | 0 | 0 | 0 = 32
        let q = QuadDescriptor::new(0, 1, 0, 1, 1);
        assert_eq!(q.as_u32(), 32);

        // col=0, row=0, layer=1, width=1, height=1
        // packed = 0 | 0 | (1 << 10) | 0 | 0 = 1024
        let q = QuadDescriptor::new(0, 0, 1, 1, 1);
        assert_eq!(q.as_u32(), 1024);
    }

    // -- layer_occupancy_empty --

    #[test]
    fn layer_occupancy_empty() {
        let lo = LayerOccupancy::empty();

        for &dir in &Direction::ALL {
            assert_eq!(lo.first_occupied_layer(dir), None);

            for layer in 0..32 {
                assert!(!lo.has_faces(dir, layer));
            }
        }
    }

    // -- layer_occupancy_single --

    #[test]
    fn layer_occupancy_single() {
        let mut lo = LayerOccupancy::empty();
        *lo.raw_mut(Direction::PosY) = 1 << 5;

        assert!(lo.has_faces(Direction::PosY, 5));
        assert!(!lo.has_faces(Direction::PosY, 4));
        assert_eq!(lo.first_occupied_layer(Direction::PosY), Some(5));

        // Other directions remain empty.
        assert_eq!(lo.first_occupied_layer(Direction::PosX), None);
    }
}
