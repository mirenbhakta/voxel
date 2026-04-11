//! Morton code (Z-order curve) encoding and decoding.
//!
//! Morton codes interleave the bits of two or three integer coordinates into a
//! single integer, producing a space-filling curve that preserves spatial
//! locality. This module provides typed encode and decode functions for 2D and
//! 3D coordinates across multiple bit widths, as well as combined variants that
//! process multiple coordinates simultaneously using bit packing.

#![allow(dead_code)]

use eden_math::Vector2;
use eden_math::Vector3;

// --- 2D Primitives ---

/// Insert a zero bit after each of the 16 low bits of `x`.
#[inline]
fn part_1_by_1(x: u32) -> u32 {
    let mut x = x & 0x0000_FFFF;
    x = (x ^ (x <<  8)) & 0x00FF_00FF;
    x = (x ^ (x <<  4)) & 0x0F0F_0F0F;
    x = (x ^ (x <<  2)) & 0x3333_3333;
    x = (x ^ (x <<  1)) & 0x5555_5555;
    x
}

/// Remove all odd-indexed bits, compacting even bits into the 16 low bits.
#[inline]
fn compact_1_by_1(x: u32) -> u32 {
    let mut x = x & 0x5555_5555;
    x = (x ^ (x >>  1)) & 0x3333_3333;
    x = (x ^ (x >>  2)) & 0x0F0F_0F0F;
    x = (x ^ (x >>  4)) & 0x00FF_00FF;
    x = (x ^ (x >>  8)) & 0x0000_FFFF;
    x
}

/// Insert a zero bit after each of the 32 low bits of `x`.
#[inline]
fn part_1_by_1_wide(x: u64) -> u64 {
    let mut x = x & 0x0000_0000_FFFF_FFFF;
    x = (x ^ (x << 16)) & 0x0000_FFFF_0000_FFFF;
    x = (x ^ (x <<  8)) & 0x00FF_00FF_00FF_00FF;
    x = (x ^ (x <<  4)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x ^ (x <<  2)) & 0x3333_3333_3333_3333;
    x = (x ^ (x <<  1)) & 0x5555_5555_5555_5555;
    x
}

/// Remove all odd-indexed bits from a 64-bit value, compacting even bits into
/// the 32 low bits.
#[inline]
fn compact_1_by_1_wide(x: u64) -> u64 {
    let mut x = x & 0x5555_5555_5555_5555;
    x = (x ^ (x >>  1)) & 0x3333_3333_3333_3333;
    x = (x ^ (x >>  2)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x ^ (x >>  4)) & 0x00FF_00FF_00FF_00FF;
    x = (x ^ (x >>  8)) & 0x0000_FFFF_0000_FFFF;
    x = (x ^ (x >> 16)) & 0x0000_0000_FFFF_FFFF;
    x
}

// --- 3D Primitives ---

/// Insert two zero bits after each of the 10 low bits of `x`.
#[inline]
fn part_1_by_2(x: u32) -> u32 {
    let mut x = x & 0x0000_03FF;
    x = (x ^ (x << 16)) & 0xFF00_00FF;
    x = (x ^ (x <<  8)) & 0x0300_F00F;
    x = (x ^ (x <<  4)) & 0x030C_30C3;
    x = (x ^ (x <<  2)) & 0x0924_9249;
    x
}

/// Remove all bits not at positions divisible by 3, compacting them into the
/// 10 low bits.
#[inline]
fn compact_1_by_2(x: u32) -> u32 {
    let mut x = x & 0x0924_9249;
    x = (x ^ (x >>  2)) & 0x030C_30C3;
    x = (x ^ (x >>  4)) & 0x0300_F00F;
    x = (x ^ (x >>  8)) & 0xFF00_00FF;
    x = (x ^ (x >> 16)) & 0x0000_03FF;
    x
}

/// Insert two zero bits after each of the 21 low bits of `x`.
#[inline]
fn part_1_by_2_wide(x: u64) -> u64 {
    let mut x = x & 0x001F_FFFF;
    x = (x ^ (x << 32)) & 0x001F_0000_0000_FFFF;
    x = (x ^ (x << 16)) & 0x001F_0000_FF00_00FF;
    x = (x ^ (x <<  8)) & 0x100F_00F0_0F00_F00F;
    x = (x ^ (x <<  4)) & 0x10C3_0C30_C30C_30C3;
    x = (x ^ (x <<  2)) & 0x1249_2492_4924_9249;
    x
}

/// Remove all bits not at positions divisible by 3 from a 64-bit value,
/// compacting them into the 21 low bits.
#[inline]
fn compact_1_by_2_wide(x: u64) -> u64 {
    let mut x = x & 0x1249_2492_4924_9249;
    x = (x ^ (x >>  2)) & 0x10C3_0C30_C30C_30C3;
    x = (x ^ (x >>  4)) & 0x100F_00F0_0F00_F00F;
    x = (x ^ (x >>  8)) & 0x001F_0000_FF00_00FF;
    x = (x ^ (x >> 16)) & 0x001F_0000_0000_FFFF;
    x = (x ^ (x >> 32)) & 0x0000_0000_001F_FFFF;
    x
}

// --- 2D Encode/Decode ---

/// Encode two 8-bit coordinates into a 16-bit Morton code.
///
/// Processes both coordinates simultaneously by packing them into separate
/// halves of a single register, requiring fewer operations than two independent
/// calls to the spread kernel.
#[inline]
pub fn encode_2d_8(v: Vector2<u8>) -> u16 {
    // Pack x at bits 0-7, y at bits 16-23. The 16-bit spacing means the
    // shift-8 step of Part1By1 is already accomplished by the packing.
    let mut t = (v.x as u32 & 0xFF) | ((v.y as u32 & 0xFF) << 16);
    t = (t ^ (t << 4)) & 0x0F0F_0F0F;
    t = (t ^ (t << 2)) & 0x3333_3333;
    t = (t ^ (t << 1)) & 0x5555_5555;

    // Merge the two halves into a contiguous 16-bit code.
    ((t & 0xFFFF) | (t >> 15)) as u16
}

/// Decode a 16-bit Morton code into two 8-bit coordinates.
///
/// Inverse of [`encode_2d_8`]. Separates x and y bits back into packed halves,
/// then compacts each half.
#[inline]
pub fn decode_2d_8(code: u16) -> Vector2<u8> {
    let code = code as u32 & 0xFFFF;

    // Separate x (even bits) and y (odd bits) back into packed halves.
    let mut v = (code & 0x5555) | ((code & 0xAAAA) << 15);
    v = (v ^ (v >> 1)) & 0x3333_3333;
    v = (v ^ (v >> 2)) & 0x0F0F_0F0F;
    v = (v ^ (v >> 4)) & 0x00FF_00FF;

    Vector2::new((v & 0xFF) as u8, ((v >> 16) & 0xFF) as u8)
}

// Generates encode_2d_{width} and decode_2d_{width} for 16- and 32-bit inputs.
macro_rules! impl_morton_2d {
    // 16-bit: each coordinate fits in u16, output is u32.
    (16) => {
        /// Encode two 16-bit coordinates into a 32-bit Morton code.
        #[inline]
        pub fn encode_2d_16(v: Vector2<u16>) -> u32 {
            part_1_by_1(v.x as u32) | (part_1_by_1(v.y as u32) << 1)
        }

        /// Decode a 32-bit Morton code into two 16-bit coordinates.
        #[inline]
        pub fn decode_2d_16(code: u32) -> Vector2<u16> {
            Vector2::new(
                compact_1_by_1(code) as u16,
                compact_1_by_1(code >> 1) as u16,
            )
        }
    };

    // 32-bit: each coordinate fits in u32, output is u64.
    (32) => {
        /// Encode two 32-bit coordinates into a 64-bit Morton code.
        #[inline]
        pub fn encode_2d_32(v: Vector2<u32>) -> u64 {
            part_1_by_1_wide(v.x as u64) | (part_1_by_1_wide(v.y as u64) << 1)
        }

        /// Decode a 64-bit Morton code into two 32-bit coordinates.
        #[inline]
        pub fn decode_2d_32(code: u64) -> Vector2<u32> {
            Vector2::new(
                compact_1_by_1_wide(code) as u32,
                compact_1_by_1_wide(code >> 1) as u32,
            )
        }
    };
}

impl_morton_2d!(16);
impl_morton_2d!(32);

// --- 3D Encode/Decode ---

// Generates encode_3d_{width} and decode_3d_{width} for 8- and 16-bit inputs.
macro_rules! impl_morton_3d {
    // 8-bit: each coordinate fits in u8 (low 8 bits used), output is u32.
    (8) => {
        /// Encode three 8-bit coordinates into a 24-bit Morton code.
        #[inline]
        pub fn encode_3d_8(v: Vector3<u8>) -> u32 {
            part_1_by_2(v.x as u32)
                | (part_1_by_2(v.y as u32) << 1)
                | (part_1_by_2(v.z as u32) << 2)
        }

        /// Decode a 24-bit Morton code into three 8-bit coordinates.
        #[inline]
        pub fn decode_3d_8(code: u32) -> Vector3<u8> {
            Vector3::new(
                compact_1_by_2(code) as u8,
                compact_1_by_2(code >> 1) as u8,
                compact_1_by_2(code >> 2) as u8,
            )
        }
    };

    // 16-bit: each coordinate fits in u16 (low 16 bits used), output is u64.
    (16) => {
        /// Encode three 16-bit coordinates into a 48-bit Morton code.
        #[inline]
        pub fn encode_3d_16(v: Vector3<u16>) -> u64 {
            part_1_by_2_wide(v.x as u64)
                | (part_1_by_2_wide(v.y as u64) << 1)
                | (part_1_by_2_wide(v.z as u64) << 2)
        }

        /// Decode a 48-bit Morton code into three 16-bit coordinates.
        #[inline]
        pub fn decode_3d_16(code: u64) -> Vector3<u16> {
            Vector3::new(
                compact_1_by_2_wide(code) as u16,
                compact_1_by_2_wide(code >> 1) as u16,
                compact_1_by_2_wide(code >> 2) as u16,
            )
        }
    };
}

impl_morton_3d!(8);
impl_morton_3d!(16);

// --- Combined 3D ---

/// Encode three 7-bit coordinates into a 21-bit Morton code.
///
/// Processes all three coordinates simultaneously by packing them at 21-bit
/// offsets in a `u64` intermediary, requiring fewer operations than three
/// separate calls to the spread kernel.
#[inline]
pub fn encode_3d_combined(v: Vector3<u32>) -> u32 {
    // Pack x at bits 0-6, y at bits 21-27, z at bits 42-48. The 21-bit
    // spacing gives each coordinate room to spread to every 3rd bit without
    // interference. The shift-16 step of Part1By2 is a no-op for 7-bit
    // inputs at these offsets, leaving only 3 steps.
    let mut t: u64 = (v.x as u64 & 0x7F)
                   | ((v.y as u64 & 0x7F) << 21)
                   | ((v.z as u64 & 0x7F) << 42);

    t = (t ^ (t << 8)) & 0x01C0_3C0E_01E0_700F;
    t = (t ^ (t << 4)) & 0x10C3_0C86_1864_30C3;
    t = (t ^ (t << 2)) & 0x1249_2492_4924_9249;

    // x bits at positions 0,3,...,18. y bits at 21,24,...,39. z bits at
    // 42,45,...,60. Merge by shifting y down by 20 and z down by 40.
    ((t | (t >> 20) | (t >> 40)) & 0x1F_FFFF) as u32
}

/// Decode a 21-bit Morton code into three 7-bit coordinates.
///
/// Inverse of [`encode_3d_combined`]. Separates x, y, and z bits and spaces
/// them back to 21-bit offsets, then compacts each group.
#[inline]
pub fn decode_3d_combined(code: u32) -> Vector3<u32> {
    let code = code as u64 & 0x1F_FFFF;

    // x at every 3rd bit from 0, y from 1, z from 2. Space them back to
    // offsets 0, 21, 42.
    let mut v = (code & 0x4_9249)
              | ((code & 0x9_2492) << 20)
              | ((code & 0x12_4924) << 40);

    // Compact (reverse of spread steps, applied in reverse order).
    v = (v ^ (v >> 2)) & 0x10C3_0C86_1864_30C3;
    v = (v ^ (v >> 4)) & 0x01C0_3C0E_01E0_700F;
    v = (v ^ (v >> 8)) & 0x0001_FC00_0FE0_007F;

    Vector3::new(
        (v & 0x7F) as u32,
        ((v >> 21) & 0x7F) as u32,
        ((v >> 42) & 0x7F) as u32,
    )
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn v2_8(x: u8, y: u8)     -> Vector2<u8>  { Vector2::new(x, y) }
    fn v2_16(x: u16, y: u16)  -> Vector2<u16> { Vector2::new(x, y) }
    fn v2_32(x: u32, y: u32)  -> Vector2<u32> { Vector2::new(x, y) }
    fn v3_8(x: u8, y: u8, z: u8)     -> Vector3<u8>  { Vector3::new(x, y, z) }
    fn v3_16(x: u16, y: u16, z: u16) -> Vector3<u16> { Vector3::new(x, y, z) }
    fn v3_32(x: u32, y: u32, z: u32) -> Vector3<u32> { Vector3::new(x, y, z) }

    #[test]
    fn roundtrip_2d_8() {
        let cases = [v2_8(0, 0), v2_8(1, 1), v2_8(255, 255), v2_8(5, 10), v2_8(0, 255)];
        for v in cases {
            let code = encode_2d_8(v);
            assert_eq!(decode_2d_8(code), v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn roundtrip_2d_16() {
        let cases = [
            v2_16(0, 0),
            v2_16(1, 1),
            v2_16(0xFFFF, 0xFFFF),
            v2_16(5, 10),
            v2_16(0, 0xFFFF),
        ];
        for v in cases {
            let code = encode_2d_16(v);
            assert_eq!(decode_2d_16(code), v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn roundtrip_2d_32() {
        let cases = [
            v2_32(0, 0),
            v2_32(1, 1),
            v2_32(0xFFFF_FFFF, 0xFFFF_FFFF),
            v2_32(5, 10),
            v2_32(0, 0xFFFF_FFFF),
        ];
        for v in cases {
            let code = encode_2d_32(v);
            assert_eq!(decode_2d_32(code), v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn roundtrip_3d_8() {
        let cases = [
            v3_8(0, 0, 0),
            v3_8(1, 1, 1),
            v3_8(255, 255, 255),
            v3_8(5, 10, 15),
            v3_8(0, 0, 255),
        ];
        for v in cases {
            let code = encode_3d_8(v);
            assert_eq!(decode_3d_8(code), v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn roundtrip_3d_16() {
        let cases = [
            v3_16(0, 0, 0),
            v3_16(1, 1, 1),
            v3_16(0xFFFF, 0xFFFF, 0xFFFF),
            v3_16(5, 10, 15),
            v3_16(0, 0, 0xFFFF),
        ];
        for v in cases {
            let code = encode_3d_16(v);
            assert_eq!(decode_3d_16(code), v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn roundtrip_3d_combined() {
        let cases = [
            v3_32(0, 0, 0),
            v3_32(1, 1, 1),
            v3_32(127, 127, 127),
            v3_32(5, 10, 15),
            v3_32(0, 0, 127),
        ];
        for v in cases {
            let code = encode_3d_combined(v);
            assert_eq!(decode_3d_combined(code), v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn known_values_2d() {
        assert_eq!(encode_2d_16(v2_16(0, 0)), 0);
        assert_eq!(encode_2d_16(v2_16(1, 0)), 1);
        assert_eq!(encode_2d_16(v2_16(0, 1)), 2);
        assert_eq!(encode_2d_16(v2_16(1, 1)), 3);
        assert_eq!(encode_2d_16(v2_16(2, 0)), 4);
    }

    #[test]
    fn known_values_3d() {
        assert_eq!(encode_3d_8(v3_8(1, 0, 0)), 1);
        assert_eq!(encode_3d_8(v3_8(0, 1, 0)), 2);
        assert_eq!(encode_3d_8(v3_8(0, 0, 1)), 4);
        assert_eq!(encode_3d_8(v3_8(1, 1, 1)), 7);
    }

    #[test]
    fn cross_validate_2d_8_vs_16() {
        // For 8-bit coordinates, encode_2d_8 and encode_2d_16 must agree.
        for x in [0u8, 1, 2, 17, 128, 254, 255] {
            for y in [0u8, 1, 2, 17, 128, 254, 255] {
                let by_8  = encode_2d_8(v2_8(x, y)) as u32;
                let by_16 = encode_2d_16(v2_16(x as u16, y as u16)) & 0xFFFF;
                assert_eq!(
                    by_8, by_16,
                    "2d mismatch for ({x}, {y}): encode_2d_8={by_8:#X}, encode_2d_16={by_16:#X}",
                );
            }
        }
    }

    #[test]
    fn cross_validate_3d_8_vs_combined() {
        // For 7-bit coordinates, encode_3d_8 and encode_3d_combined must agree.
        for x in [0u8, 1, 2, 17, 64, 126, 127] {
            for y in [0u8, 1, 2, 17, 64, 126, 127] {
                for z in [0u8, 1, 2, 17, 64, 126, 127] {
                    let by_8        = encode_3d_8(v3_8(x, y, z)) & 0x1F_FFFF;
                    let by_combined = encode_3d_combined(v3_32(x as u32, y as u32, z as u32));
                    assert_eq!(
                        by_8, by_combined,
                        "3d mismatch for ({x}, {y}, {z}): \
                         encode_3d_8={by_8:#X}, encode_3d_combined={by_combined:#X}",
                    );
                }
            }
        }
    }
}
