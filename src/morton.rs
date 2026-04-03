//! Morton code (Z-order curve) encoding and decoding.
//!
//! Morton codes interleave the bits of two or three integer coordinates into a
//! single integer, producing a space-filling curve that preserves spatial
//! locality. This module provides encode and decode functions for 2D and 3D
//! coordinates in both standard and wide (64-bit) variants, as well as combined
//! variants that process multiple coordinates simultaneously using bit packing.

use eden_math::Vector2;
use eden_math::Vector3;

// --- 2D Primitives ---

/// Insert a zero bit after each of the 16 low bits of `x`.
#[inline]
pub fn part_1_by_1(x: u32) -> u32 {
    let mut x = x & 0x0000_FFFF;
    x = (x ^ (x <<  8)) & 0x00FF_00FF;
    x = (x ^ (x <<  4)) & 0x0F0F_0F0F;
    x = (x ^ (x <<  2)) & 0x3333_3333;
    x = (x ^ (x <<  1)) & 0x5555_5555;
    x
}

/// Remove all odd-indexed bits, compacting even bits into the 16 low bits.
#[inline]
pub fn compact_1_by_1(x: u32) -> u32 {
    let mut x = x & 0x5555_5555;
    x = (x ^ (x >>  1)) & 0x3333_3333;
    x = (x ^ (x >>  2)) & 0x0F0F_0F0F;
    x = (x ^ (x >>  4)) & 0x00FF_00FF;
    x = (x ^ (x >>  8)) & 0x0000_FFFF;
    x
}

/// Insert a zero bit after each of the 32 low bits of `x`.
#[inline]
pub fn part_1_by_1_wide(x: u64) -> u64 {
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
pub fn compact_1_by_1_wide(x: u64) -> u64 {
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
pub fn part_1_by_2(x: u32) -> u32 {
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
pub fn compact_1_by_2(x: u32) -> u32 {
    let mut x = x & 0x0924_9249;
    x = (x ^ (x >>  2)) & 0x030C_30C3;
    x = (x ^ (x >>  4)) & 0x0300_F00F;
    x = (x ^ (x >>  8)) & 0xFF00_00FF;
    x = (x ^ (x >> 16)) & 0x0000_03FF;
    x
}

/// Insert two zero bits after each of the 21 low bits of `x`.
#[inline]
pub fn part_1_by_2_wide(x: u64) -> u64 {
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
pub fn compact_1_by_2_wide(x: u64) -> u64 {
    let mut x = x & 0x1249_2492_4924_9249;
    x = (x ^ (x >>  2)) & 0x10C3_0C30_C30C_30C3;
    x = (x ^ (x >>  4)) & 0x100F_00F0_0F00_F00F;
    x = (x ^ (x >>  8)) & 0x001F_0000_FF00_00FF;
    x = (x ^ (x >> 16)) & 0x001F_0000_0000_FFFF;
    x = (x ^ (x >> 32)) & 0x0000_0000_001F_FFFF;
    x
}

// --- 2D Encode/Decode ---

/// Encode two 16-bit coordinates into a 32-bit Morton code.
#[inline]
pub fn encode_2d(v: Vector2<u32>) -> u32 {
    part_1_by_1(v.x) | (part_1_by_1(v.y) << 1)
}

/// Decode a 32-bit Morton code into two 16-bit coordinates.
#[inline]
pub fn decode_2d(code: u32) -> Vector2<u32> {
    Vector2::new(compact_1_by_1(code), compact_1_by_1(code >> 1))
}

/// Encode two 32-bit coordinates into a 64-bit Morton code.
#[inline]
pub fn encode_2d_wide(v: Vector2<u32>) -> u64 {
    part_1_by_1_wide(v.x as u64) | (part_1_by_1_wide(v.y as u64) << 1)
}

/// Decode a 64-bit Morton code into two 32-bit coordinates.
#[inline]
pub fn decode_2d_wide(code: u64) -> Vector2<u32> {
    Vector2::new(
        compact_1_by_1_wide(code) as u32,
        compact_1_by_1_wide(code >> 1) as u32,
    )
}

// --- 3D Encode/Decode ---

/// Encode three 10-bit coordinates into a 30-bit Morton code.
#[inline]
pub fn encode_3d(v: Vector3<u32>) -> u32 {
    part_1_by_2(v.x) | (part_1_by_2(v.y) << 1) | (part_1_by_2(v.z) << 2)
}

/// Decode a 30-bit Morton code into three 10-bit coordinates.
#[inline]
pub fn decode_3d(code: u32) -> Vector3<u32> {
    Vector3::new(
        compact_1_by_2(code),
        compact_1_by_2(code >> 1),
        compact_1_by_2(code >> 2),
    )
}

/// Encode three 21-bit coordinates into a 63-bit Morton code.
#[inline]
pub fn encode_3d_wide(v: Vector3<u32>) -> u64 {
    part_1_by_2_wide(v.x as u64)
        | (part_1_by_2_wide(v.y as u64) << 1)
        | (part_1_by_2_wide(v.z as u64) << 2)
}

/// Decode a 63-bit Morton code into three 21-bit coordinates.
#[inline]
pub fn decode_3d_wide(code: u64) -> Vector3<u32> {
    Vector3::new(
        compact_1_by_2_wide(code) as u32,
        compact_1_by_2_wide(code >> 1) as u32,
        compact_1_by_2_wide(code >> 2) as u32,
    )
}

// --- Combined 2D ---

/// Encode two 8-bit coordinates into a 16-bit Morton code.
///
/// Processes both coordinates simultaneously by packing them into separate
/// halves of a single register, requiring fewer operations than two separate
/// calls to [`part_1_by_1`].
#[inline]
pub fn encode_2d_combined(v: Vector2<u32>) -> u32 {
    // Pack x at bits 0-7, y at bits 16-23. The 16-bit spacing means the
    // shift-8 step of Part1By1 is already accomplished by the packing.
    let mut t = (v.x & 0xFF) | ((v.y & 0xFF) << 16);
    t = (t ^ (t << 4)) & 0x0F0F_0F0F;
    t = (t ^ (t << 2)) & 0x3333_3333;
    t = (t ^ (t << 1)) & 0x5555_5555;

    // Merge the two halves into a contiguous 16-bit code.
    (t & 0xFFFF) | (t >> 15)
}

/// Decode a 16-bit Morton code into two 8-bit coordinates.
///
/// Inverse of [`encode_2d_combined`]. Separates x and y bits back into packed
/// halves, then compacts each half.
#[inline]
pub fn decode_2d_combined(code: u32) -> Vector2<u32> {
    let code = code & 0xFFFF;

    // Separate x (even bits) and y (odd bits) back into packed halves.
    let mut v = (code & 0x5555) | ((code & 0xAAAA) << 15);
    v = (v ^ (v >> 1)) & 0x3333_3333;
    v = (v ^ (v >> 2)) & 0x0F0F_0F0F;
    v = (v ^ (v >> 4)) & 0x00FF_00FF;

    Vector2::new(v & 0xFF, (v >> 16) & 0xFF)
}

// --- Combined 3D ---

/// Encode three 7-bit coordinates into a 21-bit Morton code.
///
/// Processes all three coordinates simultaneously by packing them at 21-bit
/// offsets in a `u64` intermediary, requiring fewer operations than three
/// separate calls to [`part_1_by_2`].
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

    fn v2(x: u32, y: u32) -> Vector2<u32> { Vector2::new(x, y) }
    fn v3(x: u32, y: u32, z: u32) -> Vector3<u32> { Vector3::new(x, y, z) }

    #[test]
    fn roundtrip_2d() {
        let cases = [v2(0, 0), v2(1, 1), v2(0xFFFF, 0xFFFF), v2(5, 10), v2(0, 0xFFFF)];
        for v in cases {
            let code = encode_2d(v);
            assert_eq!(decode_2d(code), v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn roundtrip_2d_wide() {
        let cases = [
            v2(0, 0),
            v2(1, 1),
            v2(0xFFFF_FFFF, 0xFFFF_FFFF),
            v2(5, 10),
            v2(0, 0xFFFF_FFFF),
        ];
        for v in cases {
            let code = encode_2d_wide(v);
            assert_eq!(decode_2d_wide(code), v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn roundtrip_3d() {
        let cases = [
            v3(0, 0, 0),
            v3(1, 1, 1),
            v3(1023, 1023, 1023),
            v3(5, 10, 15),
            v3(0, 0, 1023),
        ];
        for v in cases {
            let code = encode_3d(v);
            assert_eq!(decode_3d(code), v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn roundtrip_3d_wide() {
        let cases = [
            v3(0, 0, 0),
            v3(1, 1, 1),
            v3(0x1F_FFFF, 0x1F_FFFF, 0x1F_FFFF),
            v3(5, 10, 15),
            v3(0, 0, 0x1F_FFFF),
        ];
        for v in cases {
            let code = encode_3d_wide(v);
            assert_eq!(decode_3d_wide(code), v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn roundtrip_2d_combined() {
        let cases = [v2(0, 0), v2(1, 1), v2(255, 255), v2(5, 10), v2(0, 255)];
        for v in cases {
            let code = encode_2d_combined(v);
            assert_eq!(decode_2d_combined(code), v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn roundtrip_3d_combined() {
        let cases = [
            v3(0, 0, 0),
            v3(1, 1, 1),
            v3(127, 127, 127),
            v3(5, 10, 15),
            v3(0, 0, 127),
        ];
        for v in cases {
            let code = encode_3d_combined(v);
            assert_eq!(decode_3d_combined(code), v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn known_values_2d() {
        assert_eq!(encode_2d(v2(0, 0)), 0);
        assert_eq!(encode_2d(v2(1, 0)), 1);
        assert_eq!(encode_2d(v2(0, 1)), 2);
        assert_eq!(encode_2d(v2(1, 1)), 3);
        assert_eq!(encode_2d(v2(2, 0)), 4);
    }

    #[test]
    fn known_values_3d() {
        assert_eq!(encode_3d(v3(1, 0, 0)), 1);
        assert_eq!(encode_3d(v3(0, 1, 0)), 2);
        assert_eq!(encode_3d(v3(0, 0, 1)), 4);
        assert_eq!(encode_3d(v3(1, 1, 1)), 7);
    }

    #[test]
    fn cross_validate_2d_combined() {
        // For 8-bit coordinates, encode_2d and encode_2d_combined must agree.
        for x in [0, 1, 2, 17, 128, 254, 255] {
            for y in [0, 1, 2, 17, 128, 254, 255] {
                let v        = v2(x, y);
                let standard = encode_2d(v) & 0xFFFF;
                let combined = encode_2d_combined(v);
                assert_eq!(
                    standard, combined,
                    "2d mismatch for ({x}, {y}): standard={standard:#X}, combined={combined:#X}",
                );
            }
        }
    }

    #[test]
    fn cross_validate_3d_combined() {
        // For 7-bit coordinates, encode_3d and encode_3d_combined must agree.
        for x in [0, 1, 2, 17, 64, 126, 127] {
            for y in [0, 1, 2, 17, 64, 126, 127] {
                for z in [0, 1, 2, 17, 64, 126, 127] {
                    let v        = v3(x, y, z);
                    let standard = encode_3d(v) & 0x1F_FFFF;
                    let combined = encode_3d_combined(v);
                    assert_eq!(
                        standard, combined,
                        "3d mismatch for ({x}, {y}, {z}): \
                         standard={standard:#X}, combined={combined:#X}",
                    );
                }
            }
        }
    }
}
