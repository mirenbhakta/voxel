//! CPU-side mirrors of the light-dispatch kernel types declared in
//! `shaders/include/lights.hlsl`.
//!
//! Layout matches the HLSL `ConstantBuffer<LightList>` byte-for-byte. The
//! `layout_matches_shader_offsets` test enforces the contract.

use bytemuck::{Pod, Zeroable};

/// Maximum number of lights in a single frame's light list. Lockstep with
/// `MAX_LIGHTS` in `shaders/include/lights.hlsl`.
pub const MAX_LIGHTS: usize = 32;

/// Discriminant for [`LightDesc::kind`]. Lockstep with `LIGHT_KIND_*` in
/// `shaders/include/lights.hlsl`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightKind {
    Directional = 0,
    Point       = 1,
}

/// One entry in the light list. 48 bytes, 16-byte aligned for HLSL cbuffer
/// array stride. Three 16-byte rows:
///
/// ```text
///   row 0: position.xyz | kind
///   row 1: direction.xyz | radius
///   row 2: color.xyz | _pad
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightDesc {
    /// World-space position. Meaningful for point/spot; unused for directional.
    pub position:  [f32; 3],
    /// Discriminant — one of [`LightKind`] cast to `u32`.
    pub kind:      u32,

    /// Unit vector *toward* the light. Meaningful for directional; unused
    /// for point (the shader derives it from `position - hit`).
    pub direction: [f32; 3],
    /// Maximum effective range in world units. Meaningful for point/spot
    /// (contribution clamps to 0 at `d >= radius`). Unused for directional.
    pub radius:    f32,

    /// Linear-space radiance. For directional, the sun radiance at the
    /// surface. For point, the per-light reference radiance that combines
    /// with the distance attenuation term inside the kernel.
    pub color:     [f32; 3],
    pub _pad:      f32,
}

const _: () = assert!(
    std::mem::size_of::<LightDesc>() == 48,
    "LightDesc must be 48 bytes to match HLSL cbuffer array stride",
);

impl LightDesc {
    /// All-zero descriptor. `kind` is `Directional`, `color` is black — the
    /// shader's contribution for such a slot is zero. Safe default for
    /// unused trailing slots, though `count` should exclude them anyway.
    pub const fn zero() -> Self {
        Self {
            position:  [0.0; 3],
            kind:      LightKind::Directional as u32,
            direction: [0.0; 3],
            radius:    0.0,
            color:     [0.0; 3],
            _pad:      0.0,
        }
    }

    /// Directional light (e.g. sun). `direction` is the unit vector
    /// *toward* the light. `color` is linear-space radiance.
    pub fn directional(direction: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            position:  [0.0; 3],
            kind:      LightKind::Directional as u32,
            direction,
            radius:    0.0,
            color,
            _pad:      0.0,
        }
    }

    /// Point light. `position` is world-space, `radius` is the cutoff
    /// beyond which contribution is forced to zero, `color` is linear.
    pub fn point(position: [f32; 3], radius: f32, color: [f32; 3]) -> Self {
        Self {
            position,
            kind:      LightKind::Point as u32,
            direction: [0.0; 3],
            radius,
            color,
            _pad:      0.0,
        }
    }
}

/// Per-frame light list uploaded to the shade pass as a single
/// `ConstantBuffer<LightList>`. 1552 bytes total.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightList {
    /// Number of valid entries in `lights[0..count]`.
    pub count:  u32,
    pub _pad:   [u32; 3],
    pub lights: [LightDesc; MAX_LIGHTS],
}

const _: () = assert!(
    std::mem::size_of::<LightList>() == 16 + 48 * MAX_LIGHTS,
    "LightList size must match HLSL cbuffer layout (16-byte header + MAX_LIGHTS * 48)",
);

impl LightList {
    /// Empty list — zero lights, zero-initialised storage.
    pub const fn empty() -> Self {
        Self {
            count:  0,
            _pad:   [0; 3],
            lights: [LightDesc::zero(); MAX_LIGHTS],
        }
    }

    /// Construct from a slice, up to [`MAX_LIGHTS`]. Trailing slots are
    /// zero-padded. Panics if `src.len() > MAX_LIGHTS`.
    pub fn from_slice(src: &[LightDesc]) -> Self {
        assert!(
            src.len() <= MAX_LIGHTS,
            "LightList::from_slice: got {} lights, max is {MAX_LIGHTS}",
            src.len(),
        );
        let mut list = Self::empty();
        list.count = src.len() as u32;
        list.lights[..src.len()].copy_from_slice(src);
        list
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_desc_layout_matches_shader_offsets() {
        assert_eq!(std::mem::size_of::<LightDesc>(), 48);

        // Row 0: position at +0, kind at +12.
        // Row 1: direction at +16, radius at +28.
        // Row 2: color at +32, _pad at +44.
        let d = LightDesc {
            position:  [1.0, 2.0, 3.0],
            kind:      0x0A0B_0C0D,
            direction: [4.0, 5.0, 6.0],
            radius:    7.0,
            color:     [8.0, 9.0, 10.0],
            _pad:      0.0,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&d);

        assert_eq!(&bytes[ 0.. 4], 1.0f32.to_le_bytes());
        assert_eq!(&bytes[ 4.. 8], 2.0f32.to_le_bytes());
        assert_eq!(&bytes[ 8..12], 3.0f32.to_le_bytes());
        assert_eq!(&bytes[12..16], 0x0A0B_0C0Du32.to_le_bytes());
        assert_eq!(&bytes[16..20], 4.0f32.to_le_bytes());
        assert_eq!(&bytes[20..24], 5.0f32.to_le_bytes());
        assert_eq!(&bytes[24..28], 6.0f32.to_le_bytes());
        assert_eq!(&bytes[28..32], 7.0f32.to_le_bytes());
        assert_eq!(&bytes[32..36], 8.0f32.to_le_bytes());
        assert_eq!(&bytes[36..40], 9.0f32.to_le_bytes());
        assert_eq!(&bytes[40..44], 10.0f32.to_le_bytes());
    }

    #[test]
    fn light_list_layout_matches_shader_offsets() {
        // Header = 16 bytes; entries start at +16 with 48-byte stride.
        let expected = 16 + 48 * MAX_LIGHTS;
        assert_eq!(std::mem::size_of::<LightList>(), expected);

        let mut list = LightList::empty();
        list.count = 3;
        // Distinct values in every field so the test catches any misordering.
        list.lights[0] = LightDesc {
            position:  [1.0, 2.0, 3.0],
            kind:      LightKind::Point as u32,
            direction: [4.0, 5.0, 6.0],
            radius:    7.0,
            color:     [8.0, 9.0, 10.0],
            _pad:      0.0,
        };
        list.lights[1] = LightDesc {
            position:  [11.0, 12.0, 13.0],
            kind:      LightKind::Directional as u32,
            direction: [14.0, 15.0, 16.0],
            radius:    17.0,
            color:     [18.0, 19.0, 20.0],
            _pad:      0.0,
        };

        let bytes: &[u8] = bytemuck::bytes_of(&list);
        assert_eq!(bytes.len(), expected);

        // count at offset 0.
        assert_eq!(&bytes[0..4], 3u32.to_le_bytes());

        // lights[0] occupies offsets 16..64. Row 0: position(0..12) | kind(12..16).
        assert_eq!(&bytes[16..20], 1.0f32.to_le_bytes());
        assert_eq!(&bytes[20..24], 2.0f32.to_le_bytes());
        assert_eq!(&bytes[24..28], 3.0f32.to_le_bytes());
        assert_eq!(&bytes[28..32], (LightKind::Point as u32).to_le_bytes());
        // Row 1: direction(16..28) | radius(28..32).
        assert_eq!(&bytes[32..36], 4.0f32.to_le_bytes());
        assert_eq!(&bytes[36..40], 5.0f32.to_le_bytes());
        assert_eq!(&bytes[40..44], 6.0f32.to_le_bytes());
        assert_eq!(&bytes[44..48], 7.0f32.to_le_bytes());
        // Row 2: color(32..44) | _pad(44..48).
        assert_eq!(&bytes[48..52], 8.0f32.to_le_bytes());
        assert_eq!(&bytes[52..56], 9.0f32.to_le_bytes());
        assert_eq!(&bytes[56..60], 10.0f32.to_le_bytes());

        // lights[1] starts at offset 16 + 48 = 64.
        assert_eq!(&bytes[64..68],  11.0f32.to_le_bytes());
        assert_eq!(&bytes[76..80],  (LightKind::Directional as u32).to_le_bytes());
        assert_eq!(&bytes[92..96],  17.0f32.to_le_bytes());
        assert_eq!(&bytes[96..100], 18.0f32.to_le_bytes());
    }

    #[test]
    fn from_slice_pads_trailing_entries() {
        let input = [
            LightDesc::directional([0.0, 1.0, 0.0], [1.0, 1.0, 1.0]),
            LightDesc::point([5.0, 6.0, 7.0], 8.0, [0.5, 0.5, 0.5]),
        ];
        let list = LightList::from_slice(&input);
        assert_eq!(list.count, 2);
        assert_eq!(list.lights[0].kind, LightKind::Directional as u32);
        assert_eq!(list.lights[1].kind, LightKind::Point as u32);
        // Trailing entries are zero.
        for i in 2..MAX_LIGHTS {
            assert_eq!(list.lights[i].color, [0.0; 3]);
        }
    }

    #[test]
    #[should_panic(expected = "got 33 lights")]
    fn from_slice_panics_on_overflow() {
        let too_many = [LightDesc::zero(); MAX_LIGHTS + 1];
        let _ = LightList::from_slice(&too_many);
    }
}
