//! Block type definitions and material properties.
//!
//! Each voxel in the world has a [`BlockId`] identifying its type. The
//! [`BlockRegistry`] maps block IDs to [`Block`] definitions carrying
//! the visual [`Material`] needed by the rendering pipeline.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// BlockId
// ---------------------------------------------------------------------------

/// A global block type identifier.
///
/// The zero value represents air (empty space). All other values identify
/// a concrete block type defined in the [`BlockRegistry`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BlockId(u16);

impl BlockId {
    /// The empty block. A voxel with this ID is not occupied.
    pub const AIR: BlockId = BlockId(0);

    /// Create a block identifier from a raw value.
    pub fn new(id: u16) -> Self {
        BlockId(id)
    }

    /// Returns `true` if this is the air (empty) block.
    pub fn is_air(self) -> bool {
        self.0 == 0
    }

    /// Returns the underlying integer value.
    pub fn raw(self) -> u16 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// FaceTexture
// ---------------------------------------------------------------------------

/// Per-face texture assignment for a block type.
///
/// Direction index order: +X, -X, +Y, -Y, +Z, -Z. This matches
/// the `Direction` enum discriminants in the render module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceTexture {
    /// Same texture on all six faces.
    Uniform(u16),
    /// Different textures per face, indexed by direction.
    PerFace([u16; 6]),
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// Visual properties of a block type.
///
/// Defines how a block appears when rendered. The fragment shader reads
/// block identity from the volumetric material array and resolves it to
/// surface color and texture through the block registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Material {
    /// Diffuse color as packed RGBA bytes (sRGB, alpha reserved).
    color   : [u8; 4],
    /// Texture assignment for block faces.
    texture : FaceTexture,
}

impl Material {
    /// Create a material from RGB color components.
    ///
    /// Alpha is set to 255 (fully opaque). Texture defaults to uniform
    /// index 0 (untextured).
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Material {
            color   : [r, g, b, 255],
            texture : FaceTexture::Uniform(0),
        }
    }

    /// Set a uniform texture for all faces.
    pub fn with_texture(mut self, index: u16) -> Self {
        self.texture = FaceTexture::Uniform(index);
        self
    }

    /// Set per-face textures.
    ///
    /// Direction order: +X, -X, +Y, -Y, +Z, -Z.
    pub fn with_face_textures(mut self, textures: [u16; 6]) -> Self {
        self.texture = FaceTexture::PerFace(textures);
        self
    }

    /// Set per-face textures using a top, bottom, side scheme.
    ///
    /// Applies `top` to +Y, `bottom` to -Y, and `side` to all four
    /// horizontal faces (+X, -X, +Z, -Z).
    pub fn with_top_bottom_side(
        mut self,
        top    : u16,
        bottom : u16,
        side   : u16,
    ) -> Self
    {
        self.texture = FaceTexture::PerFace([
            side, side, top, bottom, side, side,
        ]);
        self
    }

    /// Returns the diffuse color as RGBA bytes.
    pub fn color(self) -> [u8; 4] {
        self.color
    }

    /// Returns the face texture assignment.
    pub fn face_texture(self) -> FaceTexture {
        self.texture
    }

    /// Returns the default texture array index.
    ///
    /// For uniform materials this is the shared texture. For per-face
    /// materials this returns the +X face texture.
    pub fn texture(self) -> u16 {
        match self.texture {
            FaceTexture::Uniform(idx) => idx,
            FaceTexture::PerFace(t)   => t[0],
        }
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

/// A registered block type definition.
///
/// Associates a human-readable name with visual properties. Block
/// definitions are registered at startup and stored in the
/// [`BlockRegistry`].
#[derive(Clone, Debug)]
pub struct Block {
    /// Human-readable name for debugging and display.
    name     : &'static str,
    /// Visual appearance when rendered.
    material : Material,
}

impl Block {
    /// Returns the block's name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the block's visual material.
    pub fn material(&self) -> Material {
        self.material
    }
}

// ---------------------------------------------------------------------------
// BlockRegistry
// ---------------------------------------------------------------------------

/// Central registry of all block types.
///
/// Maps [`BlockId`]s to [`Block`] definitions. Index 0 is permanently
/// reserved for air. Block types are registered with [`register`] and
/// looked up with [`get`]. The registry is append-only.
///
/// [`register`]: BlockRegistry::register
/// [`get`]: BlockRegistry::get
pub struct BlockRegistry {
    /// Block definitions, indexed by [`BlockId`] raw value.
    blocks : Vec<Block>,
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockRegistry {
    /// Create a new registry with only the air block at index 0.
    pub fn new() -> Self {
        BlockRegistry {
            blocks: vec![Block {
                name     : "air",
                material : Material::from_rgb(0, 0, 0),
            }],
        }
    }

    /// Register a new block type and return its [`BlockId`].
    ///
    /// # Panics
    ///
    /// Panics if the registry exceeds 65535 block types.
    pub fn register(
        &mut self,
        name     : &'static str,
        material : Material,
    ) -> BlockId
    {
        let id = self.blocks.len();
        assert!(id <= u16::MAX as usize, "block registry overflow");

        self.blocks.push(Block { name, material });
        BlockId::new(id as u16)
    }

    /// Returns the block definition for `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` was not returned by [`register`](Self::register).
    pub fn get(&self, id: BlockId) -> &Block {
        &self.blocks[id.raw() as usize]
    }

    /// Returns the number of registered block types, including air.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Returns whether this registry has no registered block types.
    ///
    /// Always `false` in practice: the registry is initialized with at
    /// least the Air block, but the method is provided for API symmetry
    /// with [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- air_is_default --

    #[test]
    fn air_is_default() {
        assert_eq!(BlockId::default(), BlockId::AIR);
        assert_eq!(BlockId::AIR.raw(), 0);
        assert!(BlockId::AIR.is_air());
    }

    // -- material_color_roundtrip --

    #[test]
    fn material_color_roundtrip() {
        let mat = Material::from_rgb(128, 64, 32);
        assert_eq!(mat.color(), [128, 64, 32, 255]);
    }

    // -- register_returns_sequential_ids --

    #[test]
    fn register_returns_sequential_ids() {
        let mut reg = BlockRegistry::new();

        let stone = reg.register("stone", Material::from_rgb(128, 128, 128));
        let dirt  = reg.register("dirt",  Material::from_rgb(139, 90, 43));

        assert_eq!(stone.raw(), 1);
        assert_eq!(dirt.raw(), 2);
        assert_eq!(reg.len(), 3);
    }

    // -- get_roundtrip --

    #[test]
    fn get_roundtrip() {
        let mut reg = BlockRegistry::new();
        let mat     = Material::from_rgb(200, 100, 50);
        let id      = reg.register("test_block", mat);
        let block   = reg.get(id);

        assert_eq!(block.name(), "test_block");
        assert_eq!(block.material(), mat);
    }

    // -- air_at_index_zero --

    #[test]
    fn air_at_index_zero() {
        let reg   = BlockRegistry::new();
        let block = reg.get(BlockId::AIR);

        assert_eq!(block.name(), "air");
    }

    // -- uniform_texture --

    #[test]
    fn uniform_texture() {
        let mat = Material::from_rgb(128, 128, 128).with_texture(5);

        assert_eq!(mat.texture(), 5);
        assert_eq!(mat.face_texture(), FaceTexture::Uniform(5));
    }

    // -- per_face_textures --

    #[test]
    fn per_face_textures() {
        let mat = Material::from_rgb(80, 160, 50)
            .with_top_bottom_side(3, 2, 4);

        assert_eq!(
            mat.face_texture(),
            FaceTexture::PerFace([4, 4, 3, 2, 4, 4]),
        );

        // texture() returns the +X face for per-face materials.
        assert_eq!(mat.texture(), 4);
    }
}
