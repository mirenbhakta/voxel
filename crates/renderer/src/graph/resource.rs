//! Resource handles and access types for the render graph.

// --- BufferDesc ---

/// Describes a transient buffer's requirements for pool allocation.
#[derive(Clone, Copy, Debug)]
pub struct BufferDesc {
    /// Buffer size in bytes.
    pub size  : u64,
    /// Required wgpu usage flags.
    pub usage : wgpu::BufferUsages,
}

// --- BufferHandle ---

/// Opaque handle to a buffer resource in the render graph.
///
/// Handles are [`Copy`] so pass execute closures can capture them without
/// borrowing the graph builder.  The inner index is meaningful only to the
/// [`RenderGraph`](super::RenderGraph) that issued it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BufferHandle(pub(super) u32);

// --- TextureHandle ---

/// Opaque handle to a texture resource in the render graph.
///
/// Handles are [`Copy`] so pass execute closures can capture them without
/// borrowing the graph builder.  The inner index is meaningful only to the
/// [`RenderGraph`](super::RenderGraph) that issued it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TextureHandle(pub(super) u32);

// --- ResourceId ---

/// Opaque resource identifier spanning both buffers and textures.
///
/// Both [`BufferHandle`] and [`TextureHandle`] convert into `ResourceId` via
/// [`From`].  Used in barrier metadata to identify the transitioning resource
/// without distinguishing its type.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ResourceId(pub u32);

impl From<BufferHandle> for ResourceId {
    fn from(h: BufferHandle) -> Self {
        ResourceId(h.0)
    }
}

impl From<TextureHandle> for ResourceId {
    fn from(h: TextureHandle) -> Self {
        ResourceId(h.0)
    }
}

// --- Access ---

/// How a pass accesses a buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Access {
    /// The pass reads the buffer.
    Read,
    /// The pass writes (or overwrites) the buffer.
    Write,
}

// --- TextureDesc ---

/// Describes a transient texture's requirements for pool allocation.
pub struct TextureDesc {
    /// Texture extents (width, height, depth or array layers).
    pub size            : wgpu::Extent3d,
    /// Texel format.
    pub format          : wgpu::TextureFormat,
    /// Required wgpu usage flags.
    pub usage           : wgpu::TextureUsages,
    /// Texture dimensionality (1D, 2D, or 3D).
    pub dimension       : wgpu::TextureDimension,
    /// Number of mip levels.
    pub mip_level_count : u32,
    /// MSAA sample count.
    pub sample_count    : u32,
}

impl TextureDesc {
    /// Create a 2D texture descriptor with one mip level and no MSAA.
    pub fn new_2d(
        width  : u32,
        height : u32,
        format : wgpu::TextureFormat,
        usage  : wgpu::TextureUsages,
    ) -> Self {
        Self {
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            format,
            usage,
            dimension       : wgpu::TextureDimension::D2,
            mip_level_count : 1,
            sample_count    : 1,
        }
    }
}
