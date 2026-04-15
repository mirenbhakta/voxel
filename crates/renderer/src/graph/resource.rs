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

/// SSA-versioned handle to a buffer resource in the render graph.
///
/// Each write to a resource mints a new version.  Passes declare reads against
/// a specific version (pinning the dependency on the write that produced it)
/// and produce a new version when they write.  The `resource` field identifies
/// the physical resource; `version` identifies which write produced this state.
///
/// Handles are [`Copy`] so pass execute closures can capture them without
/// borrowing the graph builder.  The inner fields are meaningful only to the
/// [`RenderGraph`](super::RenderGraph) that issued the handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BufferHandle {
    pub(super) resource : u32,
    pub(super) version  : u32,
}

// --- TextureHandle ---

/// SSA-versioned handle to a texture resource in the render graph.
///
/// Same versioning semantics as [`BufferHandle`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TextureHandle {
    pub(super) resource : u32,
    pub(super) version  : u32,
}

// --- ResourceId ---

/// Opaque resource identifier spanning both buffers and textures.
///
/// Both [`BufferHandle`] and [`TextureHandle`] convert into `ResourceId` via
/// [`From`].  Used in barrier metadata to identify the transitioning resource
/// without distinguishing its type or version — barriers are GPU-level and
/// care only about resource identity.
///
/// `ResourceId` does *not* include [`BindGroupHandle`] — bind groups are a
/// separate index space that carries no barrier semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ResourceId(pub u32);

// --- BindGroupHandle ---

/// Opaque handle to a bind group registered with the render graph.
///
/// Bind groups live in their own index space (distinct from [`ResourceId`])
/// because they are a pure output of the resolve-bind-groups step and
/// carry no barrier or versioning semantics — every frame produces a
/// fresh bind group from its template.
///
/// Passes obtain the resolved [`wgpu::BindGroup`] at execute time via
/// [`ResourceMap::bind_group`](super::ResourceMap::bind_group).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BindGroupHandle(pub(super) u32);

impl From<BufferHandle> for ResourceId {
    fn from(h: BufferHandle) -> Self {
        ResourceId(h.resource)
    }
}

impl From<TextureHandle> for ResourceId {
    fn from(h: TextureHandle) -> Self {
        ResourceId(h.resource)
    }
}

// --- Access ---

/// How a pass accesses a resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Access {
    /// The pass reads the resource.
    Read,
    /// The pass writes (or overwrites) the resource.
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
