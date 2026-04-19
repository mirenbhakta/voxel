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

// --- ResourceHandle ---

/// SSA-versioned resource identity: (resource_index, write_version).
///
/// Shared implementation detail behind [`BufferHandle`] and [`TextureHandle`].
/// The typed wrappers preserve static buffer/texture distinction at the public
/// API boundary; this struct centralises the field pair so handle-agnostic
/// internals (access recording, entry indexing) touch one place.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) struct ResourceHandle {
    pub(super) resource : u32,
    pub(super) version  : u32,
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
pub struct BufferHandle(pub(super) ResourceHandle);

// --- TextureHandle ---

/// SSA-versioned handle to a texture resource in the render graph.
///
/// Same versioning semantics as [`BufferHandle`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TextureHandle(pub(super) ResourceHandle);

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
        ResourceId(h.0.resource)
    }
}

impl From<TextureHandle> for ResourceId {
    fn from(h: TextureHandle) -> Self {
        ResourceId(h.0.resource)
    }
}

// --- BindResource ---

/// A bind-group slot's resource: either a single resource (scalar binding)
/// or a list of resources (binding array).
///
/// Scalar slots (uniform / storage buffer, sampled / storage texture) carry
/// a single [`ResourceId`]; binding-array slots (currently only
/// [`BindKind::StorageBufferReadOnlyArray`](crate::pipeline::BindKind::StorageBufferReadOnlyArray))
/// carry an ordered list whose length may be less than the declared array
/// count — under `PARTIALLY_BOUND_BINDING_ARRAY` the remaining descriptors
/// stay unbound.
///
/// `BindResource: From<ResourceId>` so existing scalar call sites keep
/// writing `(binding, some_handle.into())`.
///
/// [`BindKind::StorageBufferReadOnlyArray`]: crate::pipeline::BindKind::StorageBufferReadOnlyArray
#[derive(Clone, Debug)]
pub enum BindResource {
    /// A single resource occupies the slot. The common case.
    Single(ResourceId),
    /// An ordered list of resources. Used for binding-array slots such as
    /// the material-data pool.
    Array(Vec<ResourceId>),
}

impl From<ResourceId> for BindResource {
    fn from(id: ResourceId) -> Self {
        BindResource::Single(id)
    }
}

impl From<BufferHandle> for BindResource {
    fn from(h: BufferHandle) -> Self {
        BindResource::Single(h.into())
    }
}

impl From<TextureHandle> for BindResource {
    fn from(h: TextureHandle) -> Self {
        BindResource::Single(h.into())
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
