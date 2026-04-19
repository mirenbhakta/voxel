/// Descriptor for a single binding in a pipeline's bind group layout.
///
/// Produced by SPIR-V reflection and stored on [`ComputePipeline`] /
/// [`RenderPipeline`].  The graph's `use_bind_group` inspects `kind` to
/// derive read vs. write accesses automatically.
///
/// [`ComputePipeline`]: crate::pipeline::compute::ComputePipeline
/// [`RenderPipeline`]: crate::pipeline::render::RenderPipeline
#[derive(Clone, Debug)]
pub struct BindEntry {
    /// The bind group slot (`[[vk::binding(N, 0)]]`).
    pub binding: u32,
    /// The kind of resource (uniform / storage read-only / storage read-write).
    pub kind: BindKind,
    /// Shader stages that may access the binding. `wgpu::ShaderStages` is the
    /// one wgpu type that deliberately leaks through the primitives layer —
    /// the stage names are a closed set and a wrapper would add zero safety.
    pub visibility: wgpu::ShaderStages,
}

/// Kind of resource exposed by a [`BindEntry`].
///
/// The first three variants cover buffers; the texture variants are added
/// for primitives that read a vis buffer or write a storage image (the
/// first of which is the subchunk shade compute pass — see
/// `shaders/subchunk_shade.cs.hlsl`). Samplers remain deferred until a
/// primitive genuinely needs them.
#[derive(Clone, Copy, Debug)]
pub enum BindKind {
    /// A uniform buffer of the given size in bytes. Maps to wgpu's
    /// `BufferBindingType::Uniform`.
    UniformBuffer { size: u64 },
    /// A read-only storage buffer of the given size in bytes. Maps to wgpu's
    /// `BufferBindingType::Storage { read_only: true }`.
    StorageBufferReadOnly { size: u64 },
    /// A read-write storage buffer of the given size in bytes. Maps to wgpu's
    /// `BufferBindingType::Storage { read_only: false }`.
    StorageBufferReadWrite { size: u64 },
    /// A sampled (SRV) texture. HLSL source: `Texture2D<T>` (plus future
    /// array / cube variants via [`view_dimension`](Self::SampledTexture::view_dimension)).
    /// Read-only from the shader. Maps to wgpu's `BindingType::Texture`.
    SampledTexture {
        sample_type    : wgpu::TextureSampleType,
        view_dimension : wgpu::TextureViewDimension,
    },
    /// A storage (UAV) texture. HLSL source: `RWTexture2D<T>` with a
    /// `[[vk::image_format(...)]]` attribute so reflection can pin the
    /// texel format. Treated as a write by the graph's automatic access
    /// tracking — the one consumer today (subchunk shade) writes and
    /// never reads. Maps to wgpu's `BindingType::StorageTexture` with
    /// `StorageTextureAccess::WriteOnly`.
    StorageTexture {
        format         : wgpu::TextureFormat,
        view_dimension : wgpu::TextureViewDimension,
    },
}
