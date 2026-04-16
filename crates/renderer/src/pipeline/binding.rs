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

/// Kind of buffer resource exposed by a [`BindEntry`].
///
/// Textures and samplers are deferred until a primitive genuinely needs them.
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
}
