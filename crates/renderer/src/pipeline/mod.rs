//! Pipeline and binding-layout abstractions.

pub mod binding;
pub mod compute;
pub mod reflect;
pub mod render;

pub use binding::{BindEntry, BindKind};
pub use compute::{ComputePipeline, ComputePipelineDescriptor};
pub use reflect::Reflected;
pub use render::{RenderPipeline, RenderPipelineDescriptor};

use std::num::NonZeroU64;

// --- PipelineBindLayout ---

/// Exposes the bind group layout and entry metadata for a pipeline.
///
/// Implemented by [`ComputePipeline`] and [`RenderPipeline`].  The render
/// graph's [`create_bind_group`](crate::graph::RenderGraph::create_bind_group)
/// accepts `&dyn PipelineBindLayout` to avoid duplicating the signature for
/// each pipeline type.
pub trait PipelineBindLayout {
    /// The wgpu bind group layout built from this pipeline's reflected shader
    /// entries.
    fn bg_layout(&self) -> &wgpu::BindGroupLayout;

    /// All descriptor-set-0 binding entries, sorted by slot, with visibility
    /// populated for the pipeline's stage(s).
    fn bind_entries(&self) -> &[BindEntry];

    /// The wgpu bind group layout for descriptor set 1, or `None` if the
    /// pipeline declares no set-1 resources.  Both [`ComputePipeline`] and
    /// [`RenderPipeline`] reflect set 1 from their shaders.
    fn bg_layout_set1(&self) -> Option<&wgpu::BindGroupLayout>;

    /// All descriptor-set-1 binding entries, sorted by slot, with visibility
    /// populated for the pipeline's stage(s).  Empty if the pipeline declares
    /// no set-1 resources.
    fn bind_entries_set1(&self) -> &[BindEntry];
}

// --- bind_kind_to_wgpu_ty ---

/// Convert a [`BindKind`] to the corresponding `wgpu::BindingType`.
pub(crate) fn bind_kind_to_wgpu_ty(kind: BindKind) -> wgpu::BindingType {
    match kind {
        BindKind::UniformBuffer { size } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
        BindKind::StorageBufferReadOnly { size } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
        BindKind::StorageBufferReadWrite { size } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
        BindKind::SampledTexture { sample_type, view_dimension } =>
            wgpu::BindingType::Texture {
                sample_type,
                view_dimension,
                multisampled: false,
            },
        BindKind::StorageTexture { format, view_dimension } =>
            wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension,
            },
    }
}
