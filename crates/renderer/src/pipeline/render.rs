//! `RenderPipeline` with shared binding layout and GpuConsts injection.
//!
//! Mirrors [`ComputePipeline`] for raster workloads. Vertex and fragment
//! shaders are loaded separately via [`ShaderModule::load`] before pipeline
//! construction — reflection and `GpuConsts`-size assertion happen there.
//!
//! Raster state (vertex buffers, color targets, depth/stencil, primitive
//! topology) is intentionally minimal in this first pass. The DDA draw
//! pipeline will extend [`RenderPipelineDescriptor`] with those fields when
//! it is built.
//!
//! [`ComputePipeline`]: crate::pipeline::compute::ComputePipeline

use std::sync::Arc;

use crate::device::RendererContext;
use crate::pipeline::binding::BindingLayout;
use crate::shader::ShaderModule;

// --- RenderPipelineDescriptor ---

/// Construction parameters for a [`RenderPipeline`].
pub struct RenderPipelineDescriptor<'a> {
    /// Debug label forwarded to wgpu.
    pub label: &'a str,
    /// Vertex shader, loaded via [`ShaderModule::load`].
    pub vertex: ShaderModule,
    /// Fragment shader, loaded via [`ShaderModule::load`].
    pub fragment: ShaderModule,
    /// The binding layout this pipeline was built against.
    pub layout: Arc<BindingLayout>,
}

// --- RenderPipeline ---

/// A raster pipeline with baked-in binding layout.
///
/// Mirrors [`ComputePipeline`] for vertex + fragment workloads. Vertex
/// buffers, color targets, and depth state are deferred to the DDA pipeline
/// increment.
///
/// [`ComputePipeline`]: crate::pipeline::compute::ComputePipeline
pub struct RenderPipeline {
    // Used by the draw methods added with the DDA pipeline increment.
    #[allow(dead_code)]
    pipeline: wgpu::RenderPipeline,
    layout: Arc<BindingLayout>,
    label: String,
}

impl RenderPipeline {
    /// Construct a new `RenderPipeline` from `desc`.
    ///
    /// Reflection and `GpuConsts`-size assertions were already performed
    /// inside [`ShaderModule::load`] — this function only wires up the
    /// wgpu pipeline.
    pub fn new(ctx: &RendererContext, desc: RenderPipelineDescriptor<'_>) -> Self {
        let pipeline_layout =
            ctx.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(desc.label),
                bind_group_layouts: &[Some(desc.layout.wgpu_layout())],
                immediate_size: 0,
            });

        let pipeline =
            ctx.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(desc.label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &desc.vertex.inner,
                    entry_point: Some(&desc.vertex.entry_point),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &desc.fragment.inner,
                    entry_point: Some(&desc.fragment.entry_point),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });

        Self {
            pipeline,
            layout: desc.layout,
            label: desc.label.to_string(),
        }
    }

    /// The binding layout this pipeline was constructed with.
    pub fn layout(&self) -> &Arc<BindingLayout> {
        &self.layout
    }

    /// The debug label this pipeline was constructed with.
    pub fn label(&self) -> &str {
        &self.label
    }
}
