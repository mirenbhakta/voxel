//! `RenderPipeline` with shared binding layout and GpuConsts injection.
//!
//! Mirrors [`ComputePipeline`] for raster workloads. Vertex and fragment
//! shaders are loaded separately via [`ShaderModule::load`] before pipeline
//! construction — reflection and `GpuConsts`-size assertion happen there.
//!
//! Vertex buffers, color targets, depth/stencil, primitive state, multisample
//! state, and immediate data (previously called push constants) are all
//! configured through [`RenderPipelineDescriptor`]. Wgpu types pass through
//! directly — there is no parallel renderer-level abstraction over them.
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
    /// Vertex buffer layouts, in slot order. Empty if the vertex shader
    /// generates positions from `vertex_index` instead of reading attributes.
    pub vertex_buffers: &'a [wgpu::VertexBufferLayout<'a>],
    /// Color attachment formats and blend state, in attachment-slot order.
    /// `None` entries leave the slot disabled.
    pub color_targets: &'a [Option<wgpu::ColorTargetState>],
    /// Depth/stencil state. `None` disables depth and stencil.
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    /// Primitive topology, winding, culling, and polygon mode.
    pub primitive: wgpu::PrimitiveState,
    /// Multisample state.
    pub multisample: wgpu::MultisampleState,
    /// Immediate-data byte budget (previously called push constants). `0`
    /// means the pipeline declares no immediate data.
    pub immediate_size: u32,
}

// --- RenderPipeline ---

/// A raster pipeline with baked-in binding layout.
///
/// Mirrors [`ComputePipeline`] for vertex + fragment workloads.
///
/// [`ComputePipeline`]: crate::pipeline::compute::ComputePipeline
pub struct RenderPipeline {
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
                immediate_size: desc.immediate_size,
            });

        let pipeline =
            ctx.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(desc.label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &desc.vertex.inner,
                    entry_point: Some(&desc.vertex.entry_point),
                    buffers: desc.vertex_buffers,
                    compilation_options: Default::default(),
                },
                primitive: desc.primitive,
                depth_stencil: desc.depth_stencil,
                multisample: desc.multisample,
                fragment: Some(wgpu::FragmentState {
                    module: &desc.fragment.inner,
                    entry_point: Some(&desc.fragment.entry_point),
                    targets: desc.color_targets,
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

    /// The underlying wgpu render pipeline.
    pub(crate) fn inner(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameCount;
    use crate::pipeline::binding::{BindEntry, BindKind, BindingLayout};
    use crate::shader::{ShaderModule, ShaderSource};

    /// Compiled SPIR-V for the subchunk vertex / pixel shaders. Mirrors the
    /// `include_bytes!` paths in `subchunk_test.rs`; duplicated here so the
    /// test does not reach across module boundaries for private constants.
    const SUBCHUNK_VS_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/subchunk.vs.spv"));
    const SUBCHUNK_PS_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/subchunk.ps.spv"));

    /// Builds a `RenderPipeline` with non-default values across every
    /// descriptor field — including a non-zero `immediate_size` — to lock in
    /// the descriptor's full shape. Reaching the end of `RenderPipeline::new`
    /// without panicking is the assertion.
    ///
    /// Gated because it requires a Vulkan-capable GPU and DXC-built SPVs.
    #[test]
    #[ignore = "requires real GPU hardware (vulkan) and a DXC-built SPV; run with --ignored"]
    fn render_pipeline_builds_with_full_descriptor() {
        let ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine");

        let layout = Arc::new(
            BindingLayout::builder("render_full_desc")
                .add_entry(BindEntry {
                    binding: 1,
                    kind: BindKind::UniformBuffer { size: 64 },
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT,
                })
                .build(&ctx),
        );

        let vs = ShaderModule::load(
            &ctx,
            "subchunk.vs",
            ShaderSource::Spirv(SUBCHUNK_VS_SPV),
            "main",
        )
        .expect("subchunk vertex shader should load");

        let ps = ShaderModule::load(
            &ctx,
            "subchunk.ps",
            ShaderSource::Spirv(SUBCHUNK_PS_SPV),
            "main",
        )
        .expect("subchunk pixel shader should load");

        let _pipeline = RenderPipeline::new(
            &ctx,
            RenderPipelineDescriptor {
                label: "render_full_desc",
                vertex: vs,
                fragment: ps,
                layout,
                vertex_buffers: &[],
                color_targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                multisample: wgpu::MultisampleState::default(),
                immediate_size: 8,
            },
        );
    }
}
