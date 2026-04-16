//! `RenderPipeline` with shader-reflected bind group layout.
//!
//! Mirrors [`ComputePipeline`] for raster workloads. Vertex and fragment
//! shaders are loaded separately via [`ShaderModule::load`] before pipeline
//! construction — reflection and `GpuConsts`-size assertion happen in
//! [`RenderPipeline::new`].
//!
//! VS and PS entries are merged by slot: the kind must match on collision; the
//! visibility is unioned across stages. Slot 0 (GpuConsts) appears in both
//! stages and ends up as `VERTEX | FRAGMENT`.
//!
//! Vertex buffers, color targets, depth/stencil, primitive state, multisample
//! state, and immediate data (previously called push constants) are all
//! configured through [`RenderPipelineDescriptor`]. Wgpu types pass through
//! directly — there is no parallel renderer-level abstraction over them.
//!
//! [`ComputePipeline`]: crate::pipeline::compute::ComputePipeline

use crate::device::RendererContext;
use crate::gpu_consts::{GpuConsts, GpuConstsData};
use crate::pipeline::PipelineBindLayout;
use crate::pipeline::binding::{BindEntry, BindKind};
use crate::pipeline::bind_kind_to_wgpu_ty;
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

/// A raster pipeline with shader-reflected bind group layout.
///
/// Mirrors [`ComputePipeline`] for vertex + fragment workloads.
///
/// [`ComputePipeline`]: crate::pipeline::compute::ComputePipeline
pub struct RenderPipeline {
    pipeline     : wgpu::RenderPipeline,
    bg_layout    : wgpu::BindGroupLayout,
    bind_entries : Vec<BindEntry>,
    label        : String,
}

impl RenderPipeline {
    /// Construct a new `RenderPipeline` from `desc`.
    ///
    /// 1. Merges VS and PS bind entries by slot (kind must match; visibility
    ///    is unioned).
    /// 2. Asserts slot 0 is the GpuConsts `UniformBuffer`.
    /// 3. Builds `wgpu::BindGroupLayout` from the merged entries.
    /// 4. Creates the wgpu render pipeline.
    ///
    /// # Panics
    ///
    /// Panics if VS and PS entries at the same slot have different kinds, or if
    /// slot 0 is absent, not a `UniformBuffer`, or sized incorrectly.
    pub fn new(ctx: &RendererContext, desc: RenderPipelineDescriptor<'_>) -> Self {
        use std::mem::size_of;

        // Step 1: merge VS and PS entries.
        // Map: binding → (kind, visibility).
        let mut merged: Vec<(u32, BindKind, wgpu::ShaderStages)> = Vec::new();

        let add_entries = |merged: &mut Vec<(u32, BindKind, wgpu::ShaderStages)>,
                           entries: &[(u32, BindKind)],
                           stage: wgpu::ShaderStages,
                           shader_label: &str|
        {
            for &(binding, kind) in entries {
                if let Some(existing) = merged.iter_mut().find(|(b, _, _)| *b == binding) {
                    // Check kind matches.
                    let existing_kind_discriminant =
                        std::mem::discriminant(&existing.1);
                    let new_kind_discriminant = std::mem::discriminant(&kind);

                    if existing_kind_discriminant != new_kind_discriminant {
                        panic!(
                            "pipeline `{}`: shader `{}` binding {} kind {:?} \
                             conflicts with other stage's kind {:?} at the same slot",
                            shader_label, shader_label, binding, kind, existing.1,
                        );
                    }

                    existing.2 |= stage;
                }
                else {
                    merged.push((binding, kind, stage));
                }
            }
        };

        add_entries(
            &mut merged,
            &desc.vertex.bind_entries,
            desc.vertex.stage,
            desc.label,
        );
        add_entries(
            &mut merged,
            &desc.fragment.bind_entries,
            desc.fragment.stage,
            desc.label,
        );

        merged.sort_by_key(|(b, _, _)| *b);

        // Step 2: slot-0 GpuConsts assertion.
        let slot0 = merged.iter().find(|(b, _, _)| *b == GpuConsts::SLOT);

        match slot0 {
            Some((_, BindKind::UniformBuffer { size }, _))
                if *size as usize == size_of::<GpuConstsData>() => {}
            Some((_, BindKind::UniformBuffer { size }, _)) => panic!(
                "pipeline `{}`: slot {} is a UniformBuffer but its size is {} bytes; \
                 expected {} bytes (GpuConstsData) — \
                 check that the shader includes shaders/include/gpu_consts.hlsl",
                desc.label, GpuConsts::SLOT, size, size_of::<GpuConstsData>(),
            ),
            Some((_, kind, _)) => panic!(
                "pipeline `{}`: slot {} must be the GpuConsts uniform buffer but is \
                 {:?} — check that the shader includes shaders/include/gpu_consts.hlsl",
                desc.label, GpuConsts::SLOT, kind,
            ),
            None => panic!(
                "pipeline `{}`: slot {} (GpuConsts) is absent from the shader's \
                 descriptor set 0 — check that the shader includes \
                 shaders/include/gpu_consts.hlsl",
                desc.label, GpuConsts::SLOT,
            ),
        }

        // Step 3: build BindEntry list and wgpu::BindGroupLayout.
        let bind_entries: Vec<BindEntry> = merged.iter()
            .map(|&(binding, kind, visibility)| BindEntry { binding, kind, visibility })
            .collect();

        let wgpu_entries: Vec<wgpu::BindGroupLayoutEntry> = bind_entries.iter()
            .map(|e| wgpu::BindGroupLayoutEntry {
                binding   : e.binding,
                visibility: e.visibility,
                ty        : bind_kind_to_wgpu_ty(e.kind),
                count     : None,
            })
            .collect();

        let bg_layout = ctx.device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label  : Some(desc.label),
                entries: &wgpu_entries,
            });

        // Step 4: create the wgpu render pipeline.
        let pipeline_layout =
            ctx.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label             : Some(desc.label),
                bind_group_layouts: &[Some(&bg_layout)],
                immediate_size    : desc.immediate_size,
            });

        let pipeline =
            ctx.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label    : Some(desc.label),
                layout   : Some(&pipeline_layout),
                vertex   : wgpu::VertexState {
                    module             : &desc.vertex.inner,
                    entry_point        : Some(&desc.vertex.entry_point),
                    buffers            : desc.vertex_buffers,
                    compilation_options: Default::default(),
                },
                primitive     : desc.primitive,
                depth_stencil : desc.depth_stencil,
                multisample   : desc.multisample,
                fragment      : Some(wgpu::FragmentState {
                    module             : &desc.fragment.inner,
                    entry_point        : Some(&desc.fragment.entry_point),
                    targets            : desc.color_targets,
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache         : None,
            });

        Self {
            pipeline,
            bg_layout,
            bind_entries,
            label: desc.label.to_string(),
        }
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

impl PipelineBindLayout for RenderPipeline {
    fn bg_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bg_layout
    }

    fn bind_entries(&self) -> &[BindEntry] {
        &self.bind_entries
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameCount;
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
                label:          "render_full_desc",
                vertex:         vs,
                fragment:       ps,
                vertex_buffers: &[],
                color_targets:  &[Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                depth_stencil:  Some(wgpu::DepthStencilState {
                    format:              wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare:       Some(wgpu::CompareFunction::Less),
                    stencil:             wgpu::StencilState::default(),
                    bias:                wgpu::DepthBiasState::default(),
                }),
                primitive:      wgpu::PrimitiveState {
                    topology:   wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode:  Some(wgpu::Face::Back),
                    ..Default::default()
                },
                multisample:    wgpu::MultisampleState::default(),
                immediate_size: 8,
            },
        );
    }
}
