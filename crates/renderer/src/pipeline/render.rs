//! `RenderPipeline` with shader-reflected bind group layout.
//!
//! Mirrors [`ComputePipeline`] for raster workloads. Vertex and fragment
//! shaders are loaded separately via [`ShaderModule::load`] before pipeline
//! construction. The `GpuConstsData` layout check fires inside
//! [`ShaderModule::load`]; [`RenderPipeline::new`] only merges entries and
//! builds the bind group layout.
//!
//! VS and PS entries are merged by slot: the kind must match on collision; the
//! visibility is unioned across stages.
//!
//! Vertex buffers, color targets, depth/stencil, primitive state, multisample
//! state, and immediate data (previously called push constants) are all
//! configured through [`RenderPipelineDescriptor`]. Wgpu types pass through
//! directly — there is no parallel renderer-level abstraction over them.
//!
//! [`ComputePipeline`]: crate::pipeline::compute::ComputePipeline

use crate::device::RendererContext;
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
    pipeline          : wgpu::RenderPipeline,
    bg_layout         : wgpu::BindGroupLayout,
    bind_entries      : Vec<BindEntry>,
    bg_layout_1       : Option<wgpu::BindGroupLayout>,
    set1_bind_entries : Vec<BindEntry>,
    label             : String,
}

impl RenderPipeline {
    /// Construct a new `RenderPipeline` from `desc`.
    ///
    /// 1. Merges VS and PS bind entries by slot, for each declared descriptor
    ///    set (kind must match; visibility is unioned).
    /// 2. Builds `wgpu::BindGroupLayout`s from the merged entries — one per
    ///    non-empty set.
    /// 3. Creates the wgpu render pipeline.
    ///
    /// # Panics
    ///
    /// Panics if VS and PS entries at the same slot within a set have
    /// different kinds.
    pub fn new(ctx: &RendererContext, desc: RenderPipelineDescriptor<'_>) -> Self {
        // Step 1: merge VS and PS entries, per set.
        let bind_entries = merge_raster_entries(
            &desc.vertex.bind_entries,
            &desc.fragment.bind_entries,
            desc.vertex.stage,
            desc.fragment.stage,
            desc.label,
        );

        let set1_bind_entries = merge_raster_entries(
            &desc.vertex.set1_bind_entries,
            &desc.fragment.set1_bind_entries,
            desc.vertex.stage,
            desc.fragment.stage,
            desc.label,
        );

        // Step 2: build the set-0 BGL (always present) and the set-1 BGL
        // (only when the shaders declare set-1 resources).
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

        let bg_layout_1: Option<wgpu::BindGroupLayout> = if set1_bind_entries.is_empty() {
            None
        }
        else {
            let wgpu_set1_entries: Vec<wgpu::BindGroupLayoutEntry> =
                set1_bind_entries.iter()
                    .map(|e| wgpu::BindGroupLayoutEntry {
                        binding   : e.binding,
                        visibility: e.visibility,
                        ty        : bind_kind_to_wgpu_ty(e.kind),
                        count     : None,
                    })
                    .collect();

            Some(ctx.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label  : Some(desc.label),
                entries: &wgpu_set1_entries,
            }))
        };

        // Step 3: create the wgpu render pipeline.
        let pipeline_layout = {
            let bg_layouts: Vec<Option<&wgpu::BindGroupLayout>> = match &bg_layout_1 {
                Some(l) => vec![Some(&bg_layout), Some(l)],
                None    => vec![Some(&bg_layout)],
            };

            ctx.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label             : Some(desc.label),
                bind_group_layouts: &bg_layouts,
                immediate_size    : desc.immediate_size,
            })
        };

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
            bg_layout_1,
            set1_bind_entries,
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

    fn bg_layout_set1(&self) -> Option<&wgpu::BindGroupLayout> {
        self.bg_layout_1.as_ref()
    }

    fn bind_entries_set1(&self) -> &[BindEntry] {
        &self.set1_bind_entries
    }
}

// --- merge_raster_entries ---

/// Merge VS and PS bind entries for a single descriptor set.
///
/// Bindings at the same slot must have the same kind; the visibility flags
/// are unioned across stages.  Returns the merged entries sorted by slot.
///
/// # Panics
///
/// Panics if VS and PS declare the same binding slot with different kinds.
fn merge_raster_entries(
    vs_entries : &[(u32, BindKind)],
    ps_entries : &[(u32, BindKind)],
    vs_stage   : wgpu::ShaderStages,
    ps_stage   : wgpu::ShaderStages,
    label      : &str,
)
    -> Vec<BindEntry>
{
    let mut merged: Vec<(u32, BindKind, wgpu::ShaderStages)> = Vec::new();

    fn add_entries(
        merged  : &mut Vec<(u32, BindKind, wgpu::ShaderStages)>,
        entries : &[(u32, BindKind)],
        stage   : wgpu::ShaderStages,
        label   : &str,
    ) {
        for &(binding, kind) in entries {
            if let Some(existing) = merged.iter_mut().find(|(b, _, _)| *b == binding) {
                let existing_kind_discriminant = std::mem::discriminant(&existing.1);
                let new_kind_discriminant = std::mem::discriminant(&kind);

                if existing_kind_discriminant != new_kind_discriminant {
                    panic!(
                        "pipeline `{}`: binding {} kind {:?} \
                         conflicts with other stage's kind {:?} at the same slot",
                        label, binding, kind, existing.1,
                    );
                }

                existing.2 |= stage;
            }
            else {
                merged.push((binding, kind, stage));
            }
        }
    }

    add_entries(&mut merged, vs_entries, vs_stage, label);
    add_entries(&mut merged, ps_entries, ps_stage, label);

    merged.sort_by_key(|(b, _, _)| *b);

    merged.into_iter()
        .map(|(binding, kind, visibility)| BindEntry { binding, kind, visibility })
        .collect()
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
