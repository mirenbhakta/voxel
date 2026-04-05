//! GPU compute frustum culling stage.
//!
//! Reads CPU-written source draw commands and chunk world offsets,
//! tests each chunk's AABB against the camera frustum, and writes
//! visible draws to a compacted output buffer. An atomic counter
//! tracks the visible count for [`multi_draw_indirect_count`].

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType,
    ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineCompilationOptions, PipelineLayoutDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

// ---------------------------------------------------------------------------
// CullPipeline
// ---------------------------------------------------------------------------

/// The compute pipeline for frustum culling.
///
/// Reads source indirect draw commands and chunk offsets, tests each
/// chunk's AABB against six frustum planes, and appends visible draws
/// to a compacted output buffer with an atomic draw count.
pub struct CullPipeline {
    /// The compiled compute pipeline.
    pipeline  : ComputePipeline,
    /// The bind group layout for the cull dispatch.
    bg_layout : wgpu::BindGroupLayout,
}

impl CullPipeline {
    /// Create the frustum cull compute pipeline.
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label  : Some("cull_shader"),
            source : ShaderSource::Wgsl(
                include_str!("shaders/cull.wgsl").into(),
            ),
        });

        let bg_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                label   : Some("cull_bgl"),
                entries : &[
                    // binding 0: frustum planes (uniform)
                    BindGroupLayoutEntry {
                        binding    : 0,
                        visibility : ShaderStages::COMPUTE,
                        ty         : BindingType::Buffer {
                            ty                 : BufferBindingType::Uniform,
                            has_dynamic_offset : false,
                            min_binding_size   : None,
                        },
                        count : None,
                    },
                    // binding 1: source indirect draws (read-only storage)
                    BindGroupLayoutEntry {
                        binding    : 1,
                        visibility : ShaderStages::COMPUTE,
                        ty         : BindingType::Buffer {
                            ty                 : BufferBindingType::Storage {
                                read_only : true,
                            },
                            has_dynamic_offset : false,
                            min_binding_size   : None,
                        },
                        count : None,
                    },
                    // binding 2: chunk offsets (read-only storage)
                    BindGroupLayoutEntry {
                        binding    : 2,
                        visibility : ShaderStages::COMPUTE,
                        ty         : BindingType::Buffer {
                            ty                 : BufferBindingType::Storage {
                                read_only : true,
                            },
                            has_dynamic_offset : false,
                            min_binding_size   : None,
                        },
                        count : None,
                    },
                    // binding 3: output indirect draws (read-write storage)
                    BindGroupLayoutEntry {
                        binding    : 3,
                        visibility : ShaderStages::COMPUTE,
                        ty         : BindingType::Buffer {
                            ty                 : BufferBindingType::Storage {
                                read_only : false,
                            },
                            has_dynamic_offset : false,
                            min_binding_size   : None,
                        },
                        count : None,
                    },
                    // binding 4: atomic draw count (read-write storage)
                    BindGroupLayoutEntry {
                        binding    : 4,
                        visibility : ShaderStages::COMPUTE,
                        ty         : BindingType::Buffer {
                            ty                 : BufferBindingType::Storage {
                                read_only : false,
                            },
                            has_dynamic_offset : false,
                            min_binding_size   : None,
                        },
                        count : None,
                    },
                ],
            },
        );

        let pipeline_layout = device.create_pipeline_layout(
            &PipelineLayoutDescriptor {
                label              : Some("cull_pl"),
                bind_group_layouts : &[Some(&bg_layout)],
                immediate_size     : 4,
            },
        );

        let pipeline = device.create_compute_pipeline(
            &ComputePipelineDescriptor {
                label               : Some("cull_pipeline"),
                layout              : Some(&pipeline_layout),
                module              : &shader,
                entry_point         : Some("cull_main"),
                compilation_options : PipelineCompilationOptions::default(),
                cache               : None,
            },
        );

        CullPipeline { pipeline, bg_layout }
    }

    /// Create a bind group for the cull dispatch.
    ///
    /// # Arguments
    ///
    /// * `device`      - The wgpu device.
    /// * `frustum_buf` - Uniform buffer with 6 frustum planes (96 bytes).
    /// * `src_buf`     - Source indirect draw buffer (read-only storage).
    /// * `offsets_buf` - Chunk offset buffer (read-only storage).
    /// * `dst_buf`     - Output indirect draw buffer (read-write storage).
    /// * `count_buf`   - Atomic draw count buffer (read-write storage).
    pub fn create_bind_group(
        &self,
        device      : &Device,
        frustum_buf : &Buffer,
        src_buf     : &Buffer,
        offsets_buf : &Buffer,
        dst_buf     : &Buffer,
        count_buf   : &Buffer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label   : Some("cull_bg"),
            layout  : &self.bg_layout,
            entries : &[
                BindGroupEntry {
                    binding  : 0,
                    resource : frustum_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 1,
                    resource : src_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 2,
                    resource : offsets_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 3,
                    resource : dst_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 4,
                    resource : count_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Returns a reference to the compute pipeline.
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }
}
