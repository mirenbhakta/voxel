//! GPU compute frustum culling stage.
//!
//! Iterates all chunk slots, reads per-chunk metadata and quad ranges,
//! tests each chunk's AABB against the camera frustum, and emits
//! indirect draw commands and per-draw slot/direction metadata for
//! visible chunks. An atomic counter tracks the visible draw count
//! for [`multi_draw_indirect_count`].

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType,
    ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineCompilationOptions, PipelineLayoutDescriptor,
    ShaderModuleDescriptorPassthrough, ShaderStages,
};

// ---------------------------------------------------------------------------
// CullPipeline
// ---------------------------------------------------------------------------

/// The compute pipeline for frustum culling.
///
/// Iterates all chunk slots, reads chunk_meta_buf for quad counts
/// (skipping empty slots), tests each chunk's AABB against six frustum
/// planes, and appends visible draws to a compacted output buffer with
/// per-draw slot metadata and an atomic draw count.
pub struct CullPipeline {
    /// The compiled compute pipeline.
    pipeline  : ComputePipeline,
    /// The bind group layout for the cull dispatch.
    bg_layout : wgpu::BindGroupLayout,
}

// --- CullPipeline ---

impl CullPipeline {
    /// Create the frustum cull compute pipeline.
    pub fn new(device: &Device) -> Self {
        // Safety: SPIR-V compiled from trusted HLSL by DXC at build time.
        let shader = unsafe {
            device.create_shader_module_passthrough(
                ShaderModuleDescriptorPassthrough {
                    label : Some("cull_shader"),
                    spirv : Some(wgpu::util::make_spirv_raw(
                        include_bytes!(concat!(
                            env!("OUT_DIR"), "/cull.cs.spv"
                        )),
                    )),
                    ..Default::default()
                },
            )
        };

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
                    // binding 1: chunk offsets (read-only storage)
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
                    // binding 2: chunk_meta_buf (read-only storage)
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
                    // binding 3: quad_range_buf (read-only storage)
                    BindGroupLayoutEntry {
                        binding    : 3,
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
                    // binding 4: output indirect draws (read-write storage)
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
                    // binding 5: draw_data_buf (read-write storage)
                    BindGroupLayoutEntry {
                        binding    : 5,
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
                    // binding 6: atomic draw count (read-write storage)
                    BindGroupLayoutEntry {
                        binding    : 6,
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
                entry_point         : Some("main"),
                compilation_options : PipelineCompilationOptions::default(),
                cache               : None,
            },
        );

        CullPipeline { pipeline: pipeline, bg_layout: bg_layout }
    }

    /// Create a bind group for the cull dispatch.
    ///
    /// # Arguments
    ///
    /// * `device`         - The wgpu device.
    /// * `frustum_buf`    - Uniform buffer: 6 frustum planes + camera pos (112 bytes).
    /// * `offsets_buf`    - Chunk offset buffer (read-only storage).
    /// * `chunk_meta_buf` - Per-chunk metadata buffer (read-only storage).
    /// * `quad_range_buf` - Per-chunk quad range buffer (read-only storage).
    /// * `dst_buf`        - Output indirect draw buffer (read-write storage).
    /// * `draw_data_buf`  - Per-draw slot/direction metadata (read-write storage).
    /// * `count_buf`      - Atomic draw count buffer (read-write storage).
    pub fn create_bind_group(
        &self,
        device         : &Device,
        frustum_buf    : &Buffer,
        offsets_buf    : &Buffer,
        chunk_meta_buf : &Buffer,
        quad_range_buf : &Buffer,
        dst_buf        : &Buffer,
        draw_data_buf  : &Buffer,
        count_buf      : &Buffer,
    ) -> BindGroup
    {
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
                    resource : offsets_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 2,
                    resource : chunk_meta_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 3,
                    resource : quad_range_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 4,
                    resource : dst_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 5,
                    resource : draw_data_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 6,
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
