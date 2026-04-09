//! GPU compute build pipelines for three-pass chunk quad generation.
//!
//! The build stage runs in three passes over slot-indexed shared buffers:
//!
//! 1. **Count pass** -- derives face bitmasks from occupancy and neighbor
//!    boundaries, runs greedy merge, and writes per-(direction, layer)
//!    quad counts to `quad_range_buf` and totals to `chunk_meta_buf`.
//!
//! 2. **Alloc pass** -- reads quad counts from `chunk_meta_buf`, advances
//!    a GPU-side bump pointer, and writes `base_offset` into
//!    `quad_range_buf` for each chunk in the batch.
//!
//! 3. **Write pass** -- re-derives faces and merge, then writes packed
//!    quad descriptors to contiguous ranges in `quad_buf` at prefix-summed
//!    offsets computed from the count pass output and the base offset
//!    written by the alloc pass.
//!
//! Count and write passes use push constants (`BuildPush`) to select the
//! chunk slot. The alloc pass uses `AllocPush` with batch size and
//! capacity. All three passes execute in a single command encoder
//! submission.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBindingType, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineCompilationOptions, PipelineLayoutDescriptor,
    ShaderModuleDescriptorPassthrough, ShaderStages,
};

use crate::world::MAX_MATERIAL_SEGMENTS;

// ---------------------------------------------------------------------------
// BuildPush
// ---------------------------------------------------------------------------

/// Push constants for build shaders.
///
/// Passed via immediates at offset 0. Both the count and write passes
/// share this layout so the pipeline layouts are compatible.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BuildPush {
    /// The slot index into the shared buffers for this chunk.
    pub slot_index  : u32,
    /// The starting quad index in the quad buffer (write pass only).
    pub base_offset : u32,
}

// ---------------------------------------------------------------------------
// BuildCountPipeline
// ---------------------------------------------------------------------------

/// The compute pipeline for the build count pass (pass 1).
///
/// Reads chunk occupancy and neighbor boundary data from shared
/// slot-indexed buffers. Runs face derivation and greedy merge in
/// count-only mode, writing per-(direction, layer) quad counts to
/// `quad_range_buf` and accumulating the total in `chunk_meta_buf`.
pub struct BuildCountPipeline {
    /// The compiled compute pipeline.
    pipeline  : ComputePipeline,
    /// The bind group layout for count pass dispatches.
    bg_layout : wgpu::BindGroupLayout,
}

impl BuildCountPipeline {
    /// Create the build count compute pipeline.
    ///
    /// Load the pre-compiled SPIR-V for `build_count.cs.hlsl` and build
    /// the pipeline with the count pass bind group layout.
    pub fn new(device: &Device) -> Self {
        let count_spv = include_bytes!(
            concat!(env!("OUT_DIR"), "/build_count.cs.spv")
        );

        // Safety: SPIR-V compiled from trusted HLSL by DXC at build time.
        let shader = unsafe {
            device.create_shader_module_passthrough(
                ShaderModuleDescriptorPassthrough {
                    label : Some("build_count_shader"),
                    spirv : Some(wgpu::util::make_spirv_raw(count_spv)),
                    ..Default::default()
                },
            )
        };

        let bg_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                label   : Some("build_count_bgl"),
                entries : &[
                    // binding 0: occupancy_buf (read-only storage)
                    BindGroupLayoutEntry {
                        binding    : 0,
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
                    // binding 1: boundary_cache_buf (read-only storage)
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
                    // binding 2: chunk_meta_buf (read-write storage)
                    BindGroupLayoutEntry {
                        binding    : 2,
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
                    // binding 3: quad_range_buf (read-write storage)
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
                ],
            },
        );

        let pipeline_layout = device.create_pipeline_layout(
            &PipelineLayoutDescriptor {
                label              : Some("build_count_pl"),
                bind_group_layouts : &[Some(&bg_layout)],
                immediate_size     : 8,
            },
        );

        let pipeline = device.create_compute_pipeline(
            &ComputePipelineDescriptor {
                label               : Some("build_count_pipeline"),
                layout              : Some(&pipeline_layout),
                module              : &shader,
                entry_point         : Some("main"),
                compilation_options : PipelineCompilationOptions::default(),
                cache               : None,
            },
        );

        BuildCountPipeline { pipeline, bg_layout }
    }

    /// Create a bind group for the count pass.
    ///
    /// The bind group references the shared slot-indexed buffers. It is
    /// created once and reused for all chunk dispatches; the slot index
    /// is passed via push constants.
    ///
    /// # Arguments
    ///
    /// * `device`            - The GPU device.
    /// * `occupancy_buf`     - Shared occupancy bitmask buffer (read-only).
    /// * `boundary_cache_buf`- Shared neighbor boundary cache (read-only).
    /// * `chunk_meta_buf`    - Shared per-chunk metadata (read-write).
    /// * `quad_range_buf`    - Shared per-chunk quad range data (read-write).
    pub fn create_bind_group(
        &self,
        device             : &Device,
        occupancy_buf      : &Buffer,
        boundary_cache_buf : &Buffer,
        chunk_meta_buf     : &Buffer,
        quad_range_buf     : &Buffer,
    ) -> BindGroup
    {
        device.create_bind_group(&BindGroupDescriptor {
            label   : Some("build_count_bg"),
            layout  : &self.bg_layout,
            entries : &[
                BindGroupEntry {
                    binding  : 0,
                    resource : occupancy_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 1,
                    resource : boundary_cache_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 2,
                    resource : chunk_meta_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 3,
                    resource : quad_range_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Returns a reference to the compute pipeline.
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }
}

// ---------------------------------------------------------------------------
// AllocPush
// ---------------------------------------------------------------------------

/// Push constants for the build alloc shader.
///
/// Passed via immediates at offset 0.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AllocPush {
    /// Number of chunks in this build batch.
    pub batch_size             : u32,
    /// Total quad buffer capacity in quads.
    pub quad_capacity          : u32,
    /// Total material capacity in sub-block units across all segments.
    pub material_capacity      : u32,
    /// Sub-block units per material segment (power of two).
    pub material_segment_units : u32,
}

// ---------------------------------------------------------------------------
// BuildAllocPipeline
// ---------------------------------------------------------------------------

/// The compute pipeline for the build alloc pass (pass 2).
///
/// Runs as a single-threaded (1,1,1) dispatch between the count and write
/// passes. Reads quad counts from `chunk_meta_buf`, advances the GPU-side
/// bump pointer in `bump_state_buf`, and writes `base_offset` into
/// `quad_range_buf` for each chunk in the batch.
pub struct BuildAllocPipeline {
    /// The compiled compute pipeline.
    pipeline  : ComputePipeline,
    /// The bind group layout for alloc pass dispatches.
    bg_layout : wgpu::BindGroupLayout,
}

impl BuildAllocPipeline {
    /// Create the build alloc compute pipeline.
    ///
    /// Load the pre-compiled SPIR-V for `build_alloc.cs.hlsl` and build
    /// the pipeline with the alloc pass bind group layout.
    pub fn new(device: &Device) -> Self {
        let alloc_spv = include_bytes!(
            concat!(env!("OUT_DIR"), "/build_alloc.cs.spv")
        );

        // Safety: SPIR-V compiled from trusted HLSL by DXC at build time.
        let shader = unsafe {
            device.create_shader_module_passthrough(
                ShaderModuleDescriptorPassthrough {
                    label : Some("build_alloc_shader"),
                    spirv : Some(wgpu::util::make_spirv_raw(alloc_spv)),
                    ..Default::default()
                },
            )
        };

        let bg_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                label   : Some("build_alloc_bgl"),
                entries : &[
                    // binding 0: bump_state_buf (read-write storage)
                    BindGroupLayoutEntry {
                        binding    : 0,
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
                    // binding 1: build_batch_buf (read-only storage)
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
                    // binding 2: chunk_meta_buf (read-write storage)
                    BindGroupLayoutEntry {
                        binding    : 2,
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
                    // binding 3: quad_range_buf (read-write storage)
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
                    // binding 4: quad_free_list_buf (read-write storage)
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
                    // binding 5: material_range_buf (read-write storage)
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
                    // binding 6: material_bump_state_buf (read-write storage)
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
                    // binding 7: material_free_list_buf (read-write storage)
                    BindGroupLayoutEntry {
                        binding    : 7,
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
                    // binding 8: material_dispatch_buf (read-write storage)
                    BindGroupLayoutEntry {
                        binding    : 8,
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
                    // binding 9: chunk_alloc_buf (read-write storage)
                    BindGroupLayoutEntry {
                        binding    : 9,
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
                label              : Some("build_alloc_pl"),
                bind_group_layouts : &[Some(&bg_layout)],
                immediate_size     : 16,
            },
        );

        let pipeline = device.create_compute_pipeline(
            &ComputePipelineDescriptor {
                label               : Some("build_alloc_pipeline"),
                layout              : Some(&pipeline_layout),
                module              : &shader,
                entry_point         : Some("main"),
                compilation_options : PipelineCompilationOptions::default(),
                cache               : None,
            },
        );

        BuildAllocPipeline { pipeline, bg_layout }
    }

    /// Create a bind group for the alloc pass.
    ///
    /// # Arguments
    ///
    /// * `device`                 - The GPU device.
    /// # Arguments
    ///
    /// * `bump_state_buf`         - GPU-side quad bump pointer (read-write).
    /// * `build_batch_buf`        - Batch slot indices (read-only).
    /// * `chunk_meta_buf`         - Per-chunk metadata (read-write).
    /// * `quad_range_buf`         - Per-chunk quad range data (read-write).
    /// * `quad_free_list_buf`     - Quad free list (read-write).
    /// * `material_range_buf`     - Per-chunk material range (read-write).
    /// * `material_bump_state_buf`- Material bump pointer (read-write).
    /// * `material_free_list_buf` - Material free list (read-write).
    /// * `material_dispatch_buf`  - Indirect dispatch args for material pack (read-write).
    /// * `chunk_alloc_buf`        - Per-slot allocation page table (read-write).
    pub fn create_bind_group(
        &self,
        device                  : &Device,
        bump_state_buf          : &Buffer,
        build_batch_buf         : &Buffer,
        chunk_meta_buf          : &Buffer,
        quad_range_buf          : &Buffer,
        quad_free_list_buf      : &Buffer,
        material_range_buf      : &Buffer,
        material_bump_state_buf : &Buffer,
        material_free_list_buf  : &Buffer,
        material_dispatch_buf   : &Buffer,
        chunk_alloc_buf         : &Buffer,
    ) -> BindGroup
    {
        device.create_bind_group(&BindGroupDescriptor {
            label   : Some("build_alloc_bg"),
            layout  : &self.bg_layout,
            entries : &[
                BindGroupEntry {
                    binding  : 0,
                    resource : bump_state_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 1,
                    resource : build_batch_buf.as_entire_binding(),
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
                    resource : quad_free_list_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 5,
                    resource : material_range_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 6,
                    resource : material_bump_state_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 7,
                    resource : material_free_list_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 8,
                    resource : material_dispatch_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 9,
                    resource : chunk_alloc_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Returns a reference to the compute pipeline.
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }
}

// ---------------------------------------------------------------------------
// BuildWritePipeline
// ---------------------------------------------------------------------------

/// The compute pipeline for the build write pass (pass 2).
///
/// Re-derives faces and merge using the same occupancy and boundary
/// data, then writes packed quad descriptors into contiguous ranges
/// in `quad_buf`. The write offset for each (direction, layer) bucket
/// is computed from the prefix-summed counts stored in `quad_range_buf`
/// by the count pass.
pub struct BuildWritePipeline {
    /// The compiled compute pipeline.
    pipeline  : ComputePipeline,
    /// The bind group layout for write pass dispatches.
    bg_layout : wgpu::BindGroupLayout,
}

impl BuildWritePipeline {
    /// Create the build write compute pipeline.
    ///
    /// Load the pre-compiled SPIR-V for `build_write.cs.hlsl` and build
    /// the pipeline with the write pass bind group layout.
    pub fn new(device: &Device) -> Self {
        let write_spv = include_bytes!(
            concat!(env!("OUT_DIR"), "/build_write.cs.spv")
        );

        // Safety: SPIR-V compiled from trusted HLSL by DXC at build time.
        let shader = unsafe {
            device.create_shader_module_passthrough(
                ShaderModuleDescriptorPassthrough {
                    label : Some("build_write_shader"),
                    spirv : Some(wgpu::util::make_spirv_raw(write_spv)),
                    ..Default::default()
                },
            )
        };

        let bg_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                label   : Some("build_write_bgl"),
                entries : &[
                    // binding 0: occupancy_buf (read-only storage)
                    BindGroupLayoutEntry {
                        binding    : 0,
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
                    // binding 1: boundary_cache_buf (read-only storage)
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
                    // binding 2: quad_range_buf (read-only storage)
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
                    // binding 3: quad_buf (read-write storage)
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
                ],
            },
        );

        let pipeline_layout = device.create_pipeline_layout(
            &PipelineLayoutDescriptor {
                label              : Some("build_write_pl"),
                bind_group_layouts : &[Some(&bg_layout)],
                immediate_size     : 8,
            },
        );

        let pipeline = device.create_compute_pipeline(
            &ComputePipelineDescriptor {
                label               : Some("build_write_pipeline"),
                layout              : Some(&pipeline_layout),
                module              : &shader,
                entry_point         : Some("main"),
                compilation_options : PipelineCompilationOptions::default(),
                cache               : None,
            },
        );

        BuildWritePipeline { pipeline, bg_layout }
    }

    /// Create a bind group for the write pass.
    ///
    /// The bind group references the shared slot-indexed buffers. It is
    /// created once and reused for all chunk dispatches; the slot index
    /// and base offset are passed via push constants.
    ///
    /// # Arguments
    ///
    /// * `device`            - The GPU device.
    /// * `occupancy_buf`     - Shared occupancy bitmask buffer (read-only).
    /// * `boundary_cache_buf`- Shared neighbor boundary cache (read-only).
    /// * `quad_range_buf`    - Shared quad range data from count pass (read-only).
    /// * `quad_buf`          - Shared quad descriptor buffer (read-write).
    pub fn create_bind_group(
        &self,
        device             : &Device,
        occupancy_buf      : &Buffer,
        boundary_cache_buf : &Buffer,
        quad_range_buf     : &Buffer,
        quad_buf           : &Buffer,
    ) -> BindGroup
    {
        device.create_bind_group(&BindGroupDescriptor {
            label   : Some("build_write_bg"),
            layout  : &self.bg_layout,
            entries : &[
                BindGroupEntry {
                    binding  : 0,
                    resource : occupancy_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 1,
                    resource : boundary_cache_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 2,
                    resource : quad_range_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 3,
                    resource : quad_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Returns a reference to the compute pipeline.
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }
}

// ---------------------------------------------------------------------------
// MaterialPackPipeline
// ---------------------------------------------------------------------------

/// The compute pipeline for material sub-block packing.
///
/// Copies populated 8x8x8 sub-blocks from a transient staging buffer
/// to contiguous positions in the packed material buffer. Only sub-blocks
/// with visible faces (indicated by the sub_mask bitmask) are copied.
pub struct MaterialPackPipeline {
    /// The compiled compute pipeline.
    pipeline     : ComputePipeline,
    /// The bind group layout for material pack dispatches (set 0).
    bg_layout    : wgpu::BindGroupLayout,
    /// The bind group layout for the material buffer array (set 1).
    array_layout : wgpu::BindGroupLayout,
}

impl MaterialPackPipeline {
    /// Create the material pack compute pipeline.
    pub fn new(device: &Device) -> Self {
        let pack_spv = include_bytes!(
            concat!(env!("OUT_DIR"), "/material_pack.cs.spv")
        );

        // Safety: SPIR-V compiled from trusted HLSL by DXC at build time.
        let shader = unsafe {
            device.create_shader_module_passthrough(
                ShaderModuleDescriptorPassthrough {
                    label : Some("material_pack_shader"),
                    spirv : Some(wgpu::util::make_spirv_raw(pack_spv)),
                    ..Default::default()
                },
            )
        };

        let bg_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                label   : Some("material_pack_bgl"),
                entries : &[
                    // binding 0: material_staging_buf (read-only storage)
                    BindGroupLayoutEntry {
                        binding    : 0,
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
                    // binding 1: material_range_buf (read-only storage)
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
                ],
            },
        );

        let array_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                label   : Some("material_pack_array_bgl"),
                entries : &[
                    // binding 0: material_bufs[] (read-write storage array)
                    BindGroupLayoutEntry {
                        binding    : 0,
                        visibility : ShaderStages::COMPUTE,
                        ty         : BindingType::Buffer {
                            ty                 : BufferBindingType::Storage {
                                read_only : false,
                            },
                            has_dynamic_offset : false,
                            min_binding_size   : None,
                        },
                        count : Some(
                            std::num::NonZero::new(MAX_MATERIAL_SEGMENTS)
                                .unwrap(),
                        ),
                    },
                ],
            },
        );

        let pipeline_layout = device.create_pipeline_layout(
            &PipelineLayoutDescriptor {
                label              : Some("material_pack_pl"),
                bind_group_layouts : &[
                    Some(&bg_layout),
                    Some(&array_layout),
                ],
                immediate_size     : 8,
            },
        );

        let pipeline = device.create_compute_pipeline(
            &ComputePipelineDescriptor {
                label               : Some("material_pack_pipeline"),
                layout              : Some(&pipeline_layout),
                module              : &shader,
                entry_point         : Some("main"),
                compilation_options : PipelineCompilationOptions::default(),
                cache               : None,
            },
        );

        MaterialPackPipeline { pipeline, bg_layout, array_layout }
    }

    /// Create a bind group for the material pack pass (set 0).
    ///
    /// # Arguments
    ///
    /// * `device`               - The GPU device.
    /// * `material_staging_buf` - Transient staging buffer (read-only).
    /// * `material_range_buf`   - Per-slot material range metadata (read-only).
    pub fn create_bind_group(
        &self,
        device               : &Device,
        material_staging_buf : &Buffer,
        material_range_buf   : &Buffer,
    ) -> BindGroup
    {
        device.create_bind_group(&BindGroupDescriptor {
            label   : Some("material_pack_bg"),
            layout  : &self.bg_layout,
            entries : &[
                BindGroupEntry {
                    binding  : 0,
                    resource : material_staging_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 1,
                    resource : material_range_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Create the buffer array bind group (set 1) for material pack.
    ///
    /// Rebuild this whenever the multi-buffer grows.
    pub fn create_array_bind_group(
        &self,
        device  : &Device,
        buffers : &[Buffer],
    ) -> BindGroup
    {
        let bindings: Vec<wgpu::BufferBinding> = buffers
            .iter()
            .map(|b| wgpu::BufferBinding {
                buffer : b,
                offset : 0,
                size   : None,
            })
            .collect();

        device.create_bind_group(&BindGroupDescriptor {
            label   : Some("material_pack_array_bg"),
            layout  : &self.array_layout,
            entries : &[
                BindGroupEntry {
                    binding  : 0,
                    resource : BindingResource::BufferArray(&bindings),
                },
            ],
        })
    }

    /// Returns a reference to the compute pipeline.
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }
}
