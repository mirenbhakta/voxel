//! GPU compute build stage for chunk face derivation and greedy merge.
//!
//! Takes a chunk's occupancy bitmask, derives per-direction face bitmasks
//! via AND-NOT, performs material-agnostic greedy merge, and writes packed
//! quad descriptors to a GPU storage buffer. The output feeds the existing
//! vertex shader directly.

use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineCompilationOptions, PipelineLayoutDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum quads a single chunk can produce.
///
/// Worst case is a 3D checkerboard: every other voxel occupied, each
/// with 6 exposed faces. 32^3 / 2 * 6 = 98304 quads. Greedy merge
/// reduces this dramatically, but the buffer must handle the pathological
/// case.
const MAX_QUADS: u32 = 98_304;

// ---------------------------------------------------------------------------
// BuildPipeline
// ---------------------------------------------------------------------------

/// The compute pipeline for the build stage.
pub struct BuildPipeline {
    /// The compiled compute pipeline.
    pipeline  : ComputePipeline,
    /// The bind group layout shared by all chunk build dispatches.
    bg_layout : wgpu::BindGroupLayout,
}

impl BuildPipeline {
    /// Create the build stage compute pipeline.
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label  : Some("build_shader"),
            source : ShaderSource::Wgsl(
                include_str!("shaders/build.wgsl").into(),
            ),
        });

        let bg_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                label   : Some("build_bgl"),
                entries : &[
                    // binding 0: occupancy (read-only storage)
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
                    // binding 1: quad count (read-write storage)
                    BindGroupLayoutEntry {
                        binding    : 1,
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
                    // binding 2: quad buffer (read-write storage)
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
                ],
            },
        );

        let pipeline_layout = device.create_pipeline_layout(
            &PipelineLayoutDescriptor {
                label              : Some("build_pl"),
                bind_group_layouts : &[Some(&bg_layout)],
                immediate_size     : 0,
            },
        );

        let pipeline = device.create_compute_pipeline(
            &ComputePipelineDescriptor {
                label               : Some("build_pipeline"),
                layout              : Some(&pipeline_layout),
                module              : &shader,
                entry_point         : Some("build"),
                compilation_options : PipelineCompilationOptions::default(),
                cache               : None,
            },
        );

        BuildPipeline { pipeline, bg_layout }
    }
}

// ---------------------------------------------------------------------------
// ChunkBuildData
// ---------------------------------------------------------------------------

/// GPU resources for building one chunk's quad buffer.
pub struct ChunkBuildData {
    /// The chunk's occupancy bitmask on the GPU (4 KB).
    _occupancy_buf : Buffer,
    /// Atomic quad count (4 bytes, zeroed before dispatch).
    quad_count_buf : Buffer,
    /// Output quad buffer, read by the render pipeline.
    pub quad_buf   : Buffer,
    /// Staging buffer for reading back the quad count to CPU.
    count_staging  : Buffer,
    /// Bind group for the compute dispatch.
    bind_group     : BindGroup,
}

impl ChunkBuildData {
    /// Create GPU resources for building a chunk's quad buffer.
    ///
    /// Uploads the occupancy data and allocates output buffers. The quad
    /// count is initialized to zero.
    pub fn new(
        device   : &Device,
        pipeline : &BuildPipeline,
        occ      : &[u32; 1024],
    ) -> Self
    {
        let occupancy_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label    : Some("build_occ"),
                contents : bytemuck::cast_slice(occ),
                usage    : BufferUsages::STORAGE,
            },
        );

        // Quad count initialized to 0.
        let quad_count_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label    : Some("build_count"),
                contents : bytemuck::bytes_of(&0u32),
                usage    : BufferUsages::STORAGE
                         | BufferUsages::COPY_SRC,
            },
        );

        let quad_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("build_quads"),
            size               : u64::from(MAX_QUADS) * 4,
            usage              : BufferUsages::STORAGE,
            mapped_at_creation : false,
        });

        let count_staging = device.create_buffer(&BufferDescriptor {
            label              : Some("build_count_staging"),
            size               : 4,
            usage              : BufferUsages::COPY_DST
                               | BufferUsages::MAP_READ,
            mapped_at_creation : false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label   : Some("build_bg"),
            layout  : &pipeline.bg_layout,
            entries : &[
                BindGroupEntry {
                    binding  : 0,
                    resource : occupancy_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 1,
                    resource : quad_count_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 2,
                    resource : quad_buf.as_entire_binding(),
                },
            ],
        });

        ChunkBuildData {
            _occupancy_buf : occupancy_buf,
            quad_count_buf,
            quad_buf,
            count_staging,
            bind_group,
        }
    }

    /// Record the compute dispatch and count readback copy.
    ///
    /// After this call, submit the encoder to the queue. Then call
    /// [`read_quad_count`] to retrieve the result.
    pub fn dispatch(
        &self,
        encoder  : &mut CommandEncoder,
        pipeline : &BuildPipeline,
    )
    {
        // Compute pass: face derivation + greedy merge.
        {
            let mut pass = encoder.begin_compute_pass(
                &ComputePassDescriptor {
                    label            : Some("build"),
                    timestamp_writes : None,
                },
            );

            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(32, 6, 1);
        }

        // Copy the atomic counter to the staging buffer.
        encoder.copy_buffer_to_buffer(
            &self.quad_count_buf, 0,
            &self.count_staging,  0,
            4,
        );
    }

    /// Read back the quad count after the dispatch has completed.
    ///
    /// Blocks until the GPU finishes and the staging buffer is mapped.
    pub fn read_quad_count(&self, device: &Device) -> u32 {
        let slice = self.count_staging.slice(..);

        // Request mapping and block until the GPU delivers.
        let (tx, rx) = std::sync::mpsc::sync_channel::<()>(1);

        slice.map_async(wgpu::MapMode::Read, move |result| {
            result.unwrap();
            tx.send(()).unwrap();
        });

        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap();

        let data  = slice.get_mapped_range();
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);

        self.count_staging.unmap();

        count
    }
}
