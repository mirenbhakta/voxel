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
    ComputePipeline, ComputePipelineDescriptor, Device, Queue,
    PipelineCompilationOptions, PipelineLayoutDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

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
                    // binding 2: shared quad pool (read-write storage)
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
                    // binding 3: page table (read-only storage)
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
                ],
            },
        );

        let pipeline_layout = device.create_pipeline_layout(
            &PipelineLayoutDescriptor {
                label              : Some("build_pl"),
                bind_group_layouts : &[Some(&bg_layout)],
                immediate_size     : 4,
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
    occupancy_buf  : Buffer,
    /// Atomic quad count (4 bytes, zeroed before dispatch).
    quad_count_buf : Buffer,
    /// Staging buffer for reading back the quad count to CPU.
    count_staging  : Buffer,
    /// Bind group for the compute dispatch.
    bind_group     : BindGroup,
}

impl ChunkBuildData {
    /// Create GPU resources for building a chunk's quad buffer.
    ///
    /// Uploads the occupancy data and allocates per-chunk buffers. The
    /// bind group references the shared quad pool and page table rather
    /// than a per-chunk quad buffer.
    ///
    /// # Arguments
    ///
    /// * `device`     - The GPU device for resource creation.
    /// * `pipeline`   - The build compute pipeline (provides the BGL).
    /// * `occ`        - Initial chunk occupancy bitmask.
    /// * `quad_pool`  - The shared quad storage buffer.
    /// * `page_table` - The shared page table buffer.
    pub fn new(
        device     : &Device,
        pipeline   : &BuildPipeline,
        occ        : &[u32; 1024],
        quad_pool  : &Buffer,
        page_table : &Buffer,
    ) -> Self
    {
        let occupancy_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label    : Some("build_occ"),
                contents : bytemuck::cast_slice(occ),
                usage    : BufferUsages::STORAGE
                         | BufferUsages::COPY_DST,
            },
        );

        // Quad count initialized to 0.
        let quad_count_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label    : Some("build_count"),
                contents : bytemuck::bytes_of(&0u32),
                usage    : BufferUsages::STORAGE
                         | BufferUsages::COPY_SRC
                         | BufferUsages::COPY_DST,
            },
        );

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
                    resource : quad_pool.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 3,
                    resource : page_table.as_entire_binding(),
                },
            ],
        });

        ChunkBuildData {
            occupancy_buf  : occupancy_buf,
            quad_count_buf : quad_count_buf,
            count_staging  : count_staging,
            bind_group     : bind_group,
        }
    }

    /// Record the compute dispatch and count readback copy.
    ///
    /// Resets the atomic counter, dispatches the compute pass with the
    /// chunk's page table offset, and copies the counter to the staging
    /// buffer. After this call, submit the encoder to the queue, then
    /// call [`read_quad_count`] to retrieve the result.
    ///
    /// # Arguments
    ///
    /// * `encoder`    - The command encoder to record into.
    /// * `pipeline`   - The build compute pipeline.
    /// * `block_base` - Offset into the page table for this chunk's slot.
    pub fn dispatch(
        &self,
        encoder    : &mut CommandEncoder,
        pipeline   : &BuildPipeline,
        block_base : u32,
    )
    {
        // Reset the atomic counter to zero.
        encoder.clear_buffer(&self.quad_count_buf, 0, None);

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
            pass.set_immediates(0, bytemuck::bytes_of(&block_base));
            pass.dispatch_workgroups(32, 6, 1);
        }

        // Copy the atomic counter to the staging buffer.
        encoder.copy_buffer_to_buffer(
            &self.quad_count_buf, 0,
            &self.count_staging,  0,
            4,
        );
    }

    /// Upload new occupancy data to the GPU buffer.
    ///
    /// The data is written immediately via the queue. The caller must
    /// dispatch the build shader afterward to update the quad buffer.
    pub fn upload_occupancy(&self, queue: &Queue, occ: &[u32; 1024]) {
        queue.write_buffer(
            &self.occupancy_buf,
            0,
            bytemuck::cast_slice(occ),
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
