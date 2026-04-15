//! Sub-chunk bitmask DDA prototype — 64-instance cull + MDI draw pass.
//!
//! Rasterizes up to [`MAX_CANDIDATES`] sub-chunk bounding cubes via
//! GPU frustum cull → multi-draw-indirect. Each sub-chunk has:
//! - A world-space [`SubchunkInstance`] (origin + occupancy slot).
//! - A [`SubchunkOccupancy`] (8×8×8 bitmask, 64 bytes).
//!
//! # Binding layouts
//!
//! **Cull bind group** (set 0 for `subchunk_cull.cs.hlsl`):
//! - 0: GpuConsts (injected)
//! - 1: camera     (UniformBuffer, 64)       — COMPUTE
//! - 2: instances  (StorageReadOnly, 16×MAX) — COMPUTE
//! - 3: visible    (StorageRW, 4×MAX)        — COMPUTE
//! - 4: indirect   (StorageRW, 16)           — COMPUTE
//! - 5: count      (StorageRW, 4)            — COMPUTE
//!
//! **Render bind group** (set 0 for `subchunk.vs/.ps.hlsl`):
//! - 0: GpuConsts  (injected)
//! - 1: camera     (UniformBuffer, 64)       — VERTEX | FRAGMENT
//! - 2: instances  (StorageReadOnly, 16×MAX) — VERTEX
//! - 3: visible    (StorageReadOnly, 4×MAX)  — VERTEX
//! - 4: occ_array  (StorageReadOnly, 64×MAX) — FRAGMENT
//!
//! # wgpu-hal Vulkan binding compaction
//!
//! wgpu-hal sorts BGL entries by binding number and assigns sequential VK
//! bindings 0, 1, 2, … The SPIR-V passthrough path does not remap, so the
//! HLSL `[[vk::binding(N, 0)]]` number MUST equal N exactly. User bindings
//! start at 1 and are consecutive — no gaps. See `subchunk.ps.hlsl` comments.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::device::RendererContext;
use crate::gpu_consts::{GpuConsts, GpuConstsData};
use crate::graph::{RenderGraph, TextureHandle};
use crate::nodes::{CullArgs, DrawArgs, IndirectArgs, cull, mdi_draw};
use crate::pipeline::binding::{BindEntry, BindKind, BindingLayout};
use crate::pipeline::compute::{ComputePipeline, ComputePipelineDescriptor};
use crate::pipeline::render::{RenderPipeline, RenderPipelineDescriptor};
use crate::shader::{ShaderModule, ShaderSource};

// --- Compiled shader bytes (produced by build.rs + DXC) ---

const SUBCHUNK_VS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/subchunk.vs.spv"));

const SUBCHUNK_PS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/subchunk.ps.spv"));

const SUBCHUNK_CS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/subchunk_cull.cs.spv"));

// --- Constants ---

/// Maximum number of candidate sub-chunks.  One thread per candidate in the
/// cull dispatch; one workgroup of 64 threads handles the full set.
pub const MAX_CANDIDATES: usize = 64;

// --- Public types ---

/// Camera parameters for the sub-chunk test render pass.
///
/// Layout must match the HLSL `Camera` struct (64 bytes):
/// ```text
///   float3 pos     (+0)
///   float  fov_y   (+12)
///   float3 forward (+16)
///   float  aspect  (+28)
///   float3 right   (+32)
///   float  _pad0   (+44)
///   float3 up      (+48)
///   float  _pad1   (+60)
/// ```
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TestCamera {
    pub pos:     [f32; 3],
    pub fov_y:   f32,
    pub forward: [f32; 3],
    pub aspect:  f32,
    pub right:   [f32; 3],
    pub _pad0:   f32,
    pub up:      [f32; 3],
    pub _pad1:   f32,
}

const _: () = assert!(
    std::mem::size_of::<TestCamera>() == 64,
    "TestCamera must be 64 bytes to match HLSL Camera"
);

/// Per-sub-chunk instance data: world-space origin and occupancy slot.
///
/// Layout must match the HLSL `Instance` struct (16 bytes):
/// ```text
///   int3 origin   (+0)
///   uint occ_slot (+12)
/// ```
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SubchunkInstance {
    /// Voxel-space origin of the `[origin, origin+8)` sub-chunk box.
    pub origin:   [i32; 3],
    /// Index into the occupancy storage buffer.
    pub occ_slot: u32,
}

const _: () = assert!(
    std::mem::size_of::<SubchunkInstance>() == 16,
    "SubchunkInstance must be 16 bytes to match HLSL Instance"
);

/// 8×8×8 occupancy stored as 8 XY planes, each a 64-bit bitmask.
///
/// `planes[z * 2 .. z * 2 + 2]` encodes Z layer `z`.  In the 64-bit value,
/// bit `y * 8 + x` is set if voxel `(x, y, z)` is occupied.
///
/// In the HLSL storage buffer the 16 u32s are packed into `uint4 plane[4]`
/// (four vec4s = 64 bytes, no internal padding issues since everything is u32).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SubchunkOccupancy {
    pub planes: [u32; 16],
}

/// Build a sphere occupancy: voxel (x,y,z) is occupied when the voxel centre
/// lies within a sphere of radius 3.5 centred at (3.5, 3.5, 3.5).
pub fn sphere_occupancy() -> SubchunkOccupancy {
    let mut occ = SubchunkOccupancy { planes: [0u32; 16] };
    for z in 0u32..8 {
        for y in 0u32..8 {
            for x in 0u32..8 {
                let fx = x as f32 - 3.5;
                let fy = y as f32 - 3.5;
                let fz = z as f32 - 3.5;
                if fx * fx + fy * fy + fz * fz <= 3.5 * 3.5 {
                    let bit  = y * 8 + x;         // 0..63
                    let word = z * 2 + (bit >> 5); // 0..15
                    occ.planes[word as usize] |= 1 << (bit & 31);
                }
            }
        }
    }
    occ
}

// --- SubchunkTest ---

/// Persistent state for the sub-chunk cull + MDI draw pass.
///
/// Owns both pipelines and both bind groups, plus all six persistent buffers.
/// Camera data is updated per-frame via [`Self::write_camera`] before the
/// render graph is built.  Instance and occupancy data are committed at
/// construction and stay fixed for the lifetime of this object.
///
/// The render graph imports [`Self::indirect_buf`] and [`Self::count_buf`] as
/// persistent resources each frame (cheaply Arc-backed).
pub struct SubchunkTest {
    // --- Pipelines ---
    cull_pipeline   : Arc<ComputePipeline>,
    render_pipeline : Arc<RenderPipeline>,

    // --- Bind groups ---
    cull_bind_group   : wgpu::BindGroup,
    render_bind_group : wgpu::BindGroup,

    // --- Persistent buffers ---
    camera_buf    : wgpu::Buffer,
    // The following three buffers are retained to keep the GPU allocations
    // alive; they are bound in the cull and render bind groups and accessed
    // only by the GPU after construction.
    _instance_buf : wgpu::Buffer,
    _occ_buf      : wgpu::Buffer,
    _visible_buf  : wgpu::Buffer,
    indirect_buf  : wgpu::Buffer,
    count_buf     : wgpu::Buffer,

    // --- GpuConsts placeholders ---
    _gpu_consts_cull   : GpuConsts,
    _gpu_consts_render : GpuConsts,
}

// --- SubchunkTest ---

impl SubchunkTest {
    /// Create the pass for the current surface format.
    ///
    /// `instances` and `occ` must be exactly [`MAX_CANDIDATES`] elements each.
    /// Instance and occupancy data are written at allocation time via
    /// `create_buffer_init` — no `write_buffer` ordering hazards.
    ///
    /// # Panics
    ///
    /// Panics if `instances.len() != MAX_CANDIDATES` or
    /// `occ.len() != MAX_CANDIDATES`.
    pub fn new(
        ctx       : &RendererContext,
        instances : &[SubchunkInstance; MAX_CANDIDATES],
        occ       : &[SubchunkOccupancy; MAX_CANDIDATES],
    )
        -> Self
    {
        let surface_format = ctx
            .surface_format()
            .expect("SubchunkTest requires a windowed RendererContext");

        let device = ctx.device();

        // --- Shader modules ---

        let vs = ShaderModule::load(
            ctx, "subchunk.vs", ShaderSource::Spirv(SUBCHUNK_VS_SPV), "main",
        )
        .expect("subchunk vertex shader failed to load");

        let ps = ShaderModule::load(
            ctx, "subchunk.ps", ShaderSource::Spirv(SUBCHUNK_PS_SPV), "main",
        )
        .expect("subchunk pixel shader failed to load");

        let cs = ShaderModule::load(
            ctx, "subchunk_cull.cs", ShaderSource::Spirv(SUBCHUNK_CS_SPV), "main",
        )
        .expect("subchunk cull shader failed to load");

        // --- Buffer sizes ---

        let camera_size   = std::mem::size_of::<TestCamera>() as u64;
        let instance_size = (std::mem::size_of::<SubchunkInstance>() * MAX_CANDIDATES) as u64;
        let occ_size      = (std::mem::size_of::<SubchunkOccupancy>() * MAX_CANDIDATES) as u64;
        let visible_size  = (4 * MAX_CANDIDATES) as u64;
        let indirect_size = 16u64; // one DrawIndirectArgs = 4 × u32
        let count_size    = 4u64;  // one u32

        // --- Persistent buffers ---

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_camera"),
            size:               camera_size,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Instance and occupancy data are fixed at construction — commit via
        // create_buffer_init (bytes written at allocation, no ordering hazards).
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("subchunk_instances"),
            contents: bytemuck::cast_slice(instances),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let occ_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("subchunk_occ"),
            contents: bytemuck::cast_slice(occ),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // The following buffers are cull-output; their contents are undefined
        // until the cull shader writes them each frame.
        let visible_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_visible"),
            size:               visible_size,
            usage:              wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let indirect_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_indirect"),
            size:               indirect_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_count"),
            size:               count_size,
            usage:              wgpu::BufferUsages::STORAGE
                              | wgpu::BufferUsages::INDIRECT
                              | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Cull binding layout (bindings 1-5) ---

        let cull_layout = Arc::new(
            BindingLayout::builder("subchunk_cull")
                .add_entry(BindEntry {
                    binding:    1,
                    kind:       BindKind::UniformBuffer { size: camera_size },
                    visibility: wgpu::ShaderStages::COMPUTE,
                })
                .add_entry(BindEntry {
                    binding:    2,
                    kind:       BindKind::StorageBufferReadOnly { size: instance_size },
                    visibility: wgpu::ShaderStages::COMPUTE,
                })
                .add_entry(BindEntry {
                    binding:    3,
                    kind:       BindKind::StorageBufferReadWrite { size: visible_size },
                    visibility: wgpu::ShaderStages::COMPUTE,
                })
                .add_entry(BindEntry {
                    binding:    4,
                    kind:       BindKind::StorageBufferReadWrite { size: indirect_size },
                    visibility: wgpu::ShaderStages::COMPUTE,
                })
                .add_entry(BindEntry {
                    binding:    5,
                    kind:       BindKind::StorageBufferReadWrite { size: count_size },
                    visibility: wgpu::ShaderStages::COMPUTE,
                })
                .build(ctx),
        );

        // --- Render binding layout (bindings 1-4) ---

        let render_layout = Arc::new(
            BindingLayout::builder("subchunk")
                .add_entry(BindEntry {
                    binding:    1,
                    kind:       BindKind::UniformBuffer { size: camera_size },
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                })
                .add_entry(BindEntry {
                    binding:    2,
                    kind:       BindKind::StorageBufferReadOnly { size: instance_size },
                    visibility: wgpu::ShaderStages::VERTEX,
                })
                .add_entry(BindEntry {
                    binding:    3,
                    kind:       BindKind::StorageBufferReadOnly { size: visible_size },
                    visibility: wgpu::ShaderStages::VERTEX,
                })
                .add_entry(BindEntry {
                    binding:    4,
                    kind:       BindKind::StorageBufferReadOnly { size: occ_size },
                    visibility: wgpu::ShaderStages::FRAGMENT,
                })
                .build(ctx),
        );

        // --- Pipelines ---

        let cull_pipeline = Arc::new(ComputePipeline::new(ctx, ComputePipelineDescriptor {
            label:                 "subchunk_cull",
            shader:                cs,
            layout:                Arc::clone(&cull_layout),
            expected_workgroup_size: Some([64, 1, 1]),
            immediate_size:        0,
        }));

        // wgpu's Vulkan backend flips Y via a negative viewport height, which
        // reverses framebuffer winding relative to world-space winding; our
        // cube table is outward-CCW in world space, so Cw is the front-face
        // rule.
        //
        // Back-face culling is on. The vertex shader swaps triangle winding
        // for cubes the camera is inside of (or near-plane-close to), which
        // inverts which side the rasterizer treats as "front" — so inside
        // cubes still get fragments via their back faces, while the common
        // outside case still pays rasterization cost for only 3 of the 6
        // cube faces.
        let render_pipeline = Arc::new(RenderPipeline::new(ctx, RenderPipelineDescriptor {
            label:          "subchunk",
            vertex:         vs,
            fragment:       ps,
            layout:         Arc::clone(&render_layout),
            vertex_buffers: &[],
            color_targets:  &[Some(wgpu::ColorTargetState {
                format:     surface_format,
                blend:      None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            depth_stencil:  Some(wgpu::DepthStencilState {
                format:              DEPTH_FORMAT,
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
            immediate_size: 0,
        }));

        // --- GpuConsts placeholders (one per bind group) ---

        let gpu_consts_cull   = GpuConsts::new(ctx, GpuConstsData::default());
        let gpu_consts_render = GpuConsts::new(ctx, GpuConstsData::default());

        // --- Cull bind group ---

        let cull_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("subchunk_cull_bg"),
            layout:  cull_layout.wgpu_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: gpu_consts_cull.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: instance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  3,
                    resource: visible_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  4,
                    resource: indirect_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  5,
                    resource: count_buf.as_entire_binding(),
                },
            ],
        });

        // --- Render bind group ---

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("subchunk_render_bg"),
            layout:  render_layout.wgpu_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: gpu_consts_render.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: instance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  3,
                    resource: visible_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  4,
                    resource: occ_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            cull_pipeline,
            render_pipeline,
            cull_bind_group,
            render_bind_group,
            camera_buf,
            _instance_buf: instance_buf,
            _occ_buf:      occ_buf,
            _visible_buf:  visible_buf,
            indirect_buf,
            count_buf,
            _gpu_consts_cull:   gpu_consts_cull,
            _gpu_consts_render: gpu_consts_render,
        }
    }

    /// Upload camera data to the GPU. Call once per frame, before building
    /// the render graph.
    pub fn write_camera(&self, ctx: &RendererContext, camera: &TestCamera) {
        ctx.queue().write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(camera));
    }

    pub(crate) fn cull_pipeline(&self) -> &Arc<ComputePipeline> {
        &self.cull_pipeline
    }

    pub(crate) fn cull_bind_group(&self) -> &wgpu::BindGroup {
        &self.cull_bind_group
    }

    pub(crate) fn render_pipeline(&self) -> &Arc<RenderPipeline> {
        &self.render_pipeline
    }

    pub(crate) fn render_bind_group(&self) -> &wgpu::BindGroup {
        &self.render_bind_group
    }

    /// The persistent indirect-args buffer.  Import this as a graph resource
    /// each frame via `graph.import_buffer(test.indirect_buf().clone())`.
    pub(crate) fn indirect_buf(&self) -> &wgpu::Buffer {
        &self.indirect_buf
    }

    /// The persistent draw-count buffer.  Import alongside [`Self::indirect_buf`].
    pub(crate) fn count_buf(&self) -> &wgpu::Buffer {
        &self.count_buf
    }
}

/// Depth format used by [`subchunk_test`] and by the pipeline's `DepthStencilState`.
/// The game loop allocates a transient depth texture in this format each frame.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Register the sub-chunk cull + MDI draw passes into `graph`.
///
/// Imports the persistent `indirect_buf` and `count_buf` as graph resources,
/// dispatches the GPU frustum cull (1 workgroup × 64 threads), then issues a
/// multi-draw-indirect-count raster pass that draws only the visible sub-chunks.
///
/// Returns the versioned output handles for the color and depth textures.
pub fn subchunk_test(
    graph : &mut RenderGraph,
    test  : &Arc<SubchunkTest>,
    color : TextureHandle,
    depth : TextureHandle,
)
    -> (TextureHandle, TextureHandle)
{
    let indirect = graph.import_buffer(test.indirect_buf().clone());
    let count    = graph.import_buffer(test.count_buf().clone());
    let args     = IndirectArgs { indirect, count, max_draws: 1 };

    let args = cull(
        graph,
        test.cull_pipeline(),
        test.cull_bind_group(),
        &CullArgs { workgroups: [1, 1, 1] },
        args,
    );

    mdi_draw(
        graph,
        test.render_pipeline(),
        test.render_bind_group(),
        &args,
        &DrawArgs::default(),
        color,
        depth,
    )
}
