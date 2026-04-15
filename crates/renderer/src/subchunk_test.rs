//! Isolated sub-chunk bitmask ray cast — prototype render pass.
//!
//! Rasterizes the sub-chunk's bounding cube and runs a planar-bitmask DDA
//! inside the fragment shader, starting from the fragment's world-space hull
//! position (the rasterizer performs the ray-AABB entry intersection for us).
//!
//! Three bindings (set 0):
//! - Binding 0: `GpuConsts` (32 bytes) — injected at slot 0 by
//!   [`BindingLayout`]. Unused by this shader, but must be present because
//!   wgpu-hal's Vulkan backend sequentially compacts BGL binding numbers to
//!   VK 0, 1, 2, ... regardless of user-supplied values. The SPIR-V
//!   passthrough path does not remap bindings, so the shader's bindings must
//!   match the compacted VK bindings; skipping slot 0 causes silent
//!   off-by-one binding reads.
//! - Binding 1: Camera uniform (64 bytes, updated each frame via
//!   [`SubchunkTest::write_camera`], read by both vertex and fragment stages).
//! - Binding 2: Occupancy uniform (64 bytes, set once via `set_occupancy`).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::commands::{ColorAttachment, DepthAttachment, RasterPassDesc};
use crate::device::RendererContext;
use crate::gpu_consts::{GpuConsts, GpuConstsData};
use crate::graph::{RenderGraph, TextureHandle};
use crate::pipeline::binding::{BindEntry, BindKind, BindingLayout};
use crate::pipeline::render::{RenderPipeline, RenderPipelineDescriptor};
use crate::shader::{ShaderModule, ShaderSource};

// --- Compiled shader bytes (produced by build.rs + DXC) ---

const SUBCHUNK_VS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/subchunk.vs.spv"));

const SUBCHUNK_PS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/subchunk.ps.spv"));

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

/// 8×8×8 occupancy stored as 8 XY planes, each a 64-bit bitmask.
///
/// `planes[z * 2 .. z * 2 + 2]` encodes Z layer `z`.  In the 64-bit value,
/// bit `y * 8 + x` is set if voxel `(x, y, z)` is occupied.
///
/// In the HLSL constant buffer the 16 u32s are packed into `uint4 plane[4]`
/// (four vec4s = 64 bytes, no std140 element padding).
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

/// Persistent state for the sub-chunk ray-cast pass.
///
/// Owns the [`RenderPipeline`], bind group, camera/occupancy uniform buffers,
/// and the [`GpuConsts`] placeholder bound at slot 0. Depth and colour
/// attachments are graph-managed — the game loop hands transient / imported
/// texture handles to [`subchunk_test`] each frame.
///
/// Camera data is updated per-frame via [`Self::write_camera`] before the
/// graph is built; the node closure captures only the bind group (cheaply
/// cloneable `Arc`-backed handles), not the camera buffer directly.
pub struct SubchunkTest {
    pipeline:    RenderPipeline,
    bind_group:  wgpu::BindGroup,
    camera_buf:  wgpu::Buffer,
    occ_buf:     wgpu::Buffer,
    _gpu_consts: GpuConsts,
}

impl SubchunkTest {
    /// Create the test pass for the current surface format.
    ///
    /// Loads shaders, builds a [`BindingLayout`] (slot 0 = GpuConsts injected,
    /// slot 1 = camera, slot 2 = occupancy), creates the [`RenderPipeline`]
    /// and the two uniform buffers, then pre-fills the occupancy buffer via
    /// `create_buffer_init` (writes the bytes at allocation time, avoiding
    /// any `write_buffer` ordering hazards).
    pub fn new(ctx: &RendererContext, occ: &SubchunkOccupancy) -> Self {
        let surface_format = ctx
            .surface_format()
            .expect("SubchunkTest requires a windowed RendererContext");

        let device = ctx.device();

        let vs = ShaderModule::load(
            ctx, "subchunk.vs", ShaderSource::Spirv(SUBCHUNK_VS_SPV), "main",
        )
        .expect("subchunk vertex shader failed to load");

        let ps = ShaderModule::load(
            ctx, "subchunk.ps", ShaderSource::Spirv(SUBCHUNK_PS_SPV), "main",
        )
        .expect("subchunk pixel shader failed to load");

        let layout = Arc::new(
            BindingLayout::builder("subchunk")
                .add_entry(BindEntry {
                    binding:    1,
                    kind:       BindKind::UniformBuffer {
                        size: std::mem::size_of::<TestCamera>() as u64,
                    },
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                })
                .add_entry(BindEntry {
                    binding:    2,
                    kind:       BindKind::UniformBuffer {
                        size: std::mem::size_of::<SubchunkOccupancy>() as u64,
                    },
                    visibility: wgpu::ShaderStages::FRAGMENT,
                })
                .build(ctx),
        );

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
        let pipeline = RenderPipeline::new(ctx, RenderPipelineDescriptor {
            label:          "subchunk_test",
            vertex:         vs,
            fragment:       ps,
            layout:         Arc::clone(&layout),
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
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_test_camera"),
            size:               std::mem::size_of::<TestCamera>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pre-fill with the caller's occupancy data at creation time.
        // Using create_buffer_init (not create_buffer + write_buffer) avoids
        // queue submission ordering issues: the bytes are committed during
        // device-side allocation, before any command submission.
        let occ_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("subchunk_test_occ"),
            contents: bytemuck::bytes_of(occ),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let gpu_consts = GpuConsts::new(ctx, GpuConstsData::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("subchunk_test_bind_group"),
            layout:  layout.wgpu_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: gpu_consts.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: occ_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
            camera_buf,
            occ_buf,
            _gpu_consts: gpu_consts,
        }
    }

    /// Upload camera data to the GPU uniform. Call once per frame, before
    /// building the render graph.
    pub fn write_camera(&self, ctx: &RendererContext, camera: &TestCamera) {
        ctx.queue().write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(camera));
    }

    /// Upload occupancy data to the GPU. Call whenever the voxel data changes.
    pub fn set_occupancy(&self, ctx: &RendererContext, occ: &SubchunkOccupancy) {
        ctx.queue().write_buffer(&self.occ_buf, 0, bytemuck::bytes_of(occ));
    }

    pub(crate) fn pipeline(&self) -> &RenderPipeline {
        &self.pipeline
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

/// Depth format used by [`subchunk_test`] and by [`SubchunkTest::new`]'s
/// pipeline `DepthStencilState`. The game loop allocates a transient depth
/// texture in this format each frame.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Register the sub-chunk test pass into `graph`.
///
/// Declares a write of `color` and `depth` (producing versioned handles)
/// and records a raster pass that clears both, then draws the 36-vertex
/// cube. The pipeline's vertex shader runs a hull-entry DDA inside the
/// fragment shader starting from each fragment's world-space position.
///
/// The `test` handle is cloned into the execute closure; both its pipeline
/// and bind group are `Arc`-backed by wgpu and thus cheap to hold.
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
    graph.add_pass("subchunk_test", |pass| {
        let color_v = pass.write_texture(color);
        let depth_v = pass.write_texture(depth);

        let test = Arc::clone(test);
        pass.execute(move |ctx| {
            let color_view = ctx.resources.texture_view(color_v);
            let depth_view = ctx.resources.texture_view(depth_v);

            ctx.commands.raster_pass(
                &RasterPassDesc {
                    label : "subchunk_test",
                    color : &[ColorAttachment::clear(
                        color_view,
                        [0.05, 0.05, 0.08, 1.0],
                    )],
                    depth : Some(DepthAttachment::clear(depth_view, 1.0)),
                },
                |rp| {
                    rp.draw(test.pipeline(), test.bind_group(), 0..36, 0..1);
                },
            );
        });

        (color_v, depth_v)
    })
}
