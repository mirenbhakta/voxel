//! Isolated sub-chunk bitmask ray cast — prototype render pass.
//!
//! Rasterizes the sub-chunk's bounding cube and runs a planar-bitmask DDA
//! inside the fragment shader, starting from the fragment's world-space hull
//! position (the rasterizer performs the ray-AABB entry intersection for us).
//!
//! Self-contained: bypasses `BindingLayout` and builds its own
//! `wgpu::RenderPipeline` directly, since `RenderPipeline` in the renderer
//! crate defers colour targets to the DDA pipeline increment.
//!
//! Three bindings (set 0):
//! - Binding 0: `GpuConsts` (32 bytes) — unused by this shader, but must be
//!   present because wgpu-hal's Vulkan backend sequentially compacts BGL
//!   binding numbers to VK 0, 1, 2, ... regardless of user-supplied values.
//!   The SPIR-V passthrough path does not remap bindings, so the shader's
//!   bindings must match the compacted VK bindings; skipping slot 0 causes
//!   silent off-by-one binding reads.
//! - Binding 1: Camera uniform (64 bytes, updated each frame via queue.write_buffer,
//!   read by both vertex and fragment stages).
//! - Binding 2: Occupancy uniform (64 bytes, set once via `set_occupancy`).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::device::{FrameEncoder, RendererContext, SurfaceFrame};
use crate::gpu_consts::{GpuConsts, GpuConstsData};
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

/// Self-contained render pass that ray-casts a single sub-chunk.
///
/// Builds its own wgpu pipeline and bind group, independent of the
/// renderer's `BindingLayout` / `RenderPipeline` infrastructure. A placeholder
/// `GpuConsts` is bound at slot 0; see the module doc for why.
///
/// Owns its depth texture: the cube rasterization path needs depth so that
/// overlapping sub-chunks composite correctly. The depth view lives for the
/// lifetime of the test and is recreated on window resize via [`Self::resize`].
pub struct SubchunkTest {
    pipeline:      wgpu::RenderPipeline,
    bind_group:    wgpu::BindGroup,
    camera_buf:    wgpu::Buffer,
    occ_buf:       wgpu::Buffer,
    depth_view:    wgpu::TextureView,
    _gpu_consts:   GpuConsts,
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn create_depth_view(ctx: &RendererContext, width: u32, height: u32) -> wgpu::TextureView {
    let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
        label:           Some("subchunk_test_depth"),
        size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          DEPTH_FORMAT,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats:    &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

impl SubchunkTest {
    /// Create the test pass for the given surface format and initial window size.
    ///
    /// Loads shaders, creates the bind group layout, pipelines, the two
    /// constant buffers (camera + occupancy), and the depth texture. `occ` is
    /// uploaded into the occupancy buffer via `create_buffer_init`, which
    /// writes the bytes as part of buffer creation and avoids any
    /// `write_buffer` ordering hazards.
    ///
    /// `width` and `height` are the initial dimensions of the depth texture;
    /// they must match the swapchain size at creation time and be updated via
    /// [`Self::resize`] on window resize events.
    pub fn new(
        ctx:    &RendererContext,
        occ:    &SubchunkOccupancy,
        width:  u32,
        height: u32,
    ) -> Self {
        let surface_format = ctx
            .surface_format()
            .expect("SubchunkTest requires a windowed RendererContext");

        let device = ctx.device();

        // Load shaders (reflection runs inside ShaderModule::load).
        let vs = ShaderModule::load(
            ctx, "subchunk.vs", ShaderSource::Spirv(SUBCHUNK_VS_SPV), "main",
        )
        .expect("subchunk vertex shader failed to load");

        let ps = ShaderModule::load(
            ctx, "subchunk.ps", ShaderSource::Spirv(SUBCHUNK_PS_SPV), "main",
        )
        .expect("subchunk pixel shader failed to load");

        // Bind group layout. Slot 0 is GpuConsts (unused by this shader but
        // present so wgpu-hal's sequential binding compaction aligns slots 1
        // and 2 with the SPIR-V's binding 1 and 2). See module doc.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("subchunk_test_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   std::num::NonZeroU64::new(
                            std::mem::size_of::<GpuConstsData>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   std::num::NonZeroU64::new(
                            std::mem::size_of::<TestCamera>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   std::num::NonZeroU64::new(
                            std::mem::size_of::<SubchunkOccupancy>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        // Pipeline layout.
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("subchunk_test_pipeline_layout"),
                bind_group_layouts:   &[Some(&bgl)],
                immediate_size:       0,
            });

        // Render pipeline: sub-chunk cube, CCW front faces with back-face
        // culling (only the three camera-facing faces rasterize when the
        // camera is outside), depth-tested with write enabled.
        let pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label:  Some("subchunk_test_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module:              &vs.inner,
                    entry_point:         Some(&vs.entry_point),
                    buffers:             &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology:   wgpu::PrimitiveTopology::TriangleList,
                    // wgpu's Vulkan backend flips Y via a negative viewport
                    // height, which reverses framebuffer winding relative to
                    // world-space winding; our cube table is outward-CCW in
                    // world space, so Cw is the front-face rule.
                    //
                    // Back-face culling is on. The vertex shader swaps
                    // triangle winding for cubes the camera is inside of
                    // (or near-plane-close to), which inverts which side
                    // the rasterizer treats as "front" — so inside cubes
                    // still get fragments via their back faces, while the
                    // common outside case still pays rasterization cost
                    // for only 3 of the 6 cube faces.
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode:  Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format:              DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare:       Some(wgpu::CompareFunction::Less),
                    stencil:             wgpu::StencilState::default(),
                    bias:                wgpu::DepthBiasState::default(),
                }),
                multisample:   wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module:      &ps.inner,
                    entry_point: Some(&ps.entry_point),
                    targets:     &[Some(wgpu::ColorTargetState {
                        format:     surface_format,
                        blend:      None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache:          None,
            });

        // Buffers: both UNIFORM | COPY_DST so we can write via queue.write_buffer.
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

        // Placeholder GpuConsts. Bound at slot 0 to keep wgpu's compacted
        // binding numbers aligned with the SPIR-V's expected bindings.
        let gpu_consts = GpuConsts::new(ctx, GpuConstsData::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("subchunk_test_bind_group"),
            layout:  &bgl,
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

        let depth_view = create_depth_view(ctx, width.max(1), height.max(1));

        Self {
            pipeline,
            bind_group,
            camera_buf,
            occ_buf,
            depth_view,
            _gpu_consts: gpu_consts,
        }
    }

    /// Upload occupancy data to the GPU. Call once after construction (or
    /// whenever the voxel data changes).
    pub fn set_occupancy(&self, ctx: &RendererContext, occ: &SubchunkOccupancy) {
        ctx.queue().write_buffer(&self.occ_buf, 0, bytemuck::bytes_of(occ));
    }

    /// Recreate the depth texture at the new swapchain size. Call from the
    /// window event loop alongside `RendererContext::configure_surface`.
    pub fn resize(&mut self, ctx: &RendererContext, width: u32, height: u32) {
        self.depth_view = create_depth_view(ctx, width.max(1), height.max(1));
    }

    /// Record the sub-chunk test render pass into `fe`.
    ///
    /// Clears the surface to the background colour, clears depth to 1.0,
    /// and rasterizes the sub-chunk's bounding cube. The fragment shader
    /// runs a planar-bitmask DDA starting at each fragment's world-space
    /// hull position.
    pub fn draw(
        &self,
        ctx:    &RendererContext,
        fe:     &mut FrameEncoder,
        frame:  &SurfaceFrame,
        camera: &TestCamera,
    ) {
        // Upload camera data for this frame.
        ctx.queue().write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(camera));

        let mut pass = fe.encoder_mut().begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("subchunk_test_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           frame.view(),
                depth_slice:    None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.05, g: 0.05, b: 0.08, a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view:        &self.depth_view,
                depth_ops:   Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set:      None,
            timestamp_writes:         None,
            multiview_mask:           None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..36, 0..1);
    }
}
