//! Sub-chunk bitmask DDA prototype — 64-instance cull + MDI draw pass.
//!
//! Rasterizes up to [`MAX_CANDIDATES`] sub-chunk bounding cubes via
//! GPU frustum + directional-exposure cull → multi-draw-indirect. Each
//! sub-chunk has:
//! - A world-space [`SubchunkInstance`] (origin + occupancy slot +
//!   6-bit directional exposure mask).
//! - A [`SubchunkOccupancy`] (8×8×8 bitmask, 64 bytes).
//!
//! # Bind group layouts
//!
//! Layouts are derived automatically from the shaders via SPIR-V reflection
//! at pipeline construction time.  The entries documented here match the
//! HLSL declarations but are no longer hand-written in Rust:
//!
//! **Cull bind group** (set 0 for `subchunk_cull.cs.hlsl`):
//! - 0: camera     (UniformBuffer, 64)          — COMPUTE
//! - 1: instances  (StorageReadOnly, stride=16) — COMPUTE
//! - 2: visible    (StorageRW, stride=4)        — COMPUTE
//!
//! **Render bind group** (set 0 for `subchunk.vs/.ps.hlsl`):
//! - 0: camera     (UniformBuffer, 64)          — VERTEX | FRAGMENT
//! - 1: instances  (StorageReadOnly, stride=16) — VERTEX
//! - 2: visible    (StorageReadOnly, stride=4)  — VERTEX
//! - 3: occ_array  (StorageReadOnly, stride=64) — FRAGMENT
//!
//! Cull's indirect output lives on its own set-1 bind group, built inside
//! the cull node; callers do not supply it.
//!
//! # wgpu-hal Vulkan binding compaction
//!
//! wgpu-hal sorts BGL entries by binding number and assigns sequential VK
//! bindings 0, 1, 2, … The SPIR-V passthrough path does not remap, so the
//! HLSL `[[vk::binding(N, 0)]]` number MUST equal N exactly.  Neither shader
//! in this pass uses `GpuConsts`, so slot 0 is bound to camera — bindings
//! stay contiguous from 0 as required.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::device::RendererContext;
use crate::graph::{BufferDesc, RenderGraph, TextureHandle};
use crate::nodes::{CullArgs, DrawArgs, IndirectArgs, cull, mdi_draw};
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

/// Per-sub-chunk instance data: world-space origin plus a packed slot +
/// exposure-mask word.
///
/// Layout must match the HLSL `Instance` struct (16 bytes):
/// ```text
///   int3 origin    (+0)
///   uint slot_mask (+12)
/// ```
///
/// `slot_mask` packs two fields into one `u32` so the CPU/GPU struct stays
/// 16-aligned (no trailing vec3-alignment padding):
/// - Low 26 bits: occupancy slot index (space for up to `1 << 26`
///   candidates; current cap is [`MAX_CANDIDATES`]).
/// - High 6 bits: directional exposure mask. One bit per face direction,
///   set when any voxel in the sub-chunk has an exposed face in that
///   direction:
///
/// | bit | direction |
/// |:---:|:---------:|
/// | 0   | -X        |
/// | 1   | +X        |
/// | 2   | -Y        |
/// | 3   | +Y        |
/// | 4   | -Z        |
/// | 5   | +Z        |
///
/// The cull shader uses the mask to reject sub-chunks whose occupied
/// voxels have no exposed faces in any camera-visible direction — a
/// DDA-specific rejection with no analog in triangle rendering.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SubchunkInstance {
    /// Voxel-space origin of the `[origin, origin+8)` sub-chunk box.
    pub origin:    [i32; 3],
    /// Low 26 bits: occupancy slot. High 6 bits: exposure mask.
    /// Prefer [`SubchunkInstance::new`] over constructing directly.
    pub slot_mask: u32,
}

const _: () = assert!(
    std::mem::size_of::<SubchunkInstance>() == 16,
    "SubchunkInstance must be 16 bytes to match HLSL Instance"
);

const SLOT_BITS:    u32 = 26;
const SLOT_MASK:    u32 = (1 << SLOT_BITS) - 1;
const EXPOSURE_MAX: u8  = (1 << 6) - 1;

impl SubchunkInstance {
    /// Build a `SubchunkInstance` from its three logical components.
    ///
    /// # Panics
    ///
    /// Debug builds panic if `occ_slot >= 2^26` or `exposure_mask > 0x3F`
    /// (the packed encoding cannot represent either overflow).
    pub fn new(origin: [i32; 3], occ_slot: u32, exposure_mask: u8) -> Self {
        debug_assert!(
            occ_slot <= SLOT_MASK,
            "occ_slot must fit in {SLOT_BITS} bits",
        );
        debug_assert!(
            exposure_mask <= EXPOSURE_MAX,
            "exposure_mask must fit in 6 bits",
        );
        Self {
            origin,
            slot_mask: (occ_slot & SLOT_MASK)
                     | ((exposure_mask as u32) << SLOT_BITS),
        }
    }

    /// Occupancy slot index (low 26 bits of `slot_mask`).
    pub fn occ_slot(&self) -> u32 {
        self.slot_mask & SLOT_MASK
    }

    /// 6-bit directional exposure mask (high 6 bits of `slot_mask`).
    pub fn exposure_mask(&self) -> u8 {
        (self.slot_mask >> SLOT_BITS) as u8
    }
}

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

/// Compute the 6-bit directional exposure mask for a sub-chunk's occupancy.
///
/// Treats the sub-chunk as isolated — voxels outside `[0, 8)^3` are empty —
/// so a voxel face on the outer boundary counts as exposed. Real voxel
/// worlds would resolve exposure across sub-chunk boundaries; that's a
/// plumbing concern for the residency control plane when it arrives.
///
/// Bit layout matches [`SubchunkInstance`]: 0/1 = -X/+X, 2/3 = -Y/+Y,
/// 4/5 = -Z/+Z. A fully enclosed shape (like a sphere that doesn't touch
/// the boundary) still lights up every bit, because each direction has at
/// least one occupied voxel whose adjacent voxel in that direction is empty.
pub fn occupancy_exposure(occ: &SubchunkOccupancy) -> u8 {
    let is_set = |x: i32, y: i32, z: i32| -> bool {
        if !(0..8).contains(&x) || !(0..8).contains(&y) || !(0..8).contains(&z) {
            return false;
        }
        let bit  = (y as u32) * 8 + (x as u32);
        let word = (z as u32) * 2 + (bit >> 5);
        (occ.planes[word as usize] >> (bit & 31)) & 1 == 1
    };

    let mut mask = 0u8;
    for z in 0i32..8 {
        for y in 0i32..8 {
            for x in 0i32..8 {
                if !is_set(x, y, z) {
                    continue;
                }
                if !is_set(x - 1, y, z) { mask |= 1 << 0; }
                if !is_set(x + 1, y, z) { mask |= 1 << 1; }
                if !is_set(x, y - 1, z) { mask |= 1 << 2; }
                if !is_set(x, y + 1, z) { mask |= 1 << 3; }
                if !is_set(x, y, z - 1) { mask |= 1 << 4; }
                if !is_set(x, y, z + 1) { mask |= 1 << 5; }
                if mask == EXPOSURE_MAX {
                    return mask;
                }
            }
        }
    }
    mask
}

// --- SubchunkTest ---

/// Persistent state for the sub-chunk cull + MDI draw pass.
///
/// Owns both pipelines (which carry their reflected bind group layouts) and
/// all four persistent buffers.  Camera data is updated per-frame via
/// [`Self::write_camera`] before the render graph is built.  Instance and
/// occupancy data are committed at construction and stay fixed for the
/// lifetime of this object.
///
/// The `visible` and `indirect` buffers are graph transients — allocated from
/// the pool each frame and discarded after execute.  `count_buf` is a fixed
/// persistent constant (holds `[1u32]`) that is never written after
/// construction; it provides the GPU-side draw count cap.  Both bind groups
/// are registered per frame via `graph.create_bind_group` inside
/// [`subchunk_test`].
pub struct SubchunkTest {
    // --- Pipelines ---
    cull_pipeline   : Arc<ComputePipeline>,
    render_pipeline : Arc<RenderPipeline>,

    // --- Persistent buffers (imported into the graph each frame) ---
    camera_buf   : wgpu::Buffer,
    instance_buf : wgpu::Buffer,
    occ_buf      : wgpu::Buffer,
    count_buf    : wgpu::Buffer,
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

        let camera_size = std::mem::size_of::<TestCamera>() as u64;
        let count_size  = 4u64; // one u32

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

        let count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_count"),
            size:               count_size,
            usage:              wgpu::BufferUsages::STORAGE
                              | wgpu::BufferUsages::INDIRECT
                              | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write the fixed draw count of 1.  After this the buffer is never
        // written again — on the CPU or the GPU — for the lifetime of this
        // SubchunkTest.
        ctx.queue().write_buffer(&count_buf, 0, bytemuck::bytes_of(&1u32));

        // --- Pipelines ---

        let cull_pipeline = Arc::new(ComputePipeline::new(ctx, ComputePipelineDescriptor {
            label:                  "subchunk_cull",
            shader:                 cs,
            expected_workgroup_size: Some([64, 1, 1]),
            immediate_size:         0,
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

        Self {
            cull_pipeline,
            render_pipeline,
            camera_buf,
            instance_buf,
            occ_buf,
            count_buf,
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

    pub(crate) fn render_pipeline(&self) -> &Arc<RenderPipeline> {
        &self.render_pipeline
    }

    pub(crate) fn camera_buf(&self) -> &wgpu::Buffer {
        &self.camera_buf
    }

    pub(crate) fn instance_buf(&self) -> &wgpu::Buffer {
        &self.instance_buf
    }

    pub(crate) fn occ_buf(&self) -> &wgpu::Buffer {
        &self.occ_buf
    }

    /// The persistent draw-count buffer.  Holds `[1u32]` and is never written
    /// after construction.  Imported as a graph resource each frame to provide
    /// the GPU-side draw count cap.
    pub(crate) fn count_buf(&self) -> &wgpu::Buffer {
        &self.count_buf
    }
}

/// Depth format used by [`subchunk_test`] and by the pipeline's `DepthStencilState`.
/// The game loop allocates a transient depth texture in this format each frame.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Register the sub-chunk cull + MDI draw passes into `graph`.
///
/// Imports the four persistent buffers as graph resources, allocates `visible`
/// and `indirect` as per-frame transients, registers dynamic bind groups for
/// both passes, dispatches the GPU frustum cull (1 workgroup × 64 threads),
/// then issues a multi-draw-indirect-count raster pass that draws only the
/// visible sub-chunks.
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
    // --- Import persistent buffers ---
    let camera_h   = graph.import_buffer(test.camera_buf().clone());
    let instance_h = graph.import_buffer(test.instance_buf().clone());
    let occ_h      = graph.import_buffer(test.occ_buf().clone());
    let count_h    = graph.import_buffer(test.count_buf().clone());

    // --- Allocate cull outputs as transients ---
    let visible_size  = (4 * MAX_CANDIDATES) as u64;
    let indirect_size = 16u64;

    let visible_h = graph.create_buffer("subchunk_visible", BufferDesc {
        size  : visible_size,
        usage : wgpu::BufferUsages::STORAGE,
    });

    let indirect_h = graph.create_buffer("subchunk_indirect", BufferDesc {
        size  : indirect_size,
        usage : wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
    });

    // --- Register dynamic bind groups ---
    let cull_bg = graph.create_bind_group(
        "subchunk_cull_bg",
        test.cull_pipeline().as_ref(),
        None,
        &[
            (0, camera_h.into()),
            (1, instance_h.into()),
            (2, visible_h.into()),
        ],
    );

    let render_bg = graph.create_bind_group(
        "subchunk_render_bg",
        test.render_pipeline().as_ref(),
        None,
        &[
            (0, camera_h.into()),
            (1, instance_h.into()),
            (2, visible_h.into()),
            (3, occ_h.into()),
        ],
    );

    // --- Cull → mdi_draw chain ---
    let indirect_in = IndirectArgs { indirect: indirect_h, count: count_h, max_draws: 1 };

    let indirect_out = cull(
        graph,
        test.cull_pipeline(),
        cull_bg,
        &CullArgs { workgroups: [1, 1, 1] },
        indirect_in,
    );

    mdi_draw(
        graph,
        test.render_pipeline(),
        render_bg,
        &indirect_out,
        &DrawArgs::default(),
        color,
        depth,
    )
}
