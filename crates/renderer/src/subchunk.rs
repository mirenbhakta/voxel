//! Sub-chunk render pipeline and GPU data formats.
//!
//! Owns the two pipelines that implement the V2 sub-chunk DDA primitive
//! (`subchunk_cull.cs` + `subchunk.vs`/`subchunk.ps`) plus the dynamic
//! GPU buffers they read from. Callers (e.g. the game crate) populate the
//! buffers per frame from a residency-driven CPU shadow:
//!
//! - Camera uniform — one upload per frame.
//! - Instance array — a full overwrite per frame (one entry per active
//!   candidate sub-chunk; padding entries have `exposure_mask = 0` so the
//!   cull shader rejects them trivially).
//! - Occupancy array — a single slot is written on completion of a prep
//!   request; never a full-array upload.
//!
//! The count buffer stays at `[1u32]` permanently — the cull shader emits
//! exactly one indirect draw entry whose `instance_count` fans out to the
//! number of passing candidates.
//!
//! # Bind group layouts (reflected from SPIR-V)
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
//! Cull's indirect output rides on its own set-1 bind group, assembled
//! inside `nodes::cull`.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

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

/// Maximum number of candidate sub-chunks the cull pass handles in one
/// dispatch. Baked into the cull shader's workgroup size (`[64, 1, 1]`)
/// and into the `instance_buf` / `occ_buf` allocations.
pub const MAX_CANDIDATES: usize = 64;

/// Depth format used by the sub-chunk pipeline's `DepthStencilState`.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

// --- SubchunkCamera ---

/// Camera parameters for the sub-chunk pipeline.
///
/// Layout matches the HLSL `Camera` struct (64 bytes):
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
pub struct SubchunkCamera {
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
    std::mem::size_of::<SubchunkCamera>() == 64,
    "SubchunkCamera must be 64 bytes to match HLSL Camera"
);

// --- SubchunkInstance ---

/// Per-sub-chunk instance data: world-space origin plus a packed slot +
/// exposure-mask word.
///
/// Layout matches the HLSL `Instance` struct (16 bytes):
/// ```text
///   int3 origin    (+0)
///   uint slot_mask (+12)
/// ```
///
/// `slot_mask` packs two fields into one `u32` so the struct stays
/// 16-aligned on both sides:
/// - Low 26 bits: occupancy slot index.
/// - High 6 bits: directional exposure mask, one bit per face:
///
/// | bit | direction |
/// |:---:|:---------:|
/// | 0   | -X        |
/// | 1   | +X        |
/// | 2   | -Y        |
/// | 3   | +Y        |
/// | 4   | -Z        |
/// | 5   | +Z        |
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SubchunkInstance {
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
    /// Build an instance from its three logical components.
    ///
    /// # Panics
    ///
    /// Debug builds panic if `occ_slot >= 2^26` or `exposure_mask > 0x3F`.
    pub fn new(origin: [i32; 3], occ_slot: u32, exposure_mask: u8) -> Self {
        debug_assert!(occ_slot      <= SLOT_MASK,    "occ_slot must fit in {SLOT_BITS} bits");
        debug_assert!(exposure_mask <= EXPOSURE_MAX, "exposure_mask must fit in 6 bits");
        Self {
            origin,
            slot_mask: (occ_slot & SLOT_MASK)
                     | ((exposure_mask as u32) << SLOT_BITS),
        }
    }

    pub fn occ_slot(&self) -> u32 {
        self.slot_mask & SLOT_MASK
    }

    pub fn exposure_mask(&self) -> u8 {
        (self.slot_mask >> SLOT_BITS) as u8
    }
}

// --- SubchunkOccupancy ---

/// GPU-format 8×8×8 occupancy, laid out as 16 × u32.
///
/// `planes[z * 2 .. z * 2 + 2]` encodes Z-layer `z`. In the 64 bits of
/// that layer, bit `y * 8 + x` is set when voxel `(x, y, z)` is occupied.
/// The HLSL storage buffer views the 16 u32s as `uint4 plane[4]`.
///
/// The CPU crate (`game::world::subchunk::SubchunkOccupancy`) uses a
/// `[u64; 8]` layout with the same bit semantics and produces
/// byte-compatible output via its `to_gpu_bytes` method.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SubchunkOccupancy {
    pub planes: [u32; 16],
}

/// Debug helper: centered sphere of radius 3.5.
pub fn sphere_occupancy() -> SubchunkOccupancy {
    let mut occ = SubchunkOccupancy { planes: [0u32; 16] };
    for z in 0u32..8 {
        for y in 0u32..8 {
            for x in 0u32..8 {
                let fx = x as f32 - 3.5;
                let fy = y as f32 - 3.5;
                let fz = z as f32 - 3.5;
                if fx * fx + fy * fy + fz * fz <= 3.5 * 3.5 {
                    let bit  = y * 8 + x;
                    let word = z * 2 + (bit >> 5);
                    occ.planes[word as usize] |= 1 << (bit & 31);
                }
            }
        }
    }
    occ
}

/// 6-bit directional exposure mask for `occ`, treating the sub-chunk as
/// isolated (voxels outside `[0, 8)^3` are empty).
///
/// Useful as a reference implementation; real usage computes the mask
/// from the CPU-side occupancy.
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

// --- WorldRenderer ---

/// Pipelines + dynamic GPU buffers for the sub-chunk render path.
///
/// Created once at startup. The caller populates
/// [`WorldRenderer::write_camera`] per frame, overwrites the full instance
/// array with [`WorldRenderer::write_instances`], and uploads individual
/// occupancy slots with [`WorldRenderer::write_occupancy_slot`] as the
/// residency control plane delivers prep completions.
pub struct WorldRenderer {
    cull_pipeline:   Arc<ComputePipeline>,
    render_pipeline: Arc<RenderPipeline>,

    camera_buf:   wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    occ_buf:      wgpu::Buffer,
    count_buf:    wgpu::Buffer,
}

impl WorldRenderer {
    /// Create the pipelines and allocate dynamic buffers sized for
    /// [`MAX_CANDIDATES`] sub-chunks. Buffers are zero-initialized; the
    /// caller must populate instances/occupancy before the first render
    /// (padding instances have `slot_mask = 0` → exposure_mask 0, which
    /// the cull shader rejects trivially, so a render before the first
    /// upload produces a blank frame rather than undefined behavior).
    pub fn new(ctx: &RendererContext) -> Self {
        let surface_format = ctx
            .surface_format()
            .expect("WorldRenderer requires a windowed RendererContext");

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

        let camera_size   = std::mem::size_of::<SubchunkCamera>() as u64;
        let instance_size = (std::mem::size_of::<SubchunkInstance>() * MAX_CANDIDATES) as u64;
        let occ_size      = (std::mem::size_of::<SubchunkOccupancy>() * MAX_CANDIDATES) as u64;
        let count_size    = 4u64;

        // --- Persistent buffers ---

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_camera"),
            size:               camera_size,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_instances"),
            size:               instance_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let occ_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_occ"),
            size:               occ_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

        // Draw-count cap is always 1 — the cull shader emits one indirect
        // entry whose `instance_count` fans out to the visible candidates.
        ctx.queue().write_buffer(&count_buf, 0, bytemuck::bytes_of(&1u32));

        // --- Pipelines ---

        let cull_pipeline = Arc::new(ComputePipeline::new(ctx, ComputePipelineDescriptor {
            label:                   "subchunk_cull",
            shader:                  cs,
            expected_workgroup_size: Some([64, 1, 1]),
            immediate_size:          0,
        }));

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

    /// Overwrite the camera uniform. Call once per frame before building
    /// the render graph.
    pub fn write_camera(&self, ctx: &RendererContext, camera: &SubchunkCamera) {
        ctx.queue().write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(camera));
    }

    /// Overwrite the full instance array.
    ///
    /// `instances` must have at most [`MAX_CANDIDATES`] entries. Remaining
    /// slots are not touched by this call — callers must keep them zeroed
    /// (or otherwise ensure `exposure_mask = 0`) by passing an exactly
    /// `MAX_CANDIDATES`-sized buffer whose tail entries are sentinels.
    pub fn write_instances(&self, ctx: &RendererContext, instances: &[SubchunkInstance]) {
        assert!(
            instances.len() <= MAX_CANDIDATES,
            "write_instances: got {} instances, max is {MAX_CANDIDATES}",
            instances.len(),
        );
        ctx.queue().write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(instances));
    }

    /// Upload a single slot's 64-byte occupancy payload.
    ///
    /// `occ_bytes` must be exactly 64 bytes in the GPU layout — an
    /// 8-word sequence of little-endian u64s covering z-layers 0..8, where
    /// bit `y*8 + x` of layer `z` is set when voxel `(x, y, z)` is occupied.
    pub fn write_occupancy_slot(
        &self,
        ctx:       &RendererContext,
        slot:      u32,
        occ_bytes: &[u8; std::mem::size_of::<SubchunkOccupancy>()],
    ) {
        let offset = (slot as u64) * std::mem::size_of::<SubchunkOccupancy>() as u64;
        ctx.queue().write_buffer(&self.occ_buf, offset, occ_bytes);
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

    pub(crate) fn count_buf(&self) -> &wgpu::Buffer {
        &self.count_buf
    }
}

// --- subchunk_world render node ---

/// Register the sub-chunk cull + MDI draw passes into `graph`.
///
/// Imports the renderer's four persistent buffers, allocates `visible`
/// and `indirect` as per-frame transients, wires both dynamic bind groups,
/// dispatches the 64-thread cull (1 workgroup), and issues a
/// multi-draw-indirect-count raster that draws the surviving sub-chunks.
///
/// Returns the versioned output handles for the color and depth textures.
pub fn subchunk_world(
    graph    : &mut RenderGraph,
    renderer : &Arc<WorldRenderer>,
    color    : TextureHandle,
    depth    : TextureHandle,
)
    -> (TextureHandle, TextureHandle)
{
    let camera_h   = graph.import_buffer(renderer.camera_buf().clone());
    let instance_h = graph.import_buffer(renderer.instance_buf().clone());
    let occ_h      = graph.import_buffer(renderer.occ_buf().clone());
    let count_h    = graph.import_buffer(renderer.count_buf().clone());

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

    let cull_bg = graph.create_bind_group(
        "subchunk_cull_bg",
        renderer.cull_pipeline().as_ref(),
        None,
        &[
            (0, camera_h.into()),
            (1, instance_h.into()),
            (2, visible_h.into()),
        ],
    );

    let render_bg = graph.create_bind_group(
        "subchunk_render_bg",
        renderer.render_pipeline().as_ref(),
        None,
        &[
            (0, camera_h.into()),
            (1, instance_h.into()),
            (2, visible_h.into()),
            (3, occ_h.into()),
        ],
    );

    let indirect_in = IndirectArgs { indirect: indirect_h, count: count_h, max_draws: 1 };

    let indirect_out = cull(
        graph,
        renderer.cull_pipeline(),
        cull_bg,
        &CullArgs { workgroups: [1, 1, 1] },
        indirect_in,
    );

    mdi_draw(
        graph,
        renderer.render_pipeline(),
        render_bg,
        &indirect_out,
        &DrawArgs::default(),
        color,
        depth,
    )
}
