//! Sub-chunk render pipeline and GPU data formats.
//!
//! Owns the two pipelines that implement the V2 sub-chunk DDA primitive
//! (`subchunk_cull.cs` + `subchunk.vs`/`subchunk.ps`) plus the dynamic
//! GPU buffers they read from. Callers (e.g. the game crate) populate the
//! buffers per frame from a residency-driven CPU shadow:
//!
//! - Camera uniform — one upload per frame.
//! - Instance array — a full overwrite per frame (one entry per active
//!   candidate sub-chunk; padding entries carry
//!   [`SubchunkInstance::PADDING_BIT`] so the cull shader rejects them
//!   before touching any per-slot buffer).
//! - Occupancy array — a single slot is written on completion of a prep
//!   request; never a full-array upload.
//! - Live exposure array — one `u32` per slot, written only by the patch
//!   pass on dirty slots; the cull shader reads it by slot index.
//!
//! The count buffer stays at `[1u32]` permanently — the cull shader emits
//! exactly one indirect draw entry whose `instance_count` fans out to the
//! number of passing candidates.
//!
//! # Bind group layouts (reflected from SPIR-V)
//!
//! **Cull bind group** (set 0 for `subchunk_cull.cs.hlsl`):
//! - 0: camera        (UniformBuffer, 64)          — COMPUTE
//! - 1: instances     (StorageReadOnly, stride=16) — COMPUTE
//! - 2: visible       (StorageRW, stride=4)        — COMPUTE
//! - 3: lod_mask      (UniformBuffer, 512)         — COMPUTE
//! - 4: live_exposure (StorageReadOnly, stride=4)  — COMPUTE
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
use crate::graph::{BufferDesc, BufferHandle, RenderGraph, TextureHandle};
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

const SUBCHUNK_PREP_CS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/subchunk_prep.cs.spv"));

// --- Constants ---

/// Maximum number of candidate sub-chunks the cull pass handles in one
/// dispatch. Baked into the cull shader's workgroup size (`[256, 1, 1]`)
/// and into the `instance_buf` / `occ_buf` allocations.
///
/// Sized to accommodate several LOD levels' shells simultaneously; each
/// level of a 3³ clipmap contributes 27 slots, so 256 holds ~9 levels
/// worth of tight-radius residency.
pub const MAX_CANDIDATES: usize = 256;

/// Maximum LOD levels supported by the pipeline. Matches the 4-bit level
/// field packed into [`SubchunkInstance::slot_mask`].
pub const MAX_LEVELS: usize = 16;

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
/// level word, with a high-bit sentinel flag for padding entries.
///
/// Layout matches the HLSL `Instance` struct (16 bytes):
/// ```text
///   int3 origin    (+0)
///   uint slot_mask (+12)
/// ```
///
/// `slot_mask` packs the per-instance fields into one `u32` so the struct
/// stays 16-aligned on both sides:
/// - bits 0-21 (22 bits): occupancy slot index — up to 4 M slots.
/// - bits 22-25 (4 bits): LOD level — voxel edge = `1 << level` metres.
/// - bits 26-30 (5 bits): reserved, must be zero.
/// - bit  31    (1 bit):  [`SubchunkInstance::PADDING_BIT`] — set on tail
///   padding entries so the cull shader can reject them before touching
///   any per-slot buffer.
///
/// Directional exposure (previously packed into bits 26-31) now lives in
/// the renderer's `live_exposure_buf`, indexed by the slot field; the cull
/// shader fetches it per-candidate rather than unpacking it from the
/// instance record.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SubchunkInstance {
    pub origin:    [i32; 3],
    /// Packed (slot, level) with a high-bit padding flag. Prefer
    /// [`SubchunkInstance::new`] / [`SubchunkInstance::padding`] over
    /// constructing directly.
    pub slot_mask: u32,
}

const _: () = assert!(
    std::mem::size_of::<SubchunkInstance>() == 16,
    "SubchunkInstance must be 16 bytes to match HLSL Instance"
);

const SLOT_BITS:   u32 = 22;
const SLOT_MASK:   u32 = (1 << SLOT_BITS) - 1;
const LEVEL_BITS:  u32 = 4;
const LEVEL_SHIFT: u32 = SLOT_BITS;
const LEVEL_MAX:   u8  = (1 << LEVEL_BITS) - 1;

impl SubchunkInstance {
    /// Sentinel bit set on padding entries in the instance array. The cull
    /// shader tests this bit first and drops the candidate without reading
    /// any other per-slot buffer — so padding never depends on the
    /// contents of `live_exposure[0]` or any other slot-indexed data.
    pub const PADDING_BIT: u32 = 1 << 31;

    /// Build a real (non-padding) instance from its three logical
    /// components.
    ///
    /// # Panics
    ///
    /// Debug builds panic if `occ_slot >= 2^22` or `level > 15` — the
    /// packed encoding cannot represent either overflow.
    pub fn new(origin: [i32; 3], occ_slot: u32, level: u8) -> Self {
        debug_assert!(occ_slot <= SLOT_MASK, "occ_slot must fit in {SLOT_BITS} bits");
        debug_assert!(level    <= LEVEL_MAX, "level must fit in {LEVEL_BITS} bits");
        Self {
            origin,
            slot_mask: (occ_slot & SLOT_MASK)
                     | ((level as u32) << LEVEL_SHIFT),
        }
    }

    /// Tail-padding instance. Carries [`SubchunkInstance::PADDING_BIT`] so
    /// the cull shader rejects it before reading `live_exposure` or any
    /// other slot-indexed buffer.
    pub fn padding() -> Self {
        Self {
            origin:    [0, 0, 0],
            slot_mask: Self::PADDING_BIT,
        }
    }

    pub fn occ_slot(&self) -> u32 {
        self.slot_mask & SLOT_MASK
    }

    pub fn level(&self) -> u8 {
        ((self.slot_mask >> LEVEL_SHIFT) & ((1 << LEVEL_BITS) - 1)) as u8
    }

    pub fn is_padding(&self) -> bool {
        (self.slot_mask & Self::PADDING_BIT) != 0
    }
}

// --- LodMaskUniform ---

/// Per-level LOD-cascade mask data.
///
/// For each level `N`, `mask_lo[N]` / `mask_hi[N]` describe the world-space
/// AABB that level `N` should defer to (the next-finer configured level's
/// shell). Sub-chunks at level `N` fully inside this box are dropped by
/// the cull pass; fragments at level `N` whose DDA hit lands inside are
/// discarded by the pixel shader — so each world point is rendered by
/// exactly one level.
///
/// - `mask_lo[N].xyz` — shell lower bound (world units).
/// - `mask_hi[N].xyz` — shell upper bound (world units).
/// - `mask_hi[N].w`   — `1.0` when the entry is active, `0.0` when level
///   `N` has no finer level to defer to (level 0, or any unconfigured
///   level).
///
/// Size: `2 * 16 * 16 = 512` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LodMaskUniform {
    pub mask_lo: [[f32; 4]; MAX_LEVELS],
    pub mask_hi: [[f32; 4]; MAX_LEVELS],
}

impl LodMaskUniform {
    /// All entries inactive. Sub-chunks at every level render everywhere.
    pub fn inactive() -> Self {
        Self {
            mask_lo: [[0.0; 4]; MAX_LEVELS],
            mask_hi: [[0.0; 4]; MAX_LEVELS],
        }
    }
}

const _: () = assert!(
    std::mem::size_of::<LodMaskUniform>() == 512,
    "LodMaskUniform layout must match HLSL LodMask (512 bytes)"
);

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

// --- PrepRequest ---

/// One entry in the GPU prep input buffer.
///
/// Layout matches the HLSL `PrepRequest` struct (32 bytes):
/// ```text
///   int3 coord (+0)
///   uint level (+12)
///   uint slot  (+16)
///   uint _pad0 (+20)
///   uint _pad1 (+24)
///   uint _pad2 (+28)
/// ```
///
/// - `coord` — sub-chunk coord at this request's LOD.
/// - `level` — LOD level; voxel edge = `1 << level` metres.
/// - `slot`  — target live occupancy slot, and the matching staging slot
///   (staging and live share a 1:1 mapping).
///
/// The trailing padding rounds the struct to a 32-byte stride so the
/// HLSL `StructuredBuffer<PrepRequest>` layout matches DX-layout naturally.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PrepRequest {
    pub coord: [i32; 3],
    pub level: u32,
    pub slot:  u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(
    std::mem::size_of::<PrepRequest>() == 32,
    "PrepRequest must be 32 bytes to match HLSL PrepRequest",
);

// --- DirtyEntry ---

/// One entry in the GPU prep dirty list.
///
/// Layout matches the HLSL shader's byte-address store (8 bytes):
/// ```text
///   uint slot           (+0)
///   uint staging_offset (+4)
/// ```
///
/// `staging_offset` equals `slot` today (staging and live are 1:1); the
/// field exists so future batched-compaction work stays additive.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DirtyEntry {
    pub slot:           u32,
    pub staging_offset: u32,
}

const _: () = assert!(
    std::mem::size_of::<DirtyEntry>() == 8,
    "DirtyEntry must be 8 bytes to match the HLSL dirty-list entry",
);

// --- DirtyReport ---

/// GPU→CPU readback payload for one prep dispatch.
///
/// The shader writes the entry count at offset 0 followed by the entries
/// themselves starting at offset 16 (12 bytes of padding keep the entries
/// 8-byte-aligned and match the HLSL layout). Entries `[0..count)` are
/// valid; slots `[count..MAX_CANDIDATES)` hold undefined data and must be
/// ignored by the consumer.
///
/// Size: `16 + 8 * MAX_CANDIDATES = 2064` bytes (for MAX_CANDIDATES = 256).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DirtyReport {
    pub count:   u32,
    pub _pad0:   u32,
    pub _pad1:   u32,
    pub _pad2:   u32,
    pub entries: [DirtyEntry; MAX_CANDIDATES],
}

const _: () = assert!(
    std::mem::size_of::<DirtyReport>() == 16 + 8 * MAX_CANDIDATES,
    "DirtyReport must be 16 + 8 * MAX_CANDIDATES bytes to match the HLSL \
     dirty-list layout",
);

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
    prep_pipeline:   Arc<ComputePipeline>,
    render_pipeline: Arc<RenderPipeline>,

    camera_buf:            wgpu::Buffer,
    instance_buf:          wgpu::Buffer,
    occ_buf:               wgpu::Buffer,
    count_buf:             wgpu::Buffer,
    lod_mask_buf:          wgpu::Buffer,
    staging_occ_buf:       wgpu::Buffer,
    staging_exposure_buf:  wgpu::Buffer,
    live_exposure_buf:     wgpu::Buffer,
    dirty_list_buf:        wgpu::Buffer,
    prep_request_buf:      wgpu::Buffer,
}

impl WorldRenderer {
    /// Create the pipelines and allocate dynamic buffers sized for
    /// [`MAX_CANDIDATES`] sub-chunks. Buffers are zero-initialized; the
    /// caller must populate instances/occupancy before the first render.
    /// A render before the first instance upload produces a blank frame
    /// rather than undefined behavior — the instance buffer is zeroed, so
    /// every entry has `slot_mask = 0` (slot 0, level 0, padding bit
    /// clear). Those entries fetch `live_exposure[0]`, which is zero, so
    /// the cull shader's exposure rejection drops them.
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

        let prep_cs = ShaderModule::load(
            ctx, "subchunk_prep.cs", ShaderSource::Spirv(SUBCHUNK_PREP_CS_SPV), "main",
        )
        .expect("subchunk prep shader failed to load");

        // --- Buffer sizes ---

        let camera_size       = std::mem::size_of::<SubchunkCamera>() as u64;
        let instance_size     = (std::mem::size_of::<SubchunkInstance>() * MAX_CANDIDATES) as u64;
        let occ_size          = (std::mem::size_of::<SubchunkOccupancy>() * MAX_CANDIDATES) as u64;
        let exposure_size     = (std::mem::size_of::<u32>()              * MAX_CANDIDATES) as u64;
        let count_size        = 4u64;
        let lod_mask_size     = std::mem::size_of::<LodMaskUniform>() as u64;
        let dirty_list_size   = std::mem::size_of::<DirtyReport>() as u64;
        let prep_request_size = (std::mem::size_of::<PrepRequest>() * MAX_CANDIDATES) as u64;

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

        let lod_mask_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_lod_mask"),
            size:               lod_mask_size,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Start with every level inactive — every sub-chunk renders — until
        // the caller populates the cascade. Keeps the buffer well-defined
        // before the first `write_lod_mask`.
        ctx.queue()
            .write_buffer(&lod_mask_buf, 0, bytemuck::bytes_of(&LodMaskUniform::inactive()));

        // Staging copy of `occ_buf`, written by the GPU prep pass.
        // COPY_SRC is used by the follow-up commit patch that CPU scheduling
        // will eventually wire up (`update_slot`); it is unused by the prep
        // node itself but stays on the buffer for the downstream slice.
        let staging_occ_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_staging_occ"),
            size:               occ_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging exposure: one `u32` per slot, written unconditionally by
        // the prep pass alongside the staging occupancy payload. The patch
        // pass blits staging → live on every dirty slot, paired with the
        // occupancy patch. Six bits of payload per slot stored as a full
        // `u32` for stride alignment; waste is 26 bits × MAX_CANDIDATES =
        // under 1 KiB, negligible.
        let staging_exposure_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_staging_exposure"),
            size:               exposure_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Live exposure: read by the cull shader per candidate. COPY_DST so
        // the patch pass can blit staging slots in on every dirty
        // completion; zero-initialised here so a first-frame cull (before
        // any prep has completed) reads 0 for every slot — the same
        // "reject everything" behaviour the old `slot_mask = 0` sentinel
        // produced.
        let live_exposure_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_live_exposure"),
            size:               exposure_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        ctx.queue().write_buffer(
            &live_exposure_buf,
            0,
            bytemuck::cast_slice(&[0u32; MAX_CANDIDATES]),
        );

        // Dirty-list buffer: shader-authoritative `DirtyReport`. COPY_SRC so
        // the prep graph pass can blit it into a `ReadbackChannel` slot;
        // COPY_DST so the prep node can clear the atomic `count` header
        // each frame before the new dispatch's `InterlockedAdd` accumulates.
        let dirty_list_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_dirty_list"),
            size:               dirty_list_size,
            usage:              wgpu::BufferUsages::STORAGE
                              | wgpu::BufferUsages::COPY_SRC
                              | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Prep request buffer: CPU writes `request_count` entries per frame
        // via `write_prep_requests`.
        let prep_request_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_prep_requests"),
            size:               prep_request_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Pipelines ---

        let cull_pipeline = Arc::new(ComputePipeline::new(ctx, ComputePipelineDescriptor {
            label:                   "subchunk_cull",
            shader:                  cs,
            expected_workgroup_size: Some([256, 1, 1]),
            immediate_size:          0,
        }));

        let prep_pipeline = Arc::new(ComputePipeline::new(ctx, ComputePipelineDescriptor {
            label:                   "subchunk_prep",
            shader:                  prep_cs,
            expected_workgroup_size: Some([4, 4, 4]),
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
            prep_pipeline,
            render_pipeline,
            camera_buf,
            instance_buf,
            occ_buf,
            count_buf,
            lod_mask_buf,
            staging_occ_buf,
            staging_exposure_buf,
            live_exposure_buf,
            dirty_list_buf,
            prep_request_buf,
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
    /// slots are not touched by this call — callers must populate tail
    /// entries with [`SubchunkInstance::padding`] by passing an exactly
    /// `MAX_CANDIDATES`-sized buffer whose unused entries carry the
    /// padding sentinel.
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

    /// Overwrite the LOD cascade uniform.
    ///
    /// Call once per frame before building the render graph, after
    /// residency has recentered its shells.
    pub fn write_lod_mask(&self, ctx: &RendererContext, mask: &LodMaskUniform) {
        ctx.queue().write_buffer(&self.lod_mask_buf, 0, bytemuck::bytes_of(mask));
    }

    /// Upload `requests` into the prep-request buffer.
    ///
    /// The prep compute pass dispatches one workgroup per request (see
    /// [`subchunk_prep`]); callers must pass the same `requests.len()` as
    /// `request_count` when registering the prep node.
    ///
    /// # Panics
    ///
    /// Panics if `requests.len() > MAX_CANDIDATES` — the buffer is sized to
    /// exactly that many entries.
    pub fn write_prep_requests(&self, ctx: &RendererContext, requests: &[PrepRequest]) {
        assert!(
            requests.len() <= MAX_CANDIDATES,
            "write_prep_requests: got {} requests, max is {MAX_CANDIDATES}",
            requests.len(),
        );
        if requests.is_empty() {
            return;
        }
        ctx.queue().write_buffer(&self.prep_request_buf, 0, bytemuck::cast_slice(requests));
    }

    pub(crate) fn cull_pipeline(&self) -> &Arc<ComputePipeline> {
        &self.cull_pipeline
    }

    pub(crate) fn prep_pipeline(&self) -> &Arc<ComputePipeline> {
        &self.prep_pipeline
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

    pub(crate) fn lod_mask_buf(&self) -> &wgpu::Buffer {
        &self.lod_mask_buf
    }

    pub(crate) fn staging_occ_buf(&self) -> &wgpu::Buffer {
        &self.staging_occ_buf
    }

    pub(crate) fn staging_exposure_buf(&self) -> &wgpu::Buffer {
        &self.staging_exposure_buf
    }

    pub(crate) fn live_exposure_buf(&self) -> &wgpu::Buffer {
        &self.live_exposure_buf
    }

    pub(crate) fn dirty_list_buf(&self) -> &wgpu::Buffer {
        &self.dirty_list_buf
    }

    pub(crate) fn prep_request_buf(&self) -> &wgpu::Buffer {
        &self.prep_request_buf
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
    let camera_h        = graph.import_buffer(renderer.camera_buf().clone());
    let instance_h      = graph.import_buffer(renderer.instance_buf().clone());
    let occ_h           = graph.import_buffer(renderer.occ_buf().clone());
    let count_h         = graph.import_buffer(renderer.count_buf().clone());
    let lod_mask_h      = graph.import_buffer(renderer.lod_mask_buf().clone());
    let live_exposure_h = graph.import_buffer(renderer.live_exposure_buf().clone());

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
            (3, lod_mask_h.into()),
            (4, live_exposure_h.into()),
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

// --- subchunk_prep render node ---

/// Register the sub-chunk prep compute pass + dirty-list readback copy
/// into `graph`.
///
/// Imports the five persistent buffers (prep-requests, live occupancy,
/// staging occupancy, staging exposure, dirty list), clears the
/// dirty-list `count` header, dispatches the prep compute with one
/// workgroup per request, then records a `copy_buffer_to_buffer` into
/// `readback_dst` — an imported handle whose destination is typically a
/// [`ReadbackChannel`](crate::readback::ReadbackChannel) slot reserved for
/// this frame.
///
/// The caller must:
/// - Upload `request_count` entries via
///   [`WorldRenderer::write_prep_requests`] before graph compile.
/// - Size `readback_dst` to at least `size_of::<DirtyReport>()` bytes.
///
/// The live-occupancy buffer is read by the compute pass; callers should
/// be aware that running prep and render in the same graph produces a
/// read-after-write or write-after-read dependency on `occ_buf` and
/// `staging_occ_buf` — the graph handles the barrier automatically.
pub fn subchunk_prep(
    graph         : &mut RenderGraph,
    renderer      : &Arc<WorldRenderer>,
    readback_dst  : BufferHandle,
    request_count : u32,
)
{
    let request_h          = graph.import_buffer(renderer.prep_request_buf().clone());
    let live_occ_h         = graph.import_buffer(renderer.occ_buf().clone());
    let staging_occ_h      = graph.import_buffer(renderer.staging_occ_buf().clone());
    let staging_exposure_h = graph.import_buffer(renderer.staging_exposure_buf().clone());
    let dirty_list_h       = graph.import_buffer(renderer.dirty_list_buf().clone());

    // Clear the dirty-list count header. The shader's `InterlockedAdd` on
    // offset 0 accumulates across workgroups, so the count must start at 0
    // every dispatch. Clearing only the 16-byte header (count + 12 pad
    // bytes) is enough; entry words past the written range are undefined
    // but the consumer ignores them.
    let dirty_list_cleared = graph.add_pass("subchunk_prep_clear", |pass| {
        let cleared = pass.write_buffer(dirty_list_h);
        let dirty_buf = renderer.dirty_list_buf().clone();
        pass.execute(move |ctx| {
            ctx.commands.clear_buffer(&dirty_buf, 0, Some(16));
        });
        cleared
    });

    let prep_bg = graph.create_bind_group(
        "subchunk_prep_bg",
        renderer.prep_pipeline().as_ref(),
        None,
        &[
            (0, request_h.into()),
            (1, live_occ_h.into()),
            (2, staging_occ_h.into()),
            (3, staging_exposure_h.into()),
            (4, dirty_list_cleared.into()),
        ],
    );

    let dirty_list_written = graph.add_pass("subchunk_prep", |pass| {
        let writes     = pass.use_bind_group(prep_bg);
        let pipeline   = Arc::clone(renderer.prep_pipeline());
        let workgroups = [request_count, 1, 1];
        let dirty_out  = writes.write_of(dirty_list_cleared);
        pass.execute(move |ctx| {
            let bg = ctx.resources.bind_group(prep_bg);
            ctx.commands.dispatch(&pipeline, &[bg], workgroups, &[]);
        });
        dirty_out
    });

    let readback_out = graph.add_pass("subchunk_prep_readback_copy", |pass| {
        pass.read_buffer(dirty_list_written);
        let written = pass.write_buffer(readback_dst);
        let copy_size = std::mem::size_of::<DirtyReport>() as u64;
        pass.execute(move |ctx| {
            let src = ctx.resources.buffer(dirty_list_written);
            let dst = ctx.resources.buffer(readback_dst);
            ctx.commands.copy_buffer_to_buffer(src, 0, dst, 0, copy_size);
        });
        written
    });

    graph.mark_output(readback_out);
}

// --- subchunk_patch render node ---

/// Register a per-slot staging→live patch pass into `graph`.
///
/// For each `slot` in `dirty_slots`, records two
/// `copy_buffer_to_buffer` calls: a 64-byte occupancy copy from
/// `staging_occ_buf[slot]` into `occ_buf[slot]`, and a 4-byte exposure
/// copy from `staging_exposure_buf[slot]` into `live_exposure_buf[slot]`.
/// The copy set reflects a [`DirtyReport`](crate::subchunk::DirtyReport)
/// that the control plane drained for a previously-completed prep
/// dispatch.
///
/// A call with an empty `dirty_slots` is a no-op and records nothing —
/// callers do not need to guard at the call site.
///
/// Staging and live are the same 1:1 layout today (one entry per slot in
/// each buffer); the function will need to change if/when staging
/// compaction lands.
pub fn subchunk_patch(
    graph       : &mut RenderGraph,
    renderer    : &Arc<WorldRenderer>,
    dirty_slots : &[u32],
)
{
    if dirty_slots.is_empty() {
        return;
    }

    let staging_occ_h      = graph.import_buffer(renderer.staging_occ_buf().clone());
    let live_occ_h         = graph.import_buffer(renderer.occ_buf().clone());
    let staging_exposure_h = graph.import_buffer(renderer.staging_exposure_buf().clone());
    let live_exposure_h    = graph.import_buffer(renderer.live_exposure_buf().clone());

    let slots: Vec<u32> = dirty_slots.to_vec();

    let patched = graph.add_pass("subchunk_patch", |pass| {
        pass.read_buffer(staging_occ_h);
        pass.read_buffer(staging_exposure_h);
        let occ_written      = pass.write_buffer(live_occ_h);
        let exposure_written = pass.write_buffer(live_exposure_h);
        pass.execute(move |ctx| {
            let occ_src      = ctx.resources.buffer(staging_occ_h);
            let occ_dst      = ctx.resources.buffer(occ_written);
            let exposure_src = ctx.resources.buffer(staging_exposure_h);
            let exposure_dst = ctx.resources.buffer(exposure_written);
            let occ_bytes      = std::mem::size_of::<SubchunkOccupancy>() as u64;
            let exposure_bytes = std::mem::size_of::<u32>()               as u64;
            for slot in &slots {
                let occ_offset      = (*slot as u64) * occ_bytes;
                let exposure_offset = (*slot as u64) * exposure_bytes;
                ctx.commands.copy_buffer_to_buffer(
                    occ_src, occ_offset, occ_dst, occ_offset, occ_bytes,
                );
                ctx.commands.copy_buffer_to_buffer(
                    exposure_src, exposure_offset,
                    exposure_dst, exposure_offset,
                    exposure_bytes,
                );
            }
        });
        occ_written
    });

    // Without an output marker, the graph would cull the pass: the live
    // buffers have no downstream reader declared inside this subgraph (the
    // render node that reads them lives in a separate call).
    graph.mark_output(patched);
}
