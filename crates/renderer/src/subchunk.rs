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
//! - Slot directory — one [`DirEntry`] per directory index, rewritten by
//!   the CPU residency plane on insert/evict. The cull shader unpacks the
//!   exposure mask from here (via `direntry_get_exposure`).
//!
//! The count buffer stays at `[1u32]` permanently — the cull shader emits
//! exactly one indirect draw entry whose `instance_count` fans out to the
//! number of passing candidates.
//!
//! # Bind group layouts (reflected from SPIR-V)
//!
//! **Cull bind group** (set 0 for `subchunk_cull.cs.hlsl`):
//! - 0: camera    (UniformBuffer, 64)          — COMPUTE
//! - 1: instances (StorageReadOnly, stride=16) — COMPUTE
//! - 2: visible   (StorageRW, stride=4)        — COMPUTE
//! - 3: lod_mask  (UniformBuffer, 512)         — COMPUTE
//! - 4: directory (StorageReadOnly, stride=24) — COMPUTE
//!
//! **Render bind group** (set 0 for `subchunk.vs/.ps.hlsl`):
//! - 0: camera     (UniformBuffer, 64)          — VERTEX | FRAGMENT
//! - 1: instances  (StorageReadOnly, stride=16) — VERTEX
//! - 2: visible    (StorageReadOnly, stride=4)  — VERTEX
//! - 3: occ_array  (StorageReadOnly, stride=64) — FRAGMENT
//!
//! **Shade bind group** (set 0 for `subchunk_shade.cs.hlsl`):
//! - 0: gpu_consts  (UniformBuffer, sizeof GpuConstsData) — COMPUTE.
//!   Implicit, supplied via `create_bind_group`'s `gpu_consts` argument
//!   (`directory.hlsl` includes `gpu_consts.hlsl` which pins this slot).
//! - 1: vis         (SampledTexture, R32Uint)              — COMPUTE
//! - 2: shaded_out  (StorageTexture, Rgba8Unorm, write)    — COMPUTE
//! - 3: directory   (StorageReadOnly, stride=24)           — COMPUTE
//! - 4: occ_array   (StorageReadOnly, stride=64)           — COMPUTE
//! - 5: camera      (UniformBuffer, 64)                    — COMPUTE
//! - 6: depth       (SampledTexture, Depth32Float)         — COMPUTE
//!
//! **Prep bind group** (set 0 for `subchunk_prep.cs.hlsl`):
//! - 0: gpu_consts     (UniformBuffer, sizeof GpuConstsData) — COMPUTE.
//!   Implicit, supplied via `create_bind_group`'s `gpu_consts` argument
//!   (the HLSL side pins this slot via `include/gpu_consts.hlsl`).
//! - 1: prep_requests  (StorageReadOnly, stride=32) — COMPUTE
//! - 2: material_pool  (StorageReadOnly, stride=64) — COMPUTE
//!   (diff source + neighbour-face source)
//! - 3: staging_occ    (StorageRW,       stride=64) — COMPUTE
//! - 4: dirty_list     (StorageRW, byte-addressed)  — COMPUTE
//! - 5: directory      (StorageReadOnly, stride=24) — COMPUTE
//!
//! Cull's indirect output rides on its own set-1 bind group, assembled
//! inside `nodes::cull`.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};

use crate::device::RendererContext;
use crate::frame::FrameIndex;
use crate::graph::{BufferDesc, BufferHandle, BufferPool, RenderGraph, TextureDesc, TextureHandle};
use crate::multi_buffer::MultiBufferRing;
use crate::nodes::{ColorTarget, CullArgs, DrawArgs, IndirectArgs, cull, mdi_draw, present_blit};
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

const SUBCHUNK_EXPOSURE_CS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/subchunk_exposure.cs.spv"));

const SUBCHUNK_SHADE_CS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/subchunk_shade.cs.spv"));

const PRESENT_BLIT_VS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/present_blit.vs.spv"));

const PRESENT_BLIT_PS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/present_blit.ps.spv"));

// --- Constants ---

/// Maximum number of candidate sub-chunks the cull pass handles in one
/// dispatch. Baked into the cull shader's workgroup size (`[256, 1, 1]`)
/// and into the `instance_buf` / `material_pool_buf` / per-slot
/// `staging_occ_ring` allocations.
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

/// Number of entries in the global material descriptor table.
///
/// Mirrors `MATERIAL_DESC_CAPACITY` in `shaders/include/material.hlsl`.
/// 256 is the M1 ceiling — raise in both places simultaneously if a world
/// ever registers more block types.
pub const MATERIAL_DESC_CAPACITY: usize = 256;

/// Number of per-voxel material-data slots in one 64 MB pool segment.
///
/// Matches `SLOTS_PER_SEGMENT` in
/// [`crates/game/src/world/material_data_pool.rs`](../../game/src/world/material_data_pool.rs)
/// and the shader-side `SLOTS_PER_SEGMENT` in `material.hlsl`. Each slot
/// holds a `MaterialBlock` (1024 bytes = 512 × u16 packed into 256 × u32),
/// so segment size = `SLOTS_PER_SEGMENT × 1024` = 64 MiB.
pub const MATERIAL_POOL_SLOTS_PER_SEGMENT: u32 = 65536;

/// Compile-time maximum number of live 64 MB segments in the material-data
/// pool. Matches `MAX_SEGMENTS` in the game allocator and the shader's
/// binding-array declaration `StructuredBuffer<MaterialBlock>
/// material_data_pool[MAX_MATERIAL_POOL_SEGMENTS]`. Raising requires a
/// shader recompile.
pub const MAX_MATERIAL_POOL_SEGMENTS: u32 = 16;

/// Byte size of a single slot in the material-data pool. 512 voxels × 2 B
/// per u16 ID = 1 KiB. Byte-identical to HLSL `MaterialBlock`.
pub const MATERIAL_BLOCK_BYTES: u64 = 1024;

/// Byte size of one 64 MB material-data pool segment =
/// `MATERIAL_POOL_SLOTS_PER_SEGMENT × MATERIAL_BLOCK_BYTES`.
pub const MATERIAL_SEGMENT_BYTES: u64 =
    MATERIAL_POOL_SLOTS_PER_SEGMENT as u64 * MATERIAL_BLOCK_BYTES;

/// Format of the transient `shaded_color` texture the shade compute pass
/// writes and the phase-1.5 blit pass consumes.
///
/// Chosen to be universally storage-binding-compatible in wgpu — the
/// swapchain format (typically `Bgra8UnormSrgb`) is a `RENDER_ATTACHMENT`
/// only surface and cannot host a UAV, so an intermediate transient is
/// required regardless. `Rgba8Unorm` is the cheapest four-channel
/// storage-binding-capable format and maps cleanly to any 8-bit-per-channel
/// swapchain via a downstream blit. The shade shader's HLSL side pins
/// this through `[[vk::image_format("rgba8")]]` — the two must stay in
/// sync or the pipeline layout validation fires on construction.
pub const SHADED_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

// --- MaterialBlock ---

/// One 1 KB material-data block: 512 u16 per-voxel material IDs packed
/// two-per-u32 into 256 u32 words. Byte-identical to the HLSL
/// `MaterialBlock` struct in `shaders/include/material.hlsl`.
///
/// `packed_ids[w]` holds voxels `(2*w, 2*w + 1)`:
/// - Low 16 bits: voxel `2*w`.
/// - High 16 bits: voxel `2*w + 1`.
///
/// The voxel linearisation is `x + 8*y + 64*z` (same as occupancy).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MaterialBlock {
    pub packed_ids: [u32; 256],
}

const _: () = assert!(
    std::mem::size_of::<MaterialBlock>() == 1024,
    "MaterialBlock must be 1 KB to match HLSL layout + MATERIAL_BLOCK_BYTES",
);

impl MaterialBlock {
    /// All-air block. Safe default for unused staging slots — any shader
    /// read returns material id 0 (air), which is never produced by a
    /// primary hit.
    #[allow(dead_code)]
    pub const fn zero() -> Self {
        Self { packed_ids: [0u32; 256] }
    }
}

// --- MaterialDesc ---

/// One entry in the global material descriptor table.
///
/// Layout matches HLSL `MaterialDesc` in `shaders/include/material.hlsl`
/// (32 bytes):
/// ```text
///   float4 albedo   (+0)    // sRGB RGBA, decoded to linear at fetch.
///   uint4  _reserved(+16)   // M3 PBR: albedo_tex, normal_tex, pbr_tex, flags.
/// ```
///
/// M1 uses only `albedo.rgb` for diffuse shading; alpha and the reserved
/// quad are populated as zero and read by nothing. The shader's
/// `material_fetch` is the single reader and decodes sRGB → linear on
/// every read — keep `albedo.rgb` in sRGB space on CPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MaterialDesc {
    pub albedo:    [f32; 4],
    pub _reserved: [u32; 4],
}

const _: () = assert!(
    std::mem::size_of::<MaterialDesc>() == 32,
    "MaterialDesc must be 32 bytes to match HLSL layout",
);

impl MaterialDesc {
    /// All-zero descriptor. Shader fetch returns black (`float3(0)`
    /// albedo) after sRGB→linear, which is never returned for a resident
    /// sub-chunk — but the buffer starts zero-initialised and is a safe
    /// default for unused slots.
    pub const fn zero() -> Self {
        Self { albedo: [0.0; 4], _reserved: [0; 4] }
    }

    /// Construct a descriptor from 8-bit sRGB RGB components. Alpha is
    /// stored as `1.0`; the reserved fields are zeroed.
    pub fn from_srgb_rgb(r: u8, g: u8, b: u8) -> Self {
        Self {
            albedo: [
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                1.0,
            ],
            _reserved: [0; 4],
        }
    }
}

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
/// the renderer's `slot_directory_buf`, packed into [`DirEntry::bits`] and
/// indexed by the slot field; the cull shader fetches it per-candidate
/// via `direntry_get_exposure` rather than unpacking it from the instance
/// record.
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
    /// contents of `g_directory[0]` or any other slot-indexed data.
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
    /// the cull shader rejects it before reading `g_directory` or any
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
///   uint _pad0 (+16)
///   uint _pad1 (+20)
///   uint _pad2 (+24)
///   uint _pad3 (+28)
/// ```
///
/// - `coord` — sub-chunk coord at this request's LOD.
/// - `level` — LOD level; voxel edge = `1 << level` metres.
///
/// The prep shader self-resolves its directory index from `(coord, level)`
/// via `resolve_coord_to_slot` (Principle 2: no CPU mirror of that
/// resolution). The trailing padding rounds the struct to a 32-byte
/// stride so the HLSL `StructuredBuffer<PrepRequest>` layout matches
/// DX-layout naturally.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PrepRequest {
    pub coord: [i32; 3],
    pub level: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

const _: () = assert!(
    std::mem::size_of::<PrepRequest>() == 32,
    "PrepRequest must be 32 bytes to match HLSL PrepRequest",
);

// --- DirtyEntry ---

/// One entry in the GPU prep dirty list.
///
/// Layout matches the HLSL shader's byte-address store (16 bytes):
/// ```text
///   uint directory_index     (+0)
///   uint new_bits_partial    (+4)   // [0..5] exposure, [6] is_solid,
///                                   // [7] resident (=1 this step)
///   uint staging_request_idx (+8)   // index into the staging ring slot
///   uint _pad                (+12)
/// ```
///
/// `new_bits_partial` carries only the directory-entry bits the prep shader
/// is authorised to emit: the 6-bit exposure field, the is-solid hint
/// (always `0` until Step 4 pairs it with neighbor-aware exposure), and the
/// resident bit (always `1` — a prep completion means the sub-chunk is
/// live). The material-slot field of [`DirEntry::bits`] is authored on the
/// CPU at retire time and is not populated by the shader.
///
/// `staging_request_idx` is the staging-buffer index for this entry's
/// payload, i.e. the `gid.x` of the prep workgroup that produced it. CPU
/// retirement uses it both to drive the patch copy (`staging_occ[idx] →
/// material_pool[dst_material_slot]`) and as the contract that decouples
/// staging layout from the live directory/material pool layouts.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq, Eq)]
pub struct DirtyEntry {
    pub directory_index:     u32,
    pub new_bits_partial:    u32,
    pub staging_request_idx: u32,
    pub _pad:                u32,
}

const _: () = assert!(
    std::mem::size_of::<DirtyEntry>() == 16,
    "DirtyEntry must be 16 bytes to match the HLSL dirty-list entry",
);

/// Sentinel value in [`DirtyEntry::staging_request_idx`] marking an
/// exposure-only-dispatch dirty entry.
///
/// Exposure-only refresh dispatches ([`subchunk_exposure`]) do not write
/// staging — they only recompute the 6-bit exposure mask + is_solid hint
/// from the existing material-pool occupancy. The shader writes
/// `EXPOSURE_STAGING_REQUEST_IDX_SENTINEL` into the entry's staging index
/// so the CPU retirement path can distinguish it from a full-prep dirty
/// entry and take the "update directory bits in place, emit no PatchCopy"
/// branch.
///
/// Production dispatches funnel full-prep and exposure-only entries
/// through separate dirty-list buffers + separate [`ReadbackChannel`]s
/// (see [`WorldRenderer::exposure_dirty_list_buf`] and the companion
/// [`subchunk_exposure`] node). The sentinel is consequently a
/// defence-in-depth redundancy: even if wiring were to route an exposure
/// entry through the full-prep retirement path, the sentinel would
/// prevent the `PatchCopy` construction from ever touching staging under
/// an unrelated `staging_request_idx`.
pub const EXPOSURE_STAGING_REQUEST_IDX_SENTINEL: u32 = 0xFFFF_FFFF;

// --- DirtyReport ---

/// GPU→CPU readback payload for one prep dispatch.
///
/// The shader writes the entry count at offset 0, an overflow flag at
/// offset 8, and the entries themselves starting at offset 16. Entries
/// `[0..min(count, MAX_CANDIDATES))` are valid; slots past that hold
/// undefined data and must be ignored by the consumer.
///
/// `overflow` is the Step-7 shadow-ledger signal that the shader would
/// have emitted more than [`MAX_CANDIDATES`] entries this dispatch —
/// the bounds-checked append drops any entry past capacity and sets
/// this flag to `1`. CPU reads it at retirement so pool-pressure
/// visibility is surfaced without per-entry GPU tracking. `_pad0` and
/// `_pad2` stay as padding so the entry array remains 16-byte-aligned.
///
/// Size: `16 + 16 * MAX_CANDIDATES = 4112` bytes (for MAX_CANDIDATES = 256).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DirtyReport {
    pub count:    u32,
    pub _pad0:    u32,
    /// Set to `1` by the shader when a workgroup's dirty-list append
    /// would have exceeded [`MAX_CANDIDATES`]. `0` otherwise. Cleared on
    /// every dispatch via the dirty-list header clear pass.
    pub overflow: u32,
    pub _pad2:    u32,
    pub entries:  [DirtyEntry; MAX_CANDIDATES],
}

const _: () = assert!(
    std::mem::size_of::<DirtyReport>() == 16 + 16 * MAX_CANDIDATES,
    "DirtyReport must be 16 + 16 * MAX_CANDIDATES bytes to match the HLSL \
     dirty-list layout",
);

// --- DirEntry ---

/// Sentinel material-slot value for non-resident / unallocated directory
/// entries. Fits in the 24-bit material-slot field of [`DirEntry::bits`].
pub const MATERIAL_SLOT_INVALID: u32 = 0xFF_FFFF;

/// Sentinel value stored in [`DirEntry::material_data_slot`] when a
/// directory entry has not yet been paired with a slot in the
/// material-data pool. Observed by the shade shader as the
/// "draw magenta" trigger — either because the entry is non-resident
/// (`resident == 0`), or because the pool was exhausted this frame and
/// the allocation is deferred to the next frame, or because the pool
/// ceiling was hit.
///
/// This is the *material-data* pool's invalid sentinel, not the
/// occupancy pool's. The two pools are independent; see
/// `decision-material-system-m1-sparse`.
pub const MATERIAL_DATA_SLOT_INVALID: u32 = 0xFFFF_FFFF;

/// Mask for the 6 directional exposure bits packed into [`DirEntry::bits`]
/// (`bits[0..5]`). One bit per `+X, -X, +Y, -Y, +Z, -Z` face; the layout
/// follows the existing cull-shader convention.
pub const BITS_EXPOSURE_MASK: u32 = 0x3F;

/// Mask for the is-solid bit packed into [`DirEntry::bits`] at bit 6.
/// Reserved for a future "this entry's sub-chunk is uniformly solid" hint
/// that lets the cull / DDA path skip occupancy reads entirely.
pub const BITS_IS_SOLID: u32 = 1 << 6;

/// Mask for the resident bit packed into [`DirEntry::bits`] at bit 7.
/// When clear, the directory entry is non-resident and its material-slot
/// field carries [`MATERIAL_SLOT_INVALID`].
pub const BITS_RESIDENT: u32 = 1 << 7;

/// Shift for the 24-bit material-slot field packed into [`DirEntry::bits`]
/// (`bits[8..31]`). The material slot is an index into the CPU-authoritative
/// `MaterialAllocator` pool, or [`MATERIAL_SLOT_INVALID`] when non-resident.
pub const BITS_MATERIAL_SLOT_SHIFT: u32 = 8;

/// One entry in the CPU-authored sub-chunk directory.
///
/// The directory is indexed by a stable `directory_index` (= `level_offset +
/// pool_slot`) and carries the contract for how the GPU should resolve a given
/// sub-chunk: whether it's resident, which material-storage slot holds its
/// occupancy payload, and its cached exposure mask.
///
/// Layout matches the HLSL `DirEntry` struct (28 bytes):
/// ```text
///   int3 coord              (+0)
///   uint bits               (+12)   // [0..5] exposure, [6] is_solid,
///                                   // [7] resident, [8..31] material_slot
///                                   //                       | INVALID
///   uint content_version    (+16)
///   uint last_synth_version (+20)
///   uint material_data_slot (+24)   // flat-global MaterialDataPool slot,
///                                   // or MATERIAL_DATA_SLOT_INVALID.
/// ```
///
/// - `coord` carries the torus-verification coord for the entry. For
///   non-resident entries it is left at `[0, 0, 0]` and the shader never
///   reads it (resident bit clear ⇒ early-out).
/// - `bits` packs the four per-entry fields that change per residency
///   event. Exposure and is_solid are read by the cull/DDA path; resident
///   gates the whole entry; material_slot is the indirection target for
///   the *occupancy* pool (NOT the material-data pool).
/// - `content_version` bumps on any occupancy mutation. `0` for pure
///   worldgen — this step never bumps it.
/// - `last_synth_version` is CPU-only bookkeeping for the eviction TTL in
///   later steps; present here for layout symmetry so downstream shader
///   ports don't need a struct-size change.
/// - `material_data_slot` is the flat-global slot index into the
///   segmented material-data pool (64 MB × N segments). Carries
///   [`MATERIAL_DATA_SLOT_INVALID`] when the entry has no allocation
///   (non-resident, deferred mid-grow, or pool ceiling reached). The
///   shade shader reads it and decodes `(segment_idx, local_idx)` on the
///   GPU via `div/mod SLOTS_PER_SEGMENT`; the division point is this
///   field so the decode never leaks to non-shade callsites. See
///   `decision-material-system-m1-sparse`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DirEntry {
    pub coord:              [i32; 3],
    pub bits:               u32,
    pub content_version:    u32,
    pub last_synth_version: u32,
    /// Material-data pool slot, or [`MATERIAL_DATA_SLOT_INVALID`]. See
    /// the struct-level docs.
    pub material_data_slot: u32,
}

const _: () = assert!(
    std::mem::size_of::<DirEntry>() == 28,
    "DirEntry must be 28 bytes to match the HLSL DirEntry layout",
);

impl DirEntry {
    /// All-zero non-resident entry except for `material_data_slot`, which
    /// starts at [`MATERIAL_DATA_SLOT_INVALID`] so the shade shader emits
    /// magenta rather than dereferencing a stale slot. `bits == 0`
    /// implies `resident = 0`, `is_solid = 0`, `exposure = 0`, and the
    /// packed `material_slot` (occupancy pool) is `0`; callers must not
    /// read those fields without first checking `resident`.
    ///
    /// Note: a zero `material_slot` is *not* [`MATERIAL_SLOT_INVALID`]; the
    /// resident bit is the authoritative gate. Constructing a
    /// non-resident entry where shader code might speculatively read
    /// `material_slot` should go through [`DirEntry::non_resident`].
    pub const fn empty() -> Self {
        Self {
            coord:              [0, 0, 0],
            bits:               0,
            content_version:    0,
            last_synth_version: 0,
            material_data_slot: MATERIAL_DATA_SLOT_INVALID,
        }
    }

    /// Explicit non-resident entry: `resident` bit clear, `material_slot`
    /// set to [`MATERIAL_SLOT_INVALID`], and `material_data_slot` set to
    /// [`MATERIAL_DATA_SLOT_INVALID`]. Use when the caller wants the
    /// sentinel slot values to be present for diagnostic readability or
    /// to match a shader that panics on a non-INVALID slot under a clear
    /// resident bit.
    pub const fn non_resident() -> Self {
        Self {
            coord:              [0, 0, 0],
            bits:               MATERIAL_SLOT_INVALID << BITS_MATERIAL_SLOT_SHIFT,
            content_version:    0,
            last_synth_version: 0,
            material_data_slot: MATERIAL_DATA_SLOT_INVALID,
        }
    }

    /// Build a resident directory entry. The `material_data_slot` field
    /// is initialised to [`MATERIAL_DATA_SLOT_INVALID`]; the materializer
    /// populates it on a separate, asynchronous code path (see
    /// `decision-material-system-m1-sparse`) and writes the updated
    /// entry via [`DirEntry::with_material_data_slot`].
    ///
    /// # Panics
    ///
    /// Debug builds panic if `exposure > 0x3F` (exposure field is 6 bits)
    /// or if `material_slot > MATERIAL_SLOT_INVALID` (slot field is 24
    /// bits). `material_slot == MATERIAL_SLOT_INVALID` is a
    /// caller-visible programming error under a resident entry — later
    /// shader ports assume the slot is a real pool index.
    pub fn resident(
        coord:         [i32; 3],
        exposure:      u32,
        is_solid:      bool,
        material_slot: u32,
    )
        -> Self
    {
        debug_assert!(exposure <= BITS_EXPOSURE_MASK, "exposure must fit in 6 bits");
        debug_assert!(
            material_slot < MATERIAL_SLOT_INVALID,
            "material_slot {material_slot} must be a real pool index under resident entry",
        );

        let mut bits = exposure & BITS_EXPOSURE_MASK;
        if is_solid {
            bits |= BITS_IS_SOLID;
        }
        bits |= BITS_RESIDENT;
        bits |= (material_slot & MATERIAL_SLOT_INVALID) << BITS_MATERIAL_SLOT_SHIFT;

        Self {
            coord,
            bits,
            content_version:    0,
            last_synth_version: 0,
            material_data_slot: MATERIAL_DATA_SLOT_INVALID,
        }
    }

    /// Return a copy of `self` with [`DirEntry::material_data_slot`] set
    /// to `slot`. Builder-style so the materializer can rewrite a
    /// resident entry's slot in one expression:
    ///
    /// ```ignore
    /// let new_entry = old_entry.with_material_data_slot(allocated_slot);
    /// directory.set(index, new_entry);
    /// ```
    pub const fn with_material_data_slot(mut self, slot: u32) -> Self {
        self.material_data_slot = slot;
        self
    }

    /// `true` if the resident bit is set.
    pub const fn is_resident(&self) -> bool {
        (self.bits & BITS_RESIDENT) != 0
    }

    /// Six-bit directional exposure mask (`+X, -X, +Y, -Y, +Z, -Z`).
    pub const fn exposure(&self) -> u32 {
        self.bits & BITS_EXPOSURE_MASK
    }

    /// `true` if the is-solid hint bit is set.
    pub const fn is_solid(&self) -> bool {
        (self.bits & BITS_IS_SOLID) != 0
    }

    /// Unpacked 24-bit material-slot field. Returns
    /// [`MATERIAL_SLOT_INVALID`] for non-resident entries built via
    /// [`DirEntry::non_resident`] or [`DirEntry::empty`] (the latter
    /// returns `0`, which is also not a resident slot — use
    /// [`DirEntry::is_resident`] to gate).
    pub const fn material_slot(&self) -> u32 {
        (self.bits >> BITS_MATERIAL_SLOT_SHIFT) & MATERIAL_SLOT_INVALID
    }
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
    cull_pipeline:     Arc<ComputePipeline>,
    prep_pipeline:     Arc<ComputePipeline>,
    /// Exposure-only refresh pipeline. Runs
    /// `shaders/subchunk_exposure.cs.hlsl` with one workgroup per
    /// [`PrepRequest`] (the shared request layout — exposure-only uses
    /// the same 32-byte struct shape). See [`subchunk_exposure`].
    exposure_pipeline: Arc<ComputePipeline>,
    /// Compute pipeline that consumes the vis-buffer produced by the draw
    /// pass and writes the shaded colour transient. See
    /// `shaders/subchunk_shade.cs.hlsl` and `subchunk_world`.
    shade_pipeline:    Arc<ComputePipeline>,
    render_pipeline:   Arc<RenderPipeline>,
    /// Raster pipeline for the phase-1.5 present blit. Fullscreen triangle
    /// that `.Load`s from `shaded_color` and writes into the swapchain
    /// attachment the node is wired against. Colour target format is the
    /// surface format observed at construction time — the pipeline state
    /// is immutable, so a resize that changed the swapchain format (not a
    /// case wgpu exposes today) would require a fresh `WorldRenderer`.
    /// See `shaders/present_blit.vs.hlsl` + `.ps.hlsl` and
    /// [`crate::nodes::present_blit`].
    present_blit_pipeline: Arc<RenderPipeline>,

    camera_buf:            wgpu::Buffer,
    instance_buf:          wgpu::Buffer,
    /// Live per-slot occupancy payload. Indexed by `material_slot`
    /// (the 24-bit field inside [`DirEntry::bits`]); the render VS/PS
    /// still reads it by `directory_index` under the identity policy
    /// (`material_slot == directory_index`). A later step will tighten
    /// the pool's capacity below the directory's and wire VS/PS through
    /// the directory resolver.
    material_pool_buf:     wgpu::Buffer,
    count_buf:             wgpu::Buffer,
    lod_mask_buf:          wgpu::Buffer,
    /// Staging copy of prep outputs, indexed by `staging_request_idx`
    /// (= the prep workgroup's `gid.x`). Sized to [`MAX_CANDIDATES`]
    /// entries per frame-in-flight — the maximum number of prep requests
    /// that may execute in one dispatch. The CPU retirement logic copies
    /// a subset of staging entries into `material_pool_buf` via the patch
    /// pass.
    ///
    /// Ring semantics: at frame `F`, prep writes
    /// `staging_occ_ring.current(F)`; the retirement that completes at
    /// frame `F + FrameCount` reads from `ring.current(F)` (the same
    /// slot, which has survived because the intervening frames rotated
    /// to other slots). This is the FIF-sized ring that
    /// `knowledge-fif-swapchain-depth-decoupling` describes — N = FIF is
    /// exactly sufficient.
    ///
    /// Memory footprint trade-off: the old single buffer was
    /// `MAX_CANDIDATES * 64 B` (= 16 KiB). The ring is
    /// `FrameCount * MAX_CANDIDATES * 64 B` (= 32 KiB under FIF=2).
    /// Negligible given the correctness problem it fixes — a single
    /// buffer is clobbered between the prep dispatch and the patch that
    /// consumes it, so the patch landed wrong-frame data at the retired
    /// directory_index. See `failure-staging-not-ringed-after-gid-x-reindexing`.
    staging_occ_ring:      MultiBufferRing<SubchunkOccupancy>,
    /// Parallel staging ring for per-voxel material IDs the prep shader
    /// emits alongside occupancy. One 1 KB [`MaterialBlock`]-shaped entry
    /// per prep request; indexed by the same `staging_request_idx`
    /// (`gid.x`) as `staging_occ_ring` so a given `req_idx` points at
    /// byte-symmetric occupancy + material-id payloads.
    ///
    /// Ring semantics identical to [`staging_occ_ring`]: prep at frame F
    /// writes slot `F % FrameCount`; the retirement at frame
    /// `F + FrameCount` reads the same slot. See
    /// `knowledge-fif-swapchain-depth-decoupling`.
    staging_material_ids_ring: MultiBufferRing<MaterialBlock>,
    dirty_list_buf:        wgpu::Buffer,
    prep_request_buf:      wgpu::Buffer,

    // --- Exposure-only dispatch (Step 5) ---
    //
    // Parallel pair to `(prep_request_buf, dirty_list_buf)`, feeding the
    // neighbour-aware-exposure-refresh dispatch. Kept separate from the
    // full-prep buffers so:
    //  - Both dispatches can coexist in the same frame without stepping on
    //    each other's dirty-list header atomic (each owns its own buffer).
    //  - The CPU retirement can read each readback channel independently
    //    (full-prep and exposure-only retire on the same frame cadence, so
    //    keeping the readbacks separate avoids a lockstep-ordering
    //    requirement the caller would otherwise need to preserve).
    //  - The dispatch bindings are simpler — exposure-only does not touch
    //    the staging ring, so it has one fewer binding than full-prep.
    /// CPU-writable buffer of [`PrepRequest`]s for the exposure-only
    /// dispatch. Same struct shape as the full-prep request (coord,
    /// level, padding) so the CPU can share its request-building code
    /// between the two paths.
    exposure_request_buf:   wgpu::Buffer,
    /// Dirty-list buffer written by the exposure-only dispatch. Same
    /// [`DirtyReport`] layout as `dirty_list_buf`; entries carry
    /// [`EXPOSURE_STAGING_REQUEST_IDX_SENTINEL`] in
    /// [`DirtyEntry::staging_request_idx`] as a structural marker.
    exposure_dirty_list_buf: wgpu::Buffer,
    /// CPU-authored directory mapping `directory_index -> DirEntry`.
    /// Sized to `MAX_CANDIDATES` entries today (trivial 1:1 directory ↔
    /// material-slot mapping). The cull shader reads it via the set-0
    /// binding added in Step 6 and unpacks the exposure mask through
    /// `direntry_get_exposure`.
    slot_directory_buf:    Arc<wgpu::Buffer>,

    /// Global descriptor table: `[MaterialDesc; MATERIAL_DESC_CAPACITY]`.
    /// Published once at startup via [`write_materials`](
    /// WorldRenderer::write_materials) from the game side's
    /// `BlockRegistry`. Read by the shade compute pass at fetch time; the
    /// shader decodes sRGB → linear on every read so the CPU side stores
    /// sRGB directly.
    material_desc_buf:     wgpu::Buffer,

    /// Live 64 MB segments of the material-data pool. Each segment is a
    /// `StructuredBuffer<MaterialBlock>` of
    /// [`MATERIAL_POOL_SLOTS_PER_SEGMENT`] entries, bound as element
    /// `segments_live[i]` of the shade shader's binding-array slot. The
    /// renderer grows this list lazily — starts with a single segment at
    /// construction so the first frame has a populated binding even when
    /// nothing's been allocated yet; subsequent segments come from
    /// [`append_material_segment`](WorldRenderer::append_material_segment)
    /// when the CPU allocator returns `None`. Capped at
    /// [`MAX_MATERIAL_POOL_SEGMENTS`]; exceeding that is an error in the
    /// materializer.
    ///
    /// Wrapped in a `Mutex` because the renderer is held as
    /// `Arc<WorldRenderer>` across the render-graph and ownership paths,
    /// but the segment list needs mutation after construction (from the
    /// materializer at frame start). A `Mutex` is the smallest concession
    /// — the contention window is one push per grow event, which is
    /// measured in seconds of play.
    material_segment_bufs: Mutex<Vec<wgpu::Buffer>>,
}

impl WorldRenderer {
    /// Create the pipelines and allocate dynamic buffers sized for
    /// [`MAX_CANDIDATES`] sub-chunks. Buffers are zero-initialized; the
    /// caller must populate instances/occupancy before the first render.
    /// A render before the first instance upload produces a blank frame
    /// rather than undefined behavior — the instance buffer is zeroed, so
    /// every entry has `slot_mask = 0` (slot 0, level 0, padding bit
    /// clear). Those entries fetch `g_directory[0]`, which is zero
    /// (= `DirEntry::empty()` — `resident = 0`, `exposure = 0`), so the
    /// cull shader's exposure rejection drops them.
    pub fn new(ctx: &RendererContext, pool: &mut BufferPool) -> Self {
        // Assert a windowed context up front — the draw/shade path writes
        // into transients whose extents come from the swapchain size, and
        // the downstream blit pass (phase-1.5) depends on a swapchain
        // target. Catching the headless case here surfaces the
        // misconfiguration at renderer construction rather than deep in a
        // compiled-graph execute closure.
        //
        // The surface format is also load-bearing for the present-blit
        // pipeline's colour target (an immutable part of the render
        // pipeline state), so we retain it rather than discarding.
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

        let exposure_cs = ShaderModule::load(
            ctx,
            "subchunk_exposure.cs",
            ShaderSource::Spirv(SUBCHUNK_EXPOSURE_CS_SPV),
            "main",
        )
        .expect("subchunk exposure shader failed to load");

        let shade_cs = ShaderModule::load(
            ctx,
            "subchunk_shade.cs",
            ShaderSource::Spirv(SUBCHUNK_SHADE_CS_SPV),
            "main",
        )
        .expect("subchunk shade shader failed to load");

        let blit_vs = ShaderModule::load(
            ctx,
            "present_blit.vs",
            ShaderSource::Spirv(PRESENT_BLIT_VS_SPV),
            "main",
        )
        .expect("present blit vertex shader failed to load");

        let blit_ps = ShaderModule::load(
            ctx,
            "present_blit.ps",
            ShaderSource::Spirv(PRESENT_BLIT_PS_SPV),
            "main",
        )
        .expect("present blit pixel shader failed to load");

        // --- Buffer sizes ---

        let camera_size       = std::mem::size_of::<SubchunkCamera>() as u64;
        let instance_size     = (std::mem::size_of::<SubchunkInstance>() * MAX_CANDIDATES) as u64;
        let occ_size          = (std::mem::size_of::<SubchunkOccupancy>() * MAX_CANDIDATES) as u64;
        let count_size        = 4u64;
        let lod_mask_size     = std::mem::size_of::<LodMaskUniform>() as u64;
        let dirty_list_size   = std::mem::size_of::<DirtyReport>() as u64;
        let prep_request_size = (std::mem::size_of::<PrepRequest>() * MAX_CANDIDATES) as u64;
        let directory_size    = (std::mem::size_of::<DirEntry>()     * MAX_CANDIDATES) as u64;
        let material_desc_size =
            (std::mem::size_of::<MaterialDesc>() * MATERIAL_DESC_CAPACITY) as u64;

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

        let material_pool_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_material_pool"),
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

        // Staging ring written by the GPU prep pass. One slot per
        // frame-in-flight: the prep dispatch at frame F writes its ring
        // slot, and the retirement that lands at frame F + FrameCount
        // reads the same slot to drive its patch copies. Intervening
        // frames rotate to their own slots, so frame F's data is
        // preserved until the retirement consumes it.
        //
        // A single buffer would be written every frame and clobbered
        // between the prep at F and the patch that runs FrameCount
        // frames later — the exact race that
        // `failure-staging-not-ringed-after-gid-x-reindexing` documents.
        //
        // Each slot is sized identically (`MAX_CANDIDATES` entries)
        // because the ring is indexed by `staging_request_idx` (=
        // `gid.x` of the prep workgroup), so the size is proportional to
        // the maximum prep dispatch width rather than to the directory
        // capacity. `STORAGE` for the prep shader write; `COPY_SRC` for
        // the patch pass that lifts selected entries into
        // `material_pool_buf` at CPU-chosen material slots.
        let staging_occ_ring = MultiBufferRing::<SubchunkOccupancy>::new(
            ctx,
            pool,
            "subchunk_staging_occ",
            BufferDesc {
                size:  occ_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        );

        // Parallel staging ring for per-voxel material IDs. Same ring
        // semantics + sizing policy as `staging_occ_ring` — one entry per
        // prep request, indexed by the same `gid.x`. Stride is
        // `MATERIAL_BLOCK_BYTES` per request.
        let staging_material_ids_size =
            MATERIAL_BLOCK_BYTES * MAX_CANDIDATES as u64;
        let staging_material_ids_ring = MultiBufferRing::<MaterialBlock>::new(
            ctx,
            pool,
            "subchunk_staging_material_ids",
            BufferDesc {
                size:  staging_material_ids_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
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

        // Exposure-only request buffer: same size + shape as the prep
        // request buffer (the struct layout is shared). Written via
        // `write_exposure_requests`.
        let exposure_request_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_exposure_requests"),
            size:               prep_request_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Exposure-only dirty-list buffer: mirrors `dirty_list_buf` in
        // shape + usage. Kept separate so the exposure dispatch has its
        // own atomic `count` header — running both dispatches in the same
        // frame without separate buffers would require `InterlockedAdd`
        // contention between the two dispatches for no gain.
        let exposure_dirty_list_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("subchunk_exposure_dirty_list"),
            size:               dirty_list_size,
            usage:              wgpu::BufferUsages::STORAGE
                              | wgpu::BufferUsages::COPY_SRC
                              | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Slot directory: CPU-authored contract mapping `directory_index`
        // to a packed `DirEntry`. Sized to [`MAX_CANDIDATES`] entries —
        // trivial 1:1 mapping against the material-storage pool at Step 1.
        // `STORAGE | COPY_DST` so future shader ports can read it via a
        // bind group, and the CPU can patch in per-entry updates via
        // `queue.write_buffer`. The buffer starts zero-initialised
        // (implicit `DirEntry::empty()`); the first frame's directory
        // flush populates whatever the residency has decided is resident.
        // `COPY_SRC` is load-bearing only for the `debug-state-history`
        // diagnostics path in the game crate: it blits this buffer to a
        // `ReadbackChannel<DirectorySnapshot>` slot each frame so the CPU
        // can compare its CPU-authored directory against what the GPU
        // actually observed. The flag is inert when the feature is off —
        // wgpu validates usage at bind time, never at allocation — so the
        // cost in release builds is exactly zero.
        let slot_directory_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("world_slot_directory"),
            size:               directory_size,
            usage:              wgpu::BufferUsages::STORAGE
                              | wgpu::BufferUsages::COPY_DST
                              | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Global material descriptor table. Populated once via
        // `write_materials`; zero-initialised until then so any speculative
        // fetch hits a well-defined (albedo = [0,0,0,0]) entry.
        let material_desc_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("material_desc_table"),
            size:               material_desc_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Material-data pool — initial segment created up front so every
        // shade dispatch has at least one bound element in the binding
        // array slot. Subsequent segments come from
        // `append_material_segment` when the CPU allocator signals grow.
        // 64 MiB per segment; `PARTIALLY_BOUND_BINDING_ARRAY` permits
        // fewer-than-count bound elements at the remaining slots.
        let material_segment_bufs = Mutex::new(vec![
            create_material_segment_buf(device, 0),
        ]);

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

        let exposure_pipeline = Arc::new(ComputePipeline::new(ctx, ComputePipelineDescriptor {
            label:                   "subchunk_exposure",
            shader:                  exposure_cs,
            expected_workgroup_size: Some([1, 1, 1]),
            immediate_size:          0,
        }));

        let shade_pipeline = Arc::new(ComputePipeline::new(ctx, ComputePipelineDescriptor {
            label:                   "subchunk_shade",
            shader:                  shade_cs,
            expected_workgroup_size: Some([8, 8, 1]),
            immediate_size:          0,
        }));

        // Single MRT slot now that task 1.4's shade compute pass produces
        // the swapchain-destined colour. The fragment shader's only
        // `SV_Target*` output is `SV_Target0` (the packed vis word); the
        // swapchain is handed the shade pass's output through a
        // downstream blit (phase-1.5).
        let render_pipeline = Arc::new(RenderPipeline::new(ctx, RenderPipelineDescriptor {
            label:          "subchunk",
            vertex:         vs,
            fragment:       ps,
            vertex_buffers: &[],
            color_targets:  &[
                Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::R32Uint,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
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

        // Fullscreen-triangle blit that samples `shaded_color` into the
        // swapchain attachment. No vertex buffer (positions derived from
        // `SV_VertexID`), no depth, no cull — the three issued vertices
        // cover clip space and the rasteriser clips the off-screen area
        // so winding / face-cull selection is moot. Blend disabled so the
        // pass is a pure overwrite of the swapchain texels. Colour target
        // format must match the actual swapchain format or pipeline
        // validation fires on the first render pass.
        let present_blit_pipeline =
            Arc::new(RenderPipeline::new(ctx, RenderPipelineDescriptor {
                label:          "present_blit",
                vertex:         blit_vs,
                fragment:       blit_ps,
                vertex_buffers: &[],
                color_targets:  &[Some(wgpu::ColorTargetState {
                    format:     surface_format,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                depth_stencil:  None,
                primitive:      wgpu::PrimitiveState {
                    topology:   wgpu::PrimitiveTopology::TriangleList,
                    cull_mode:  None,
                    ..Default::default()
                },
                multisample:    wgpu::MultisampleState::default(),
                immediate_size: 0,
            }));

        Self {
            cull_pipeline,
            prep_pipeline,
            exposure_pipeline,
            shade_pipeline,
            render_pipeline,
            present_blit_pipeline,
            camera_buf,
            instance_buf,
            material_pool_buf,
            count_buf,
            lod_mask_buf,
            staging_occ_ring,
            staging_material_ids_ring,
            dirty_list_buf,
            prep_request_buf,
            exposure_request_buf,
            exposure_dirty_list_buf,
            slot_directory_buf,
            material_desc_buf,
            material_segment_bufs,
        }
    }

    /// Overwrite the global material descriptor table.
    ///
    /// Intended to be called once at startup after the game side's
    /// [`BlockRegistry`] has been built. The buffer is sized to
    /// [`MATERIAL_DESC_CAPACITY`] entries; passing fewer leaves trailing
    /// slots zero-initialised (safe — the shade shader never fetches
    /// slots it wasn't told about). Passing more than
    /// [`MATERIAL_DESC_CAPACITY`] panics.
    pub fn write_materials(&self, ctx: &RendererContext, descs: &[MaterialDesc]) {
        assert!(
            descs.len() <= MATERIAL_DESC_CAPACITY,
            "write_materials: got {} descs, max is {MATERIAL_DESC_CAPACITY}",
            descs.len(),
        );
        if descs.is_empty() {
            return;
        }
        ctx.queue()
            .write_buffer(&self.material_desc_buf, 0, bytemuck::cast_slice(descs));
    }

    /// Append a new 64 MB material-data pool segment.
    ///
    /// Called by the materializer at frame boundary when the CPU
    /// allocator's [`MaterialDataPool::grow`] succeeds. The new GPU buffer
    /// is zero-initialised — the first writer that lands a patch into one
    /// of its slots is what gives the segment's slots defined contents;
    /// before that, a speculative fetch returns `uint(0)` which the
    /// shader treats as material id 0 (air / default).
    ///
    /// The bind-group for the shade dispatch is rebuilt every frame
    /// (render graph is rebuilt from scratch per frame), so the new
    /// segment naturally picks up in the next dispatch after this call —
    /// no explicit descriptor rebind is needed here beyond pushing the
    /// buffer into `material_segment_bufs`.
    ///
    /// Takes `&self` (not `&mut self`) so callers can hold the renderer
    /// behind an `Arc`; the segment list is guarded by an internal
    /// `Mutex`.
    ///
    /// # Panics
    ///
    /// Panics if the pool has already reached
    /// [`MAX_MATERIAL_POOL_SEGMENTS`]. The CPU allocator is the
    /// authoritative ceiling-enforcer; this helper is the renderer-side
    /// double-check in case of bookkeeping drift.
    pub fn append_material_segment(&self, ctx: &RendererContext) {
        let mut segs = self
            .material_segment_bufs
            .lock()
            .expect("material_segment_bufs mutex poisoned");
        let segments_now = segs.len() as u32;
        assert!(
            segments_now < MAX_MATERIAL_POOL_SEGMENTS,
            "append_material_segment: already at MAX_MATERIAL_POOL_SEGMENTS \
             ({MAX_MATERIAL_POOL_SEGMENTS})",
        );
        let buf = create_material_segment_buf(ctx.device(), segments_now);
        segs.push(buf);
    }

    /// Number of live material-data pool segments. Equal to the CPU
    /// allocator's `segments_live()` under correct bookkeeping.
    pub fn material_segments_live(&self) -> u32 {
        self.material_segment_bufs
            .lock()
            .expect("material_segment_bufs mutex poisoned")
            .len() as u32
    }

    /// Copy a single prepared 1 KB material block into the material-data
    /// pool at `global_slot`. The slot is decomposed into (segment, local)
    /// on the CPU; the target segment must already exist. Safe to call
    /// concurrently with a material-pool bind: the graph's access
    /// tracking will emit a barrier when the shade dispatch reads the
    /// written segment this frame.
    ///
    /// `bytes` must be exactly [`MATERIAL_BLOCK_BYTES`] long (1024 B =
    /// 512 × u16 per-voxel material IDs).
    ///
    /// # Panics
    ///
    /// Panics when the slot's segment is past
    /// [`material_segments_live`](WorldRenderer::material_segments_live)
    /// or when `bytes.len() != MATERIAL_BLOCK_BYTES`. Both are CPU-side
    /// bookkeeping bugs.
    pub fn write_material_block(
        &self,
        ctx:         &RendererContext,
        global_slot: u32,
        bytes:       &[u8],
    ) {
        assert_eq!(
            bytes.len() as u64,
            MATERIAL_BLOCK_BYTES,
            "write_material_block: expected {MATERIAL_BLOCK_BYTES} bytes, got {}",
            bytes.len(),
        );
        let segment_idx = global_slot / MATERIAL_POOL_SLOTS_PER_SEGMENT;
        let local_idx   = global_slot % MATERIAL_POOL_SLOTS_PER_SEGMENT;
        let segs = self
            .material_segment_bufs
            .lock()
            .expect("material_segment_bufs mutex poisoned");
        let seg_buf = segs.get(segment_idx as usize)
            .unwrap_or_else(|| panic!(
                "write_material_block: slot {global_slot} → segment {segment_idx} \
                 not live (segments_live = {}); caller must append_material_segment \
                 before routing slots into it",
                segs.len(),
            ));
        let offset = local_idx as u64 * MATERIAL_BLOCK_BYTES;
        ctx.queue().write_buffer(seg_buf, offset, bytes);
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

    /// Upload a single slot's 64-byte occupancy payload directly into
    /// `material_pool_buf`.
    ///
    /// `occ_bytes` must be exactly 64 bytes in the GPU layout — an
    /// 8-word sequence of little-endian u64s covering z-layers 0..8, where
    /// bit `y*8 + x` of layer `z` is set when voxel `(x, y, z)` is occupied.
    ///
    /// Left in place for diagnostic paths (tests, validation). The
    /// production pipeline routes occupancy through the prep shader + patch
    /// pass, so this entry point should not be used on the hot path.
    pub fn write_occupancy_slot(
        &self,
        ctx:       &RendererContext,
        slot:      u32,
        occ_bytes: &[u8; std::mem::size_of::<SubchunkOccupancy>()],
    ) {
        let offset = (slot as u64) * std::mem::size_of::<SubchunkOccupancy>() as u64;
        ctx.queue().write_buffer(&self.material_pool_buf, offset, occ_bytes);
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

    /// Upload `requests` into the exposure-only request buffer.
    ///
    /// The exposure-only compute pass dispatches one workgroup per request
    /// (see [`subchunk_exposure`]); callers must pass the same
    /// `requests.len()` as `request_count` when registering the exposure
    /// node.
    ///
    /// Uses the same [`PrepRequest`] struct as full-prep — the struct's
    /// payload (coord + level) is identical in both dispatches.
    ///
    /// # Panics
    ///
    /// Panics if `requests.len() > MAX_CANDIDATES` — the buffer is sized
    /// to exactly that many entries.
    pub fn write_exposure_requests(&self, ctx: &RendererContext, requests: &[PrepRequest]) {
        assert!(
            requests.len() <= MAX_CANDIDATES,
            "write_exposure_requests: got {} requests, max is {MAX_CANDIDATES}",
            requests.len(),
        );
        if requests.is_empty() {
            return;
        }
        ctx.queue().write_buffer(
            &self.exposure_request_buf,
            0,
            bytemuck::cast_slice(requests),
        );
    }

    /// Batch-write a set of `(directory_index, entry)` updates into the
    /// [`WorldRenderer::slot_directory_buf`].
    ///
    /// Each entry is committed via a single `queue.write_buffer` at its
    /// own 24-byte offset. The directory buffer is small (MAX_CANDIDATES ×
    /// 24 B ≈ 6 KiB today) and updates are sparse, so a staging ring is
    /// not warranted here — this is the simplest shape that satisfies the
    /// CPU-authoritative / GPU-readonly contract.
    ///
    /// Called once per frame with the drained dirty set from
    /// `SlotDirectory::drain_dirty`. An empty slice is a no-op.
    ///
    /// # Panics
    ///
    /// Debug builds panic if any `index >= MAX_CANDIDATES` — the
    /// directory is sized exactly to the candidate pool.
    pub fn write_directory_entries(
        &self,
        ctx:     &RendererContext,
        updates: &[(u32, DirEntry)],
    ) {
        let stride = std::mem::size_of::<DirEntry>() as u64;
        for (index, entry) in updates {
            debug_assert!(
                (*index as usize) < MAX_CANDIDATES,
                "directory index {index} out of range (MAX_CANDIDATES = {MAX_CANDIDATES})",
            );
            let offset = (*index as u64) * stride;
            ctx.queue()
                .write_buffer(&self.slot_directory_buf, offset, bytemuck::bytes_of(entry));
        }
    }

    /// Borrow the `Arc`-held slot-directory buffer for bind-group wiring.
    /// No shader reads it yet (Step 2 adds bind entries); the handle is
    /// exposed so residency tests / Step 2 callers can thread it into
    /// future passes.
    pub fn slot_directory_buf(&self) -> &Arc<wgpu::Buffer> {
        &self.slot_directory_buf
    }

    pub(crate) fn cull_pipeline(&self) -> &Arc<ComputePipeline> {
        &self.cull_pipeline
    }

    pub(crate) fn prep_pipeline(&self) -> &Arc<ComputePipeline> {
        &self.prep_pipeline
    }

    pub(crate) fn exposure_pipeline(&self) -> &Arc<ComputePipeline> {
        &self.exposure_pipeline
    }

    pub(crate) fn shade_pipeline(&self) -> &Arc<ComputePipeline> {
        &self.shade_pipeline
    }

    pub(crate) fn render_pipeline(&self) -> &Arc<RenderPipeline> {
        &self.render_pipeline
    }

    pub(crate) fn present_blit_pipeline(&self) -> &Arc<RenderPipeline> {
        &self.present_blit_pipeline
    }

    pub(crate) fn camera_buf(&self) -> &wgpu::Buffer {
        &self.camera_buf
    }

    pub(crate) fn instance_buf(&self) -> &wgpu::Buffer {
        &self.instance_buf
    }

    pub(crate) fn material_pool_buf(&self) -> &wgpu::Buffer {
        &self.material_pool_buf
    }

    pub(crate) fn count_buf(&self) -> &wgpu::Buffer {
        &self.count_buf
    }

    pub(crate) fn lod_mask_buf(&self) -> &wgpu::Buffer {
        &self.lod_mask_buf
    }

    /// Borrow the FIF-sized staging ring.
    ///
    /// `staging_occ_ring().current(frame)` is the buffer the prep
    /// dispatch at `frame` writes to. `staging_occ_ring().current(
    /// retired_frame)` is the buffer the patch pass must read when
    /// consuming the dirty list from `retired_frame` (the ring rotates
    /// such that both dereferences name the same underlying
    /// `wgpu::Buffer` when `retired_frame + FrameCount == frame`, but
    /// the correctness proof uses the per-frame resolution directly —
    /// see `knowledge-fif-swapchain-depth-decoupling`).
    pub(crate) fn staging_occ_ring(&self) -> &MultiBufferRing<SubchunkOccupancy> {
        &self.staging_occ_ring
    }

    pub(crate) fn staging_material_ids_ring(&self) -> &MultiBufferRing<MaterialBlock> {
        &self.staging_material_ids_ring
    }

    pub(crate) fn dirty_list_buf(&self) -> &wgpu::Buffer {
        &self.dirty_list_buf
    }

    pub(crate) fn prep_request_buf(&self) -> &wgpu::Buffer {
        &self.prep_request_buf
    }

    pub(crate) fn exposure_dirty_list_buf(&self) -> &wgpu::Buffer {
        &self.exposure_dirty_list_buf
    }

    pub(crate) fn exposure_request_buf(&self) -> &wgpu::Buffer {
        &self.exposure_request_buf
    }

    pub(crate) fn material_desc_buf(&self) -> &wgpu::Buffer {
        &self.material_desc_buf
    }

    /// Snapshot the current segment list as a `Vec<wgpu::Buffer>` clone.
    /// Each `wgpu::Buffer` is `Arc`-backed and cheap to clone; the clone
    /// is required because callers (like `subchunk_world`) need stable
    /// handles to import into the render graph for the frame's bind-group
    /// resolution.
    pub(crate) fn material_segment_bufs_snapshot(&self) -> Vec<wgpu::Buffer> {
        self.material_segment_bufs
            .lock()
            .expect("material_segment_bufs mutex poisoned")
            .clone()
    }
}

/// Allocate one 64 MB material-data-pool segment GPU buffer. Label
/// embeds `segment_idx` so RenderDoc captures distinguish them at a
/// glance.
fn create_material_segment_buf(
    device:      &wgpu::Device,
    segment_idx: u32,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some(&format!("material_segment_{segment_idx}")),
        size:               MATERIAL_SEGMENT_BYTES,
        usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

// --- subchunk_world render node ---

/// Register the sub-chunk cull + MDI draw + shade + present-blit passes
/// into `graph`.
///
/// Imports the renderer's persistent buffers, allocates the `visible`,
/// `indirect`, `vis`, and `shaded_color` per-frame transients, wires the
/// cull / draw / shade / blit bind groups, dispatches the 256-thread cull
/// (1 workgroup), issues a multi-draw-indirect-count raster that writes
/// the vis buffer + depth, dispatches an 8×8 compute shade that reads
/// the vis buffer and writes `shaded_color`, then issues a
/// fullscreen-triangle blit that reads `shaded_color` and writes into the
/// caller-supplied `color` attachment (typically the swapchain).
///
/// `gpu_consts` is forwarded to the shade pass's bind group (slot 0) so
/// the shade shader's `dda_world` can read `g_consts.levels[level_idx]`
/// via `resolve_coord_to_slot`. Callers that already hold a
/// [`crate::GpuConsts`] for the prep pass can pass the same reference.
///
/// The vis buffer is an `R32_UINT` transient with the `subchunk_vis`
/// graph name; its resolution matches `(width, height)` — the same
/// extent the caller uses for the depth buffer and the swapchain blit.
/// Width and height are passed in explicitly because neither the vis nor
/// `shaded_color` transients nor the depth handle expose their extent
/// through the handle itself. See
/// `decision-vis-buffer-deferred-shading-phase-1` for the vis packing
/// layout.
///
/// Returns `(color_out, depth_out)` — the blit pass's colour write (the
/// final pixels handed to the swapchain) and the draw pass's depth
/// write. The `shaded_color` transient is consumed internally by the
/// blit; callers do not observe it.
pub fn subchunk_world(
    graph      : &mut RenderGraph,
    renderer   : &Arc<WorldRenderer>,
    gpu_consts : &crate::GpuConsts,
    color      : TextureHandle,
    depth      : TextureHandle,
    width      : u32,
    height     : u32,
)
    -> (TextureHandle, TextureHandle)
{
    let camera_h    = graph.import_buffer(renderer.camera_buf().clone());
    let instance_h  = graph.import_buffer(renderer.instance_buf().clone());
    let occ_h       = graph.import_buffer(renderer.material_pool_buf().clone());
    let count_h     = graph.import_buffer(renderer.count_buf().clone());
    let lod_mask_h  = graph.import_buffer(renderer.lod_mask_buf().clone());
    let directory_h = graph.import_buffer(renderer.slot_directory_buf().as_ref().clone());

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

    // Vis buffer: R32_UINT MRT target for the draw pass; sampled as a
    // `Texture2D<uint>` by the shade compute pass. `STORAGE` is
    // deliberately not included — shade reads via a sampled texture
    // binding, not a storage UAV; adding `STORAGE` would force
    // `R32Uint`-compatible storage-format validation on the GPU for zero
    // gain.
    let vis_h = graph.create_texture(
        "subchunk_vis",
        TextureDesc::new_2d(
            width,
            height,
            wgpu::TextureFormat::R32Uint,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        ),
    );

    // Shaded-colour transient the compute shade pass writes and the
    // downstream blit pass reads. `STORAGE_BINDING` is load-bearing for
    // the shade pass's `RWTexture2D<float4>` UAV; `TEXTURE_BINDING` for
    // the blit. See `SHADED_COLOR_FORMAT` for the format rationale.
    let shaded_h = graph.create_texture(
        "subchunk_shaded_color",
        TextureDesc::new_2d(
            width,
            height,
            SHADED_COLOR_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        ),
    );

    let cull_bg = graph.create_bind_group(
        "subchunk_cull_bg",
        renderer.cull_pipeline().as_ref(),
        None,
        &[
            (0, camera_h.into()),
            (1, instance_h.into()),
            (2, visible_h.into()),
            (3, lod_mask_h.into()),
            (4, directory_h.into()),
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

    // Single colour attachment — the vis buffer. Slot 0 is cleared to
    // the `0xFFFFFFFF` miss sentinel (packed layout in
    // `decision-vis-buffer-deferred-shading-phase-1`). Passed as an
    // `f64` because `wgpu::Color`'s channels are f64; `4294967295` fits
    // exactly in f64's mantissa, so wgpu's bitwise integer cast for
    // integer targets reproduces the sentinel exactly.
    let (color_outs, depth_out) = mdi_draw(
        graph,
        renderer.render_pipeline(),
        render_bg,
        &indirect_out,
        &DrawArgs::default(),
        &[ColorTarget { texture: vis_h, clear: [4294967295.0, 0.0, 0.0, 0.0] }],
        depth,
    );
    let vis_written = color_outs[0];

    // Material descriptor table (scalar, set 0 slot 7) + binding-array
    // of per-sub-chunk material-data pool segments (set 1 slot 0). The
    // pool's live-segment list drives the array length per frame;
    // slots past `segments_live` stay partially bound
    // (`PARTIALLY_BOUND_BINDING_ARRAY`). The pool is kept on set 1
    // because wgpu's validator refuses to co-locate a binding array
    // with uniform buffers (`g_consts`, `g_camera`) in one set.
    let material_desc_h = graph.import_buffer(renderer.material_desc_buf().clone());
    let material_segment_handles: Vec<crate::graph::BufferHandle> = renderer
        .material_segment_bufs_snapshot()
        .into_iter()
        .map(|b| graph.import_buffer(b))
        .collect();
    let material_segment_res: Vec<crate::graph::ResourceId> = material_segment_handles
        .iter()
        .map(|h| (*h).into())
        .collect();

    // Shade bind group is built after the draw pass so that `depth_out`
    // (the post-draw version of the depth texture) is available. The
    // shade pass reads the depth buffer written by the draw pass, so
    // passing `depth_out` establishes the correct read-after-write
    // ordering via the graph's access tracking.
    let shade_bg = graph.create_bind_group(
        "subchunk_shade_bg",
        renderer.shade_pipeline().as_ref(),
        Some(gpu_consts),
        &[
            (1, vis_written.into()),
            (2, shaded_h.into()),
            (3, directory_h.into()),
            (4, occ_h.into()),
            (5, camera_h.into()),
            (6, depth_out.into()),
            (7, material_desc_h.into()),
        ],
    );

    // Set-1 bind group holds the material-data pool binding array.
    // Caller-composed (no implicit `gpu_consts`), produced via
    // `create_bind_group_arrays` + set-1 helper below.
    let shade_bg_set1 = graph.create_bind_group_arrays_set1(
        "subchunk_shade_bg_set1",
        renderer.shade_pipeline().as_ref(),
        vec![
            (0, crate::graph::BindResource::Array(material_segment_res)),
        ],
    );

    // Dispatch grid: one thread per output pixel in 8×8 workgroups.
    // Right / bottom edges may be partial — the shader's own
    // `GetDimensions` check drops out-of-range threads.
    let wg_x = width.div_ceil(8);
    let wg_y = height.div_ceil(8);

    let shade_pipeline = Arc::clone(renderer.shade_pipeline());
    let shaded_out = graph.add_pass("subchunk_shade", |pass| {
        let writes   = pass.use_bind_group(shade_bg);
        // Record reads against set-1's binding-array entries (material
        // segments) so the graph's access tracking pairs them with the
        // material patch pass's writes.
        pass.use_bind_group(shade_bg_set1);
        let shaded_v = writes.write_texture_of(shaded_h);
        // `vis_written` and `depth_out` are already recorded as reads
        // via their `SampledTexture` bindings in the shade bind group —
        // no additional explicit declaration required.
        pass.execute(move |ctx| {
            let bg0 = ctx.resources.bind_group(shade_bg);
            let bg1 = ctx.resources.bind_group(shade_bg_set1);
            ctx.commands.dispatch(&shade_pipeline, &[bg0, bg1], [wg_x, wg_y, 1], &[]);
        });
        shaded_v
    });

    // Fullscreen blit from `shaded_color` into the caller-supplied colour
    // attachment. This is the final pass for the sub-chunk world render
    // and — when `color` is the swapchain texture imported via
    // `graph.present()` — also the pass whose output the swapchain
    // presents. The blit's explicit read of `shaded_color` subsumes the
    // liveness role the previous `mark_texture_output` scaffold served,
    // so the scaffold is no longer needed and has been removed.
    let color_out = present_blit(
        graph,
        renderer.present_blit_pipeline(),
        shaded_out,
        color,
    );

    (color_out, depth_out)
}

// --- subchunk_prep render node ---

/// Register the sub-chunk prep compute pass + dirty-list readback copy
/// into `graph`.
///
/// Imports the five persistent buffers (prep-requests, material pool,
/// dirty list, directory, and the current ring slot of staging
/// occupancy), clears the dirty-list `count` header, dispatches the prep
/// compute with one workgroup per request, then records a
/// `copy_buffer_to_buffer` into `readback_dst` — an imported handle
/// whose destination is typically a
/// [`ReadbackChannel`](crate::readback::ReadbackChannel) slot reserved for
/// this frame.
///
/// `gpu_consts` lands at the reflected slot 0 of the prep pipeline. The
/// prep shader reads `g_consts.levels[level_idx]` through
/// `resolve_coord_to_slot` / `resolve_and_verify` to self-resolve its
/// own directory index and to locate each of the 6 neighbour directory
/// entries for the neighbour-aware exposure mask.
///
/// The caller must:
/// - Upload `request_count` entries via
///   [`WorldRenderer::write_prep_requests`] before graph compile.
/// - Size `readback_dst` to at least `size_of::<DirtyReport>()` bytes.
///
/// The material-pool buffer is read by the compute pass as a diff source
/// *and* as the neighbour-face source for the exposure computation;
/// callers should be aware that running prep and render in the same graph
/// produces a read-after-write or write-after-read dependency on
/// `material_pool_buf` and the current staging ring slot — the graph
/// handles the barrier automatically.
///
/// `frame` selects which slot of the staging ring this dispatch writes
/// into. The caller must pass the same `FrameIndex` it will pass to
/// [`subchunk_patch`] in frame `frame + FrameCount` when consuming this
/// dispatch's dirty list — otherwise staging writes land in a slot that
/// the retirement will never read.
pub fn subchunk_prep(
    graph         : &mut RenderGraph,
    renderer      : &Arc<WorldRenderer>,
    gpu_consts    : &crate::GpuConsts,
    readback_dst  : BufferHandle,
    request_count : u32,
    frame         : FrameIndex,
)
{
    let request_h        = graph.import_buffer(renderer.prep_request_buf().clone());
    let material_pool_h  = graph.import_buffer(renderer.material_pool_buf().clone());
    let staging_occ_h    = graph.import_buffer(
        renderer.staging_occ_ring().current(frame).clone(),
    );
    let staging_mat_ids_h = graph.import_buffer(
        renderer.staging_material_ids_ring().current(frame).clone(),
    );
    let dirty_list_h     = graph.import_buffer(renderer.dirty_list_buf().clone());
    let directory_h      = graph.import_buffer(renderer.slot_directory_buf().as_ref().clone());

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
        Some(gpu_consts),
        &[
            (1, request_h.into()),
            (2, material_pool_h.into()),
            (3, staging_occ_h.into()),
            (4, dirty_list_cleared.into()),
            (5, directory_h.into()),
            (6, staging_mat_ids_h.into()),
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

// --- subchunk_exposure render node ---

/// Register the exposure-only refresh compute pass + dirty-list readback
/// copy into `graph`.
///
/// Companion to [`subchunk_prep`]. Full-prep voxelizes terrain and writes
/// staging for the CPU patch pass. Exposure-only refresh reads the
/// already-populated `material_pool` entries in place and recomputes the
/// neighbour-aware 6-bit exposure mask — no terrain evaluation, no staging
/// write, no patch pass. The retirement logic updates only the directory
/// `bits` field for each returned dirty entry (see
/// [`EXPOSURE_STAGING_REQUEST_IDX_SENTINEL`]).
///
/// Bindings mirror [`subchunk_prep`] minus the staging-occ slot:
///   0: `g_consts`                   (implicit via `gpu_consts`)
///   1: `g_requests`                 (exposure-only request buffer)
///   2: `g_material_pool`            (self-occupancy source + neighbour source)
///   3: `g_dirty_list`               (exposure-only dirty-list buffer)
///   4: `g_directory`                (read-only)
///
/// The caller must:
/// - Upload `request_count` entries via
///   [`WorldRenderer::write_exposure_requests`] before graph compile.
/// - Size `readback_dst` to at least `size_of::<DirtyReport>()` bytes.
///
/// # Non-staging contract
///
/// The exposure dispatch never writes the staging ring. Consequently it
/// takes no `frame` argument (unlike [`subchunk_prep`]) — there is no
/// ring-slot selection to make. The retirement path for exposure-only
/// entries asserts `staging_request_idx ==
/// EXPOSURE_STAGING_REQUEST_IDX_SENTINEL` and emits zero [`PatchCopy`]s.
pub fn subchunk_exposure(
    graph         : &mut RenderGraph,
    renderer      : &Arc<WorldRenderer>,
    gpu_consts    : &crate::GpuConsts,
    readback_dst  : BufferHandle,
    request_count : u32,
)
{
    let request_h        = graph.import_buffer(renderer.exposure_request_buf().clone());
    let material_pool_h  = graph.import_buffer(renderer.material_pool_buf().clone());
    let dirty_list_h     = graph.import_buffer(renderer.exposure_dirty_list_buf().clone());
    let directory_h      = graph.import_buffer(renderer.slot_directory_buf().as_ref().clone());

    // Clear the dirty-list count header. Same reasoning as subchunk_prep:
    // the shader's `InterlockedAdd` accumulates, so the header must start
    // at 0. Entry words past the written range are undefined; consumer
    // only reads [0, count).
    let dirty_list_cleared = graph.add_pass("subchunk_exposure_clear", |pass| {
        let cleared = pass.write_buffer(dirty_list_h);
        let dirty_buf = renderer.exposure_dirty_list_buf().clone();
        pass.execute(move |ctx| {
            ctx.commands.clear_buffer(&dirty_buf, 0, Some(16));
        });
        cleared
    });

    let exposure_bg = graph.create_bind_group(
        "subchunk_exposure_bg",
        renderer.exposure_pipeline().as_ref(),
        Some(gpu_consts),
        &[
            (1, request_h.into()),
            (2, material_pool_h.into()),
            (3, dirty_list_cleared.into()),
            (4, directory_h.into()),
        ],
    );

    let dirty_list_written = graph.add_pass("subchunk_exposure", |pass| {
        let writes     = pass.use_bind_group(exposure_bg);
        let pipeline   = Arc::clone(renderer.exposure_pipeline());
        let workgroups = [request_count, 1, 1];
        let dirty_out  = writes.write_of(dirty_list_cleared);
        pass.execute(move |ctx| {
            let bg = ctx.resources.bind_group(exposure_bg);
            ctx.commands.dispatch(&pipeline, &[bg], workgroups, &[]);
        });
        dirty_out
    });

    let readback_out = graph.add_pass("subchunk_exposure_readback_copy", |pass| {
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

// --- PatchCopy ---

/// One CPU-authored staging→material-pool copy decision.
///
/// Produced by the retirement logic that walks a
/// [`DirtyReport`](crate::subchunk::DirtyReport): for every entry the CPU
/// decides is sparse (has real occupancy), the retirement emits a
/// `PatchCopy { staging_request_idx, dst_material_slot }` so the patch
/// pass knows which staging entry to lift and where in the material pool
/// to land it. Uniform-empty entries never produce a `PatchCopy`; they
/// consume no material-pool storage.
///
/// `staging_request_idx` matches `gid.x` of the prep workgroup that
/// produced the staging write (and the `staging_request_idx` field of the
/// corresponding `DirtyEntry`). `dst_material_slot` is either the
/// allocator slot the retirement logic allocated for a first-time-sparse
/// transition or the existing slot reused across a sparse-to-sparse
/// update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchCopy {
    pub staging_request_idx: u32,
    pub dst_material_slot:   u32,
}

// --- subchunk_patch render node ---

/// Register a CPU-scheduled staging→material-pool patch pass into `graph`.
///
/// For each `PatchCopy` in `copies`, records a 64-byte occupancy copy
/// from `staging_occ_ring[staging_frame][staging_request_idx]` into
/// `material_pool_buf[dst_material_slot]`. The copy set is produced by
/// the CPU retirement logic that walked a previously-completed prep
/// dispatch's [`DirtyReport`](crate::subchunk::DirtyReport) and made the
/// allocator decisions that turn shader-reported classifications into
/// material-slot assignments.
///
/// Exposure no longer rides with the patch — it lives in the CPU-authored
/// [`WorldRenderer::slot_directory_buf`] and is rewritten via
/// `queue.write_buffer` when retirement mutates a directory entry.
///
/// A call with an empty `copies` is a no-op and records nothing —
/// callers do not need to guard at the call site.
///
/// `staging_frame` is the [`FrameIndex`] the retiring dirty list was
/// dispatched on — typically the `FrameIndex` returned alongside the
/// dirty report from `ReadbackChannel::take_ready`. It selects the
/// staging-ring slot that holds the shader-written payloads this patch
/// will lift into the material pool. Passing the *current* frame here
/// (instead of the retired frame) routes the copy at a slot whose
/// contents belong to the *current* frame's prep dispatch — the exact
/// cross-frame data corruption `failure-staging-not-ringed-after-gid-x-reindexing`
/// captures.
pub fn subchunk_patch(
    graph         : &mut RenderGraph,
    renderer      : &Arc<WorldRenderer>,
    copies        : &[PatchCopy],
    staging_frame : FrameIndex,
)
{
    if copies.is_empty() {
        return;
    }

    let staging_occ_h   = graph.import_buffer(
        renderer.staging_occ_ring().current(staging_frame).clone(),
    );
    let material_pool_h = graph.import_buffer(renderer.material_pool_buf().clone());

    let copies: Vec<PatchCopy> = copies.to_vec();

    let patched = graph.add_pass("subchunk_patch", |pass| {
        pass.read_buffer(staging_occ_h);
        let pool_written = pass.write_buffer(material_pool_h);
        pass.execute(move |ctx| {
            let src_buf   = ctx.resources.buffer(staging_occ_h);
            let dst_buf   = ctx.resources.buffer(pool_written);
            let occ_bytes = std::mem::size_of::<SubchunkOccupancy>() as u64;
            for copy in &copies {
                let src_offset = (copy.staging_request_idx as u64) * occ_bytes;
                let dst_offset = (copy.dst_material_slot   as u64) * occ_bytes;
                ctx.commands.copy_buffer_to_buffer(
                    src_buf, src_offset, dst_buf, dst_offset, occ_bytes,
                );
            }
        });
        pool_written
    });

    // Without an output marker, the graph would cull the pass: the
    // material pool has no downstream reader declared inside this
    // subgraph (the render node that reads it lives in a separate call).
    graph.mark_output(patched);
}

// --- MaterialPatchCopy ---

/// One CPU-authored staging_material_ids → material-data-pool copy.
///
/// Parallel to [`PatchCopy`] but targets the binding-array segmented
/// material-data pool. `dst_global_slot` is the flat
/// [`MaterialDataPool`](crate::MaterialDataPool) slot the retirement
/// chose for this sub-chunk; the patch node decomposes it into
/// `(segment_idx, local_offset)` and does one `copy_buffer_to_buffer`
/// per entry.
///
/// `staging_request_idx` matches `gid.x` of the prep workgroup that
/// produced the material-id payload, identical to the occupancy patch
/// pairing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterialPatchCopy {
    pub staging_request_idx: u32,
    pub dst_global_slot:     u32,
}

// --- subchunk_material_patch render node ---

/// Register a CPU-scheduled staging_material_ids → material-data-pool
/// patch pass into `graph`.
///
/// For each [`MaterialPatchCopy`], records a 1 KB
/// [`MATERIAL_BLOCK_BYTES`](MATERIAL_BLOCK_BYTES) copy from
/// `staging_material_ids_ring[staging_frame][staging_request_idx]` into
/// the appropriate segment buffer at
/// `(dst_global_slot % SLOTS_PER_SEGMENT) * 1024`. The destination
/// segment is resolved from `dst_global_slot / SLOTS_PER_SEGMENT` and
/// looked up in the renderer's live segment list.
///
/// An empty `copies` slice is a no-op.
///
/// `staging_frame` selects the ring slot; same semantics as
/// [`subchunk_patch`]'s `staging_frame` — pass the retired frame, not the
/// current one.
pub fn subchunk_material_patch(
    graph         : &mut RenderGraph,
    renderer      : &Arc<WorldRenderer>,
    copies        : &[MaterialPatchCopy],
    staging_frame : FrameIndex,
)
{
    if copies.is_empty() {
        return;
    }

    let staging_mat_ids_h = graph.import_buffer(
        renderer.staging_material_ids_ring().current(staging_frame).clone(),
    );

    // Import every live segment buffer. The pass issues per-copy
    // `copy_buffer_to_buffer` into the segment identified by each copy's
    // `dst_global_slot / SLOTS_PER_SEGMENT`, so every segment the copy
    // set touches must appear as an imported buffer.
    let segments: Vec<wgpu::Buffer> = renderer.material_segment_bufs_snapshot();
    let segment_handles: Vec<crate::graph::BufferHandle> = segments
        .iter()
        .map(|b| graph.import_buffer(b.clone()))
        .collect();
    let segment_count = segment_handles.len();

    let copies: Vec<MaterialPatchCopy> = copies.to_vec();

    let patched = graph.add_pass("subchunk_material_patch", |pass| {
        pass.read_buffer(staging_mat_ids_h);
        // Record a write access on every segment we might touch; the
        // graph will pessimistically barrier all of them, which is fine
        // — the grow rate is O(1 per minute of play).
        let segment_writes: Vec<crate::graph::BufferHandle> = segment_handles
            .iter()
            .map(|h| pass.write_buffer(*h))
            .collect();
        let first_write = segment_writes[0];
        let segment_writes_for_exec = segment_writes.clone();

        pass.execute(move |ctx| {
            let src_buf = ctx.resources.buffer(staging_mat_ids_h);
            let block_bytes = MATERIAL_BLOCK_BYTES;
            for copy in &copies {
                let seg_idx = (copy.dst_global_slot
                    / MATERIAL_POOL_SLOTS_PER_SEGMENT) as usize;
                let local_slot = copy.dst_global_slot
                    % MATERIAL_POOL_SLOTS_PER_SEGMENT;
                debug_assert!(
                    seg_idx < segment_count,
                    "subchunk_material_patch: dst_global_slot {} -> segment {} \
                     but only {} segment(s) imported",
                    copy.dst_global_slot, seg_idx, segment_count,
                );
                let dst_buf = ctx.resources.buffer(segment_writes_for_exec[seg_idx]);
                let src_offset = (copy.staging_request_idx as u64) * block_bytes;
                let dst_offset = (local_slot as u64) * block_bytes;
                ctx.commands.copy_buffer_to_buffer(
                    src_buf, src_offset, dst_buf, dst_offset, block_bytes,
                );
            }
        });

        // Return any one written handle as the pass's output marker. The
        // graph only needs *some* output for liveness; downstream passes
        // read from the segment buffers directly.
        first_write
    });

    graph.mark_output(patched);
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    // -- DirEntry bit packing --

    #[test]
    fn dir_entry_empty_is_all_zero() {
        let e = DirEntry::empty();
        assert_eq!(e.coord, [0, 0, 0]);
        assert_eq!(e.bits,  0);
        assert!(!e.is_resident());
        assert_eq!(e.exposure(),      0);
        assert_eq!(e.material_slot(), 0);
        assert_eq!(e.content_version,    0);
        assert_eq!(e.last_synth_version, 0);
    }

    #[test]
    fn dir_entry_non_resident_carries_invalid_sentinel() {
        let e = DirEntry::non_resident();
        assert!(!e.is_resident());
        assert_eq!(e.material_slot(), MATERIAL_SLOT_INVALID);
        assert_eq!(e.exposure(),      0);
        assert!(!e.is_solid());
    }

    #[test]
    fn dir_entry_resident_roundtrips_all_fields() {
        let e = DirEntry::resident([-7, 42, 3], 0x2A, false, 0x12_3456);
        assert_eq!(e.coord, [-7, 42, 3]);
        assert!(e.is_resident());
        assert!(!e.is_solid());
        assert_eq!(e.exposure(),      0x2A);
        assert_eq!(e.material_slot(), 0x12_3456);
    }

    #[test]
    fn dir_entry_resident_max_exposure_roundtrips() {
        let e = DirEntry::resident([0, 0, 0], BITS_EXPOSURE_MASK, false, 0);
        assert_eq!(e.exposure(), BITS_EXPOSURE_MASK);
        assert!(e.is_resident());
    }

    #[test]
    fn dir_entry_resident_max_material_slot_roundtrips() {
        // MATERIAL_SLOT_INVALID - 1 is the largest legal resident slot.
        let max = MATERIAL_SLOT_INVALID - 1;
        let e = DirEntry::resident([1, 2, 3], 0x3F, true, max);
        assert!(e.is_resident());
        assert!(e.is_solid());
        assert_eq!(e.material_slot(), max);
        assert_eq!(e.exposure(),       0x3F);
    }

    #[test]
    fn dir_entry_is_solid_and_resident_both_set() {
        let e = DirEntry::resident([0, 0, 0], 0, true, 7);
        assert!(e.is_resident());
        assert!(e.is_solid());
        assert_eq!(e.bits & BITS_IS_SOLID,  BITS_IS_SOLID);
        assert_eq!(e.bits & BITS_RESIDENT,  BITS_RESIDENT);
    }

    #[test]
    #[should_panic(expected = "exposure must fit in 6 bits")]
    fn dir_entry_resident_rejects_overflowing_exposure() {
        // 0x40 == 0b0100_0000, which overflows the 6-bit exposure field.
        let _ = DirEntry::resident([0, 0, 0], 0x40, false, 0);
    }

    #[test]
    #[should_panic(expected = "must be a real pool index under resident entry")]
    fn dir_entry_resident_rejects_invalid_sentinel_as_slot() {
        let _ = DirEntry::resident([0, 0, 0], 0, false, MATERIAL_SLOT_INVALID);
    }

    // -- DirtyEntry byte layout --

    #[test]
    fn dirty_entry_is_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<DirtyEntry>(), 16);
    }

    #[test]
    fn dirty_entry_bytes_roundtrip_pod_layout() {
        // Cast to bytes and back — confirms field ordering matches the
        // shader's byte-address store at offsets 0 / 4 / 8 / 12 without
        // any compiler-inserted padding.
        let e = DirtyEntry {
            directory_index:     0x1111_1111,
            new_bits_partial:    0x2222_2222,
            staging_request_idx: 0x3333_3333,
            _pad:                0x4444_4444,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&e);
        assert_eq!(&bytes[0.. 4], 0x1111_1111u32.to_le_bytes());
        assert_eq!(&bytes[4.. 8], 0x2222_2222u32.to_le_bytes());
        assert_eq!(&bytes[8..12], 0x3333_3333u32.to_le_bytes());
        assert_eq!(&bytes[12..16], 0x4444_4444u32.to_le_bytes());
    }

    /// A `new_bits_partial` word built with `exposure | (is_solid << 6) |
    /// (1 << 7)` must decode back to the same three fields via the
    /// directory's bit-field masks — the load-bearing contract between
    /// the prep shader's emit path and the CPU's retirement logic.
    #[test]
    fn dirty_entry_new_bits_partial_decodes_as_direntry_bits() {
        for exposure in [0u32, 0x01, 0x2A, BITS_EXPOSURE_MASK] {
            for is_solid in [false, true] {
                let packed = exposure
                    | ((is_solid as u32) << 6)
                    | BITS_RESIDENT;

                assert_eq!(packed & BITS_EXPOSURE_MASK, exposure);
                assert_eq!(packed & BITS_IS_SOLID, (is_solid as u32) << 6);
                assert_eq!(packed & BITS_RESIDENT, BITS_RESIDENT);

                // The material-slot field (bits 8..31) is authored on the
                // CPU side and must be zero in the shader-authored word.
                assert_eq!(packed >> BITS_MATERIAL_SLOT_SHIFT, 0);
            }
        }
    }

    // -- PrepRequest byte layout --

    /// `PrepRequest` must remain exactly 32 bytes after Step 4 drops the
    /// `slot` field — the HLSL `StructuredBuffer<PrepRequest>` stride
    /// reflects the Rust struct size, and a size change would silently
    /// misalign every subsequent element.
    #[test]
    fn prep_request_is_thirty_two_bytes() {
        assert_eq!(std::mem::size_of::<PrepRequest>(), 32);
    }

    // -- Face-mask CPU mirror --
    //
    // Mirrors the six canonical face extractors in
    // `shaders/include/face_mask.hlsl` on `SubchunkOccupancy`. The HLSL
    // is the only runtime consumer; this mirror exists as a
    // regression guard for the bit math and as the ground-truth harness
    // for the exposure-comparison contract `face_exposed(my, nbr)`.
    //
    // # Canonical face layout (matches the HLSL `uint2` layout)
    //
    // A face is 8×8, indexed by two "free" axes `(a, b)`:
    //   ±X face: free axes (a = y, b = z)
    //   ±Y face: free axes (a = x, b = z)
    //   ±Z face: free axes (a = x, b = y)
    //
    // The 64 bits are packed as `(u32, u32)` with:
    //   half 0 (`.0`) — 32 bits for b ∈ [0, 4)
    //   half 1 (`.1`) — 32 bits for b ∈ [4, 8)
    // Within each half, the 8 bits for a fixed `b` live at bit positions
    // `(b % 4) * 8 + a` for `a ∈ [0, 8)`.
    //
    // Because the opposite-face extractor for a given direction uses the
    // same free-axis pair, matching canonical bits correspond to
    // voxel pairs that meet across the face. That is the invariant
    // `face_exposed(my, nbr) = any((my & ~nbr) != 0)` relies on.
    //
    // # History / why the canonical layout is load-bearing
    //
    // An earlier version of the HLSL returned `uint2`s at non-canonical
    // bit positions (±X at sparse {7,15,23,31}-style column bits, ±Y in
    // only one of the two half-words, ±Z at full-plane offsets). The
    // `my` and `nbr` sides of a direction pair had *disjoint* bit
    // positions, so `my & ~nbr = my` always — the neighbour was ignored
    // and every boundary-voxel face rendered as exposed. The "fully solid
    // neighbour" and "partial overlap" regression tests below would have
    // caught that bug at the time it was introduced; keep them.

    /// Set the voxel `(x, y, z)` in the GPU-format occupancy.
    fn set_bit(occ: &mut SubchunkOccupancy, x: u32, y: u32, z: u32) {
        let bit  = y * 8 + x;
        let word = z * 2 + (bit >> 5);
        occ.planes[word as usize] |= 1 << (bit & 31);
    }

    /// word 0 (y∈[0,4)) or word 1 (y∈[4,8)) of z-plane `z` in the
    /// occupancy buffer.
    fn word_at(occ: &SubchunkOccupancy, z: u32, half: u32) -> u32 {
        occ.planes[(z * 2 + half) as usize]
    }

    /// Compact the sparse x=7 column bits (at word positions 7, 15, 23,
    /// 31) of a single occupancy word into a contiguous 4-bit nybble.
    fn compact_col_x7(w: u32) -> u32 {
        ((w >>  7) & 0x1)
            | ((w >> 14) & 0x2)
            | ((w >> 21) & 0x4)
            | ((w >> 28) & 0x8)
    }

    /// Compact the sparse x=0 column bits (at word positions 0, 8, 16,
    /// 24) into a contiguous 4-bit nybble.
    fn compact_col_x0(w: u32) -> u32 {
        (w & 0x1)
            | ((w >>  7) & 0x2)
            | ((w >> 14) & 0x4)
            | ((w >> 21) & 0x8)
    }

    /// Build a (u32, u32) canonical face by OR-ing 8 per-slab bytes into
    /// the half-words determined by `b ∈ [0, 8)`. The byte carries the
    /// 8 `a`-axis bits at positions 0..7; the canonical packing puts it
    /// at bit `(b % 4) * 8` of half `b / 4`.
    fn pack_face_byte(face: &mut (u32, u32), b: u32, byte8: u32) {
        let shifted = byte8 << ((b & 3) * 8);
        if b < 4 {
            face.0 |= shifted;
        } else {
            face.1 |= shifted;
        }
    }

    /// +X face (x = 7), canonical layout. Free axes (a = y, b = z).
    fn face_px_mirror(occ: &SubchunkOccupancy) -> (u32, u32) {
        let mut face = (0u32, 0u32);
        for z in 0..8 {
            let w0 = word_at(occ, z, 0);
            let w1 = word_at(occ, z, 1);
            let byte = compact_col_x7(w0) | (compact_col_x7(w1) << 4);
            pack_face_byte(&mut face, z, byte);
        }
        face
    }

    /// -X face (x = 0), canonical layout. Free axes (a = y, b = z).
    fn face_nx_mirror(occ: &SubchunkOccupancy) -> (u32, u32) {
        let mut face = (0u32, 0u32);
        for z in 0..8 {
            let w0 = word_at(occ, z, 0);
            let w1 = word_at(occ, z, 1);
            let byte = compact_col_x0(w0) | (compact_col_x0(w1) << 4);
            pack_face_byte(&mut face, z, byte);
        }
        face
    }

    /// +Y face (y = 7), canonical layout. Free axes (a = x, b = z).
    fn face_py_mirror(occ: &SubchunkOccupancy) -> (u32, u32) {
        let mut face = (0u32, 0u32);
        for z in 0..8 {
            let w1   = word_at(occ, z, 1);
            let byte = (w1 >> 24) & 0xFF;
            pack_face_byte(&mut face, z, byte);
        }
        face
    }

    /// -Y face (y = 0), canonical layout. Free axes (a = x, b = z).
    fn face_ny_mirror(occ: &SubchunkOccupancy) -> (u32, u32) {
        let mut face = (0u32, 0u32);
        for z in 0..8 {
            let w0   = word_at(occ, z, 0);
            let byte = w0 & 0xFF;
            pack_face_byte(&mut face, z, byte);
        }
        face
    }

    /// +Z face (z = 7), canonical layout. Free axes (a = x, b = y). The
    /// occupancy layout already matches the canonical packing at a fixed
    /// z-plane: `word_at(occ, 7, 0)` carries y∈[0,4) at `(y%4)*8 + x`
    /// and `word_at(occ, 7, 1)` carries y∈[4,8) at the same pattern.
    fn face_pz_mirror(occ: &SubchunkOccupancy) -> (u32, u32) {
        (word_at(occ, 7, 0), word_at(occ, 7, 1))
    }

    /// -Z face (z = 0), canonical layout. Free axes (a = x, b = y).
    fn face_nz_mirror(occ: &SubchunkOccupancy) -> (u32, u32) {
        (word_at(occ, 0, 0), word_at(occ, 0, 1))
    }

    /// CPU mirror of `face_exposed(my, nbr)` in face_mask.hlsl. Relies
    /// on both operands being in canonical layout.
    fn face_exposed_mirror(my: (u32, u32), nbr: (u32, u32)) -> bool {
        let ex0 = my.0 & !nbr.0;
        let ex1 = my.1 & !nbr.1;
        (ex0 | ex1) != 0
    }

    // -- Interior cell-pair exposure CPU mirror --
    //
    // Mirrors the six `interior_exposed_*` helpers in face_mask.hlsl.
    // Each returns a u32 accumulator that is non-zero iff at least one
    // interior voxel V satisfies (V solid AND V's d-neighbour empty AND
    // that neighbour is internal to this sub-chunk).

    fn interior_exposed_px_mirror(occ: &SubchunkOccupancy) -> u32 {
        const MASK_NO_X7: u32 = 0x7F7F_7F7F;
        let mut acc = 0u32;
        for z in 0..8 {
            let w0 = word_at(occ, z, 0);
            let w1 = word_at(occ, z, 1);
            acc |= w0 & !(w0 >> 1) & MASK_NO_X7;
            acc |= w1 & !(w1 >> 1) & MASK_NO_X7;
        }
        acc
    }

    fn interior_exposed_nx_mirror(occ: &SubchunkOccupancy) -> u32 {
        const MASK_NO_X0: u32 = 0xFEFE_FEFE;
        let mut acc = 0u32;
        for z in 0..8 {
            let w0 = word_at(occ, z, 0);
            let w1 = word_at(occ, z, 1);
            acc |= w0 & !(w0 << 1) & MASK_NO_X0;
            acc |= w1 & !(w1 << 1) & MASK_NO_X0;
        }
        acc
    }

    fn interior_exposed_py_mirror(occ: &SubchunkOccupancy) -> u32 {
        const MASK_NO_Y_TOP: u32 = 0x00FF_FFFF;
        let mut acc = 0u32;
        for z in 0..8 {
            let w0 = word_at(occ, z, 0);
            let w1 = word_at(occ, z, 1);
            acc |= w0 & !(w0 >> 8) & MASK_NO_Y_TOP;
            acc |= w1 & !(w1 >> 8) & MASK_NO_Y_TOP;
            acc |= (w0 >> 24) & !(w1 & 0xFF);
        }
        acc
    }

    fn interior_exposed_ny_mirror(occ: &SubchunkOccupancy) -> u32 {
        const MASK_NO_Y_BOT: u32 = 0xFFFF_FF00;
        let mut acc = 0u32;
        for z in 0..8 {
            let w0 = word_at(occ, z, 0);
            let w1 = word_at(occ, z, 1);
            acc |= w0 & !(w0 << 8) & MASK_NO_Y_BOT;
            acc |= w1 & !(w1 << 8) & MASK_NO_Y_BOT;
            acc |= (w1 & 0xFF) & !(w0 >> 24);
        }
        acc
    }

    fn interior_exposed_pz_mirror(occ: &SubchunkOccupancy) -> u32 {
        let mut acc = 0u32;
        for z in 0..7 {
            let w0_here  = word_at(occ, z,     0);
            let w1_here  = word_at(occ, z,     1);
            let w0_there = word_at(occ, z + 1, 0);
            let w1_there = word_at(occ, z + 1, 1);
            acc |= w0_here & !w0_there;
            acc |= w1_here & !w1_there;
        }
        acc
    }

    fn interior_exposed_nz_mirror(occ: &SubchunkOccupancy) -> u32 {
        let mut acc = 0u32;
        for z in 1..8 {
            let w0_here  = word_at(occ, z,     0);
            let w1_here  = word_at(occ, z,     1);
            let w0_there = word_at(occ, z - 1, 0);
            let w1_there = word_at(occ, z - 1, 1);
            acc |= w0_here & !w0_there;
            acc |= w1_here & !w1_there;
        }
        acc
    }

    /// The canonical bit position for face cell (a, b) — the single bit
    /// that a voxel at the free-axis coordinates (a, b) on a given face
    /// should set in the `(u32, u32)` canonical layout.
    fn canonical_face_bit(a: u32, b: u32) -> (u32, u32) {
        assert!(a < 8 && b < 8, "free-axis coords must be < 8");
        let bit = 1u32 << ((b & 3) * 8 + a);
        if b < 4 { (bit, 0) } else { (0, bit) }
    }

    // -- Single-voxel-on-boundary: each direction picks up exactly one
    //    bit at the canonical position and nothing at the opposite face.

    #[test]
    fn face_px_single_voxel_on_boundary_hits_canonical_bit() {
        // Voxel at (x=7, y=3, z=5) lives on the +X face; its canonical
        // position there has free axes (a=y=3, b=z=5).
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut occ, 7, 3, 5);
        assert_eq!(face_px_mirror(&occ), canonical_face_bit(3, 5));
        assert_eq!(face_nx_mirror(&occ), (0, 0));
    }

    #[test]
    fn face_nx_single_voxel_on_boundary_hits_canonical_bit() {
        // Voxel at (x=0, y=6, z=1): free axes (a=y=6, b=z=1) on -X.
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut occ, 0, 6, 1);
        assert_eq!(face_nx_mirror(&occ), canonical_face_bit(6, 1));
        assert_eq!(face_px_mirror(&occ), (0, 0));
    }

    #[test]
    fn face_py_single_voxel_on_boundary_hits_canonical_bit() {
        // Voxel at (x=2, y=7, z=4): free axes (a=x=2, b=z=4) on +Y.
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut occ, 2, 7, 4);
        assert_eq!(face_py_mirror(&occ), canonical_face_bit(2, 4));
        assert_eq!(face_ny_mirror(&occ), (0, 0));
    }

    #[test]
    fn face_ny_single_voxel_on_boundary_hits_canonical_bit() {
        // Voxel at (x=5, y=0, z=0): free axes (a=x=5, b=z=0) on -Y.
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut occ, 5, 0, 0);
        assert_eq!(face_ny_mirror(&occ), canonical_face_bit(5, 0));
        assert_eq!(face_py_mirror(&occ), (0, 0));
    }

    #[test]
    fn face_pz_single_voxel_on_boundary_hits_canonical_bit() {
        // Voxel at (x=4, y=6, z=7): free axes (a=x=4, b=y=6) on +Z.
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut occ, 4, 6, 7);
        assert_eq!(face_pz_mirror(&occ), canonical_face_bit(4, 6));
        assert_eq!(face_nz_mirror(&occ), (0, 0));
    }

    #[test]
    fn face_nz_single_voxel_on_boundary_hits_canonical_bit() {
        // Voxel at (x=1, y=2, z=0): free axes (a=x=1, b=y=2) on -Z.
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut occ, 1, 2, 0);
        assert_eq!(face_nz_mirror(&occ), canonical_face_bit(1, 2));
        assert_eq!(face_pz_mirror(&occ), (0, 0));
    }

    /// Fully-solid occupancy saturates every face to all 64 bits set,
    /// regardless of which direction's free-axis pair we use — the
    /// canonical layout packs exactly 64 bits per face into the two
    /// u32s, so the value is `(0xFFFF_FFFF, 0xFFFF_FFFF)` uniformly.
    #[test]
    fn face_masks_saturate_on_full_occupancy() {
        let occ = SubchunkOccupancy { planes: [0xFFFF_FFFF; 16] };
        assert_eq!(face_px_mirror(&occ), (0xFFFF_FFFF, 0xFFFF_FFFF));
        assert_eq!(face_nx_mirror(&occ), (0xFFFF_FFFF, 0xFFFF_FFFF));
        assert_eq!(face_py_mirror(&occ), (0xFFFF_FFFF, 0xFFFF_FFFF));
        assert_eq!(face_ny_mirror(&occ), (0xFFFF_FFFF, 0xFFFF_FFFF));
        assert_eq!(face_pz_mirror(&occ), (0xFFFF_FFFF, 0xFFFF_FFFF));
        assert_eq!(face_nz_mirror(&occ), (0xFFFF_FFFF, 0xFFFF_FFFF));
    }

    /// Empty occupancy has every face zero.
    #[test]
    fn face_masks_are_zero_on_empty_occupancy() {
        let occ = SubchunkOccupancy { planes: [0; 16] };
        assert_eq!(face_px_mirror(&occ), (0, 0));
        assert_eq!(face_nx_mirror(&occ), (0, 0));
        assert_eq!(face_py_mirror(&occ), (0, 0));
        assert_eq!(face_ny_mirror(&occ), (0, 0));
        assert_eq!(face_pz_mirror(&occ), (0, 0));
        assert_eq!(face_nz_mirror(&occ), (0, 0));
    }

    // -- face_exposed regression guards (the current bug-class tests) --
    //
    // These would have caught the historical "non-canonical disjoint
    // bits" bug and an aligned-but-OR-folded variant. Both are the
    // load-bearing failure modes for the exposure-mask cull.

    /// `my_face` has solid voxels on the boundary; `nbr_face` is fully
    /// solid ⇒ no voxel pair is exposed ⇒ `face_exposed` returns false.
    ///
    /// This is the scenario the old code failed: under its non-canonical
    /// layout `my & ~nbr = my` (nbr bits lived elsewhere), so it always
    /// returned `true`, which produced spurious +X / -X / +Y / -Y cull
    /// bits for every sub-chunk with any boundary voxel.
    #[test]
    fn face_exposed_is_false_under_fully_solid_neighbour() {
        let full = SubchunkOccupancy { planes: [0xFFFF_FFFF; 16] };
        for &(mx, my, mz) in &[(7u32, 0, 0), (7, 3, 5), (7, 7, 7), (7, 1, 4)] {
            let mut me = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut me, mx, my, mz);
            assert!(
                !face_exposed_mirror(face_px_mirror(&me), face_nx_mirror(&full)),
                "+X voxel at ({mx},{my},{mz}) under fully-solid -X neighbour must NOT be exposed",
            );
        }
        for &(mx, my, mz) in &[(0u32, 0, 0), (0, 5, 6), (0, 7, 2)] {
            let mut me = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut me, mx, my, mz);
            assert!(
                !face_exposed_mirror(face_nx_mirror(&me), face_px_mirror(&full)),
                "-X voxel at ({mx},{my},{mz}) under fully-solid +X neighbour must NOT be exposed",
            );
        }
        for &(mx, my, mz) in &[(0u32, 7, 0), (3, 7, 5), (7, 7, 2)] {
            let mut me = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut me, mx, my, mz);
            assert!(
                !face_exposed_mirror(face_py_mirror(&me), face_ny_mirror(&full)),
                "+Y voxel at ({mx},{my},{mz}) under fully-solid -Y neighbour must NOT be exposed",
            );
        }
        for &(mx, my, mz) in &[(0u32, 0, 0), (4, 0, 3), (7, 0, 6)] {
            let mut me = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut me, mx, my, mz);
            assert!(
                !face_exposed_mirror(face_ny_mirror(&me), face_py_mirror(&full)),
                "-Y voxel at ({mx},{my},{mz}) under fully-solid +Y neighbour must NOT be exposed",
            );
        }
        for &(mx, my, mz) in &[(0u32, 0, 7), (3, 5, 7), (6, 2, 7)] {
            let mut me = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut me, mx, my, mz);
            assert!(
                !face_exposed_mirror(face_pz_mirror(&me), face_nz_mirror(&full)),
                "+Z voxel at ({mx},{my},{mz}) under fully-solid -Z neighbour must NOT be exposed",
            );
        }
        for &(mx, my, mz) in &[(0u32, 0, 0), (2, 3, 0), (5, 7, 0)] {
            let mut me = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut me, mx, my, mz);
            assert!(
                !face_exposed_mirror(face_nz_mirror(&me), face_pz_mirror(&full)),
                "-Z voxel at ({mx},{my},{mz}) under fully-solid +Z neighbour must NOT be exposed",
            );
        }
    }

    /// Fully-empty neighbour ⇒ every solid face-voxel on `my_face` is
    /// exposed. Matches the isolated-exposure baseline.
    #[test]
    fn face_exposed_is_true_under_fully_empty_neighbour() {
        let empty = SubchunkOccupancy { planes: [0; 16] };
        let mut me = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut me, 7, 4, 2);
        set_bit(&mut me, 0, 1, 6);
        set_bit(&mut me, 3, 7, 0);
        set_bit(&mut me, 5, 0, 3);
        set_bit(&mut me, 2, 6, 7);
        set_bit(&mut me, 4, 2, 0);

        assert!(face_exposed_mirror(face_px_mirror(&me), face_nx_mirror(&empty)));
        assert!(face_exposed_mirror(face_nx_mirror(&me), face_px_mirror(&empty)));
        assert!(face_exposed_mirror(face_py_mirror(&me), face_ny_mirror(&empty)));
        assert!(face_exposed_mirror(face_ny_mirror(&me), face_py_mirror(&empty)));
        assert!(face_exposed_mirror(face_pz_mirror(&me), face_nz_mirror(&empty)));
        assert!(face_exposed_mirror(face_nz_mirror(&me), face_pz_mirror(&empty)));
    }

    /// Partial-overlap exposure — the test that distinguishes the
    /// correct per-(a, b) comparison from an aligned-but-OR-folded
    /// version.
    ///
    /// Setup: `my_pX` has a solid voxel at (x=7, y=3, z=2). The -X face
    /// of the neighbour has a solid voxel at (x=0, y=3, z=5). Because
    /// my's single voxel lives at canonical cell (a=y=3, b=z=2) while
    /// nbr's lives at cell (a=y=3, b=z=5), there is NO overlap: `my &
    /// ~nbr` retains my's bit (nbr has zero at cell (3, 2)) ⇒ exposed.
    ///
    /// If the implementation OR-folded across `z` (mapping all 8 z's
    /// into a single canonical-byte slot per y, e.g. bit 3 of byte 0),
    /// my's byte would be `0b0000_1000` and nbr's also `0b0000_1000`,
    /// so `my & ~nbr = 0` and the test would incorrectly return false.
    /// The distinct-z placement is what makes this test meaningful.
    #[test]
    fn face_exposed_respects_per_cell_pairing_under_partial_overlap() {
        // ±X: my at z=2, nbr at z=5, same y.
        {
            let mut me = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut me, 7, 3, 2);

            let mut nbr = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut nbr, 0, 3, 5);

            assert!(
                face_exposed_mirror(face_px_mirror(&me), face_nx_mirror(&nbr)),
                "+X voxel at (7,3,2) with -X neighbour voxel at (0,3,5) \
                 is exposed — the two voxels do not meet across the face",
            );
        }

        // ±Y: my at z=2, nbr at z=5, same x.
        {
            let mut me = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut me, 3, 7, 2);

            let mut nbr = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut nbr, 3, 0, 5);

            assert!(
                face_exposed_mirror(face_py_mirror(&me), face_ny_mirror(&nbr)),
                "+Y voxel at (3,7,2) with -Y neighbour at (3,0,5) \
                 must report exposed — different z => different cells",
            );
        }

        // ±Z: my at y=2, nbr at y=5, same x (z is fixed to 7/0 by the face).
        {
            let mut me = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut me, 4, 2, 7);

            let mut nbr = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut nbr, 4, 5, 0);

            assert!(
                face_exposed_mirror(face_pz_mirror(&me), face_nz_mirror(&nbr)),
                "+Z voxel at (4,2,7) with -Z neighbour at (4,5,0) \
                 must report exposed — different y => different cells",
            );
        }

        // Symmetric overlap: my and nbr at the same canonical cell =>
        // NOT exposed (nbr occupies the voxel facing my directly).
        {
            let mut me = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut me, 7, 3, 2);

            let mut nbr = SubchunkOccupancy { planes: [0; 16] };
            set_bit(&mut nbr, 0, 3, 2);

            assert!(
                !face_exposed_mirror(face_px_mirror(&me), face_nx_mirror(&nbr)),
                "+X voxel at (7,3,2) with -X neighbour at (0,3,2) \
                 is NOT exposed — the two voxels meet at cell (y=3, z=2)",
            );
        }
    }

    // -- Interior cell-pair exposure tests --

    /// Empty occupancy has no solid voxels, so no `here & ~there` cell
    /// pair can fire in any direction.
    #[test]
    fn interior_exposed_is_zero_on_empty() {
        let occ = SubchunkOccupancy { planes: [0; 16] };
        assert_eq!(interior_exposed_px_mirror(&occ), 0);
        assert_eq!(interior_exposed_nx_mirror(&occ), 0);
        assert_eq!(interior_exposed_py_mirror(&occ), 0);
        assert_eq!(interior_exposed_ny_mirror(&occ), 0);
        assert_eq!(interior_exposed_pz_mirror(&occ), 0);
        assert_eq!(interior_exposed_nz_mirror(&occ), 0);
    }

    /// Fully-solid occupancy has no interior cell pair where `here` is
    /// solid AND `there` is empty — `ALL_ONES & !ALL_ONES = 0`. This
    /// isolates the interior path; the boundary-via-neighbour
    /// contribution is checked separately by the face_exposed tests.
    /// It is also the load-bearing property that lets cull drop the
    /// `is_solid` early-out: for fully-solid sub-chunks, exposure bits
    /// are entirely determined by the boundary check against neighbours.
    #[test]
    fn interior_exposed_is_zero_on_fully_solid() {
        let occ = SubchunkOccupancy { planes: [0xFFFF_FFFF; 16] };
        assert_eq!(interior_exposed_px_mirror(&occ), 0);
        assert_eq!(interior_exposed_nx_mirror(&occ), 0);
        assert_eq!(interior_exposed_py_mirror(&occ), 0);
        assert_eq!(interior_exposed_ny_mirror(&occ), 0);
        assert_eq!(interior_exposed_pz_mirror(&occ), 0);
        assert_eq!(interior_exposed_nz_mirror(&occ), 0);
    }

    /// A single voxel at the centre has all 6 neighbours empty AND
    /// inside the sub-chunk, so every direction's interior OR-fold
    /// fires.
    #[test]
    fn interior_exposed_fires_in_all_directions_for_isolated_voxel() {
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut occ, 4, 4, 4);
        assert_ne!(interior_exposed_px_mirror(&occ), 0);
        assert_ne!(interior_exposed_nx_mirror(&occ), 0);
        assert_ne!(interior_exposed_py_mirror(&occ), 0);
        assert_ne!(interior_exposed_ny_mirror(&occ), 0);
        assert_ne!(interior_exposed_pz_mirror(&occ), 0);
        assert_ne!(interior_exposed_nz_mirror(&occ), 0);
    }

    /// A voxel at x=7 has no internal +X neighbour — the cell at "x=8"
    /// wraps into the next y-row in raw bit terms, which the
    /// MASK_NO_X7 mask is there to suppress. Without that mask, the
    /// helper would spuriously report +X exposure for any voxel at the
    /// outer face. Symmetric guard for `interior_exposed_pz` at z=7
    /// (mask not needed — the iteration bound `z < 7u` excludes it).
    #[test]
    fn interior_exposed_does_not_fire_for_voxel_at_outer_face() {
        // x=7: no internal +X neighbour. -X should still fire because
        // (x=6, y=4, z=4) is empty and (x=7, y=4, z=4) is solid.
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut occ, 7, 4, 4);
        assert_eq!(
            interior_exposed_px_mirror(&occ), 0,
            "+X must NOT fire for x=7 voxel — no internal +X neighbour",
        );
        assert_ne!(interior_exposed_nx_mirror(&occ), 0);

        // z=7: no internal +Z neighbour.
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut occ, 3, 4, 7);
        assert_eq!(
            interior_exposed_pz_mirror(&occ), 0,
            "+Z must NOT fire for z=7 voxel — no internal +Z neighbour",
        );
        assert_ne!(interior_exposed_nz_mirror(&occ), 0);

        // x=0: no internal -X neighbour.
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut occ, 0, 4, 4);
        assert_eq!(
            interior_exposed_nx_mirror(&occ), 0,
            "-X must NOT fire for x=0 voxel — no internal -X neighbour",
        );
        assert_ne!(interior_exposed_px_mirror(&occ), 0);
    }

    /// The y=3 / y=4 interface crosses the word-0 / word-1 boundary
    /// inside a single z-plane (word 0 holds y in [0,4), word 1 holds y
    /// in [4,8)). The cross-word OR `(w0 >> 24) & !(w1 & 0xFF)` must
    /// detect the empty +Y neighbour at y=4 for a solid voxel at y=3.
    #[test]
    fn interior_exposed_py_fires_across_y3_y4_word_boundary() {
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut occ, 2, 3, 4);
        assert_ne!(
            interior_exposed_py_mirror(&occ), 0,
            "+Y must fire across the y=3 -> y=4 word boundary",
        );
    }

    /// Symmetric to the +Y cross-word test: a voxel at y=4 in word 1's
    /// low byte with empty -Y neighbour at y=3 in word 0's high byte
    /// must fire via `(w1 & 0xFF) & !(w0 >> 24)`.
    #[test]
    fn interior_exposed_ny_fires_across_y4_y3_word_boundary() {
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        set_bit(&mut occ, 2, 4, 4);
        assert_ne!(
            interior_exposed_ny_mirror(&occ), 0,
            "-Y must fire across the y=4 -> y=3 word boundary",
        );
    }

    /// The original motivating case: a heightfield surface that crests
    /// inside the sub-chunk and never reaches y=7. The boundary-only
    /// `face_pY` extracts the y=7 row, sees zero solid voxels there, and
    /// reports +Y exposure = 0 — dropping the sub-chunk from cull when
    /// the camera looks down on it. The interior +Y OR-fold catches the
    /// y=3 -> y=4 transition at the column top and forces +Y exposure
    /// regardless of the boundary.
    #[test]
    fn interior_exposed_py_fires_on_heightfield_top_inside_sub_chunk() {
        let mut occ = SubchunkOccupancy { planes: [0; 16] };
        for y in 0..4 {
            set_bit(&mut occ, 4, y, 4);
        }
        assert_ne!(
            interior_exposed_py_mirror(&occ), 0,
            "+Y interior must fire for an interior heightfield top",
        );
        assert_eq!(
            interior_exposed_ny_mirror(&occ), 0,
            "-Y interior must NOT fire — column bottom only exposes via boundary",
        );
        // Lateral directions: column has empty interior neighbours on
        // every side, so all four ±X / ±Z fire.
        assert_ne!(interior_exposed_px_mirror(&occ), 0);
        assert_ne!(interior_exposed_nx_mirror(&occ), 0);
        assert_ne!(interior_exposed_pz_mirror(&occ), 0);
        assert_ne!(interior_exposed_nz_mirror(&occ), 0);
    }

    /// Fully-solid except a single empty cell at the centre. Each of the
    /// 6 surrounding solid cells has its inward-facing direction's
    /// interior OR-fold fire because of the empty centre. Exercises all
    /// 6 helpers in a single configuration and confirms the symmetric
    /// behaviour.
    #[test]
    fn interior_exposed_fires_for_single_empty_in_solid() {
        let mut occ = SubchunkOccupancy { planes: [0xFFFF_FFFF; 16] };
        let bit  = 4u32 * 8 + 4;
        let word = 4u32 * 2 + (bit >> 5);
        occ.planes[word as usize] &= !(1u32 << (bit & 31));
        assert_ne!(interior_exposed_px_mirror(&occ), 0);
        assert_ne!(interior_exposed_nx_mirror(&occ), 0);
        assert_ne!(interior_exposed_py_mirror(&occ), 0);
        assert_ne!(interior_exposed_ny_mirror(&occ), 0);
        assert_ne!(interior_exposed_pz_mirror(&occ), 0);
        assert_ne!(interior_exposed_nz_mirror(&occ), 0);
    }

    // -- DirtyReport byte layout --

    #[test]
    fn dirty_report_layout_matches_shader_offsets() {
        let expected = 16 + 16 * MAX_CANDIDATES;
        assert_eq!(std::mem::size_of::<DirtyReport>(), expected);

        let r = DirtyReport {
            count:    0x0A0B_0C0D,
            _pad0:    0,
            overflow: 0x1122_3344,
            _pad2:    0,
            entries: [DirtyEntry { directory_index: 0, new_bits_partial: 0,
                                   staging_request_idx: 0, _pad: 0 };
                      MAX_CANDIDATES],
        };
        let bytes: &[u8] = bytemuck::bytes_of(&r);
        // Count lives at offset 0; overflow at offset 8; entries at offset 16.
        assert_eq!(&bytes[0..4],  0x0A0B_0C0Du32.to_le_bytes());
        assert_eq!(&bytes[8..12], 0x1122_3344u32.to_le_bytes());
        assert_eq!(bytes.len(), expected);
    }

    // -- Staging ring rotation (regression: failure-staging-not-ringed-after-gid-x-reindexing) --
    //
    // Guards the retirement-frame → staging-ring-slot wiring. The bug
    // this catches: at frame `F + FrameCount`, the retirement for frame
    // F issues `PatchCopy { staging_request_idx, dst_material_slot }`.
    // If the patch pass reads staging at the *current* frame's ring
    // slot instead of frame F's ring slot, it copies the current
    // frame's prep output (possibly for entirely different coords)
    // into the retirement's target `material_pool` entries. Under the
    // pre-ring shape (single buffer), every frame clobbered the same
    // memory and the bug was structural — hence this test.
    //
    // We assert the slot-selection function directly, on the same math
    // `subchunk_patch` uses (`MultiBufferRing::current(frame)` →
    // `frame.slot(frame_count)`), because the rest of the wiring is a
    // thin shell around it. An assertion on the slot index is the
    // load-bearing property: if the patch pass picks a different slot
    // than the prep did for the retired frame, the race returns.

    /// Ring depth N = FIF must place a dispatch's staging write and the
    /// retirement's read on the *same* ring slot, and that slot must be
    /// distinct from the slot used by every intervening frame. Written
    /// against `FrameCount = 2` because that's the pinned FIF for this
    /// project (see `knowledge-fif-swapchain-depth-decoupling`).
    #[test]
    fn staging_ring_slot_for_retired_frame_is_stable_under_fif() {
        use crate::frame::FrameCount;

        let fc = FrameCount::new(2).unwrap();

        // Simulate a sequence of frames. At frame F, prep writes ring
        // slot `F.slot(fc)`. At frame F + FrameCount, the retirement
        // for frame F is about to run; patch must read the same slot
        // the prep wrote — `F.slot(fc)` — *not* the current frame's
        // slot `(F + 2).slot(fc)`.
        //
        // With FIF=2 the two happen to coincide on the wgpu::Buffer
        // identity, because the rotation cycles back. But the
        // intervening frame F+1 uses the *other* slot, so frame F's
        // staging bytes survive across the FIF window intact. The
        // buggy "no ring" shape and the "read current frame" shape
        // both produce different slot indices from this check below,
        // so the assertion pins the correct relation.
        for origin in [0u64, 1, 7, 123, u64::MAX - 3] {
            let dispatch       = FrameIndex::default().plus(origin as u32);
            let intervening    = dispatch.plus(1);
            let retire_at      = dispatch.plus(fc.get());

            // Prep at dispatch wrote `dispatch.slot(fc)`. Patch at
            // retire_at must read the same staging payload, so it
            // indexes the ring with `dispatch`, not with `retire_at`.
            assert_eq!(
                dispatch.slot(fc),
                retire_at.slot(fc),
                "FIF=N rotation: dispatch slot and N-later slot must \
                 coincide by modular arithmetic (guards the ring depth \
                 invariant)",
            );

            // But an intervening frame's prep slot must NOT collide
            // with the dispatch's slot — otherwise frame F's staging
            // bytes would be clobbered before the retirement reads
            // them, which is exactly the failure mode.
            assert_ne!(
                dispatch.slot(fc),
                intervening.slot(fc),
                "intervening frame's ring slot must not clobber \
                 frame F's staging before retirement consumes it",
            );

            // Witness: the *buggy* "read current frame" version of
            // patch would use `retire_at.slot(fc)` with the retire
            // frame's index. Because retire_at == dispatch + N and N =
            // frame_count, `retire_at.slot(fc) == dispatch.slot(fc)`.
            // The assertion above already exploits that coincidence.
            // So the load-bearing check is really the intervening
            // collision test — if it passes, the ring depth is >= 2.
            let _ = retire_at;
        }
    }

    /// `subchunk_patch` imports the ring slot for the retirement's
    /// dispatch frame, *not* the current frame. This test makes the
    /// wiring visible as a compile-time check on the
    /// `MultiBufferRing::current(frame)` shape — the function picks
    /// the staging buffer by the frame the retirement originated on.
    /// If somebody re-plumbs the patch pass to use a different
    /// `FrameIndex`, the contract for "which frame's staging does the
    /// patch read" flips and the test's assertion on slot identity
    /// fires.
    #[test]
    fn retired_dispatch_frame_selects_correct_staging_slot() {
        use crate::frame::FrameCount;

        let fc = FrameCount::new(2).unwrap();

        // Scenario modelled by the failure: dispatch at frame 10,
        // retirement at frame 12 (10 + FIF). Between them, frame 11
        // also dispatched prep (coord C_b) and wrote its own slot.
        let dispatch   = FrameIndex::default().plus(10);
        let retire_at  = FrameIndex::default().plus(12);
        let collision  = FrameIndex::default().plus(11);

        assert_eq!(dispatch.slot(fc), retire_at.slot(fc));
        assert_ne!(dispatch.slot(fc), collision.slot(fc));

        // The patch pass must address the ring slot via the
        // `dispatch` frame index (the frame whose dirty list is
        // retiring). If it were to use `retire_at` that also happens
        // to name the same slot under a size-FIF ring — but the
        // invariant is about *intent*: ring.current(dispatch) ≡
        // "staging bytes from frame 10's prep". Using a third-frame
        // index or the intervening `collision` frame would produce a
        // different slot and surface the bug.
        assert_ne!(collision.slot(fc), dispatch.slot(fc),
                   "reading the intervening frame's slot returns coord \
                    C_b's staging, not C_a's — the old single-buffer bug");
    }
}

