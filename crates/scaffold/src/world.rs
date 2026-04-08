//! GPU world manager for chunk lifecycle and rendering.
//!
//! Manages shared slot-indexed GPU buffers, a contiguous quad allocator,
//! and per-chunk state. Handles occupancy upload, build dispatch, quad
//! count readback, allocation management, frustum culling, and draw call
//! emission.
//!
//! The buffer layout uses a single set of shared buffers indexed by slot
//! rather than per-chunk GPU resources. The build shader reads occupancy
//! and boundary data from shared buffers via push-constant slot indices.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytemuck::{Pod, Zeroable};
use voxel::block::{BlockId, BlockRegistry, FaceTexture};
use voxel::chunk::ChunkPos;
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayout, BindingResource, Buffer, BufferDescriptor, BufferUsages,
    CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, Device,
    Extent3d, FilterMode, Queue, RenderPass, SamplerDescriptor,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor, TextureViewDimension,
};

use crate::build::{BuildCountPipeline, BuildWritePipeline, MaterialPackPipeline};
use crate::cull::CullPipeline;
use crate::multi_buffer::MultiBuffer;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum concurrent loaded chunks (slot count).
const MAX_CHUNKS: u32 = 4096;

/// Total quad buffer size in bytes (64 MB).
const QUAD_BUF_SIZE: u64 = 64 * 1024 * 1024;

/// Total quad buffer capacity in quads (one quad = 4 bytes).
const QUAD_BUF_QUADS: u32 = (QUAD_BUF_SIZE / 4) as u32;

/// Occupancy words per slot (32x32 = 1024 bits packed into 32 words
/// per layer, 32 layers).
const OCC_WORDS_PER_SLOT: u32 = 1024;

/// Neighbor boundary words per slot (6 directions x 32 words each).
const BOUNDARY_WORDS_PER_SLOT: u32 = 192;

/// Bytes per quad range entry. Layout: buffer_index (4) + base_offset (4)
/// + dir_layer_counts (6 x 32 x 4 = 768) = 776 bytes.
const QUAD_RANGE_BYTES: u32 = 776;

/// Bytes per chunk metadata entry (quad_count, flags, reserved, reserved).
const CHUNK_META_BYTES: u32 = 16;

/// Bytes per draw data entry (slot, direction).
const DRAW_DATA_BYTES: u32 = 8;

/// Conservative upper bound on quads per chunk. Used for capacity
/// estimation when throttling chunk loads. Derived from the worst-case
/// checkerboard pattern (98304 faces) rounded to a convenient number.
pub(crate) const MAX_CHUNK_QUADS: u32 = 98304;

/// Words per neighbor boundary slice (one 32x32 face layer).
const SLICE_WORDS: usize = 32;

/// Total words for all 6 neighbor boundary slices.
const NEIGHBOR_WORDS: usize = 6 * SLICE_WORDS;

/// Neighbor slice offset for +X direction within the 192-word region.
const DIR_POS_X: usize = 0 * SLICE_WORDS;

/// Neighbor slice offset for -X direction.
const DIR_NEG_X: usize = 1 * SLICE_WORDS;

/// Neighbor slice offset for +Y direction.
const DIR_POS_Y: usize = 2 * SLICE_WORDS;

/// Neighbor slice offset for -Y direction.
const DIR_NEG_Y: usize = 3 * SLICE_WORDS;

/// Neighbor slice offset for +Z direction.
const DIR_POS_Z: usize = 4 * SLICE_WORDS;

/// Neighbor slice offset for -Z direction.
const DIR_NEG_Z: usize = 5 * SLICE_WORDS;

/// Maximum number of chunk builds that can be in flight at once.
///
/// Determines the size of the async build staging buffer. The staging
/// buffer holds one full `ChunkMeta` (16 bytes: quad count + flags +
/// sub_mask) per in-flight build.
const MAX_BUILDS_IN_FLIGHT: u32 = 64;

/// Bytes per entry in the build staging buffer. One full ChunkMeta:
/// quad_count(4) + flags(4) + sub_mask_lo(4) + sub_mask_hi(4).
const BUILD_STAGING_ENTRY_BYTES: u32 = 16;

/// Bytes per entry in material_range_buf.
const MATERIAL_RANGE_BYTES: u32 = 16;

/// Initial packed material buffer size in bytes (16 MB).
const MATERIAL_BUF_INITIAL: u64 = 16 * 1024 * 1024;

/// Bytes per material sub-block (8x8x8 voxels, u16 each).
const MATERIAL_SUB_BLOCK: u32 = 1024;

/// Maximum number of material buffer segments in the binding array.
pub(crate) const MAX_MATERIAL_SEGMENTS: u32 = 16;

/// Material segment capacity in sub-block units (16 MB / 1024 B = 16384).
const MATERIAL_SEGMENT_UNITS: u32 =
    (MATERIAL_BUF_INITIAL / MATERIAL_SUB_BLOCK as u64) as u32;

// ---------------------------------------------------------------------------
// GpuMaterial
// ---------------------------------------------------------------------------

/// GPU-side material entry for the material property table.
///
/// Each entry maps a block type to its packed color and texture
/// configuration. Laid out to match the shader's `array<vec4<u32>>`
/// binding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuMaterial {
    /// Packed RGBA color in little-endian byte order.
    pub color_rgba  : u32,
    /// Default texture index (used when `face_offset` is 0).
    pub texture_idx : u32,
    /// Offset into the face texture table for per-face blocks.
    /// Zero means uniform (all faces use `texture_idx`).
    pub face_offset : u32,
    /// Reserved for future use.
    pub _pad        : u32,
}

// ---------------------------------------------------------------------------
// build_material_tables
// ---------------------------------------------------------------------------

/// Build GPU material tables from a block registry.
///
/// Scans the registry and produces two tables:
/// - A per-block material entry with color, default texture, and face
///   offset.
/// - A flat face texture array containing per-direction texture indices
///   for blocks with non-uniform face textures.
///
/// Uniform blocks get `face_offset = 0` and use the single texture
/// index directly. Per-face blocks get an offset into the face table
/// where 6 consecutive entries store per-direction indices.
pub fn build_material_tables(
    registry : &BlockRegistry,
) -> (Vec<GpuMaterial>, Vec<u32>)
{
    let mut materials  = Vec::with_capacity(registry.len());
    // Index 0 is reserved as the "uniform" sentinel, so real offsets
    // start at 1 and are always nonzero.
    let mut face_table = vec![0u32];

    for i in 0..registry.len() {
        let block        = registry.get(BlockId::new(i as u16));
        let mat          = block.material();
        let [r, g, b, a] = mat.color();
        let color_rgba   = u32::from_le_bytes([r, g, b, a]);

        let (texture_idx, face_offset) = match mat.face_texture() {
            FaceTexture::Uniform(idx) => {
                (idx as u32, 0u32)
            }

            FaceTexture::PerFace(faces) => {
                let base = face_table.len() as u32;
                for &t in &faces {
                    face_table.push(t as u32);
                }
                // Default texture is the +X face (index 0).
                (faces[0] as u32, base)
            }
        };

        materials.push(GpuMaterial {
            color_rgba  : color_rgba,
            texture_idx : texture_idx,
            face_offset : face_offset,
            _pad        : 0,
        });
    }

    (materials, face_table)
}

// ---------------------------------------------------------------------------
// ContiguousAllocator
// ---------------------------------------------------------------------------

/// Contiguous range allocator for the quad buffer.
///
/// Manages a single buffer as a bump allocator with a coalescing free
/// list. New allocations advance the bump pointer when the free list
/// cannot satisfy the request. Freed ranges go to the free list and
/// are reused first-fit, coalescing with adjacent free ranges to reduce
/// fragmentation.
struct ContiguousAllocator {
    /// High-water mark in the quad buffer (next bump allocation offset).
    bump_offset : u32,
    /// Free ranges sorted by offset for coalescing. Each entry is
    /// (offset, size) in quads.
    free_list   : Vec<(u32, u32)>,
    /// Total capacity in quads.
    capacity    : u32,
}

// --- ContiguousAllocator ---

impl ContiguousAllocator {
    /// Create a new allocator with the given capacity.
    fn new(capacity: u32) -> Self {
        ContiguousAllocator {
            bump_offset : 0,
            free_list   : Vec::new(),
            capacity    : capacity,
        }
    }

    /// Allocate a contiguous range of `size` quads.
    ///
    /// Returns the offset into the quad buffer on success, or `None` if
    /// the allocator cannot satisfy the request. Tries the free list
    /// first (first-fit), then falls back to the bump pointer.
    fn alloc(&mut self, size: u32) -> Option<u32> {
        if size == 0 {
            return Some(0);
        }

        // First-fit search through free list.
        for i in 0..self.free_list.len() {
            let (offset, free_size) = self.free_list[i];

            if free_size >= size {
                if free_size == size {
                    self.free_list.remove(i);
                }
                else {
                    // Shrink the free range from the front.
                    self.free_list[i] = (offset + size, free_size - size);
                }

                return Some(offset);
            }
        }

        // Bump allocation.
        let remaining = self.capacity - self.bump_offset;

        if remaining >= size {
            let offset = self.bump_offset;
            self.bump_offset += size;
            return Some(offset);
        }

        None
    }

    /// Free a contiguous range starting at `offset` with the given `size`.
    ///
    /// Coalesces with adjacent free ranges to prevent fragmentation.
    /// If the freed range is at the bump pointer, the pointer is
    /// retracted instead of adding to the free list.
    fn free(&mut self, offset: u32, size: u32) {
        if size == 0 {
            return;
        }

        let end = offset + size;

        // If this range is at the top of the bump region, retract.
        if end == self.bump_offset {
            self.bump_offset = offset;

            // Check if any free list entry now abuts the new bump pointer.
            self.coalesce_bump();
            return;
        }

        // Insert into the free list maintaining sort order by offset,
        // then coalesce with neighbors.
        let insert_pos = self.free_list
            .partition_point(|&(o, _)| o < offset);

        self.free_list.insert(insert_pos, (offset, size));
        self.coalesce_at(insert_pos);
    }

    /// Returns the number of free units available (free list + remaining
    /// bump space).
    fn free_quads(&self) -> u32 {
        let free_list_total: u32 = self.free_list
            .iter()
            .map(|&(_, size)| size)
            .sum();

        free_list_total + (self.capacity - self.bump_offset)
    }

    /// Returns the total capacity in allocation units.
    fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Extend the capacity by `extra` units. The new units are available
    /// at the end of the bump region.
    fn grow(&mut self, extra: u32) {
        self.capacity += extra;
    }

    /// Coalesce the free list entry at `idx` with its neighbors.
    fn coalesce_at(&mut self, idx: usize) {
        // Merge with the entry after idx, if adjacent.
        if idx + 1 < self.free_list.len() {
            let (off_a, size_a) = self.free_list[idx];
            let (off_b, size_b) = self.free_list[idx + 1];

            if off_a + size_a == off_b {
                self.free_list[idx] = (off_a, size_a + size_b);
                self.free_list.remove(idx + 1);
            }
        }

        // Merge with the entry before idx, if adjacent.
        if idx > 0 {
            let (off_prev, size_prev) = self.free_list[idx - 1];
            let (off_cur, size_cur)   = self.free_list[idx];

            if off_prev + size_prev == off_cur {
                self.free_list[idx - 1] = (off_prev, size_prev + size_cur);
                self.free_list.remove(idx);
            }
        }
    }

    /// Retract the bump pointer by absorbing any free list entry that
    /// now abuts it from below.
    fn coalesce_bump(&mut self) {
        while let Some(&(offset, size)) = self.free_list.last() {
            if offset + size == self.bump_offset {
                self.bump_offset = offset;
                self.free_list.pop();
            }
            else {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GpuChunk
// ---------------------------------------------------------------------------

/// Per-chunk GPU state.
///
/// Tracks the slot index, quad buffer allocation, and CPU-side occupancy
/// shadow. All actual GPU data lives in shared slot-indexed buffers
/// managed by [`GpuWorld`].
struct GpuChunk {
    /// Slot index in the shared buffers.
    slot            : u32,
    /// Offset into the quad buffer (from the contiguous allocator).
    quad_offset     : u32,
    /// Allocated size in quads (may exceed actual quad count).
    quad_alloc      : u32,
    /// Actual quad count from the last completed build.
    quad_count      : u32,
    /// CPU-side occupancy shadow for neighbor boundary extraction.
    occ             : Box<[u32; 1024]>,
    /// Whether this chunk needs rebuilding.
    dirty            : bool,
    /// Whether a count pass is in flight for this chunk.
    count_in_flight  : bool,
    /// Offset into packed material buffer (in sub-block units).
    material_offset  : u32,
    /// Allocated size in sub-blocks.
    material_alloc   : u32,
    /// Sub-block visibility mask from the last completed build.
    sub_mask         : u64,
}

/// A chunk whose count pass has been dispatched but whose write pass
/// has not yet run.
struct PendingBuild {
    /// Chunk world position.
    pos           : ChunkPos,
    /// Slot index in the shared buffers.
    slot          : u32,
    /// Index into the build staging buffer (read at offset * 4).
    staging_index : u32,
}

// ---------------------------------------------------------------------------
// RenderStats
// ---------------------------------------------------------------------------

/// Rendering statistics from the GPU world.
///
/// A snapshot of current chunk loading, quad generation, and allocator
/// utilization. All values are derived from CPU-side state at zero cost.
pub struct RenderStats {
    /// Total loaded chunks.
    pub chunks_loaded  : u32,
    /// Chunks visible after frustum culling (one frame latency).
    pub chunks_visible : u32,
    /// Total quads across all chunks.
    pub total_quads    : u32,
    /// Quad buffer bytes currently allocated.
    pub quad_buf_used  : u64,
    /// Quad buffer total capacity in bytes.
    pub quad_buf_total : u64,
    /// Material buffer bytes currently allocated.
    pub material_buf_used  : u64,
    /// Material buffer total capacity in bytes.
    pub material_buf_total : u64,
    /// Total bytes used by fixed per-slot buffers (occupancy, boundary
    /// cache, chunk meta, quad range, chunk offsets, material range,
    /// indirect draws, draw data).
    pub slot_buf_total     : u64,
    /// Total GPU buffer memory (quad buffer + slot buffers + small buffers).
    pub gpu_mem_total      : u64,
}

// ---------------------------------------------------------------------------
// GpuWorld
// ---------------------------------------------------------------------------

/// Manages GPU resources for all loaded chunks.
///
/// Owns shared slot-indexed buffers and a contiguous quad allocator.
/// The build and cull compute pipelines are created externally (in
/// `build.rs` and `cull.rs`) and referenced here for dispatching.
///
/// Per-frame workflow:
///
/// 1. Call [`process_build_feedback`](Self::process_build_feedback) to
///    harvest results from the previous frame's count passes.
/// 2. Modify chunks via [`update_occupancy`](Self::update_occupancy).
/// 3. Call [`dispatch_counts`](Self::dispatch_counts) or
///    [`rebuild`](Self::rebuild) to dispatch count passes.
/// 4. Call [`dispatch_cull`](Self::dispatch_cull) to run frustum culling.
/// 5. In the render pass, call [`draw`](Self::draw) to issue draw calls.
pub struct GpuWorld {
    // -- Shared buffers (slot-indexed) --

    /// Shared quad storage buffer (64 MB, contiguous per chunk).
    quad_buf             : Buffer,
    /// Occupancy bitmasks, slot-indexed (4096 x 4096 B = 16 MB).
    occupancy_buf        : Buffer,
    /// Neighbor boundary cache, slot-indexed (4096 x 768 B = 3 MB).
    boundary_cache_buf   : Buffer,
    /// Per-chunk metadata: {quad_count, flags, _, _} per slot.
    chunk_meta_buf       : Buffer,
    /// Per-chunk quad range data: {buffer_index, base_offset,
    /// dir_layer_counts} per slot.
    quad_range_buf       : Buffer,
    /// Per-slot chunk world offsets (ivec4 per slot).
    chunk_offset_buf     : Buffer,
    /// Per-slot material range metadata for sparse sub-block packing.
    material_range_buf   : Buffer,
    /// Segmented packed material buffers (visibility-driven sub-blocks).
    material_bufs        : MultiBuffer,
    /// Transient staging buffer for CPU material upload during builds.
    material_staging_buf : Buffer,

    // -- Cull/draw buffers --

    /// Output indirect draw buffer (GPU-written, visible chunks only).
    dst_indirect_buf     : Buffer,
    /// Per-draw metadata: {slot, direction} written by cull shader.
    draw_data_buf        : Buffer,
    /// GPU-side atomic draw count for `multi_draw_indirect_count`.
    draw_count_buf       : Buffer,
    /// Frustum plane uniform buffer (6 x vec4<f32>).
    frustum_buf          : Buffer,

    // -- Pipelines and bind groups --

    /// The build count compute pipeline (pass 1, created in build.rs).
    build_count_pipeline : BuildCountPipeline,
    /// Bind group for build count dispatches (shared buffers).
    build_count_bg       : BindGroup,
    /// The build write compute pipeline (pass 2, created in build.rs).
    build_write_pipeline : BuildWritePipeline,
    /// Bind group for build write dispatches (shared buffers).
    build_write_bg       : BindGroup,
    /// The frustum culling compute pipeline (created in cull.rs).
    cull_pipeline        : CullPipeline,
    /// Bind group for the cull dispatch.
    cull_bg              : BindGroup,
    /// The material pack compute pipeline (copies sub-blocks to packed buf).
    material_pack_pipeline : MaterialPackPipeline,
    /// Bind group for material pack dispatches.
    material_pack_bg       : BindGroup,
    /// Material buffer array bind group for render pass (set 1, read-only).
    material_array_ro_bg   : BindGroup,
    /// Material buffer array bind group for material pack pass (set 1, read-write).
    material_array_rw_bg   : BindGroup,
    /// Bind group layout for the read-only material buffer array (set 1).
    material_array_bgl     : BindGroupLayout,
    /// The render bind group layout (created in main.rs).
    render_bgl           : BindGroupLayout,
    /// Shared render bind group (camera + quad + offsets + draw_data +
    /// materials + textures).
    render_bg            : BindGroup,

    // -- Render resources --

    /// The shared camera uniform buffer.
    camera_buf           : Buffer,
    /// The material property table buffer.
    material_table_buf   : Buffer,
    /// Per-face texture override table for non-uniform blocks.
    face_texture_buf     : Buffer,
    /// The block texture array view.
    texture_view         : TextureView,
    /// The texture sampler.
    tex_sampler          : wgpu::Sampler,

    // -- Allocator and chunk state --

    /// Contiguous quad buffer allocator.
    allocator            : ContiguousAllocator,
    /// Cached material buffer generation for bind group rebuild detection.
    material_gen         : u64,
    /// Free slot indices (LIFO stack).
    free_slots           : Vec<u32>,
    /// Per-chunk GPU state keyed by world position.
    chunks               : HashMap<ChunkPos, GpuChunk>,
    /// Positions of chunks that need rebuilding.
    dirty                : Vec<ChunkPos>,

    // -- Build synchronization (async) --

    /// Staging buffer for async readback of quad counts after count passes.
    build_staging        : Buffer,
    /// Whether an async build staging map is in flight.
    build_pending        : bool,
    /// Set by the map callback when the build staging data is ready.
    build_ready          : Arc<AtomicBool>,
    /// Chunks awaiting write pass dispatch (count in flight).
    pending_builds       : Vec<PendingBuild>,

    // -- Visible count async readback --

    /// Staging buffer for async readback of the visible draw count.
    count_staging        : Buffer,
    /// Set by the map callback when the visible count is ready.
    count_ready          : Arc<AtomicBool>,
    /// Whether a count readback request is currently in flight.
    count_pending        : bool,
    /// Last known visible chunk count (one frame latency).
    visible_count        : u32,

    // -- Draw state --

    /// Total slots with non-zero quad counts (upper bound for cull).
    active_slot_count    : u32,
}

// --- GpuWorld ---

impl GpuWorld {
    /// Create a new GPU world manager.
    ///
    /// # Arguments
    ///
    /// * `device`         - The GPU device for pipeline and resource creation.
    /// * `queue`          - The queue for texture upload.
    /// * `render_bgl`         - The bind group layout used by the render pipeline
    ///   (set 0).
    /// * `material_array_bgl` - The bind group layout for the read-only
    ///   material buffer array (set 1).
    /// * `camera_buf`         - The shared camera uniform buffer. A handle clone
    ///   is stored internally. The caller retains ownership for writing
    ///   camera updates.
    /// * `materials`      - Material property table entries, one per block type.
    /// * `face_textures`  - Per-face texture indices for non-uniform blocks,
    ///   produced by [`build_material_tables`].
    /// * `texture_pixels` - Raw RGBA pixel data for all texture array layers,
    ///   packed sequentially.
    /// * `texture_size`   - Width and height of each texture layer in pixels.
    /// * `texture_layers` - Number of layers in the texture array.
    pub fn new(
        device             : &Device,
        queue              : &Queue,
        render_bgl         : BindGroupLayout,
        material_array_bgl : BindGroupLayout,
        camera_buf         : Buffer,
        materials          : &[GpuMaterial],
        face_textures      : &[u32],
        texture_pixels     : &[u8],
        texture_size       : u32,
        texture_layers     : u32,
    ) -> Self
    {
        // -- Shared slot-indexed buffers --

        let quad_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("quad_buf"),
            size               : QUAD_BUF_SIZE,
            usage              : BufferUsages::STORAGE,
            mapped_at_creation : false,
        });

        let occupancy_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("occupancy_buf"),
            size               : u64::from(MAX_CHUNKS)
                               * u64::from(OCC_WORDS_PER_SLOT) * 4,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        let boundary_cache_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("boundary_cache_buf"),
            size               : u64::from(MAX_CHUNKS)
                               * u64::from(BOUNDARY_WORDS_PER_SLOT) * 4,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        let chunk_meta_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("chunk_meta_buf"),
            size               : u64::from(MAX_CHUNKS)
                               * u64::from(CHUNK_META_BYTES),
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_SRC
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        let quad_range_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("quad_range_buf"),
            size               : u64::from(MAX_CHUNKS)
                               * u64::from(QUAD_RANGE_BYTES),
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        let chunk_offset_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("chunk_offsets"),
            size               : u64::from(MAX_CHUNKS) * 16,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        let material_range_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("material_range_buf"),
            size               : u64::from(MAX_CHUNKS)
                               * u64::from(MATERIAL_RANGE_BYTES),
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        let material_bufs = MultiBuffer::new(
            device,
            MATERIAL_SEGMENT_UNITS,
            MATERIAL_SUB_BLOCK,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            "material_buf",
        );

        let material_staging_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("material_staging"),
            size               : u64::from(MAX_BUILDS_IN_FLIGHT) * 65536,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        // -- Cull / draw buffers --

        let dst_indirect_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("dst_indirect_buf"),
            size               : u64::from(MAX_CHUNKS) * 16,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::INDIRECT,
            mapped_at_creation : false,
        });

        let draw_data_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("draw_data_buf"),
            size               : u64::from(MAX_CHUNKS)
                               * u64::from(DRAW_DATA_BYTES),
            usage              : BufferUsages::STORAGE,
            mapped_at_creation : false,
        });

        let draw_count_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("draw_count_buf"),
            size               : 4,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::INDIRECT
                               | BufferUsages::COPY_SRC
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        let frustum_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("frustum_buf"),
            size               : 96,
            usage              : BufferUsages::UNIFORM
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        // -- Build synchronization (async) --

        let build_staging = device.create_buffer(&BufferDescriptor {
            label              : Some("build_staging"),
            size               : u64::from(MAX_BUILDS_IN_FLIGHT)
                               * u64::from(BUILD_STAGING_ENTRY_BYTES),
            usage              : BufferUsages::COPY_DST
                               | BufferUsages::MAP_READ,
            mapped_at_creation : false,
        });

        // -- Visible count readback --

        let count_staging = device.create_buffer(&BufferDescriptor {
            label              : Some("cull_count_staging"),
            size               : 4,
            usage              : BufferUsages::COPY_DST
                               | BufferUsages::MAP_READ,
            mapped_at_creation : false,
        });

        // -- Material table --

        let material_table_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("material_table"),
            size               : (materials.len()
                                    * std::mem::size_of::<GpuMaterial>())
                                    .max(16) as u64,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        queue.write_buffer(
            &material_table_buf,
            0,
            bytemuck::cast_slice(materials),
        );

        // -- Face texture table (minimum 4 bytes for wgpu) --

        let face_texture_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("face_textures"),
            size               : (face_textures.len() * 4).max(4) as u64,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        if !face_textures.is_empty() {
            queue.write_buffer(
                &face_texture_buf,
                0,
                bytemuck::cast_slice(face_textures),
            );
        }

        // -- Block texture array --

        let texture = device.create_texture(&TextureDescriptor {
            label           : Some("block_textures"),
            size            : Extent3d {
                width                 : texture_size,
                height                : texture_size,
                depth_or_array_layers : texture_layers,
            },
            mip_level_count : 1,
            sample_count    : 1,
            dimension       : TextureDimension::D2,
            format          : TextureFormat::Rgba8UnormSrgb,
            usage           : TextureUsages::TEXTURE_BINDING
                            | TextureUsages::COPY_DST,
            view_formats    : &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture   : &texture,
                mip_level : 0,
                origin    : wgpu::Origin3d::ZERO,
                aspect    : wgpu::TextureAspect::All,
            },
            texture_pixels,
            wgpu::TexelCopyBufferLayout {
                offset         : 0,
                bytes_per_row  : Some(texture_size * 4),
                rows_per_image : Some(texture_size),
            },
            Extent3d {
                width                 : texture_size,
                height                : texture_size,
                depth_or_array_layers : texture_layers,
            },
        );

        let texture_view = texture.create_view(&TextureViewDescriptor {
            dimension : Some(TextureViewDimension::D2Array),
            ..Default::default()
        });

        let tex_sampler = device.create_sampler(&SamplerDescriptor {
            label          : Some("block_sampler"),
            mag_filter     : FilterMode::Nearest,
            min_filter     : FilterMode::Nearest,
            address_mode_u : AddressMode::Repeat,
            address_mode_v : AddressMode::Repeat,
            address_mode_w : AddressMode::Repeat,
            ..Default::default()
        });

        // Dummy buffer for binding 5 (material data now comes from set 1).
        let dummy_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("dummy"),
            size               : 4,
            usage              : BufferUsages::STORAGE,
            mapped_at_creation : false,
        });

        // -- Render bind group (matches bindings.hlsl Render layout) --
        //
        //   binding 0: camera           (uniform)
        //   binding 1: quad_buf         (read-only storage)
        //   binding 2: chunk_offsets     (read-only storage)
        //   binding 3: draw_data_buf    (read-only storage)
        //   binding 4: material_range   (read-only storage)
        //   binding 5: dummy            (read-only storage, unused)
        //   binding 6: material_table   (read-only storage)
        //   binding 7: face_textures    (read-only storage)
        //   binding 8: block_textures   (Texture2DArray)
        //   binding 9: tex_sampler      (SamplerState)

        let render_bg = device.create_bind_group(&BindGroupDescriptor {
            label   : Some("render_bg"),
            layout  : &render_bgl,
            entries : &[
                BindGroupEntry {
                    binding  : 0,
                    resource : camera_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 1,
                    resource : quad_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 2,
                    resource : chunk_offset_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 3,
                    resource : draw_data_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 4,
                    resource : material_range_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 5,
                    resource : dummy_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 6,
                    resource : material_table_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 7,
                    resource : face_texture_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 8,
                    resource : BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding  : 9,
                    resource : BindingResource::Sampler(&tex_sampler),
                },
            ],
        });

        // -- Build count pipeline + bind group (pass 1) --
        //
        //   binding 0: occupancy_buf       (read-only storage)
        //   binding 1: boundary_cache_buf  (read-only storage)
        //   binding 2: chunk_meta_buf      (read-write storage)
        //   binding 3: quad_range_buf      (read-write storage)

        let build_count_pipeline = BuildCountPipeline::new(device);

        let build_count_bg = build_count_pipeline.create_bind_group(
            device,
            &occupancy_buf,
            &boundary_cache_buf,
            &chunk_meta_buf,
            &quad_range_buf,
        );

        // -- Build write pipeline + bind group (pass 2) --
        //
        //   binding 0: occupancy_buf       (read-only storage)
        //   binding 1: boundary_cache_buf  (read-only storage)
        //   binding 2: quad_range_buf      (read-only storage)
        //   binding 3: quad_buf            (read-write storage)

        let build_write_pipeline = BuildWritePipeline::new(device);

        let build_write_bg = build_write_pipeline.create_bind_group(
            device,
            &occupancy_buf,
            &boundary_cache_buf,
            &quad_range_buf,
            &quad_buf,
        );

        // -- Material pack pipeline + bind group --

        let material_pack_pipeline = MaterialPackPipeline::new(device);

        let material_pack_bg = material_pack_pipeline.create_bind_group(
            device,
            &material_staging_buf,
            &material_range_buf,
        );

        // Material buffer array bind groups (set 1).
        let material_array_ro_bg = Self::create_material_array_bg(
            device,
            &material_array_bgl,
            material_bufs.buffers(),
        );

        let material_array_rw_bg = material_pack_pipeline.create_array_bind_group(
            device,
            material_bufs.buffers(),
        );

        let material_gen = material_bufs.generation();

        // -- Cull pipeline + bind group --
        //
        //   binding 0: frustum_planes  (uniform)
        //   binding 1: chunk_offsets   (read-only storage)
        //   binding 2: chunk_meta_buf  (read-only storage)
        //   binding 3: quad_range_buf  (read-only storage)
        //   binding 4: dst_draws       (read-write storage)
        //   binding 5: draw_data_buf   (read-write storage)
        //   binding 6: draw_count      (read-write storage)

        let cull_pipeline = CullPipeline::new(device);

        let cull_bg = cull_pipeline.create_bind_group(
            device,
            &frustum_buf,
            &chunk_offset_buf,
            &chunk_meta_buf,
            &quad_range_buf,
            &dst_indirect_buf,
            &draw_data_buf,
            &draw_count_buf,
        );

        // All slots start free.
        let free_slots = (0..MAX_CHUNKS).rev().collect();

        GpuWorld {
            quad_buf               : quad_buf,
            occupancy_buf          : occupancy_buf,
            boundary_cache_buf     : boundary_cache_buf,
            chunk_meta_buf         : chunk_meta_buf,
            quad_range_buf         : quad_range_buf,
            chunk_offset_buf       : chunk_offset_buf,
            material_range_buf     : material_range_buf,
            material_bufs          : material_bufs,
            material_staging_buf   : material_staging_buf,
            dst_indirect_buf       : dst_indirect_buf,
            draw_data_buf          : draw_data_buf,
            draw_count_buf         : draw_count_buf,
            frustum_buf            : frustum_buf,
            build_count_pipeline   : build_count_pipeline,
            build_count_bg         : build_count_bg,
            build_write_pipeline   : build_write_pipeline,
            build_write_bg         : build_write_bg,
            cull_pipeline          : cull_pipeline,
            cull_bg                : cull_bg,
            material_pack_pipeline : material_pack_pipeline,
            material_pack_bg       : material_pack_bg,
            material_array_ro_bg   : material_array_ro_bg,
            material_array_rw_bg   : material_array_rw_bg,
            material_array_bgl     : material_array_bgl,
            render_bgl             : render_bgl,
            render_bg              : render_bg,
            camera_buf             : camera_buf,
            material_table_buf     : material_table_buf,
            face_texture_buf       : face_texture_buf,
            texture_view           : texture_view,
            tex_sampler            : tex_sampler,
            allocator              : ContiguousAllocator::new(QUAD_BUF_QUADS),
            material_gen           : material_gen,
            free_slots             : free_slots,
            chunks                 : HashMap::new(),
            dirty                  : Vec::new(),
            build_staging          : build_staging,
            build_pending          : false,
            build_ready            : Arc::new(AtomicBool::new(false)),
            pending_builds         : Vec::new(),
            count_staging          : count_staging,
            count_ready            : Arc::new(AtomicBool::new(false)),
            count_pending          : false,
            visible_count          : 0,
            active_slot_count      : 0,
        }
    }

    /// Insert a chunk with initial occupancy data.
    ///
    /// Allocates a slot and uploads occupancy, boundary, and offset data
    /// to the shared buffers. No quad or material allocation is performed
    /// until the first build determines actual counts and sub-block
    /// visibility. The chunk is marked dirty for building.
    ///
    /// # Arguments
    ///
    /// * `device` - The GPU device (unused in current flow, reserved).
    /// * `queue`  - The queue for buffer writes.
    /// * `pos`    - The chunk position.
    /// * `occ`    - Initial chunk occupancy bitmask.
    pub fn insert(
        &mut self,
        _device : &Device,
        queue   : &Queue,
        pos     : ChunkPos,
        occ     : &[u32; 1024],
    )
    {
        let slot = self.alloc_slot();

        // Upload occupancy to the shared buffer at this slot's region.
        self.write_occupancy(queue, slot, occ);

        // Upload chunk world offset.
        self.write_chunk_offset(queue, slot, pos);

        // Zero the chunk metadata so the cull shader skips this slot
        // until the first build completes.
        self.zero_chunk_meta(queue, slot);

        self.chunks.insert(pos, GpuChunk {
            slot             : slot,
            quad_offset      : 0,
            quad_alloc       : 0,
            quad_count       : 0,
            occ              : Box::new(*occ),
            dirty            : true,
            count_in_flight  : false,
            material_offset  : 0,
            material_alloc   : 0,
            sub_mask         : 0,
        });

        self.dirty.push(pos);

        // Neighbors need rebuilt -- their boundary faces may change.
        self.dirty_neighbors(pos);
    }

    /// Remove a chunk and release its GPU resources.
    ///
    /// Frees the slot and quad allocation. Zeros the chunk metadata so
    /// the cull shader stops processing this slot.
    pub fn remove(&mut self, queue: &Queue, pos: ChunkPos) {
        if let Some(chunk) = self.chunks.remove(&pos) {
            if chunk.quad_alloc > 0 {
                self.allocator.free(chunk.quad_offset, chunk.quad_alloc);
            }

            if chunk.material_alloc > 0 {
                self.material_bufs.free(
                    chunk.material_offset, chunk.material_alloc,
                );
            }

            self.free_slot(chunk.slot);
            self.zero_chunk_meta(queue, chunk.slot);
        }

        self.dirty.retain(|&p| p != pos);

        // Neighbors' boundary faces are now exposed.
        self.dirty_neighbors(pos);

        self.recount_active_slots();
    }

    /// Remove all loaded chunks from the GPU.
    ///
    /// Frees every slot and quad allocation. Used when switching world
    /// generators to start with a clean slate.
    pub fn clear(&mut self, queue: &Queue) {
        let positions: Vec<ChunkPos> =
            self.chunks.keys().copied().collect();

        for pos in positions {
            self.remove(queue, pos);
        }
    }

    /// Upload new occupancy data for a chunk and mark it for rebuilding.
    ///
    /// The occupancy buffer is written immediately via the queue. The
    /// build shader is not dispatched until
    /// [`rebuild`](Self::rebuild) or [`dispatch_counts`](Self::dispatch_counts).
    pub fn update_occupancy(
        &mut self,
        queue : &Queue,
        pos   : ChunkPos,
        occ   : &[u32; 1024],
    )
    {
        let Some(chunk) = self.chunks.get_mut(&pos)
        else {
            return;
        };

        *chunk.occ = *occ;
        let slot   = chunk.slot;

        if !chunk.dirty {
            chunk.dirty = true;
            self.dirty.push(pos);
        }

        self.write_occupancy(queue, slot, occ);

        // Neighbors may need new boundary slices.
        self.dirty_neighbors(pos);
    }

    /// Returns whether a chunk is loaded on the GPU at `pos`.
    pub fn is_loaded(&self, pos: ChunkPos) -> bool {
        self.chunks.contains_key(&pos)
    }

    /// Returns the number of free slots.
    pub fn free_slots(&self) -> u32 {
        self.free_slots.len() as u32
    }

    /// Returns the number of free quads in the allocator.
    pub fn free_quads(&self) -> u32 {
        self.allocator.free_quads()
    }

    /// Dispatch count passes for a batch of dirty chunks.
    ///
    /// Uploads neighbor boundary slices, dispatches the build count
    /// compute shader for each chunk, copies quad counts to a staging
    /// buffer, and starts an async map. Results are harvested next frame
    /// by [`process_build_feedback`](Self::process_build_feedback).
    ///
    /// If a previous batch is still in flight (`build_pending`), this
    /// method returns immediately and the chunks remain dirty.
    pub fn dispatch_counts(
        &mut self,
        device       : &Device,
        queue        : &Queue,
        positions    : &[ChunkPos],
        rejected     : &HashSet<ChunkPos>,
        material_for : impl Fn(ChunkPos) -> Option<[u16; 32768]>,
    )
    {
        if self.build_pending {
            return;
        }

        // Collect the subset of requested positions that are actually
        // dirty and not already in flight.
        let to_build: Vec<ChunkPos> = positions.iter()
            .copied()
            .filter(|pos| {
                self.chunks.get(pos)
                    .is_some_and(|c| c.dirty && !c.count_in_flight)
            })
            .take(MAX_BUILDS_IN_FLIGHT as usize)
            .collect();

        if to_build.is_empty() {
            return;
        }

        // Upload neighbor boundary slices for chunks being counted.
        for &pos in &to_build {
            let slices = build_neighbor_slices(
                &self.chunks, pos, rejected,
            );

            if let Some(chunk) = self.chunks.get(&pos) {
                self.write_boundary(queue, chunk.slot, &slices);
            }
        }

        // Batch all count passes into one encoder.
        let mut encoder = device.create_command_encoder(
            &CommandEncoderDescriptor::default(),
        );

        for (i, &pos) in to_build.iter().enumerate() {
            // Stage material for this chunk.
            if let Some(mat) = material_for(pos) {
                queue.write_buffer(
                    &self.material_staging_buf,
                    u64::from(i as u32) * 65536,
                    bytemuck::cast_slice(&mat),
                );
            }

            let Some(chunk) = self.chunks.get(&pos)
            else {
                continue;
            };

            let slot = chunk.slot;

            // Zero the chunk_meta entry so the count starts at 0.
            encoder.clear_buffer(
                &self.chunk_meta_buf,
                u64::from(slot) * u64::from(CHUNK_META_BYTES),
                Some(u64::from(CHUNK_META_BYTES)),
            );

            // Dispatch the build count pass.
            {
                let mut pass = encoder.begin_compute_pass(
                    &ComputePassDescriptor {
                        label            : Some("build_count"),
                        timestamp_writes : None,
                    },
                );

                pass.set_pipeline(
                    self.build_count_pipeline.pipeline(),
                );
                pass.set_bind_group(0, &self.build_count_bg, &[]);

                let push = [slot, 0u32];
                pass.set_immediates(
                    0, bytemuck::cast_slice(&push),
                );

                pass.dispatch_workgroups(32, 6, 1);
            }

            // Copy the full ChunkMeta (quad_count + flags + sub_mask)
            // to the staging buffer at this chunk's index position.
            encoder.copy_buffer_to_buffer(
                &self.chunk_meta_buf,
                u64::from(slot) * u64::from(CHUNK_META_BYTES),
                &self.build_staging,
                u64::from(i as u32) * u64::from(BUILD_STAGING_ENTRY_BYTES),
                u64::from(BUILD_STAGING_ENTRY_BYTES),
            );

            self.pending_builds.push(PendingBuild {
                pos           : pos,
                slot          : slot,
                staging_index : i as u32,
            });
        }

        queue.submit(Some(encoder.finish()));

        // The count pass just wrote a new quad_count into chunk_meta_buf
        // on the GPU. The cull shader runs later this frame and would
        // read that count before the write pass has allocated quads or
        // set base_offset, producing garbage draws. Restore each
        // chunk's current CPU-side quad_count so:
        //   - New chunks (quad_count = 0): cull shader skips them.
        //   - Existing chunks (quad_count > 0): cull shader renders
        //     the old quads at the old base_offset for one more frame.
        // The staged writes are flushed at the next queue.submit (the
        // main frame encoder). process_build_feedback sets the correct
        // values next frame after the write pass.
        for pb in &self.pending_builds {
            let qc = self.chunks.get(&pb.pos)
                .map_or(0u32, |c| c.quad_count);

            queue.write_buffer(
                &self.chunk_meta_buf,
                u64::from(pb.slot) * u64::from(CHUNK_META_BYTES),
                bytemuck::bytes_of(&qc),
            );
        }

        // Mark counted chunks: clear dirty, set in-flight.
        for pb in &self.pending_builds {
            if let Some(chunk) = self.chunks.get_mut(&pb.pos) {
                chunk.dirty           = false;
                chunk.count_in_flight = true;
            }
        }

        // Start async map of the staging buffer.
        self.build_pending = true;
        self.build_ready.store(false, Ordering::Release);

        let ready = self.build_ready.clone();

        self.build_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                result.unwrap();
                ready.store(true, Ordering::Release);
            });

        // Clean up the dirty list.
        self.dirty.retain(|pos| {
            self.chunks.get(pos).is_some_and(|c| c.dirty)
        });
    }

    /// Harvest async build count results and dispatch write passes.
    ///
    /// Reads quad counts from the staging buffer mapped by the previous
    /// frame's [`dispatch_counts`](Self::dispatch_counts), allocates or
    /// resizes quad ranges, and dispatches the build write compute
    /// shader for each chunk. Updates `chunk_meta_buf` and
    /// `quad_range_buf` so the cull shader sees the new data.
    ///
    /// Call at the start of the frame, before `sync_dirty_to_gpu` and
    /// `dispatch_counts`.
    pub fn process_build_feedback(
        &mut self,
        device : &Device,
        queue  : &Queue,
    )
    {
        if !self.build_pending {
            return;
        }

        // Non-blocking poll to advance the map callback.
        let _ = device.poll(wgpu::PollType::Poll);

        if !self.build_ready.load(Ordering::Acquire) {
            return;
        }

        // Read quad counts and sub-block masks from the mapped staging
        // buffer. Each entry is a full ChunkMeta (16 bytes).
        let pending = std::mem::take(&mut self.pending_builds);

        struct BuildResult {
            quad_count  : u32,
            sub_mask_lo : u32,
            sub_mask_hi : u32,
        }

        let results: Vec<BuildResult>;

        {
            let slice = self.build_staging.slice(..);
            let data  = slice.get_mapped_range();

            results = pending.iter().map(|pb| {
                let off = pb.staging_index as usize
                        * BUILD_STAGING_ENTRY_BYTES as usize;

                BuildResult {
                    quad_count  : u32::from_le_bytes([
                        data[off],     data[off + 1],
                        data[off + 2], data[off + 3],
                    ]),
                    // Skip flags at offset+4..+8.
                    sub_mask_lo : u32::from_le_bytes([
                        data[off + 8],  data[off + 9],
                        data[off + 10], data[off + 11],
                    ]),
                    sub_mask_hi : u32::from_le_bytes([
                        data[off + 12], data[off + 13],
                        data[off + 14], data[off + 15],
                    ]),
                }
            }).collect();

            drop(data);
        }

        self.build_staging.unmap();
        self.build_pending = false;
        self.build_ready.store(false, Ordering::Release);

        // Allocate or resize quad ranges for each pending chunk.
        //
        // Collect (slot, base_offset, quad_count) for the write pass
        // dispatch that follows.
        struct WriteJob {
            slot          : u32,
            base_offset   : u32,
            quad_count    : u32,
            staging_index : u32,
            popcount      : u32,
        }

        let mut write_jobs: Vec<WriteJob> = Vec::with_capacity(pending.len());

        for (pb, br) in pending.iter().zip(results.iter()) {
            let quad_count = br.quad_count;
            let popcount   = br.sub_mask_lo.count_ones()
                           + br.sub_mask_hi.count_ones();

            // Phase 1: quad allocation and material free.
            let Some(chunk) = self.chunks.get_mut(&pb.pos)
            else {
                continue;
            };

            // Free old quad allocation if the new count doesn't fit.
            if quad_count > chunk.quad_alloc {
                if chunk.quad_alloc > 0 {
                    self.allocator.free(
                        chunk.quad_offset, chunk.quad_alloc,
                    );

                    chunk.quad_offset = 0;
                    chunk.quad_alloc  = 0;
                }

                if let Some(offset) = self.allocator.alloc(quad_count) {
                    chunk.quad_offset = offset;
                    chunk.quad_alloc  = quad_count;
                }
                else {
                    // Allocator exhausted. Re-mark dirty for retry.
                    chunk.quad_count      = 0;
                    chunk.count_in_flight = false;

                    if !chunk.dirty {
                        chunk.dirty = true;
                        self.dirty.push(pb.pos);
                    }

                    continue;
                }
            }

            chunk.quad_count = quad_count;
            let base_offset  = chunk.quad_offset;

            // Free old material allocation if needed.
            let needs_mat_alloc = if popcount > chunk.material_alloc {
                if chunk.material_alloc > 0 {
                    self.material_bufs.free(
                        chunk.material_offset, chunk.material_alloc,
                    );

                    chunk.material_offset = 0;
                    chunk.material_alloc  = 0;
                }

                popcount > 0
            }
            else {
                false
            };

            // Phase 2: material allocation (auto-grows on exhaustion).
            if needs_mat_alloc {
                let offset = self.material_bufs.alloc(device, popcount);
                let chunk  = self.chunks.get_mut(&pb.pos).unwrap();
                chunk.material_offset = offset;
                chunk.material_alloc  = popcount;
            }

            // Phase 3: write GPU metadata and collect write jobs.
            let chunk = self.chunks.get_mut(&pb.pos).unwrap();

            chunk.sub_mask = u64::from(br.sub_mask_lo)
                           | (u64::from(br.sub_mask_hi) << 32);

            // Write base_offset into quad_range_buf so the cull shader
            // can read it when constructing MDI entries.
            queue.write_buffer(
                &self.quad_range_buf,
                u64::from(pb.slot) * u64::from(QUAD_RANGE_BYTES) + 4,
                bytemuck::bytes_of(&base_offset),
            );

            // Write material range so the vertex shader can read it.
            let (seg_idx, local_bytes) =
                self.material_bufs.resolve(chunk.material_offset);
            let mat_range: [u32; 4] = [
                seg_idx,
                local_bytes,
                br.sub_mask_lo,
                br.sub_mask_hi,
            ];

            queue.write_buffer(
                &self.material_range_buf,
                u64::from(pb.slot) * u64::from(MATERIAL_RANGE_BYTES),
                bytemuck::cast_slice(&mat_range),
            );

            write_jobs.push(WriteJob {
                slot          : pb.slot,
                base_offset   : base_offset,
                quad_count    : quad_count,
                staging_index : pb.staging_index,
                popcount      : popcount,
            });
        }

        // Batch all write and material pack passes into one encoder.
        let has_work = write_jobs.iter()
            .any(|j| j.quad_count > 0 || j.popcount > 0);

        if has_work {
            let mut encoder = device.create_command_encoder(
                &CommandEncoderDescriptor::default(),
            );

            for job in &write_jobs {
                if job.quad_count == 0 {
                    continue;
                }

                let mut pass = encoder.begin_compute_pass(
                    &ComputePassDescriptor {
                        label            : Some("build_write"),
                        timestamp_writes : None,
                    },
                );

                pass.set_pipeline(
                    self.build_write_pipeline.pipeline(),
                );
                pass.set_bind_group(0, &self.build_write_bg, &[]);

                let push = [job.slot, job.base_offset];
                pass.set_immediates(
                    0, bytemuck::cast_slice(&push),
                );

                pass.dispatch_workgroups(32, 6, 1);
            }

            // Dispatch material pack passes for chunks with populated
            // sub-blocks.
            for job in &write_jobs {
                if job.popcount == 0 {
                    continue;
                }

                let mut pass = encoder.begin_compute_pass(
                    &ComputePassDescriptor {
                        label            : Some("material_pack"),
                        timestamp_writes : None,
                    },
                );

                pass.set_pipeline(
                    self.material_pack_pipeline.pipeline(),
                );
                pass.set_bind_group(0, &self.material_pack_bg, &[]);
                pass.set_bind_group(1, &self.material_array_rw_bg, &[]);

                let push = [
                    job.staging_index,
                    job.slot,
                ];
                pass.set_immediates(
                    0, bytemuck::cast_slice(&push),
                );

                pass.dispatch_workgroups(job.popcount, 1, 1);
            }

            queue.submit(Some(encoder.finish()));
        }

        // Rebuild material bind groups if the buffer grew.
        if self.material_bufs.generation() != self.material_gen {
            self.material_gen = self.material_bufs.generation();
            self.material_array_ro_bg = Self::create_material_array_bg(
                device,
                &self.material_array_bgl,
                self.material_bufs.buffers(),
            );

            self.material_array_rw_bg =
                self.material_pack_pipeline.create_array_bind_group(
                    device,
                    self.material_bufs.buffers(),
                );
        }

        // Write quad_count to chunk_meta_buf so the cull shader can
        // read it. Staged here, flushed at the next queue.submit
        // (the main frame encoder with cull + render).
        for job in &write_jobs {
            queue.write_buffer(
                &self.chunk_meta_buf,
                u64::from(job.slot) * u64::from(CHUNK_META_BYTES),
                bytemuck::bytes_of(&job.quad_count),
            );
        }

        // Clear in-flight flags for all pending chunks.
        for pb in &pending {
            if let Some(chunk) = self.chunks.get_mut(&pb.pos) {
                chunk.count_in_flight = false;
            }
        }

        self.dirty.retain(|pos| {
            self.chunks.get(pos).is_some_and(|c| c.dirty)
        });

        self.recount_active_slots();
    }

    /// Dispatch the build shader for all dirty chunks.
    ///
    /// Convenience wrapper around [`dispatch_counts`](Self::dispatch_counts)
    /// that processes every dirty chunk.
    pub fn rebuild(
        &mut self,
        device   : &Device,
        queue    : &Queue,
        rejected : &HashSet<ChunkPos>,
    )
    {
        let all_dirty: Vec<ChunkPos> = self.dirty.clone();
        self.dispatch_counts(
            device, queue, &all_dirty, rejected,
            |_| None,
        );
    }

    /// Returns dirty chunk positions that are not currently in flight.
    ///
    /// Chunks with a count pass in flight are excluded so they are not
    /// double-counted.
    pub fn dirty_positions(&self) -> Vec<ChunkPos> {
        self.dirty.iter()
            .copied()
            .filter(|pos| {
                self.chunks.get(pos)
                    .is_some_and(|c| !c.count_in_flight)
            })
            .collect()
    }

    /// Returns a snapshot of current rendering statistics.
    ///
    /// All values come from CPU-side state. No GPU queries are issued.
    pub fn stats(&self) -> RenderStats {
        let total_quads: u32 = self.chunks.values()
            .map(|c| c.quad_count)
            .sum();

        let quad_buf_used = u64::from(
            QUAD_BUF_QUADS - self.allocator.free_quads()
        ) * 4;

        let mat_cap = self.material_bufs.capacity();

        let material_buf_used = u64::from(
            mat_cap - self.material_bufs.free_units()
        ) * u64::from(MATERIAL_SUB_BLOCK);

        let material_buf_total = self.material_bufs.segment_byte_size()
            * u64::from(self.material_bufs.segment_count());

        // Per-slot buffer sizes (all pre-allocated for MAX_CHUNKS slots).
        let slot_buf_total =
            u64::from(MAX_CHUNKS) * u64::from(OCC_WORDS_PER_SLOT) * 4      // occupancy
          + u64::from(MAX_CHUNKS) * u64::from(BOUNDARY_WORDS_PER_SLOT) * 4 // boundary
          + u64::from(MAX_CHUNKS) * u64::from(CHUNK_META_BYTES)            // chunk meta
          + u64::from(MAX_CHUNKS) * u64::from(QUAD_RANGE_BYTES)            // quad range
          + u64::from(MAX_CHUNKS) * 16                                     // chunk offsets
          + u64::from(MAX_CHUNKS) * u64::from(MATERIAL_RANGE_BYTES)        // material range
          + u64::from(MAX_CHUNKS) * 16                                     // dst indirect
          + u64::from(MAX_CHUNKS) * u64::from(DRAW_DATA_BYTES);           // draw data

        // Small constant buffers (frustum, staging, counters).
        let small_bufs = 96 + 4 + 16 + 4;

        // Variable-size buffers (packed material + material staging).
        let material_bufs = material_buf_total
            + u64::from(MAX_BUILDS_IN_FLIGHT) * 65536;

        RenderStats {
            chunks_loaded      : self.chunks.len() as u32,
            chunks_visible     : self.visible_count,
            total_quads        : total_quads,
            quad_buf_used      : quad_buf_used,
            quad_buf_total     : QUAD_BUF_SIZE,
            material_buf_used  : material_buf_used,
            material_buf_total : material_buf_total,
            slot_buf_total     : slot_buf_total,
            gpu_mem_total      : QUAD_BUF_SIZE + slot_buf_total
                               + material_bufs + small_bufs,
        }
    }

    /// Record the frustum cull compute pass.
    ///
    /// Clears the draw count buffer, uploads frustum planes, and
    /// dispatches the cull compute shader. The cull shader iterates all
    /// slots directly, reading chunk_meta_buf to skip empty slots.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder to record into.
    /// * `queue`   - The queue for frustum plane upload.
    /// * `planes`  - The six frustum planes from the camera.
    pub fn dispatch_cull(
        &self,
        encoder : &mut CommandEncoder,
        queue   : &Queue,
        planes  : &[[f32; 4]; 6],
    )
    {
        if self.active_slot_count == 0 {
            return;
        }

        // Upload frustum planes.
        queue.write_buffer(
            &self.frustum_buf,
            0,
            bytemuck::cast_slice(planes.as_slice()),
        );

        // Clear the atomic draw count to zero.
        encoder.clear_buffer(&self.draw_count_buf, 0, None);

        // Frustum cull compute pass: iterate all slots, skip empty.
        {
            let mut pass = encoder.begin_compute_pass(
                &ComputePassDescriptor {
                    label            : Some("cull_frustum"),
                    timestamp_writes : None,
                },
            );

            pass.set_pipeline(self.cull_pipeline.pipeline());
            pass.set_bind_group(0, &self.cull_bg, &[]);

            // Push constant: total_slots (the cull shader iterates
            // [0, total_slots) and skips slots with quad_count == 0).
            pass.set_immediates(
                0,
                bytemuck::bytes_of(&MAX_CHUNKS),
            );

            let workgroups = (MAX_CHUNKS + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Copy the GPU-written visible draw count to the staging buffer.
    ///
    /// Call after the cull compute pass but before `encoder.finish()`.
    /// Results arrive one frame later via
    /// [`poll_visible_count`](Self::poll_visible_count).
    pub fn resolve_visible_count(&mut self, encoder: &mut CommandEncoder) {
        if self.active_slot_count == 0 {
            return;
        }

        // If a previous readback completed, consume it first so the
        // staging buffer is unmapped before we copy into it.
        if self.count_pending {
            if self.count_ready.load(Ordering::Acquire) {
                self.consume_count_readback();
            }
            else {
                // Previous map still in flight. Skip this frame.
                return;
            }
        }

        encoder.copy_buffer_to_buffer(
            &self.draw_count_buf, 0,
            &self.count_staging,  0,
            4,
        );
    }

    /// Request async readback of the visible draw count.
    ///
    /// Call after `queue.submit()`. The result becomes available in the
    /// next frame's [`poll_visible_count`](Self::poll_visible_count).
    pub fn request_count_readback(&mut self) {
        if self.active_slot_count == 0 || self.count_pending {
            return;
        }

        self.count_pending = true;
        self.count_ready.store(false, Ordering::Release);

        let ready = self.count_ready.clone();

        self.count_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                result.unwrap();
                ready.store(true, Ordering::Release);
            });
    }

    /// Poll for completed visible count readback.
    ///
    /// Call at the start of each frame. Updates the visible count if
    /// results are available.
    pub fn poll_visible_count(&mut self, device: &Device) {
        if !self.count_pending {
            return;
        }

        let _ = device.poll(wgpu::PollType::Poll);

        if self.count_ready.load(Ordering::Acquire) {
            self.consume_count_readback();
        }
    }

    /// Issue a single multi-draw-indirect-count call for all visible
    /// chunks.
    ///
    /// Sets the shared render bind group and dispatches all visible
    /// draws from the compacted indirect buffer. The caller must have
    /// already set the render pipeline on the pass.
    pub fn draw<'a>(&'a self, pass: &mut RenderPass<'a>) {
        if self.active_slot_count == 0 {
            return;
        }

        pass.set_bind_group(0, &self.render_bg, &[]);
        pass.set_bind_group(1, &self.material_array_ro_bg, &[]);
        pass.multi_draw_indirect_count(
            &self.dst_indirect_buf,
            0,
            &self.draw_count_buf,
            0,
            MAX_CHUNKS,
        );
    }

    // -----------------------------------------------------------------------
    // Private: material bind groups
    // -----------------------------------------------------------------------

    /// Create the read-only material buffer array bind group (set 1).
    fn create_material_array_bg(
        device  : &Device,
        layout  : &BindGroupLayout,
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
            label   : Some("material_array_ro_bg"),
            layout  : layout,
            entries : &[
                BindGroupEntry {
                    binding  : 0,
                    resource : BindingResource::BufferArray(&bindings),
                },
            ],
        })
    }

    // -----------------------------------------------------------------------
    // Private: slot management
    // -----------------------------------------------------------------------

    /// Allocate a slot index from the free stack.
    ///
    /// # Panics
    ///
    /// Panics if no slots are available.
    fn alloc_slot(&mut self) -> u32 {
        self.free_slots.pop().expect("chunk slots exhausted")
    }

    /// Return a slot to the free stack.
    fn free_slot(&mut self, slot: u32) {
        self.free_slots.push(slot);
    }

    // -----------------------------------------------------------------------
    // Private: shared buffer writes
    // -----------------------------------------------------------------------

    /// Write occupancy data to the shared occupancy buffer at a slot.
    fn write_occupancy(
        &self,
        queue : &Queue,
        slot  : u32,
        occ   : &[u32; 1024],
    )
    {
        let offset = u64::from(slot)
                   * u64::from(OCC_WORDS_PER_SLOT) * 4;

        queue.write_buffer(
            &self.occupancy_buf,
            offset,
            bytemuck::cast_slice(occ),
        );
    }

    /// Write neighbor boundary slices to the shared boundary cache at a slot.
    fn write_boundary(
        &self,
        queue  : &Queue,
        slot   : u32,
        slices : &[u32; NEIGHBOR_WORDS],
    )
    {
        let offset = u64::from(slot)
                   * u64::from(BOUNDARY_WORDS_PER_SLOT) * 4;

        queue.write_buffer(
            &self.boundary_cache_buf,
            offset,
            bytemuck::cast_slice(slices),
        );
    }

    /// Write a chunk's world-space voxel offset into the offset buffer.
    fn write_chunk_offset(
        &self,
        queue : &Queue,
        slot  : u32,
        pos   : ChunkPos,
    )
    {
        let data: [i32; 4] = [pos.x * 32, pos.y * 32, pos.z * 32, 0];

        queue.write_buffer(
            &self.chunk_offset_buf,
            u64::from(slot) * 16,
            bytemuck::cast_slice(&data),
        );
    }

    /// Zero the chunk metadata entry for a slot.
    ///
    /// Sets quad_count to 0 so the cull shader skips this slot.
    fn zero_chunk_meta(&self, queue: &Queue, slot: u32) {
        let zeros = [0u32; 4];

        queue.write_buffer(
            &self.chunk_meta_buf,
            u64::from(slot) * u64::from(CHUNK_META_BYTES),
            bytemuck::cast_slice(&zeros),
        );
    }

    // -----------------------------------------------------------------------
    // Private: visible count readback
    // -----------------------------------------------------------------------

    /// Read mapped staging data and update the visible count.
    fn consume_count_readback(&mut self) {
        let slice = self.count_staging.slice(..);
        let data  = slice.get_mapped_range();
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);

        self.count_staging.unmap();

        self.count_pending = false;
        self.count_ready.store(false, Ordering::Release);
        self.visible_count = count;
    }

    // -----------------------------------------------------------------------
    // Private: dirty tracking
    // -----------------------------------------------------------------------

    /// Mark the six cardinal neighbors of a position as dirty.
    ///
    /// Called when a chunk is inserted, removed, or has its occupancy
    /// updated, since neighboring chunks' boundary face derivation
    /// depends on this chunk's boundary layer.
    fn dirty_neighbors(&mut self, pos: ChunkPos) {
        for &[dx, dy, dz] in &[
            [ 1,  0,  0], [-1,  0,  0],
            [ 0,  1,  0], [ 0, -1,  0],
            [ 0,  0,  1], [ 0,  0, -1],
        ] {
            let npos = ChunkPos::new(pos.x + dx, pos.y + dy, pos.z + dz);

            if let Some(chunk) = self.chunks.get_mut(&npos) {
                if !chunk.dirty {
                    chunk.dirty = true;
                    self.dirty.push(npos);
                }
            }
        }
    }

    /// Recompute the count of slots with non-zero quad counts.
    ///
    /// Used as the upper bound for the cull shader dispatch and as a
    /// fast early-exit check for draw and cull methods.
    fn recount_active_slots(&mut self) {
        self.active_slot_count = self.chunks.values()
            .filter(|c| c.quad_count > 0)
            .count() as u32;
    }
}

// ---------------------------------------------------------------------------
// Neighbor boundary slices
// ---------------------------------------------------------------------------

/// Build neighbor boundary slices for a chunk's build shader.
///
/// Extract the 6 boundary layers from neighboring chunks' occupancy
/// shadows and pack them into a 192-word array matching the shader's
/// expected layout.
///
/// Unloaded neighbors default to fully occupied (all-ones) so that
/// boundary faces against the unloaded void are culled. Neighbors
/// in the `rejected` set are known-empty and produce zero slices,
/// keeping boundary faces visible against confirmed air.
fn build_neighbor_slices(
    chunks   : &HashMap<ChunkPos, GpuChunk>,
    pos      : ChunkPos,
    rejected : &HashSet<ChunkPos>,
) -> [u32; NEIGHBOR_WORDS]
{
    // Default: assume unloaded neighbors are solid (cull boundary
    // faces). Loaded and rejected neighbors overwrite their direction.
    let mut slices = [!0u32; NEIGHBOR_WORDS];

    // Look up a neighbor. Returns:
    //   Some(Some(occ)) — loaded, use actual occupancy
    //   Some(None)      — rejected (known empty), zero the slice
    //   None            — unloaded, leave as all-ones
    let neighbor = |dx: i32, dy: i32, dz: i32|
        -> Option<Option<&[u32; 1024]>>
    {
        let npos = ChunkPos::new(pos.x + dx, pos.y + dy, pos.z + dz);

        if let Some(c) = chunks.get(&npos) {
            return Some(Some(c.occ.as_ref()));
        }

        if rejected.contains(&npos) {
            return Some(None);
        }

        None
    };

    // +X: x=0 column of neighbor at (+1,0,0). Word[z], bit y.
    if let Some(data) = neighbor(1, 0, 0) {
        for z in 0..32 {
            let mut word = 0u32;
            if let Some(occ) = data {
                for y in 0..32 {
                    word |= (occ[z * 32 + y] & 1) << y;
                }
            }
            slices[DIR_POS_X + z] = word;
        }
    }

    // -X: x=31 column of neighbor at (-1,0,0). Word[z], bit y.
    if let Some(data) = neighbor(-1, 0, 0) {
        for z in 0..32 {
            let mut word = 0u32;
            if let Some(occ) = data {
                for y in 0..32 {
                    word |= ((occ[z * 32 + y] >> 31) & 1) << y;
                }
            }
            slices[DIR_NEG_X + z] = word;
        }
    }

    // +Y: y=0 row of neighbor at (0,+1,0). Word[z], bit x.
    if let Some(data) = neighbor(0, 1, 0) {
        for z in 0..32 {
            slices[DIR_POS_Y + z] = match data {
                Some(occ) => occ[z * 32],
                None      => 0,
            };
        }
    }

    // -Y: y=31 row of neighbor at (0,-1,0). Word[z], bit x.
    if let Some(data) = neighbor(0, -1, 0) {
        for z in 0..32 {
            slices[DIR_NEG_Y + z] = match data {
                Some(occ) => occ[z * 32 + 31],
                None      => 0,
            };
        }
    }

    // +Z: z=0 layer of neighbor at (0,0,+1). Word[y], bit x.
    if let Some(data) = neighbor(0, 0, 1) {
        match data {
            Some(occ) => {
                slices[DIR_POS_Z..DIR_POS_Z + 32]
                    .copy_from_slice(&occ[..32]);
            }
            None => {
                slices[DIR_POS_Z..DIR_POS_Z + 32].fill(0);
            }
        }
    }

    // -Z: z=31 layer of neighbor at (0,0,-1). Word[y], bit x.
    if let Some(data) = neighbor(0, 0, -1) {
        match data {
            Some(occ) => {
                slices[DIR_NEG_Z..DIR_NEG_Z + 32]
                    .copy_from_slice(&occ[992..]);
            }
            None => {
                slices[DIR_NEG_Z..DIR_NEG_Z + 32].fill(0);
            }
        }
    }

    slices
}
