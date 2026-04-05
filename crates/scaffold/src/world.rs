//! GPU world manager for chunk lifecycle and rendering.
//!
//! Manages the build compute pipeline, a shared block pool for quad
//! storage, and per-chunk GPU resources. Handles occupancy upload,
//! build dispatch, quad count readback, block allocation/trimming,
//! and draw call emission.

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
    util::DrawIndirectArgs,
};

use crate::build::{BuildPipeline, ChunkBuildData};
use crate::cull::CullPipeline;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Quads per block. Each quad is one u32 (4 bytes).
const BLOCK_SIZE: u32       = 256;

/// Bytes per block.
const BLOCK_BYTES: u64      = BLOCK_SIZE as u64 * 4;

/// Total blocks in the pool. 65536 blocks = 64 MB total quad storage.
const POOL_BLOCKS: u32      = 65536;

/// Maximum blocks any single chunk can use (worst case: 98304 / 256).
pub(crate) const MAX_CHUNK_BLOCKS: u32 = 384;

/// Maximum concurrent loaded chunks.
const MAX_CHUNKS: u32       = 4096;

/// Quads per page-table slot (blocks per slot * quads per block).
///
/// Used as the `first_instance` stride in indirect draw args so that
/// `instance_index / 256` yields the correct global block index.
const SLOT_INSTANCE_STRIDE: u32 = MAX_CHUNK_BLOCKS * BLOCK_SIZE;

/// Words per neighbor boundary slice (one 32x32 face layer).
const SLICE_WORDS: usize    = 32;

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
            color_rgba,
            texture_idx,
            face_offset,
            _pad : 0,
        });
    }

    (materials, face_table)
}

// ---------------------------------------------------------------------------
// QuadPool
// ---------------------------------------------------------------------------

/// Shared GPU quad storage pool with fixed-size block allocation.
///
/// One large buffer divided into blocks of [`BLOCK_SIZE`] quads each.
/// A CPU-side free stack manages block allocation. A separate page
/// table buffer maps per-chunk logical block indices to physical
/// block IDs in the pool.
struct QuadPool {
    /// The shared quad storage buffer.
    quad_buf            : Buffer,
    /// The page table mapping logical to physical block IDs.
    page_table          : Buffer,
    /// Per-slot chunk world offsets (`vec4<i32>` stride, xyz in voxel units).
    chunk_offset_buf    : Buffer,
    /// Volumetric material buffer for per-voxel block IDs.
    material_volume_buf : Buffer,
    /// Free block IDs (LIFO stack).
    free_blocks         : Vec<u32>,
    /// Free page table slot indices (LIFO stack).
    free_slots          : Vec<u32>,
}

// --- QuadPool ---

impl QuadPool {
    /// Create a new quad pool with all blocks and slots free.
    fn new(device: &Device) -> Self {
        let quad_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("quad_pool"),
            size               : u64::from(POOL_BLOCKS) * BLOCK_BYTES,
            usage              : BufferUsages::STORAGE,
            mapped_at_creation : false,
        });

        let page_table = device.create_buffer(&BufferDescriptor {
            label              : Some("page_table"),
            size               : u64::from(MAX_CHUNKS)
                               * u64::from(MAX_CHUNK_BLOCKS)
                               * 4,
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

        let material_volume_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("material_volume"),
            size               : u64::from(MAX_CHUNKS) * 32768,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        // Block 0 is reserved as a null block. Stale page table entries
        // are zeroed after trimming so that any overflow writes from the
        // build shader land in block 0 instead of corrupting another
        // chunk's storage.
        let free_blocks = (1..POOL_BLOCKS).rev().collect();
        let free_slots  = (0..MAX_CHUNKS).rev().collect();

        QuadPool {
            quad_buf            : quad_buf,
            page_table          : page_table,
            chunk_offset_buf    : chunk_offset_buf,
            material_volume_buf : material_volume_buf,
            free_blocks         : free_blocks,
            free_slots          : free_slots,
        }
    }

    /// Allocate `count` blocks from the pool.
    ///
    /// # Panics
    ///
    /// Panics if the pool has fewer than `count` free blocks.
    fn alloc_blocks(&mut self, count: u32) -> Vec<u32> {
        let len = self.free_blocks.len();
        assert!(
            count as usize <= len,
            "quad pool exhausted: need {count}, have {len}",
        );
        self.free_blocks.split_off(len - count as usize)
    }

    /// Return blocks to the pool.
    fn free_blocks(&mut self, blocks: &[u32]) {
        self.free_blocks.extend_from_slice(blocks);
    }

    /// Allocate a page table slot for a new chunk.
    ///
    /// # Panics
    ///
    /// Panics if no slots are available.
    fn alloc_slot(&mut self) -> u32 {
        self.free_slots.pop().expect("page table slots exhausted")
    }

    /// Free a page table slot.
    fn free_slot(&mut self, slot: u32) {
        self.free_slots.push(slot);
    }

    /// Write block IDs into the page table for a chunk's slot.
    fn write_page_table(
        &self,
        queue     : &Queue,
        slot      : u32,
        block_ids : &[u32],
    )
    {
        let offset = u64::from(slot)
                   * u64::from(MAX_CHUNK_BLOCKS)
                   * 4;

        queue.write_buffer(
            &self.page_table,
            offset,
            bytemuck::cast_slice(block_ids),
        );
    }

    /// Zero page table entries from `start` to `MAX_CHUNK_BLOCKS` for a slot.
    ///
    /// Called after trimming to invalidate stale entries that would
    /// otherwise point to freed blocks. Zeroed entries resolve to the
    /// null block (block 0), containing any overflow writes harmlessly.
    fn clear_page_table_tail(
        &self,
        queue : &Queue,
        slot  : u32,
        start : u32,
    )
    {
        let count = MAX_CHUNK_BLOCKS - start;

        if count == 0 {
            return;
        }

        let offset = u64::from(slot)
                   * u64::from(MAX_CHUNK_BLOCKS)
                   * 4
                   + u64::from(start) * 4;

        let zeros = vec![0u32; count as usize];

        queue.write_buffer(
            &self.page_table,
            offset,
            bytemuck::cast_slice(&zeros),
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

    /// Write per-voxel material data for a chunk's slot.
    fn write_material(
        &self,
        queue    : &Queue,
        slot     : u32,
        material : &[u8; 32768],
    )
    {
        queue.write_buffer(
            &self.material_volume_buf,
            u64::from(slot) * 32768,
            material,
        );
    }
}

// ---------------------------------------------------------------------------
// ChunkAlloc
// ---------------------------------------------------------------------------

/// Per-chunk block allocation state.
struct ChunkAlloc {
    /// Slot index in the page table.
    slot      : u32,
    /// Physical block IDs currently allocated to this chunk.
    block_ids : Vec<u32>,
}

// ---------------------------------------------------------------------------
// GpuChunk
// ---------------------------------------------------------------------------

/// Per-chunk GPU resources and rendering state.
struct GpuChunk {
    /// Build-stage GPU resources (occupancy, quad count, staging, compute bg).
    build : ChunkBuildData,
    /// Block allocation state for this chunk.
    alloc : ChunkAlloc,
    /// CPU-side occupancy shadow for neighbor boundary slice extraction.
    occ   : Box<[u32; 1024]>,
    /// Current quad count from the last completed build.
    count : u32,
    /// Whether this chunk's occupancy has changed since the last build.
    dirty : bool,
}

// ---------------------------------------------------------------------------
// RenderStats
// ---------------------------------------------------------------------------

/// Rendering statistics from the GPU world.
///
/// A snapshot of current chunk loading, quad generation, and pool
/// utilization. All values are derived from CPU-side state at zero cost.
pub struct RenderStats {
    /// Total loaded chunks.
    pub chunks_loaded     : u32,
    /// Chunks included in draw calls (non-zero quad count).
    pub chunks_drawn      : u32,
    /// Chunks visible after frustum culling (one frame latency).
    pub chunks_visible    : u32,
    /// Total quads across all chunks.
    pub total_quads       : u32,
    /// Pool blocks currently allocated.
    pub pool_blocks_used  : u32,
    /// Pool blocks total capacity.
    pub pool_blocks_total : u32,
    /// Page table slots currently allocated.
    pub pool_slots_used   : u32,
    /// Page table slots total capacity.
    pub pool_slots_total  : u32,
}

// ---------------------------------------------------------------------------
// GpuWorld
// ---------------------------------------------------------------------------

/// Manages GPU resources for all loaded chunks.
///
/// Owns the build compute pipeline, a shared block pool for quad
/// storage, and per-chunk GPU state. The typical per-frame workflow:
///
/// 1. Modify chunks via [`update_occupancy`](Self::update_occupancy).
/// 2. Call [`rebuild`](Self::rebuild) to dispatch the build shader for
///    dirty chunks and read back quad counts.
/// 3. In the render pass, call [`draw`](Self::draw) to issue draw calls.
pub struct GpuWorld {
    /// The shared build compute pipeline.
    build_pipeline     : BuildPipeline,
    /// The render bind group layout.
    render_bgl         : BindGroupLayout,
    /// The shared camera uniform buffer.
    camera_buf         : Buffer,
    /// Shared render bind group (camera + pool + page table + materials + textures).
    render_bg          : BindGroup,
    /// The shared block pool.
    pool               : QuadPool,
    /// Per-chunk GPU state.
    chunks             : HashMap<ChunkPos, GpuChunk>,
    /// Positions of chunks that need rebuilding.
    dirty              : Vec<ChunkPos>,
    /// The frustum culling compute pipeline.
    cull_pipeline      : CullPipeline,
    /// Source indirect draw buffer (CPU-written, all chunks with quads).
    src_indirect_buf   : Buffer,
    /// Output indirect draw buffer (GPU-written, visible chunks only).
    dst_indirect_buf   : Buffer,
    /// GPU-side atomic draw count for `multi_draw_indirect_count`.
    draw_count_buf     : Buffer,
    /// Frustum plane uniform buffer (6 x `vec4<f32>`).
    frustum_buf        : Buffer,
    /// Cull pass bind group.
    cull_bg            : BindGroup,
    /// Total source draws before culling.
    src_draw_count     : u32,
    /// Staging buffer for async readback of the visible draw count.
    count_staging      : Buffer,
    /// Set by the map callback when the visible count is ready.
    count_ready        : Arc<AtomicBool>,
    /// Whether a count readback request is currently in flight.
    count_pending      : bool,
    /// Last known visible chunk count (one frame latency).
    visible_count      : u32,
    /// The material property table buffer (per-block color + texture config).
    material_table_buf : Buffer,
    /// Per-face texture override table for non-uniform blocks.
    face_texture_buf   : Buffer,
    /// The block texture array view.
    texture_view       : TextureView,
    /// The texture sampler.
    tex_sampler        : wgpu::Sampler,
}

// --- GpuWorld ---

impl GpuWorld {
    /// Create a new GPU world manager.
    ///
    /// # Arguments
    ///
    /// * `device`         - The GPU device for pipeline and resource creation.
    /// * `queue`          - The queue for texture upload.
    /// * `render_bgl`     - The bind group layout used by the render pipeline.
    /// * `camera_buf`     - The shared camera uniform buffer. A handle clone
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
        device         : &Device,
        queue          : &Queue,
        render_bgl     : BindGroupLayout,
        camera_buf     : Buffer,
        materials      : &[GpuMaterial],
        face_textures  : &[u32],
        texture_pixels : &[u8],
        texture_size   : u32,
        texture_layers : u32,
    ) -> Self
    {
        let pool = QuadPool::new(device);

        // Create the material table buffer from the provided entries.
        let material_table_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("material_table"),
            size               : (materials.len() * std::mem::size_of::<GpuMaterial>())
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

        // Create the face texture buffer. Minimum 4 bytes for wgpu.
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

        // Create the block texture array.
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

        // Upload all layers in a single write.
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

        // Build the render bind group with all 9 bindings.
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
                    resource : pool.quad_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 2,
                    resource : pool.page_table.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 3,
                    resource : pool.chunk_offset_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 4,
                    resource : pool.material_volume_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 5,
                    resource : material_table_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 6,
                    resource : face_texture_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 7,
                    resource : BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding  : 8,
                    resource : BindingResource::Sampler(&tex_sampler),
                },
            ],
        });

        // Culling buffers: source (CPU-written), destination (GPU-written),
        // and an atomic draw count that feeds multi_draw_indirect_count.
        let src_indirect_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("src_indirect_buf"),
            size               : u64::from(MAX_CHUNKS) * 16,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        let dst_indirect_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("dst_indirect_buf"),
            size               : u64::from(MAX_CHUNKS) * 16,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::INDIRECT,
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

        let count_staging = device.create_buffer(&BufferDescriptor {
            label              : Some("cull_count_staging"),
            size               : 4,
            usage              : BufferUsages::COPY_DST
                               | BufferUsages::MAP_READ,
            mapped_at_creation : false,
        });

        let frustum_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("frustum_buf"),
            size               : 96,
            usage              : BufferUsages::UNIFORM
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        let cull_pipeline = CullPipeline::new(device);

        let cull_bg = cull_pipeline.create_bind_group(
            device,
            &frustum_buf,
            &src_indirect_buf,
            &pool.chunk_offset_buf,
            &dst_indirect_buf,
            &draw_count_buf,
        );

        GpuWorld {
            build_pipeline     : BuildPipeline::new(device),
            render_bgl         : render_bgl,
            camera_buf         : camera_buf,
            render_bg          : render_bg,
            pool               : pool,
            chunks             : HashMap::new(),
            dirty              : Vec::new(),
            cull_pipeline      : cull_pipeline,
            src_indirect_buf   : src_indirect_buf,
            dst_indirect_buf   : dst_indirect_buf,
            draw_count_buf     : draw_count_buf,
            frustum_buf        : frustum_buf,
            cull_bg            : cull_bg,
            src_draw_count     : 0,
            count_staging      : count_staging,
            count_ready        : Arc::new(AtomicBool::new(false)),
            count_pending      : false,
            visible_count      : 0,
            material_table_buf : material_table_buf,
            face_texture_buf   : face_texture_buf,
            texture_view       : texture_view,
            tex_sampler        : tex_sampler,
        }
    }

    /// Insert a chunk with initial occupancy and material data.
    ///
    /// Allocates a page table slot and block storage, writes the page
    /// table and material volume, and marks the chunk for building. Call
    /// [`rebuild`](Self::rebuild) to dispatch the build compute shader.
    ///
    /// # Arguments
    ///
    /// * `device`   - The GPU device for resource creation.
    /// * `queue`    - The queue for page table and material writes.
    /// * `pos`      - The chunk position.
    /// * `occ`      - Initial chunk occupancy bitmask.
    /// * `material` - Per-voxel block IDs (32768 bytes, one byte per voxel).
    pub fn insert(
        &mut self,
        device   : &Device,
        queue    : &Queue,
        pos      : ChunkPos,
        occ      : &[u32; 1024],
        material : &[u8; 32768],
    )
    {
        let slot      = self.pool.alloc_slot();
        let block_ids = self.pool.alloc_blocks(MAX_CHUNK_BLOCKS);

        // Write block IDs to the page table and chunk world offset.
        self.pool.write_page_table(queue, slot, &block_ids);
        self.pool.write_chunk_offset(queue, slot, pos);
        self.pool.write_material(queue, slot, material);

        let build = ChunkBuildData::new(
            device,
            &self.build_pipeline,
            occ,
            &self.pool.quad_buf,
            &self.pool.page_table,
        );

        self.chunks.insert(pos, GpuChunk {
            build : build,
            alloc : ChunkAlloc {
                slot      : slot,
                block_ids : block_ids,
            },
            occ   : Box::new(*occ),
            count : 0,
            dirty : true,
        });

        self.dirty.push(pos);

        // Neighbors need rebuilt -- their boundary faces may change.
        self.dirty_neighbors(pos);
    }

    /// Remove a chunk and release its GPU resources.
    ///
    /// Rebuilds the indirect draw buffer so that `draw` never references
    /// freed page table slots.
    pub fn remove(&mut self, queue: &Queue, pos: ChunkPos) {
        if let Some(chunk) = self.chunks.remove(&pos) {
            self.pool.free_blocks(&chunk.alloc.block_ids);
            self.pool.free_slot(chunk.alloc.slot);
        }

        self.dirty.retain(|&p| p != pos);

        // Neighbors' boundary faces are now exposed.
        self.dirty_neighbors(pos);

        self.rebuild_indirect(queue);
    }

    /// Remove all loaded chunks from the GPU.
    ///
    /// Frees every slot and zeroes occupancy for all positions. Used when
    /// switching world generators to start with a clean slate.
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
    /// build shader is not dispatched until [`rebuild`](Self::rebuild).
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
        chunk.build.upload_occupancy(queue, occ);

        if !chunk.dirty {
            chunk.dirty = true;
            self.dirty.push(pos);
        }

        // Neighbors may need new boundary slices.
        self.dirty_neighbors(pos);
    }

    /// Returns whether a chunk is loaded on the GPU at `pos`.
    pub fn is_loaded(&self, pos: ChunkPos) -> bool {
        self.chunks.contains_key(&pos)
    }

    /// Returns the number of free page table slots.
    pub fn free_slots(&self) -> u32 {
        self.pool.free_slots.len() as u32
    }

    /// Returns the number of free pool blocks.
    pub fn free_blocks(&self) -> u32 {
        self.pool.free_blocks.len() as u32
    }

    /// Upload new material data for an existing chunk.
    ///
    /// Use when voxel block types change without affecting occupancy.
    /// The chunk is not marked dirty for rebuild -- call
    /// [`update_occupancy`](Self::update_occupancy) separately if
    /// occupancy also changed.
    pub fn update_material(
        &mut self,
        queue    : &Queue,
        pos      : ChunkPos,
        material : &[u8; 32768],
    )
    {
        if let Some(chunk) = self.chunks.get(&pos) {
            self.pool.write_material(queue, chunk.alloc.slot, material);
        }
    }

    /// Rebuild a specific set of dirty chunks.
    ///
    /// Only chunks in `positions` that are currently marked dirty will be
    /// dispatched. Chunks not in the set remain dirty for a future call.
    /// After this call, the indirect draw buffer is rebuilt to reflect
    /// updated quad counts.
    pub fn rebuild_subset(
        &mut self,
        device   : &Device,
        queue    : &Queue,
        positions : &[ChunkPos],
        rejected : &HashSet<ChunkPos>,
    )
    {
        // Collect the subset of requested positions that are actually dirty.
        let to_build: Vec<ChunkPos> = positions.iter()
            .copied()
            .filter(|pos| {
                self.chunks.get(pos).is_some_and(|c| c.dirty)
            })
            .collect();

        if to_build.is_empty() {
            return;
        }

        // Upload neighbor boundary slices for chunks being rebuilt.
        for &pos in &to_build {
            let slices = build_neighbor_slices(
                &self.chunks, pos, rejected,
            );

            if let Some(chunk) = self.chunks.get(&pos) {
                chunk.build.upload_neighbor_slices(queue, &slices);
            }
        }

        // Encode all rebuilds into one command buffer.
        let mut encoder = device.create_command_encoder(
            &CommandEncoderDescriptor::default(),
        );

        for &pos in &to_build {
            if let Some(chunk) = self.chunks.get(&pos) {
                let block_base = chunk.alloc.slot * MAX_CHUNK_BLOCKS;
                chunk.build.dispatch(
                    &mut encoder, &self.build_pipeline, block_base,
                );
            }
        }

        queue.submit(Some(encoder.finish()));

        // Read back quad counts, handle overflow, and trim.
        for pos in to_build {
            let Some(chunk) = self.chunks.get_mut(&pos)
            else {
                continue;
            };

            let raw_count = chunk.build.read_quad_count(device);
            let needed    = ((raw_count + BLOCK_SIZE - 1) / BLOCK_SIZE)
                .max(1) as usize;
            let current   = chunk.alloc.block_ids.len();

            if needed <= current {
                // Normal case: enough blocks allocated.
                chunk.count = raw_count;
                chunk.dirty = false;

                if needed < current {
                    let excess = chunk.alloc.block_ids.split_off(needed);
                    self.pool.free_blocks(&excess);

                    // Zero freed page table entries so future overflow
                    // writes land in the null block (block 0).
                    self.pool.clear_page_table_tail(
                        queue, chunk.alloc.slot, needed as u32,
                    );
                }
            }
            else {
                // Overflow: the build produced more quads than blocks
                // allocated. Greedy merge is non-monotonic across
                // neighbor configurations, so a rebuild can exceed a
                // previous trim. Grow the allocation by the exact
                // deficit and re-dispatch to write all quads correctly.
                let deficit = (needed - current) as u32;

                if (self.pool.free_blocks.len() as u32) >= deficit {
                    let new_blocks = self.pool.alloc_blocks(deficit);
                    chunk.alloc.block_ids.extend_from_slice(&new_blocks);
                    self.pool.write_page_table(
                        queue,
                        chunk.alloc.slot,
                        &chunk.alloc.block_ids,
                    );

                    // Re-dispatch with the grown allocation.
                    let block_base = chunk.alloc.slot * MAX_CHUNK_BLOCKS;
                    let mut enc    = device.create_command_encoder(
                        &CommandEncoderDescriptor::default(),
                    );

                    chunk.build.dispatch(
                        &mut enc, &self.build_pipeline, block_base,
                    );
                    queue.submit(Some(enc.finish()));

                    chunk.count = chunk.build.read_quad_count(device);
                    chunk.dirty = false;

                    // Trim the retry result.
                    let needed2 = ((chunk.count + BLOCK_SIZE - 1)
                        / BLOCK_SIZE)
                        .max(1) as usize;

                    if needed2 < chunk.alloc.block_ids.len() {
                        let excess =
                            chunk.alloc.block_ids.split_off(needed2);
                        self.pool.free_blocks(&excess);
                        self.pool.clear_page_table_tail(
                            queue, chunk.alloc.slot, needed2 as u32,
                        );
                    }
                }
                else {
                    // Pool exhausted. Cap to allocated capacity and
                    // leave dirty for retry next frame.
                    chunk.count = current as u32 * BLOCK_SIZE;
                }
            }
        }

        // Remove rebuilt positions from the master dirty list.
        self.dirty.retain(|pos| {
            self.chunks.get(pos).is_some_and(|c| c.dirty)
        });

        self.rebuild_indirect(queue);
    }

    /// Dispatch the build shader for all dirty chunks and read back
    /// quad counts.
    ///
    /// Encodes compute dispatches into a single command buffer, submits
    /// it, then blocks until the GPU finishes. After this call all chunks
    /// have up-to-date quad counts, excess blocks are freed, and the
    /// dirty list is empty.
    pub fn rebuild(
        &mut self,
        device   : &Device,
        queue    : &Queue,
        rejected : &HashSet<ChunkPos>,
    )
    {
        let all_dirty: Vec<ChunkPos> = self.dirty.clone();
        self.rebuild_subset(device, queue, &all_dirty, rejected);
    }

    /// Returns the current list of dirty chunk positions.
    pub fn dirty_positions(&self) -> &[ChunkPos] {
        &self.dirty
    }

    /// Returns a snapshot of current rendering statistics.
    ///
    /// All values come from CPU-side state. No GPU queries are issued.
    pub fn stats(&self) -> RenderStats {
        let total_quads: u32 = self.chunks.values()
            .map(|c| c.count)
            .sum();

        RenderStats {
            chunks_loaded     : self.chunks.len() as u32,
            chunks_drawn      : self.src_draw_count,
            chunks_visible    : self.visible_count,
            total_quads       : total_quads,
            pool_blocks_used  : POOL_BLOCKS
                              - self.pool.free_blocks.len() as u32,
            pool_blocks_total : POOL_BLOCKS,
            pool_slots_used   : MAX_CHUNKS
                              - self.pool.free_slots.len() as u32,
            pool_slots_total  : MAX_CHUNKS,
        }
    }

    /// Record the frustum cull compute pass.
    ///
    /// Clears the draw count buffer, uploads frustum planes, and
    /// dispatches the cull compute shader. Must be called after
    /// `rebuild_indirect` and before the render pass.
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
        if self.src_draw_count == 0 {
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

        // Frustum cull compute pass.
        {
            let mut pass = encoder.begin_compute_pass(
                &ComputePassDescriptor {
                    label            : Some("cull_frustum"),
                    timestamp_writes : None,
                },
            );

            pass.set_pipeline(self.cull_pipeline.pipeline());
            pass.set_bind_group(0, &self.cull_bg, &[]);
            pass.set_immediates(
                0,
                bytemuck::bytes_of(&self.src_draw_count),
            );

            let workgroups = (self.src_draw_count + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Copy the GPU-written visible draw count to the staging buffer.
    ///
    /// Call after the cull compute pass but before `encoder.finish()`.
    /// Results arrive one frame later via [`poll_visible_count`](Self::poll_visible_count).
    pub fn resolve_visible_count(&mut self, encoder: &mut CommandEncoder) {
        if self.src_draw_count == 0 {
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
        if self.src_draw_count == 0 || self.count_pending {
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
    /// Call at the start of each frame. Updates `visible_count` if
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

    /// Issue a single multi-draw-indirect call for all chunks with quads.
    ///
    /// Sets the shared render bind group and dispatches all chunk draws
    /// from the indirect buffer. The caller must have already set the
    /// render pipeline on the pass.
    pub fn draw<'a>(&'a self, pass: &mut RenderPass<'a>) {
        if self.src_draw_count == 0 {
            return;
        }

        pass.set_bind_group(0, &self.render_bg, &[]);
        pass.multi_draw_indirect_count(
            &self.dst_indirect_buf,
            0,
            &self.draw_count_buf,
            0,
            self.src_draw_count,
        );
    }

    /// Rebuild the indirect draw buffer from all loaded chunks.
    ///
    /// Packs [`DrawIndirectArgs`] for every chunk with a non-zero quad
    /// count and writes them to the GPU indirect buffer.
    fn rebuild_indirect(&mut self, queue: &Queue) {
        let mut args: Vec<DrawIndirectArgs> = Vec::new();

        for chunk in self.chunks.values() {
            if chunk.count == 0 {
                continue;
            }

            args.push(DrawIndirectArgs {
                vertex_count   : 6,
                instance_count : chunk.count,
                first_vertex   : 0,
                first_instance : chunk.alloc.slot * SLOT_INSTANCE_STRIDE,
            });
        }

        self.src_draw_count = args.len() as u32;

        if !args.is_empty() {
            queue.write_buffer(
                &self.src_indirect_buf,
                0,
                bytemuck::cast_slice(&args),
            );
        }
    }

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
