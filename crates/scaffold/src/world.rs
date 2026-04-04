//! GPU world manager for chunk lifecycle and rendering.
//!
//! Manages the build compute pipeline, a shared block pool for quad
//! storage, and per-chunk GPU resources. Handles occupancy upload,
//! build dispatch, quad count readback, block allocation/trimming,
//! and draw call emission.

use std::collections::HashMap;

use voxel::chunk::ChunkPos;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    Device, Queue, RenderPass,
    util::DrawIndirectArgs,
};

use crate::build::{BuildPipeline, ChunkBuildData};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Quads per block. Each quad is one u32 (4 bytes).
const BLOCK_SIZE: u32       = 256;

/// Bytes per block.
const BLOCK_BYTES: u64      = BLOCK_SIZE as u64 * 4;

/// Total blocks in the pool. 4096 blocks = 4 MB total quad storage.
const POOL_BLOCKS: u32      = 4096;

/// Maximum blocks any single chunk can use (worst case: 98304 / 256).
const MAX_CHUNK_BLOCKS: u32 = 384;

/// Maximum concurrent loaded chunks.
const MAX_CHUNKS: u32       = 256;

/// Quads per page-table slot (blocks per slot * quads per block).
///
/// Used as the `first_instance` stride in indirect draw args so that
/// `instance_index / 256` yields the correct global block index.
const SLOT_INSTANCE_STRIDE: u32 = MAX_CHUNK_BLOCKS * BLOCK_SIZE;

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
    quad_buf    : Buffer,
    /// The page table mapping logical to physical block IDs.
    page_table  : Buffer,
    /// Free block IDs (LIFO stack).
    free_blocks : Vec<u32>,
    /// Free page table slot indices (LIFO stack).
    free_slots  : Vec<u32>,
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

        let free_blocks = (0..POOL_BLOCKS).rev().collect();
        let free_slots  = (0..MAX_CHUNKS).rev().collect();

        QuadPool {
            quad_buf    : quad_buf,
            page_table  : page_table,
            free_blocks : free_blocks,
            free_slots  : free_slots,
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
    /// Current quad count from the last completed build.
    count : u32,
    /// Whether this chunk's occupancy has changed since the last build.
    dirty : bool,
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
    build_pipeline : BuildPipeline,
    /// The render bind group layout.
    render_bgl     : BindGroupLayout,
    /// The shared camera uniform buffer.
    camera_buf     : Buffer,
    /// Shared render bind group (camera + quad pool + page table).
    render_bg      : BindGroup,
    /// The shared block pool.
    pool           : QuadPool,
    /// Per-chunk GPU state.
    chunks         : HashMap<ChunkPos, GpuChunk>,
    /// Positions of chunks that need rebuilding.
    dirty          : Vec<ChunkPos>,
    /// Packed `DrawIndirectArgs` for all drawable chunks.
    indirect_buf   : Buffer,
    /// Number of valid draw commands in `indirect_buf`.
    draw_count     : u32,
}

// --- GpuWorld ---

impl GpuWorld {
    /// Create a new GPU world manager.
    ///
    /// # Arguments
    ///
    /// * `device`     - The GPU device for pipeline and resource creation.
    /// * `render_bgl` - The bind group layout used by the render pipeline.
    ///   Binding 0 is a camera uniform, binding 1 is the shared quad pool,
    ///   and binding 2 is the page table.
    /// * `camera_buf` - The shared camera uniform buffer. A handle clone
    ///   is stored internally. The caller retains ownership for writing
    ///   camera updates.
    pub fn new(
        device     : &Device,
        render_bgl : BindGroupLayout,
        camera_buf : Buffer,
    ) -> Self
    {
        let pool = QuadPool::new(device);

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
            ],
        });

        let indirect_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("indirect_buf"),
            size               : u64::from(MAX_CHUNKS) * 16,
            usage              : BufferUsages::INDIRECT
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        GpuWorld {
            build_pipeline : BuildPipeline::new(device),
            render_bgl     : render_bgl,
            camera_buf     : camera_buf,
            render_bg      : render_bg,
            pool           : pool,
            chunks         : HashMap::new(),
            dirty          : Vec::new(),
            indirect_buf   : indirect_buf,
            draw_count     : 0,
        }
    }

    /// Insert a chunk with initial occupancy data.
    ///
    /// Allocates a page table slot and block storage, writes the page
    /// table, and marks the chunk for building. Call
    /// [`rebuild`](Self::rebuild) to dispatch the build compute shader.
    ///
    /// # Arguments
    ///
    /// * `device` - The GPU device for resource creation.
    /// * `queue`  - The queue for page table writes.
    /// * `pos`    - The chunk position.
    /// * `occ`    - Initial chunk occupancy bitmask.
    pub fn insert(
        &mut self,
        device : &Device,
        queue  : &Queue,
        pos    : ChunkPos,
        occ    : &[u32; 1024],
    )
    {
        let slot      = self.pool.alloc_slot();
        let block_ids = self.pool.alloc_blocks(MAX_CHUNK_BLOCKS);

        // Write block IDs to the page table.
        self.pool.write_page_table(queue, slot, &block_ids);

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
            count : 0,
            dirty : true,
        });

        self.dirty.push(pos);
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
        self.rebuild_indirect(queue);
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

        chunk.build.upload_occupancy(queue, occ);

        if !chunk.dirty {
            chunk.dirty = true;
            self.dirty.push(pos);
        }
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
        device : &Device,
        queue  : &Queue,
    )
    {
        if self.dirty.is_empty() {
            return;
        }

        // Encode all dirty chunk rebuilds into one command buffer.
        let mut encoder = device.create_command_encoder(
            &CommandEncoderDescriptor::default(),
        );

        for &pos in &self.dirty {
            if let Some(chunk) = self.chunks.get(&pos) {
                let block_base = chunk.alloc.slot * MAX_CHUNK_BLOCKS;
                chunk.build.dispatch(
                    &mut encoder, &self.build_pipeline, block_base,
                );
            }
        }

        queue.submit(Some(encoder.finish()));

        // Read back quad counts and trim excess block allocations.
        for pos in self.dirty.drain(..) {
            if let Some(chunk) = self.chunks.get_mut(&pos) {
                chunk.count = chunk.build.read_quad_count(device);
                chunk.dirty = false;

                // Trim excess blocks. Keep at least one block so the
                // page table entry remains valid for zero-count chunks.
                let needed = (chunk.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let needed = needed.max(1) as usize;

                if needed < chunk.alloc.block_ids.len() {
                    let excess = chunk.alloc.block_ids.split_off(needed);
                    self.pool.free_blocks(&excess);
                }
            }
        }

        // Rebuild the indirect draw buffer from all chunks.
        self.rebuild_indirect(queue);
    }

    /// Issue a single multi-draw-indirect call for all chunks with quads.
    ///
    /// Sets the shared render bind group and dispatches all chunk draws
    /// from the indirect buffer. The caller must have already set the
    /// render pipeline on the pass.
    pub fn draw<'a>(&'a self, pass: &mut RenderPass<'a>) {
        if self.draw_count == 0 {
            return;
        }

        pass.set_bind_group(0, &self.render_bg, &[]);
        pass.multi_draw_indirect(&self.indirect_buf, 0, self.draw_count);
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

        self.draw_count = args.len() as u32;

        if !args.is_empty() {
            queue.write_buffer(
                &self.indirect_buf,
                0,
                bytemuck::cast_slice(&args),
            );
        }
    }
}
