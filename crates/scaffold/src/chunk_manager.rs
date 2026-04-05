//! Chunk loading, unloading, and GPU synchronization.
//!
//! The [`ChunkManager`] bridges the CPU [`World`] and GPU [`GpuWorld`],
//! loading chunks within a view distance of a center position and keeping
//! the GPU representation in sync with the source data.

use glam::Vec3;
use voxel::chunk::ChunkPos;
use voxel::world::{ChunkProvider, World};
use wgpu::{Device, Queue};

use crate::world::{GpuWorld, MAX_CHUNK_BLOCKS};

// ---------------------------------------------------------------------------
// ChunkManager
// ---------------------------------------------------------------------------

/// Coordinates chunk loading, unloading, and GPU synchronization.
///
/// Tracks a center position (typically the camera) and a view distance
/// in chunk units. Each frame, determines which chunks should be loaded
/// or unloaded, generates new chunks via a [`ChunkProvider`], syncs
/// modified chunks to the GPU, and requests rebuilds within a per-frame
/// budget.
pub struct ChunkManager {
    /// The CPU-side world (source of truth for voxel data).
    world            : World,
    /// The view distance in chunk units.
    view_distance    : i32,
    /// The current center position in chunk coordinates.
    center           : ChunkPos,
    /// Positions queued for loading, sorted nearest-first.
    load_queue       : Vec<ChunkPos>,
    /// Maximum chunk generations per frame.
    loads_per_frame  : usize,
    /// Maximum GPU rebuilds per frame.
    builds_per_frame : usize,
}

// --- ChunkManager ---

impl ChunkManager {
    /// Create a chunk manager with the given view distance and per-frame
    /// budgets.
    ///
    /// # Arguments
    ///
    /// * `view_distance`    - Radius in chunk units around the center to
    ///   keep loaded.
    /// * `loads_per_frame`  - Maximum new chunks generated per frame.
    /// * `builds_per_frame` - Maximum GPU chunk rebuilds per frame.
    pub fn new(
        view_distance    : i32,
        loads_per_frame  : usize,
        builds_per_frame : usize,
    ) -> Self
    {
        ChunkManager {
            world            : World::new(),
            view_distance    : view_distance,
            center           : ChunkPos::new(0, 0, 0),
            load_queue       : Vec::new(),
            loads_per_frame  : loads_per_frame,
            builds_per_frame : builds_per_frame,
        }
    }

    /// Set the center position from a world-space floating point position.
    ///
    /// Converts to chunk coordinates and, if the center chunk changed,
    /// recomputes the load queue with positions sorted nearest-first.
    /// Chunks outside the view distance (plus hysteresis) are marked
    /// for unloading on the next [`update`](Self::update) call.
    pub fn set_center(&mut self, world_pos: Vec3) {
        let cx = (world_pos.x / 32.0).floor() as i32;
        let cy = (world_pos.y / 32.0).floor() as i32;
        let cz = (world_pos.z / 32.0).floor() as i32;

        let new_center = ChunkPos::new(cx, cy, cz);

        if new_center == self.center && !self.load_queue.is_empty() {
            return;
        }

        self.center = new_center;
        self.rebuild_load_queue();
    }

    /// Run one frame of chunk management.
    ///
    /// 1. Unloads chunks outside the view distance from both the CPU
    ///    world and the GPU.
    /// 2. Loads up to `loads_per_frame` new chunks from the provider.
    /// 3. Syncs dirty CPU chunks (from voxel edits) to the GPU.
    /// 4. Triggers GPU rebuild for up to `builds_per_frame` dirty chunks,
    ///    prioritized by distance to center.
    pub fn update(
        &mut self,
        provider : &dyn ChunkProvider,
        gpu      : &mut GpuWorld,
        device   : &Device,
        queue    : &Queue,
    )
    {
        self.unload_distant_chunks(gpu, queue);
        self.load_queued_chunks(provider, gpu, device, queue);
        self.sync_dirty_to_gpu(gpu, queue);
        self.rebuild_dirty(gpu, device, queue);
    }

    /// Returns a reference to the CPU world.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Returns a mutable reference to the CPU world.
    ///
    /// Voxel edits through the returned reference are tracked by the
    /// world's dirty list and synced to the GPU on the next
    /// [`update`](Self::update).
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Returns the current view distance in chunk units.
    pub fn view_distance(&self) -> i32 {
        self.view_distance
    }

    /// Returns the number of positions waiting in the load queue.
    pub fn load_queue_len(&self) -> usize {
        self.load_queue.len()
    }

    /// Returns the current center chunk position.
    pub fn center(&self) -> ChunkPos {
        self.center
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    /// Recompute the load queue for the current center.
    ///
    /// Enumerates all positions within the view distance sphere, filters
    /// out positions already loaded in the CPU world, and sorts by
    /// squared distance (nearest first).
    fn rebuild_load_queue(&mut self) {
        let vd  = self.view_distance;
        let vd2 = vd * vd;
        let cx  = self.center.x;
        let cy  = self.center.y;
        let cz  = self.center.z;

        self.load_queue.clear();

        for dx in -vd..=vd {
            for dy in -vd..=vd {
                for dz in -vd..=vd {
                    let dist2 = dx * dx + dy * dy + dz * dz;
                    if dist2 > vd2 {
                        continue;
                    }

                    let pos = ChunkPos::new(cx + dx, cy + dy, cz + dz);

                    if self.world.chunk(pos).is_none() {
                        self.load_queue.push(pos);
                    }
                }
            }
        }

        // Sort farthest-first so pop() yields nearest chunks.
        let center = self.center;
        self.load_queue.sort_by_key(|pos| {
            let dx = pos.x - center.x;
            let dy = pos.y - center.y;
            let dz = pos.z - center.z;
            std::cmp::Reverse(dx * dx + dy * dy + dz * dz)
        });
    }

    /// Remove chunks beyond the view distance from CPU and GPU.
    ///
    /// Uses `view_distance + 1` as the threshold to provide hysteresis
    /// and prevent load/unload thrashing at the boundary.
    fn unload_distant_chunks(
        &mut self,
        gpu   : &mut GpuWorld,
        queue : &Queue,
    )
    {
        let threshold = (self.view_distance + 1) * (self.view_distance + 1);
        let cx        = self.center.x;
        let cy        = self.center.y;
        let cz        = self.center.z;

        // Collect positions to unload (can't mutate while iterating).
        let to_unload: Vec<ChunkPos> = self.world.positions()
            .filter(|pos| {
                let dx = pos.x - cx;
                let dy = pos.y - cy;
                let dz = pos.z - cz;
                dx * dx + dy * dy + dz * dz > threshold
            })
            .collect();

        for pos in to_unload {
            self.world.remove_chunk(pos);
            gpu.remove(queue, pos);
        }
    }

    /// Load chunks from the front of the queue via the provider.
    ///
    /// Respects both the per-frame load budget and GPU pool capacity.
    fn load_queued_chunks(
        &mut self,
        provider : &dyn ChunkProvider,
        gpu      : &mut GpuWorld,
        device   : &Device,
        queue    : &Queue,
    )
    {
        let mut loaded = 0;

        while loaded < self.loads_per_frame {
            // Check GPU capacity before popping. If the pool is full,
            // stop and let the rebuild trim free blocks for next frame.
            if gpu.free_slots() == 0
                || gpu.free_blocks() < MAX_CHUNK_BLOCKS
            {
                break;
            }

            let Some(pos) = self.load_queue.pop()
            else {
                break;
            };

            // Skip if already loaded (can happen after center change
            // if the position was already in the world).
            if self.world.chunk(pos).is_some() {
                continue;
            }

            let Some(chunk) = provider.generate(pos)
            else {
                continue;
            };

            // Extract GPU-format data before inserting into the world.
            let occ = chunk.occupancy_words();
            let mat = chunk.material_block_ids();

            self.world.insert_chunk(pos, chunk);
            gpu.insert(device, queue, pos, &occ, &mat);

            loaded += 1;
        }
    }

    /// Sync voxel edits from the CPU world to the GPU.
    ///
    /// Drains the world's dirty list and re-uploads occupancy and
    /// material data for each modified chunk.
    fn sync_dirty_to_gpu(
        &mut self,
        gpu   : &mut GpuWorld,
        queue : &Queue,
    )
    {
        let dirty = self.world.drain_dirty();

        for pos in dirty {
            // Skip chunks not on the GPU (e.g. just inserted chunks
            // which were already uploaded during load).
            if !gpu.is_loaded(pos) {
                continue;
            }

            let Some(chunk) = self.world.chunk(pos)
            else {
                continue;
            };

            let occ = chunk.occupancy_words();
            let mat = chunk.material_block_ids();

            gpu.update_occupancy(queue, pos, &occ);
            gpu.update_material(queue, pos, &mat);
        }
    }

    /// Rebuild dirty GPU chunks within the per-frame budget.
    ///
    /// Prioritizes chunks nearest to the center so nearby geometry
    /// updates are visible first.
    fn rebuild_dirty(
        &mut self,
        gpu    : &mut GpuWorld,
        device : &Device,
        queue  : &Queue,
    )
    {
        let dirty = gpu.dirty_positions();
        if dirty.is_empty() {
            return;
        }

        // Sort dirty positions by distance to center, take the budget.
        let center = self.center;
        let mut prioritized: Vec<ChunkPos> = dirty.to_vec();

        prioritized.sort_by_key(|pos| {
            let dx = pos.x - center.x;
            let dy = pos.y - center.y;
            let dz = pos.z - center.z;
            dx * dx + dy * dy + dz * dz
        });

        prioritized.truncate(self.builds_per_frame);

        gpu.rebuild_subset(device, queue, &prioritized);
    }
}
