//! Chunk loading, unloading, and GPU synchronization.
//!
//! The [`ChunkManager`] bridges the CPU [`World`] and GPU [`GpuWorld`],
//! loading chunks within a view distance of a center position and keeping
//! the GPU representation in sync with the source data. Chunk generation
//! runs in parallel via rayon with a per-frame time budget.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use glam::Vec3;
use rayon::prelude::*;
use voxel::chunk::{Chunk, ChunkPos};
use voxel::world::{ChunkProvider, World};
use wgpu::{Device, Queue};

use crate::world::{GpuWorld, MAX_CHUNK_BLOCKS};

// ---------------------------------------------------------------------------
// GenOutcome
// ---------------------------------------------------------------------------

/// Result of a single parallel chunk generation attempt.
enum GenOutcome {
    /// The provider produced a chunk.
    Generated(ChunkPos, Chunk),
    /// The provider returned `None` for this position.
    Rejected(ChunkPos),
    /// Generation was skipped because the frame budget expired.
    Skipped(ChunkPos),
}

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
    /// Positions where the provider returned `None`. Tracked to avoid
    /// re-queuing empty positions every time the load queue is rebuilt.
    rejected         : HashSet<ChunkPos>,
    /// Maximum chunk generations per frame.
    loads_per_frame  : usize,
    /// Maximum GPU rebuilds per frame.
    builds_per_frame : usize,
    /// Wall-clock budget for parallel chunk generation.
    gen_budget       : Duration,
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
    /// * `gen_budget`       - Wall-clock time budget for parallel chunk
    ///   generation. Workers that have not started when the budget
    ///   expires are skipped and retried next frame.
    pub fn new(
        view_distance    : i32,
        loads_per_frame  : usize,
        builds_per_frame : usize,
        gen_budget       : Duration,
    ) -> Self
    {
        ChunkManager {
            world            : World::new(),
            view_distance    : view_distance,
            center           : ChunkPos::new(0, 0, 0),
            load_queue       : Vec::new(),
            rejected         : HashSet::new(),
            loads_per_frame  : loads_per_frame,
            builds_per_frame : builds_per_frame,
            gen_budget       : gen_budget,
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
    /// 2. Loads up to `loads_per_frame` new chunks from the provider,
    ///    generating in parallel with a time budget.
    /// 3. Syncs dirty CPU chunks (from voxel edits) to the GPU.
    /// 4. Triggers GPU rebuild for up to `builds_per_frame` dirty chunks,
    ///    prioritized by distance to center.
    pub fn update(
        &mut self,
        provider : &(dyn ChunkProvider + Sync),
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

    /// Set the view distance in chunk units.
    ///
    /// If the distance changed, rebuilds the load queue so new positions
    /// within the expanded radius are queued. Chunks outside the new
    /// distance are unloaded on the next [`update`](Self::update) call.
    pub fn set_view_distance(&mut self, distance: i32) {
        if distance == self.view_distance {
            return;
        }

        self.view_distance = distance;
        self.rebuild_load_queue();
    }

    /// Returns the current center chunk position.
    pub fn center(&self) -> ChunkPos {
        self.center
    }

    /// Clear all loaded chunks and reset queues.
    ///
    /// Used when switching world generators. The caller should also clear
    /// all corresponding GPU chunks, then call
    /// [`set_center`](Self::set_center) to repopulate the load queue.
    pub fn reset(&mut self) {
        self.world      = World::new();
        self.load_queue.clear();
        self.rejected.clear();
        self.center     = ChunkPos::new(i32::MAX, i32::MAX, i32::MAX);
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    /// Recompute the load queue for the current center.
    ///
    /// Enumerates positions within a sphere capped to the smaller of the
    /// view distance and the radius that fits the remaining chunk budget.
    /// This prevents combinatorial explosion when the view distance is
    /// much larger than what the GPU can hold.
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

                    if self.world.chunk(pos).is_none()
                        && !self.rejected.contains(&pos)
                    {
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

        // Evict stale rejected positions outside the unload threshold
        // so the set doesn't grow unboundedly as the camera moves.
        self.rejected.retain(|pos| {
            let dx = pos.x - cx;
            let dy = pos.y - cy;
            let dz = pos.z - cz;
            dx * dx + dy * dy + dz * dz <= threshold
        });
    }

    /// Generate and load chunks in parallel using rayon.
    ///
    /// Pops up to `loads_per_frame` positions from the load queue,
    /// dispatches generation across the thread pool, and collects
    /// results. An atomic flag provides early exit: once the wall-clock
    /// budget expires, workers that have not yet started their
    /// generation skip immediately. In-progress workers finish
    /// naturally (bounded by single-chunk generation time). Skipped
    /// positions are pushed back onto the load queue for next frame.
    fn load_queued_chunks(
        &mut self,
        provider : &(dyn ChunkProvider + Sync),
        gpu      : &mut GpuWorld,
        device   : &Device,
        queue    : &Queue,
    )
    {
        // Determine batch size from per-frame budget and GPU capacity.
        let available_slots  = gpu.free_slots() as usize;
        let available_blocks = gpu.free_blocks() as usize
                             / MAX_CHUNK_BLOCKS as usize;
        let capacity         = available_slots.min(available_blocks);
        let max_loads        = self.loads_per_frame.min(capacity);

        // Pop nearest positions from the queue.
        let mut positions = Vec::with_capacity(max_loads);

        while positions.len() < max_loads {
            let Some(pos) = self.load_queue.pop()
            else {
                break;
            };

            // Skip if already loaded (can happen after center change
            // if the position was already in the world).
            if self.world.chunk(pos).is_some() {
                continue;
            }

            positions.push(pos);
        }

        if positions.is_empty() {
            return;
        }

        // Fork: generate chunks in parallel with early-exit flag.
        let deadline = Instant::now() + self.gen_budget;
        let expired  = AtomicBool::new(false);

        let outcomes: Vec<GenOutcome> = positions
            .par_iter()
            .map(|&pos| {
                if expired.load(Ordering::Relaxed) {
                    return GenOutcome::Skipped(pos);
                }

                match provider.generate(pos) {
                    Some(chunk) => {
                        if Instant::now() > deadline {
                            expired.store(true, Ordering::Relaxed);
                        }

                        GenOutcome::Generated(pos, chunk)
                    }

                    None => GenOutcome::Rejected(pos),
                }
            })
            .collect();

        // Join: process outcomes serially.
        for outcome in outcomes {
            match outcome {
                GenOutcome::Generated(pos, chunk) => {
                    let occ = chunk.occupancy_words();
                    let mat = chunk.material_block_ids();

                    self.world.insert_chunk(pos, chunk);
                    gpu.insert(device, queue, pos, &occ, &mat);
                }

                GenOutcome::Rejected(pos) => {
                    self.rejected.insert(pos);
                }

                GenOutcome::Skipped(pos) => {
                    self.load_queue.push(pos);
                }
            }
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

        gpu.rebuild_subset(device, queue, &prioritized, &self.rejected);
    }
}
