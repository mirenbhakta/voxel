//! Residency-driven view of the world for a single frame's render.
//!
//! [`WorldView`] owns the CPU residency state (a multi-level
//! [`Residency`]) and the GPU-side [`WorldRenderer`]. It exposes a single
//! `update` entry point that the main loop calls once per frame with the
//! current camera position; the view handles eviction, prep synthesis,
//! pool commits, and the CPU→GPU uploads that keep the render buffers in
//! sync with the resident set.
//!
//! # Slot namespaces
//!
//! Each level's [`SlotPool`](crate::world::pool::SlotPool) has its own
//! [`SlotId`](crate::world::pool::SlotId) space (`0..pool_capacity`). The
//! renderer's GPU occupancy buffer is a single shared array, so every
//! level gets an exclusive `[offset, offset + capacity)` range inside it;
//! the global slot for `(level_idx, pool_slot)` is `offset + pool_slot`.
//! Sum of per-level capacities must not exceed [`SUBCHUNK_MAX_CANDIDATES`].
//!
//! # LOD selection (deferred)
//!
//! Every level emits an instance for each of its resident sub-chunks.
//! Coarse-level sub-chunks that overlap finer levels' shells are not yet
//! skipped — visual overlap is a known limitation until the LOD-cascade
//! masking pass lands.

use std::sync::Arc;

use renderer::{
    LodMaskUniform, RendererContext, SUBCHUNK_MAX_CANDIDATES, SUBCHUNK_MAX_LEVELS,
    SubchunkInstance, WorldRenderer,
};

use crate::world::coord::{Level, SubchunkCoord};
use crate::world::pool::SlotId;
use crate::world::residency::{LevelConfig, OccupancySummary, Residency};
use crate::world::subchunk::{SUBCHUNK_EDGE, SubchunkOccupancy};

// --- WorldView ---

/// Per-frame integrator between the CPU residency and the GPU render path.
pub struct WorldView {
    residency:   Residency<SubchunkOccupancy>,
    renderer:    Arc<WorldRenderer>,
    /// Cloned configs so the view can recompute shell bounds each frame.
    configs:     Vec<LevelConfig>,
    /// Parallel to `configs`. `level_slots[i]` describes the GPU-side
    /// slot range assigned to `configs[i]`.
    level_slots: Vec<LevelSlotRange>,
    /// Shadow of the GPU instance array. Rebuilt from every level's
    /// occupied pool each update; indices past the resident count are
    /// left as sentinels with `exposure_mask = 0` so the cull shader
    /// rejects them trivially.
    instances:   [SubchunkInstance; SUBCHUNK_MAX_CANDIDATES],

    // --- Eviction bookkeeping ---
    evicted_last_update: usize,
    evicted_total:       u64,
}

impl WorldView {
    /// Construct the view with the given levels, centered on
    /// `initial_camera`. Runs the initial prep-and-upload cycle so the
    /// renderer has resident data before the first frame.
    ///
    /// # Panics
    ///
    /// Panics if the total resident capacity across all levels exceeds
    /// [`SUBCHUNK_MAX_CANDIDATES`].
    pub fn new(
        ctx:            &RendererContext,
        configs:        &[LevelConfig],
        initial_camera: [f32; 3],
    )
        -> Self
    {
        let mut level_slots = Vec::with_capacity(configs.len());
        let mut offset      = 0u32;
        for cfg in configs {
            let capacity = (2 * cfg.radius[0])
                         * (2 * cfg.radius[1])
                         * (2 * cfg.radius[2]);
            level_slots.push(LevelSlotRange { offset, capacity });
            offset = offset
                .checked_add(capacity)
                .expect("level slot offsets overflow u32");
        }

        assert!(
            offset as usize <= SUBCHUNK_MAX_CANDIDATES,
            "total resident capacity {offset} exceeds MAX_CANDIDATES ({SUBCHUNK_MAX_CANDIDATES})",
        );

        // The LOD-cascade mask assumes contiguous, nested configs.
        // `build_lod_mask` derives each level's mask entry from the
        // next-finer entry in `configs`, so callers passing e.g. L0 + L2
        // (skipping L1) get a mask that doesn't protect L2 from L0
        // overlap. Validate the contiguity invariant up front.
        for w in configs.windows(2) {
            assert_eq!(
                w[1].level.0, w[0].level.0 + 1,
                "WorldView requires contiguous levels; got {:?} then {:?}",
                w[0].level, w[1].level,
            );
        }
        assert!(
            configs.iter().all(|c| (c.level.0 as usize) < SUBCHUNK_MAX_LEVELS),
            "level index must fit in the pipeline's LOD table ({SUBCHUNK_MAX_LEVELS} entries)",
        );
        // Radii must be even for `L_(n-1)` shell to align to an integer
        // cluster of `L_n` sub-chunks — otherwise the cull's "fully inside
        // finer shell" test straddles boundaries and culls nothing. See
        // `decision-lod-nested-shells-hierarchical-occupancy` for the
        // geometric argument.
        assert!(
            configs.iter().all(|c| c.radius.iter().all(|r| *r >= 2 && r % 2 == 0)),
            "WorldView requires every radius component to be even and >= 2 for nested LOD",
        );

        let residency = Residency::<SubchunkOccupancy>::new(configs, initial_camera);
        let renderer  = Arc::new(WorldRenderer::new(ctx));

        let mut view = Self {
            residency,
            renderer,
            configs:     configs.to_vec(),
            level_slots,
            instances: [SubchunkInstance::new([0, 0, 0], 0, 0, 0); SUBCHUNK_MAX_CANDIDATES],
            evicted_last_update: 0,
            evicted_total:       0,
        };

        view.process_pending_prep(ctx);
        view.rebuild_instances(ctx);
        view.upload_lod_mask(ctx);
        view
    }

    /// Borrow the GPU renderer for render-graph registration.
    pub fn renderer(&self) -> &Arc<WorldRenderer> {
        &self.renderer
    }

    pub fn evicted_last_update(&self) -> usize {
        self.evicted_last_update
    }

    pub fn evicted_total(&self) -> u64 {
        self.evicted_total
    }

    /// Update residency against the current camera position and resync the
    /// GPU buffers.
    pub fn update(&mut self, ctx: &RendererContext, world_pos: [f32; 3]) {
        let evictions = self.residency.update_camera(world_pos);
        self.evicted_last_update = evictions.len();
        self.evicted_total       = self.evicted_total.saturating_add(evictions.len() as u64);
        drop(evictions);

        self.process_pending_prep(ctx);
        self.rebuild_instances(ctx);
        self.upload_lod_mask(ctx);
    }

    // --- internal ---

    /// Drain every outstanding prep request, synthesize occupancy, upload
    /// the result into the right global slot, and commit it into the
    /// residency pool.
    fn process_pending_prep(&mut self, ctx: &RendererContext) {
        let requests = self.residency.take_prep_requests();
        for req in requests {
            let Some(level_idx) = self.level_index(req.level) else {
                // Residency emitted a request for a level we don't track;
                // safest to drop without polluting pool state.
                debug_assert!(false, "prep request for unknown level {:?}", req.level);
                continue;
            };
            let global = self.global_slot(level_idx, req.slot);

            let occ     = synthesize_occupancy(req.coord, req.level);
            let summary = summarize(&occ);
            let bytes   = occ.to_gpu_bytes();

            // Upload before commit so the GPU slot is populated before the
            // CPU pool starts handing the coord out to consumers.
            self.renderer.write_occupancy_slot(ctx, global, &bytes);
            self.residency.complete_prep(req.id, summary, occ);
        }
    }

    /// Rebuild `instances[..]` from every level's occupied slots and push
    /// the full array to the GPU.
    fn rebuild_instances(&mut self, ctx: &RendererContext) {
        // Reset to sentinel so any unused tail is rejected by the cull shader.
        self.instances.fill(SubchunkInstance::new([0, 0, 0], 0, 0, 0));

        let mut i = 0usize;
        for (level_idx, cfg) in self.configs.iter().enumerate() {
            let level = cfg.level;
            let Some(pool) = self.residency.pool(level) else { continue; };
            let base = self.level_slots[level_idx].offset;
            for (coord, pool_slot, occ) in pool.occupied() {
                if i >= SUBCHUNK_MAX_CANDIDATES {
                    debug_assert!(false, "resident count exceeds MAX_CANDIDATES");
                    break;
                }
                self.instances[i] = SubchunkInstance::new(
                    world_origin(coord, level),
                    base + pool_slot.0,
                    level.0,
                    occ.isolated_exposure_mask(),
                );
                i += 1;
            }
        }

        self.renderer.write_instances(ctx, &self.instances);
    }

    /// Compute and upload the per-level LOD-cascade mask.
    ///
    /// Entry `[cfg.level]` of the mask is set to the shell bounds of the
    /// next-finer configured level (the one immediately below in
    /// `self.configs`). Level 0 and any unconfigured level stay inactive.
    fn upload_lod_mask(&self, ctx: &RendererContext) {
        let mut mask = LodMaskUniform::inactive();

        // Levels beyond index 0 have a configured finer level at index i-1.
        for i in 1..self.configs.len() {
            let this_cfg  = &self.configs[i];
            let finer_cfg = &self.configs[i - 1];
            let finer_level = finer_cfg.level;

            let Some(corner) = self.residency.corner(finer_level) else { continue; };
            let (lo_w, hi_w) = shell_world_bounds(corner, finer_cfg.radius, finer_level);

            let idx = this_cfg.level.0 as usize;
            mask.mask_lo[idx] = [lo_w[0], lo_w[1], lo_w[2], 0.0];
            mask.mask_hi[idx] = [hi_w[0], hi_w[1], hi_w[2], 1.0];
        }

        self.renderer.write_lod_mask(ctx, &mask);
    }

    fn level_index(&self, level: Level) -> Option<usize> {
        self.configs.iter().position(|c| c.level == level)
    }

    fn global_slot(&self, level_idx: usize, pool_slot: SlotId) -> u32 {
        let range = &self.level_slots[level_idx];
        debug_assert!(
            pool_slot.0 < range.capacity,
            "pool slot {} at level idx {level_idx} exceeds capacity {}",
            pool_slot.0,
            range.capacity,
        );
        range.offset + pool_slot.0
    }
}

// --- LevelSlotRange ---

struct LevelSlotRange {
    offset:   u32,
    capacity: u32,
}

// --- helpers ---

/// World-space voxel origin of a sub-chunk at `(level, coord)`, in L0
/// voxel units (= metres, since L0 voxels are 1 m).
///
/// A sub-chunk at level N spans `8 * 2^N` voxels per axis; its origin is
/// `coord * 8 * 2^N`.
fn world_origin(coord: SubchunkCoord, level: Level) -> [i32; 3] {
    let extent = (SUBCHUNK_EDGE as i32) << level.0;
    [coord.x * extent, coord.y * extent, coord.z * extent]
}

/// World-space AABB of a level's clipmap shell.
///
/// The shell at level N spans coords `[corner - radius, corner + radius - 1]`
/// per axis (inclusive-exclusive — `2 * radius` sub-chunks per axis).
/// Each sub-chunk at level N spans `8 * 2^N` world units, so the shell's
/// world box extends from `(corner - radius) * 8 * 2^N` to
/// `(corner + radius) * 8 * 2^N`.
fn shell_world_bounds(
    corner: SubchunkCoord,
    radius: [u32; 3],
    level:  Level,
) -> ([f32; 3], [f32; 3]) {
    let extent = (SUBCHUNK_EDGE as i32) << level.0;
    let lo_c = [
        corner.x - radius[0] as i32,
        corner.y - radius[1] as i32,
        corner.z - radius[2] as i32,
    ];
    let hi_c = [
        corner.x + radius[0] as i32,
        corner.y + radius[1] as i32,
        corner.z + radius[2] as i32,
    ];
    let lo = [
        (lo_c[0] * extent) as f32,
        (lo_c[1] * extent) as f32,
        (lo_c[2] * extent) as f32,
    ];
    let hi = [
        (hi_c[0] * extent) as f32,
        (hi_c[1] * extent) as f32,
        (hi_c[2] * extent) as f32,
    ];
    (lo, hi)
}

fn summarize(occ: &SubchunkOccupancy) -> OccupancySummary {
    if occ.is_empty() {
        OccupancySummary::Empty
    }
    else if occ.is_full() {
        OccupancySummary::Full
    }
    else {
        OccupancySummary::Mixed
    }
}

/// Stand-in procedural content: voxelize a continuous world-space terrain
/// field at the sub-chunk's LOD resolution.
///
/// Sampling the same `terrain_density` field at every level means coarser
/// LOD sub-chunks look like lower-resolution versions of the finer ones —
/// LOD transitions show up as voxel-size stairstepping of the same surface
/// rather than as unrelated content. Not a worldgen implementation; real
/// generation will replace this once GPU-side prep lands.
fn synthesize_occupancy(coord: SubchunkCoord, level: Level) -> SubchunkOccupancy {
    let voxel_size_m = level.voxel_size_m();
    let base_x = (coord.x as f32) * 8.0 * voxel_size_m;
    let base_y = (coord.y as f32) * 8.0 * voxel_size_m;
    let base_z = (coord.z as f32) * 8.0 * voxel_size_m;

    let mut occ = SubchunkOccupancy::empty();
    for z in 0u8..8 {
        for y in 0u8..8 {
            for x in 0u8..8 {
                let wx = base_x + (x as f32 + 0.5) * voxel_size_m;
                let wy = base_y + (y as f32 + 0.5) * voxel_size_m;
                let wz = base_z + (z as f32 + 0.5) * voxel_size_m;
                if terrain_occupied(wx, wy, wz) {
                    occ.set(x, y, z, true);
                }
            }
        }
    }
    occ
}

/// Continuous world-space terrain test: a voxel's world-space centre is
/// solid iff it lies below a layered-sinusoid ground surface.
///
/// Height field: two low-frequency sinusoids (coarse rolling terrain) plus
/// two high-frequency ones (fine detail) plus a constant offset, so the
/// surface near the origin sits around `y = 0`. Purely a placeholder shape
/// that varies across many wavelengths.
fn terrain_occupied(wx: f32, wy: f32, wz: f32) -> bool {
    let h = (wx * 0.05).sin() * 4.0
          + (wz * 0.05).cos() * 4.0
          + (wx * 0.20).sin() * 1.0
          + (wz * 0.20).cos() * 1.0
          - 5.0;
    wy < h
}
