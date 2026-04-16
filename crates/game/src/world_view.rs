//! Residency-driven view of the world for a single frame's render.
//!
//! [`WorldView`] owns the CPU residency state (a [`Residency`] of L0
//! sub-chunks) and the GPU-side [`WorldRenderer`]. It exposes a single
//! `update` entry point that the main loop calls once per frame with the
//! current camera position; the view handles eviction, prep synthesis,
//! pool commits, and the CPU→GPU uploads that keep the render buffers in
//! sync with the resident set.
//!
//! The current implementation is single-level (L0 only), content-synthesis
//! is a simple centered-sphere-per-sub-chunk (stand-in for real worldgen),
//! and evictions are discarded. All three are slated to grow as further
//! slices land.

use std::sync::Arc;

use renderer::{
    RendererContext, SUBCHUNK_MAX_CANDIDATES, SubchunkInstance, WorldRenderer,
};

use crate::world::coord::{Level, SubchunkCoord};
use crate::world::residency::{LevelConfig, OccupancySummary, Residency};
use crate::world::subchunk::{SUBCHUNK_EDGE, SubchunkOccupancy};

// --- WorldView ---

/// Per-frame integrator between the CPU residency and the GPU render path.
pub struct WorldView {
    residency: Residency<SubchunkOccupancy>,
    renderer:  Arc<WorldRenderer>,
    /// Shadow of the GPU instance array. Rebuilt from the resident pool
    /// each update; indices past the resident count are left as sentinels
    /// with `exposure_mask = 0` so the cull shader rejects them trivially.
    instances: [SubchunkInstance; SUBCHUNK_MAX_CANDIDATES],

    // --- Eviction bookkeeping ---
    evicted_last_update: usize,
    evicted_total:       u64,
}

impl WorldView {
    /// Construct the view centered on `initial_camera`. Runs the initial
    /// prep-and-upload cycle so the renderer has resident data before the
    /// first frame.
    pub fn new(ctx: &RendererContext, initial_camera: [f32; 3]) -> Self {
        let configs = [LevelConfig {
            level:  Level::ZERO,
            radius: [1, 1, 1],
        }];
        let residency = Residency::<SubchunkOccupancy>::new(&configs, initial_camera);
        let renderer      = Arc::new(WorldRenderer::new(ctx));

        let mut view = Self {
            residency,
            renderer,
            instances: [SubchunkInstance::new([0, 0, 0], 0, 0); SUBCHUNK_MAX_CANDIDATES],
            evicted_last_update: 0,
            evicted_total:       0,
        };

        view.process_pending_prep(ctx);
        view.rebuild_instances(ctx);
        view
    }

    /// Borrow the GPU renderer for render-graph registration.
    pub fn renderer(&self) -> &Arc<WorldRenderer> {
        &self.renderer
    }

    /// Number of sub-chunks evicted on the most recent [`Self::update`].
    pub fn evicted_last_update(&self) -> usize {
        self.evicted_last_update
    }

    /// Cumulative eviction count since construction.
    pub fn evicted_total(&self) -> u64 {
        self.evicted_total
    }

    /// Update residency against the current camera position and resync the
    /// GPU buffers.
    pub fn update(&mut self, ctx: &RendererContext, world_pos: [f32; 3]) {
        let evictions = self.residency.update_camera(world_pos);
        self.evicted_last_update = evictions.len();
        self.evicted_total       = self.evicted_total.saturating_add(evictions.len() as u64);

        // Content is a pure function of coord — nothing to persist. The
        // evicted slot's stale GPU bytes are harmless: no instance in the
        // rebuilt shadow list references the slot until a new prep commits
        // to it, at which point `write_occupancy_slot` overwrites the bytes.
        drop(evictions);

        self.process_pending_prep(ctx);
        self.rebuild_instances(ctx);
    }

    // --- internal ---

    /// Drain every outstanding prep request, synthesize occupancy for it,
    /// upload the result to the GPU, and commit it into the residency pool.
    ///
    /// Synchronous CPU prep — the request drains to empty in one call.
    fn process_pending_prep(&mut self, ctx: &RendererContext) {
        let requests = self.residency.take_prep_requests();
        for req in requests {
            let occ     = synthesize_occupancy(req.coord);
            let summary = summarize(&occ);
            let bytes   = occ.to_gpu_bytes();

            // Upload before commit so the GPU slot is populated before the
            // CPU pool starts handing the coord out to consumers.
            self.renderer.write_occupancy_slot(ctx, req.slot.0, &bytes);
            self.residency.complete_prep(req.id, summary, occ);
        }
    }

    /// Rebuild `instances[..]` from the L0 pool's occupied slots and push
    /// the full array to the GPU.
    fn rebuild_instances(&mut self, ctx: &RendererContext) {
        // Reset to sentinel so any slot past the resident count is rejected
        // by the cull shader (exposure_mask 0).
        self.instances.fill(SubchunkInstance::new([0, 0, 0], 0, 0));

        let Some(pool) = self.residency.pool(Level::ZERO) else {
            self.renderer.write_instances(ctx, &self.instances);
            return;
        };

        for (i, (coord, slot, occ)) in pool.occupied().enumerate() {
            if i >= SUBCHUNK_MAX_CANDIDATES {
                // Residency was sized to fit; any overflow is a programming
                // error in the shell radius / MAX_CANDIDATES relationship.
                debug_assert!(false, "resident count exceeds MAX_CANDIDATES");
                break;
            }
            self.instances[i] = SubchunkInstance::new(
                world_origin(coord),
                slot.0,
                occ.isolated_exposure_mask(),
            );
        }

        self.renderer.write_instances(ctx, &self.instances);
    }
}

// --- helpers ---

/// World-space voxel origin of an L0 sub-chunk at `coord`.
fn world_origin(coord: SubchunkCoord) -> [i32; 3] {
    let e = SUBCHUNK_EDGE as i32;
    [coord.x * e, coord.y * e, coord.z * e]
}

/// Classify the occupancy for the readback report that the GPU prep path
/// would emit. The CPU prep path computes it directly from the data.
fn summarize(occ: &SubchunkOccupancy) -> OccupancySummary {
    if occ.is_empty() { OccupancySummary::Empty }
    else if occ.is_full() { OccupancySummary::Full }
    else { OccupancySummary::Mixed }
}

/// Stand-in procedural content: a per-coord sphere whose radius and center
/// derive from a hash of the sub-chunk coord, so each resident slot looks
/// distinctive and residency rolling is visually observable.
///
/// Deliberate stand-in for real worldgen; replaced once GPU-side prep lands.
fn synthesize_occupancy(coord: SubchunkCoord) -> SubchunkOccupancy {
    let h      = coord_hash(coord);
    let off    = |shift: u32| (((h >> shift) & 0x7) as f32 - 3.5) * 0.35;
    let center = [3.5 + off(0), 3.5 + off(3), 3.5 + off(6)];
    // Radius in [1.5, 3.6], quantised into 8 discrete sizes.
    let radius = 1.5 + ((h >> 9) & 0x7) as f32 * 0.3;

    let mut occ = SubchunkOccupancy::empty();
    for z in 0u8..8 {
        for y in 0u8..8 {
            for x in 0u8..8 {
                let fx = x as f32 - center[0];
                let fy = y as f32 - center[1];
                let fz = z as f32 - center[2];
                if fx * fx + fy * fy + fz * fz <= radius * radius {
                    occ.set(x, y, z, true);
                }
            }
        }
    }
    occ
}

/// Cheap integer hash of a sub-chunk coord, used purely for content variation
/// in the stand-in synthesis. Not cryptographic.
fn coord_hash(c: SubchunkCoord) -> u32 {
    let x = c.x as u32;
    let y = c.y as u32;
    let z = c.z as u32;
    x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791)
}
