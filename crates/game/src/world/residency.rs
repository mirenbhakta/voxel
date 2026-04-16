//! Multi-level clipmap residency coordinator.
//!
//! [`Residency`] owns one [`Shell`] + [`SlotPool`] per LOD level and drives
//! the CPU half of the prep/commit handshake described in
//! `docs/world_streaming.md` §CPU/GPU Data Flow.
//!
//! The caller drives it through three calls per frame:
//!
//! 1. [`Residency::update_camera`] — recenter every level's shell against
//!    the current camera position and return any sub-chunks that rolled
//!    out of residency (their stored data is handed back to the caller
//!    for persistence / disposal).
//! 2. [`Residency::take_prep_requests`] — drain the list of newly-issued
//!    prep requests. The caller runs whatever prep logic produces `T`
//!    (sync CPU, async GPU, file read, …) and eventually calls…
//! 3. [`Residency::complete_prep`] — hand back the prepared `T` for a
//!    request id. The residency commits it to the right slot pool iff
//!    the coord is still resident.
//!
//! The flow is deliberately pull-based: requests accumulate inside the
//! residency until the caller asks for them, and completions are pushed
//! back at whatever cadence the prep producer achieves. This maps
//! cleanly onto both a synchronous CPU path (drain → compute → complete,
//! all in one frame) and the asynchronous GPU staging/readback path
//! (requests dispatched this frame, completions arriving 1–2 frames
//! later).
//!
//! # Cross-level corner derivation
//!
//! Each level's shell is anchored at a grid-vertex "origin corner" rather
//! than a containing sub-chunk. To keep shells nested across levels, all
//! corners coincide in world space: the coarsest level picks an anchor by
//! rounding the camera to its own grid, and every finer level shifts that
//! anchor into its own coord system by a factor of `2^(coarsest - level)`.
//! Consequence: all levels' shells jump at the coarsest level's cadence
//! (every half-extent of camera motion), which keeps coarser shells from
//! slipping relative to finer ones.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::world::coord::{Level, SubchunkCoord};
use crate::world::pool::{SlotId, SlotPool};
use crate::world::shell::Shell;

// --- LevelConfig ---

/// Per-level configuration: LOD index + shell radius.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelConfig {
    pub level:  Level,
    pub radius: [u32; 3],
}

// --- RequestId ---

/// Monotonic identifier for a single prep request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

// --- OccupancySummary ---

/// Compact summary of a prepared sub-chunk's occupancy, carried on every
/// completion report. Mirrors the GPU staging readback's `occupancy_summary`
/// field — the prep shader can write this cheaply, and it lets the control
/// plane short-circuit storage for uniform sub-chunks in a future slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancySummary {
    Empty,
    Full,
    Mixed,
}

// --- PrepRequest ---

/// A single prep request: prepare a sub-chunk for a specific (level, coord)
/// and deliver the result to `slot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrepRequest {
    pub id:    RequestId,
    pub level: Level,
    pub coord: SubchunkCoord,
    pub slot:  SlotId,
}

// --- EvictEntry ---

/// A sub-chunk that has rolled out of residency, with the data it held.
///
/// The caller owns disposition — persist to disk, discard, forward to an
/// upload ring, etc. The residency itself does not retain the data.
pub struct EvictEntry<T> {
    pub level: Level,
    pub coord: SubchunkCoord,
    pub slot:  SlotId,
    pub data:  T,
}

// --- Residency ---

/// Multi-level clipmap residency coordinator.
///
/// Owns one shell + pool per level. `T` is the per-slot payload; use
/// `SubchunkOccupancy` (or a compound thereof) in CPU-authored workflows
/// and `()` once data lives exclusively in a GPU buffer indexed by
/// [`SlotId`].
pub struct Residency<T> {
    levels:          Vec<LevelState<T>>,
    /// Highest level among `levels`; drives the shared corner derivation.
    /// All finer levels' corners are bit-shifted copies of the coarsest's,
    /// so corners coincide in world space across the cascade.
    coarsest_level:  Level,
    next_request_id: u64,
    new_requests:    Vec<PrepRequest>,
    pending:         HashMap<RequestId, PendingEntry>,
}

impl<T> Residency<T> {
    /// Create a residency for the given levels, centered on `initial_camera`.
    ///
    /// Issues an initial batch of prep requests covering every resident
    /// sub-chunk at every level. These are immediately available via
    /// [`Residency::take_prep_requests`].
    ///
    /// # Panics
    ///
    /// Panics if `levels` is empty — the coarsest level drives the anchor
    /// derivation and is required.
    pub fn new(levels: &[LevelConfig], initial_camera: [f32; 3]) -> Self {
        assert!(!levels.is_empty(), "Residency requires at least one level");

        let coarsest_level = levels
            .iter()
            .map(|c| c.level)
            .max()
            .expect("levels non-empty");

        let level_states: Vec<LevelState<T>> = levels
            .iter()
            .map(|cfg| {
                let corner = compute_corner(coarsest_level, cfg.level, initial_camera);
                let dims   = [
                    2 * cfg.radius[0],
                    2 * cfg.radius[1],
                    2 * cfg.radius[2],
                ];
                LevelState {
                    level: cfg.level,
                    shell: Shell::new(cfg.radius, corner),
                    pool:  SlotPool::new(dims),
                }
            })
            .collect();

        let mut r = Self {
            levels:          level_states,
            coarsest_level,
            next_request_id: 0,
            new_requests:    Vec::new(),
            pending:         HashMap::new(),
        };

        // Seed prep requests for the initial resident set at every level.
        for level_idx in 0..r.levels.len() {
            let level  = r.levels[level_idx].level;
            let coords: Vec<_> = r.levels[level_idx].shell.residents().collect();
            for coord in coords {
                r.enqueue_prep(level_idx, level, coord);
            }
        }

        r
    }

    /// Recenter every level against `world_pos` and return evicted slots.
    ///
    /// All corners derive from the coarsest level's anchor, so finer levels
    /// inherit its cadence: no level rolls until camera motion crosses the
    /// coarsest's half-extent boundary. On a jump, finer levels shift by a
    /// larger coord delta (in their own grids) but in world units all
    /// levels move together by one coarsest half-extent. Pending prep
    /// requests for evicted coords are cancelled; completions that arrive
    /// later for those ids are silently dropped.
    pub fn update_camera(&mut self, world_pos: [f32; 3]) -> Vec<EvictEntry<T>> {
        let mut evictions = Vec::new();

        let coarsest = self.coarsest_level;
        for level_idx in 0..self.levels.len() {
            let level      = self.levels[level_idx].level;
            let new_corner = compute_corner(coarsest, level, world_pos);
            let diff       = self.levels[level_idx].shell.recenter(new_corner);

            for coord in &diff.removed {
                let slot = self.levels[level_idx].pool.slot_id(*coord);
                if let Some(data) = self.levels[level_idx].pool.remove(*coord) {
                    evictions.push(EvictEntry { level, coord: *coord, slot, data });
                }
                self.cancel_pending(level, *coord);
            }

            for coord in &diff.added {
                self.enqueue_prep(level_idx, level, *coord);
            }
        }

        evictions
    }

    /// Drain and return prep requests not yet handed to the caller.
    ///
    /// Each request is returned once; subsequent calls only yield requests
    /// queued after the previous drain. The corresponding in-flight tracking
    /// entry remains in the residency's `pending` map until
    /// [`Residency::complete_prep`] retires it (or an eviction cancels it).
    pub fn take_prep_requests(&mut self) -> Vec<PrepRequest> {
        std::mem::take(&mut self.new_requests)
    }

    /// Number of requests currently in flight (issued, not yet completed).
    pub fn pending_request_count(&self) -> usize {
        self.pending.len()
    }

    /// Deliver prepared data for a previously-issued request.
    ///
    /// If the request is unknown (cancelled by eviction, or a completion
    /// arriving after the coord rolled out), the completion is dropped.
    /// If the coord is still resident, the data is committed to the level's
    /// slot pool.
    pub fn complete_prep(
        &mut self,
        id:       RequestId,
        _summary: OccupancySummary,
        data:     T,
    ) {
        let Some(entry) = self.pending.remove(&id) else {
            return;
        };

        let Some(level_idx) = self.level_index(entry.level) else {
            return;
        };

        if !self.levels[level_idx].shell.contains(entry.coord) {
            return;
        }

        self.levels[level_idx].pool.insert(entry.coord, data);
    }

    // --- read-only accessors ---

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Current anchor corner for `level`, in that level's sub-chunk grid.
    ///
    /// Corners coincide in world space across every resident level — the
    /// coarsest picks the anchor, finer levels are bit-shifted copies.
    pub fn corner(&self, level: Level) -> Option<SubchunkCoord> {
        self.level_state(level).map(|s| s.shell.corner())
    }

    /// Read the stored payload at `(level, coord)`, if any.
    pub fn get(&self, level: Level, coord: SubchunkCoord) -> Option<&T> {
        self.level_state(level)?.pool.get(coord)
    }

    /// Borrow a level's entire slot pool for iteration / diagnostics.
    pub fn pool(&self, level: Level) -> Option<&SlotPool<T>> {
        self.level_state(level).map(|s| &s.pool)
    }

    // --- internal ---

    fn level_state(&self, level: Level) -> Option<&LevelState<T>> {
        self.levels.iter().find(|s| s.level == level)
    }

    fn level_index(&self, level: Level) -> Option<usize> {
        self.levels.iter().position(|s| s.level == level)
    }

    fn enqueue_prep(&mut self, level_idx: usize, level: Level, coord: SubchunkCoord) {
        let slot = self.levels[level_idx].pool.slot_id(coord);
        let id   = RequestId(self.next_request_id);
        self.next_request_id += 1;

        self.new_requests.push(PrepRequest { id, level, coord, slot });
        self.pending.insert(id, PendingEntry { level, coord });
    }

    fn cancel_pending(&mut self, level: Level, coord: SubchunkCoord) {
        let stale: Vec<RequestId> = self.pending.iter()
            .filter(|(_, e)| e.level == level && e.coord == coord)
            .map(|(id, _)| *id)
            .collect();
        for id in stale {
            self.pending.remove(&id);
        }
        self.new_requests.retain(|r| !(r.level == level && r.coord == coord));
    }
}

// --- internal state types ---

struct LevelState<T> {
    level: Level,
    shell: Shell,
    pool:  SlotPool<T>,
}

struct PendingEntry {
    level: Level,
    coord: SubchunkCoord,
}

// --- corner derivation ---

/// Anchor corner for `level`, derived from the camera position through
/// the coarsest level's grid.
///
/// The coarsest level picks the nearest grid vertex to the camera by
/// rounding `world_pos / extent_coarsest`. Finer levels' corners are the
/// same world point expressed in their own coord system: shift left by
/// `coarsest.0 - level.0` to convert (each step down doubles the coord
/// since sub-chunk extent halves).
///
/// Rounding the coarsest and deriving the rest by shift guarantees:
/// - all corners coincide in world space (no cross-level drift);
/// - `L_n` shell boundaries land on `L_(n+1)` sub-chunk boundaries when
///   radii are even, so the "fully inside finer shell" cull test can
///   express an integer number of coarser cells rather than straddling.
fn compute_corner(coarsest: Level, level: Level, world_pos: [f32; 3]) -> SubchunkCoord {
    debug_assert!(
        level.0 <= coarsest.0,
        "level {level:?} must be finer-or-equal to coarsest {coarsest:?}",
    );
    let coarsest_extent = coarsest.subchunk_extent_m();
    let kx = (world_pos[0] / coarsest_extent).round() as i32;
    let ky = (world_pos[1] / coarsest_extent).round() as i32;
    let kz = (world_pos[2] / coarsest_extent).round() as i32;
    let shift = (coarsest.0 - level.0) as u32;
    SubchunkCoord::new(kx << shift, ky << shift, kz << shift)
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn single_level(level: Level, radius: [u32; 3]) -> Vec<LevelConfig> {
        vec![LevelConfig { level, radius }]
    }

    // -- construction --

    #[test]
    fn new_issues_request_per_resident() {
        let cfg = single_level(Level::ZERO, [1, 1, 1]);
        let r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        // 2×2×2 shell → 8 initial requests.
        assert_eq!(r.pending_request_count(), 8);
    }

    #[test]
    fn new_seeds_requests_for_all_levels() {
        let cfg = vec![
            LevelConfig { level: Level::ZERO, radius: [2, 2, 2] }, // 64
            LevelConfig { level: Level(1),    radius: [1, 1, 1] }, // 8
        ];
        let r = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        assert_eq!(r.pending_request_count(), 72);
    }

    #[test]
    fn take_prep_requests_drains_once() {
        let cfg     = single_level(Level::ZERO, [1, 1, 1]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let first   = r.take_prep_requests();
        let second  = r.take_prep_requests();
        assert_eq!(first.len(),  8);
        assert_eq!(second.len(), 0);
        assert_eq!(r.pending_request_count(), 8, "requests still in-flight after drain");
    }

    // -- complete_prep --

    #[test]
    fn complete_prep_commits_to_pool() {
        let cfg     = single_level(Level::ZERO, [1, 1, 1]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        // Pick a request for a known-resident coord: anchor at (0,0,0)
        // makes the shell occupy {-1, 0}³, so (0,0,0) is in it.
        let reqs    = r.take_prep_requests();
        let req     = reqs.iter()
            .find(|r| r.coord == SubchunkCoord::new(0, 0, 0))
            .expect("(0,0,0) should be resident in shell anchored at origin");

        r.complete_prep(req.id, OccupancySummary::Full, 42);
        assert_eq!(r.get(Level::ZERO, SubchunkCoord::new(0, 0, 0)), Some(&42));
        assert_eq!(r.pending_request_count(), 7);
    }

    #[test]
    fn complete_prep_unknown_id_is_dropped() {
        let cfg     = single_level(Level::ZERO, [1, 1, 1]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        r.complete_prep(RequestId(9999), OccupancySummary::Mixed, 1);
        assert_eq!(r.get(Level::ZERO, SubchunkCoord::new(0, 0, 0)), None);
    }

    // -- camera update --

    #[test]
    fn small_motion_within_level_cell_is_noop() {
        // Coarsest = L0. Anchor = round(pos/8). Anchor jumps at pos=4
        // (the vertex midpoint). Motion of 2 m keeps round(2/8) = 0.
        let cfg     = single_level(Level::ZERO, [1, 1, 1]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let _       = r.take_prep_requests();

        let evictions = r.update_camera([2.0, 0.0, 0.0]);
        assert!(evictions.is_empty());
        assert!(r.take_prep_requests().is_empty());
    }

    #[test]
    fn crossing_level_cell_boundary_triggers_diff() {
        let cfg     = single_level(Level::ZERO, [1, 1, 1]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let _       = r.take_prep_requests();

        // Move past 4 m on X — crosses an anchor-vertex midpoint at L0.
        let evictions = r.update_camera([5.0, 0.0, 0.0]);
        // No prep completed yet → no evicted data to return, but shell rolled.
        assert_eq!(evictions.len(), 0, "no pool data yet (prep never completed)");
        let reqs = r.take_prep_requests();
        // Anchor moved from (0,0,0) to (1,0,0); 2r=2 shell → 4-coord face per axis
        // rolls in, 4-coord face rolls out.
        assert_eq!(reqs.len(), 4);
        for req in &reqs {
            assert_eq!(req.coord.x, 1);
        }
    }

    #[test]
    fn coarser_level_drives_cadence_for_all_levels() {
        // Coarsest = L2 (32 m, half-extent = 16 m). Moving 9 m doesn't move
        // the L2 anchor → finer levels also hold (their corners derive from
        // L2's). At 17 m motion the L2 anchor would shift.
        let cfg = vec![
            LevelConfig { level: Level::ZERO, radius: [1, 1, 1] },
            LevelConfig { level: Level(2),    radius: [1, 1, 1] },
        ];
        let mut r = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let _     = r.take_prep_requests();

        r.update_camera([9.0, 0.0, 0.0]);
        let reqs = r.take_prep_requests();
        assert!(reqs.is_empty(), "no level should move before L2 anchor jumps");
    }

    #[test]
    fn eviction_returns_stored_data() {
        let cfg     = single_level(Level::ZERO, [1, 1, 1]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let reqs    = r.take_prep_requests();
        // Prepare the coord that will roll out: (-1, 0, 0) is resident at
        // anchor=(0,0,0) but leaves when anchor moves to (1,0,0).
        let target  = reqs.iter()
            .find(|r| r.coord == SubchunkCoord::new(-1, 0, 0))
            .expect("(-1,0,0) should be resident initially");
        r.complete_prep(target.id, OccupancySummary::Full, 77);

        // Cross the anchor-vertex midpoint at x = 4.
        let evictions = r.update_camera([5.0, 0.0, 0.0]);
        // Eviction face is the 2×2 slab at x = -1.
        let target_evict = evictions.iter()
            .find(|e| e.coord == SubchunkCoord::new(-1, 0, 0))
            .expect("(-1,0,0) should evict");
        assert_eq!(target_evict.level, Level::ZERO);
        assert_eq!(target_evict.data,  77);
    }

    // -- cancellation on eviction --

    #[test]
    fn eviction_cancels_pending_request() {
        let cfg     = single_level(Level::ZERO, [1, 1, 1]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let reqs    = r.take_prep_requests();
        let target  = reqs.iter()
            .find(|r| r.coord == SubchunkCoord::new(-1, 0, 0))
            .expect("(-1,0,0) should be in the initial shell");

        // Evict before completing. The request should be cancelled.
        r.update_camera([5.0, 0.0, 0.0]);
        let _ = r.take_prep_requests(); // discard the new requests

        // Completing the stale id is a no-op — the coord is long gone from
        // the shell and the request was cancelled.
        r.complete_prep(target.id, OccupancySummary::Full, 999);
        assert_eq!(r.get(Level::ZERO, SubchunkCoord::new(-1, 0, 0)), None);
    }

    #[test]
    fn completion_after_coord_rolled_out_is_dropped() {
        // Same shape as above but exercise the "shell no longer contains"
        // guard even if cancellation somehow leaked the id.
        let cfg     = single_level(Level::ZERO, [1, 1, 1]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let reqs    = r.take_prep_requests();
        let target  = reqs.iter()
            .find(|r| r.coord == SubchunkCoord::new(0, 0, 0))
            .expect("(0,0,0) should be in the initial shell");
        let target_id = target.id;

        r.update_camera([1000.0, 0.0, 0.0]);
        let _ = r.take_prep_requests();

        r.complete_prep(target_id, OccupancySummary::Full, 123);
        assert_eq!(r.get(Level::ZERO, SubchunkCoord::new(0, 0, 0)), None);
    }

    // -- end-to-end sanity --

    #[test]
    fn full_sync_roundtrip_populates_all_levels() {
        let cfg = vec![
            LevelConfig { level: Level::ZERO, radius: [2, 2, 2] },
            LevelConfig { level: Level(1),    radius: [1, 1, 1] },
        ];
        let mut r = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);

        let reqs = r.take_prep_requests();
        for req in &reqs {
            r.complete_prep(req.id, OccupancySummary::Mixed, req.id.0 as u32);
        }
        assert_eq!(r.pending_request_count(), 0);

        // Every resident at every level is now readable. Shell at
        // anchor=(0,0,0), r=2 spans x ∈ [-2, 1]; at r=1 spans x ∈ [-1, 0].
        let sample_coords = [
            (Level::ZERO, SubchunkCoord::new(-2, -2, -2)),
            (Level::ZERO, SubchunkCoord::new( 1,  1,  1)),
            (Level::ZERO, SubchunkCoord::new( 0,  0,  0)),
            (Level(1),    SubchunkCoord::new(-1, -1, -1)),
            (Level(1),    SubchunkCoord::new( 0,  0,  0)),
        ];
        for (level, coord) in sample_coords {
            assert!(r.get(level, coord).is_some(), "missing {coord:?} at {level:?}");
        }
    }

    // -- corner derivation --

    #[test]
    fn compute_corner_aligns_across_levels() {
        // Coarsest = L1 (16 m). Anchor = round(16.5 / 16) = 1 (L1 coord).
        // L0 corner = 1 << 1 = 2 (L0 coord). In world: L1 anchor = 16,
        // L0 anchor = 16 — same point.
        let pos = [16.5, 0.0, 0.0];
        assert_eq!(compute_corner(Level(1), Level(1), pos), SubchunkCoord::new(1, 0, 0));
        assert_eq!(compute_corner(Level(1), Level(0), pos), SubchunkCoord::new(2, 0, 0));
    }

    #[test]
    fn compute_corner_rounds_halves_away_from_zero() {
        // Coarsest = L0 (8 m). Anchor flips at the vertex midpoint x = 4.
        // f32::round rounds half away from zero, so x = 4.0 rounds to 1,
        // x = -4.0 rounds to -1.
        assert_eq!(
            compute_corner(Level::ZERO, Level::ZERO, [ 4.0, 0.0, 0.0]),
            SubchunkCoord::new(1, 0, 0),
        );
        assert_eq!(
            compute_corner(Level::ZERO, Level::ZERO, [-4.0, 0.0, 0.0]),
            SubchunkCoord::new(-1, 0, 0),
        );
    }

    #[test]
    fn compute_corner_finer_is_shift_of_coarsest() {
        // Coarsest = L2, coarser coord = 3. Finer corners = 3<<1 = 6 (L1),
        // 3<<2 = 12 (L0). World: all three anchors at 3*32 = 96.
        let pos = [96.0, 0.0, 0.0];
        assert_eq!(compute_corner(Level(2), Level(2), pos), SubchunkCoord::new(3, 0, 0));
        assert_eq!(compute_corner(Level(2), Level(1), pos), SubchunkCoord::new(6, 0, 0));
        assert_eq!(compute_corner(Level(2), Level(0), pos), SubchunkCoord::new(12, 0, 0));
    }
}
