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
    pub fn new(levels: &[LevelConfig], initial_camera: [f32; 3]) -> Self {
        let level_states: Vec<LevelState<T>> = levels
            .iter()
            .map(|cfg| {
                let center = camera_coord_at(cfg.level, initial_camera);
                let dims   = [
                    2 * cfg.radius[0] + 1,
                    2 * cfg.radius[1] + 1,
                    2 * cfg.radius[2] + 1,
                ];
                LevelState {
                    level: cfg.level,
                    shell: Shell::new(cfg.radius, center),
                    pool:  SlotPool::new(dims),
                }
            })
            .collect();

        let mut r = Self {
            levels:          level_states,
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
    /// Levels that didn't cross a sub-chunk boundary produce no diff —
    /// coarser levels are inherently slower to change. Pending prep requests
    /// for evicted coords are cancelled; completions that arrive later for
    /// those ids are silently dropped.
    pub fn update_camera(&mut self, world_pos: [f32; 3]) -> Vec<EvictEntry<T>> {
        let mut evictions = Vec::new();

        for level_idx in 0..self.levels.len() {
            let level      = self.levels[level_idx].level;
            let new_center = camera_coord_at(level, world_pos);
            let diff       = self.levels[level_idx].shell.recenter(new_center);

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

    /// Current camera-aligned center for `level`, in that level's sub-chunk
    /// grid.
    pub fn center(&self, level: Level) -> Option<SubchunkCoord> {
        self.level_state(level).map(|s| s.shell.center())
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

// --- camera coord derivation ---

/// Convert a world-space position (meters) to a sub-chunk coord at `level`.
fn camera_coord_at(level: Level, world_pos: [f32; 3]) -> SubchunkCoord {
    let extent = level.subchunk_extent_m();
    SubchunkCoord::new(
        (world_pos[0] / extent).floor() as i32,
        (world_pos[1] / extent).floor() as i32,
        (world_pos[2] / extent).floor() as i32,
    )
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
        // 3×3×3 shell → 27 initial requests.
        assert_eq!(r.pending_request_count(), 27);
    }

    #[test]
    fn new_seeds_requests_for_all_levels() {
        let cfg = vec![
            LevelConfig { level: Level::ZERO, radius: [1, 1, 1] }, // 27
            LevelConfig { level: Level(1),    radius: [0, 0, 0] }, // 1
        ];
        let r = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        assert_eq!(r.pending_request_count(), 28);
    }

    #[test]
    fn take_prep_requests_drains_once() {
        let cfg     = single_level(Level::ZERO, [0, 0, 0]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let first   = r.take_prep_requests();
        let second  = r.take_prep_requests();
        assert_eq!(first.len(),  1);
        assert_eq!(second.len(), 0);
        assert_eq!(r.pending_request_count(), 1, "request still in-flight after drain");
    }

    // -- complete_prep --

    #[test]
    fn complete_prep_commits_to_pool() {
        let cfg     = single_level(Level::ZERO, [0, 0, 0]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let req     = r.take_prep_requests()[0];

        r.complete_prep(req.id, OccupancySummary::Full, 42);
        assert_eq!(r.get(Level::ZERO, SubchunkCoord::new(0, 0, 0)), Some(&42));
        assert_eq!(r.pending_request_count(), 0);
    }

    #[test]
    fn complete_prep_unknown_id_is_dropped() {
        let cfg     = single_level(Level::ZERO, [0, 0, 0]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        r.complete_prep(RequestId(9999), OccupancySummary::Mixed, 1);
        assert_eq!(r.get(Level::ZERO, SubchunkCoord::new(0, 0, 0)), None);
    }

    // -- camera update --

    #[test]
    fn small_motion_within_level_cell_is_noop() {
        // L0 sub-chunk extent is 8 m. Moving 4 m stays in the same L0 cell.
        let cfg     = single_level(Level::ZERO, [0, 0, 0]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let _       = r.take_prep_requests();

        let evictions = r.update_camera([4.0, 0.0, 0.0]);
        assert!(evictions.is_empty());
        assert!(r.take_prep_requests().is_empty());
    }

    #[test]
    fn crossing_level_cell_boundary_triggers_diff() {
        let cfg     = single_level(Level::ZERO, [0, 0, 0]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let _       = r.take_prep_requests();

        // Move past 8 m on X — crosses an L0 boundary.
        let evictions = r.update_camera([9.0, 0.0, 0.0]);
        // Shell radius 0 → center moves, old coord evicts, new coord enqueues.
        assert_eq!(evictions.len(), 0, "no pool data yet (prep never completed)");
        let reqs = r.take_prep_requests();
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].coord, SubchunkCoord::new(1, 0, 0));
    }

    #[test]
    fn coarser_level_ignores_sub_boundary_motion() {
        // L0 = 8 m sub-chunks; L2 = 32 m. Moving 9 m crosses L0 but not L2.
        let cfg = vec![
            LevelConfig { level: Level::ZERO, radius: [0, 0, 0] },
            LevelConfig { level: Level(2),    radius: [0, 0, 0] },
        ];
        let mut r = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let _     = r.take_prep_requests();

        r.update_camera([9.0, 0.0, 0.0]);
        let reqs = r.take_prep_requests();
        assert_eq!(reqs.len(), 1, "only L0 should produce new work");
        assert_eq!(reqs[0].level, Level::ZERO);
    }

    #[test]
    fn eviction_returns_stored_data() {
        let cfg     = single_level(Level::ZERO, [0, 0, 0]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let req     = r.take_prep_requests()[0];
        r.complete_prep(req.id, OccupancySummary::Full, 77);

        // Move out of (0,0,0).
        let evictions = r.update_camera([9.0, 0.0, 0.0]);
        assert_eq!(evictions.len(), 1);
        assert_eq!(evictions[0].coord, SubchunkCoord::new(0, 0, 0));
        assert_eq!(evictions[0].level, Level::ZERO);
        assert_eq!(evictions[0].data,  77);
    }

    // -- cancellation on eviction --

    #[test]
    fn eviction_cancels_pending_request() {
        let cfg     = single_level(Level::ZERO, [0, 0, 0]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let req     = r.take_prep_requests()[0];

        // Evict before completing. The request should be cancelled.
        r.update_camera([9.0, 0.0, 0.0]);
        let _ = r.take_prep_requests(); // discard the new (1,0,0) request

        // Completing the stale id is a no-op — the coord is long gone from
        // the shell and the request was cancelled.
        r.complete_prep(req.id, OccupancySummary::Full, 999);
        assert_eq!(r.get(Level::ZERO, SubchunkCoord::new(0, 0, 0)), None);
    }

    #[test]
    fn completion_after_coord_rolled_out_is_dropped() {
        // Same shape as above but exercise the "shell no longer contains"
        // guard even if cancellation somehow leaked the id.
        let cfg     = single_level(Level::ZERO, [1, 0, 0]);
        let mut r   = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);
        let reqs    = r.take_prep_requests();
        // Pick the request for (1, 0, 0) — will roll out when camera jumps far.
        let target  = reqs.iter().find(|r| r.coord == SubchunkCoord::new(1, 0, 0)).unwrap();
        let target_id = target.id;

        r.update_camera([1000.0, 0.0, 0.0]);
        let _ = r.take_prep_requests();

        r.complete_prep(target_id, OccupancySummary::Full, 123);
        assert_eq!(r.get(Level::ZERO, SubchunkCoord::new(1, 0, 0)), None);
    }

    // -- end-to-end sanity --

    #[test]
    fn full_sync_roundtrip_populates_all_levels() {
        let cfg = vec![
            LevelConfig { level: Level::ZERO, radius: [1, 1, 1] },
            LevelConfig { level: Level(1),    radius: [0, 0, 0] },
        ];
        let mut r = Residency::<u32>::new(&cfg, [0.0, 0.0, 0.0]);

        let reqs = r.take_prep_requests();
        for req in &reqs {
            r.complete_prep(req.id, OccupancySummary::Mixed, req.id.0 as u32);
        }
        assert_eq!(r.pending_request_count(), 0);

        // Every resident at every level is now readable.
        for level_idx in 0..r.level_count() {
            let level = if level_idx == 0 { Level::ZERO } else { Level(1) };
            let coords: Vec<_> = match level.0 {
                0 => {
                    let mut v = Vec::new();
                    for dz in -1..=1i32 {
                        for dy in -1..=1i32 {
                            for dx in -1..=1i32 {
                                v.push(SubchunkCoord::new(dx, dy, dz));
                            }
                        }
                    }
                    v
                }
                _ => vec![SubchunkCoord::new(0, 0, 0)],
            };
            for c in coords {
                assert!(r.get(level, c).is_some(), "missing {c:?} at {level:?}");
            }
        }
    }

    // -- camera coord derivation --

    #[test]
    fn camera_coord_scales_with_level() {
        // At L0 (8 m), world x=16 is sub-chunk 2. At L1 (16 m), it's 1.
        let pos = [16.5, 0.0, 0.0];
        assert_eq!(camera_coord_at(Level::ZERO, pos), SubchunkCoord::new(2, 0, 0));
        assert_eq!(camera_coord_at(Level(1),    pos), SubchunkCoord::new(1, 0, 0));
    }

    #[test]
    fn camera_coord_negative_floors_toward_minus_infinity() {
        // At L0 (8 m), x=-0.5 floors to sub-chunk -1, not 0.
        let pos = [-0.5, 0.0, 0.0];
        assert_eq!(camera_coord_at(Level::ZERO, pos), SubchunkCoord::new(-1, 0, 0));
    }
}
