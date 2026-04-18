//! Per-frame CPU-state ledger for the `debug-state-history` divergence
//! detector.
//!
//! The refactor that introduced the sub-chunk directory + material pool
//! has, twice so far, surfaced bugs that only manifest as "CPU thinks the
//! directory holds X, GPU observes Y" — silent cross-wiring with no
//! crash, no panic, no failed test. The pattern that finally makes these
//! diagnosable is the shadow-ledger/readback pair described in
//! `decision-scaffold-rewrite-principles` §Principle 6: the CPU writes a
//! witness of what it sent, the GPU publishes a snapshot of what it
//! holds, and a comparator classifies the difference on retirement.
//!
//! This module owns the CPU half — a bounded ring of [`FrameRecord`]s
//! describing everything the CPU authored on a given frame. It is the
//! source of truth the comparator (in [`crate::world::divergence`]) uses
//! when a GPU-side readback for frame `F` lands on the CPU.
//!
//! All types and storage here are gated behind the `debug-state-history`
//! cargo feature.

#![cfg(feature = "debug-state-history")]
// Many fields here are populated by the instrumentation hooks but are
// read only by the divergence reporter and by whoever is interactively
// inspecting a `FrameRecord` from a debugger. That makes them look
// "unused" to rustc under the current compare_directory_snapshots
// signature (which reads only the directory_snapshot); the ledger is
// nonetheless load-bearing for the diagnostic we're building. Silence
// the lint at the module level rather than tagging every field — this
// module is explicitly a debug-only observation surface.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use bytemuck::{Pod, Zeroable};

use renderer::{DirEntry, DirtyEntry, FrameIndex, PatchCopy, SUBCHUNK_MAX_CANDIDATES};

use crate::world::coord::{Level, SubchunkCoord};

// --- DirectorySnapshot ---

/// Fixed-size byte-compatible snapshot of the GPU's `slot_directory_buf`.
///
/// Sized to exactly the renderer's directory capacity so the
/// `ReadbackChannel<DirectorySnapshot>` slot buffer fits a whole-frame
/// readback in one `copy_buffer_to_buffer`. At 24 B × 256 entries this is
/// 6 KiB per slot × frame_count slots — small enough that the default
/// `FrameCount::MIN = 2` gives 12 KiB of mapped-read GPU memory, well
/// within any backend's budget and entirely absent from release builds
/// (feature-gated out).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DirectorySnapshot {
    pub entries: [DirEntry; SUBCHUNK_MAX_CANDIDATES],
}

impl DirectorySnapshot {
    /// Borrow the entry array as a slice for comparison against a CPU
    /// [`FrameRecord::directory_snapshot`].
    pub fn entries(&self) -> &[DirEntry] {
        &self.entries
    }
}

// --- FrameRecord ---

/// Every CPU-authored mutation + the end-of-frame CPU state that results,
/// for one frame of the world update loop.
///
/// Built up by the `*_builder` methods on [`StateHistory`] as the
/// `WorldView::update` walks its phases, then finalized and pushed into
/// the ring at the end of the update. The directory snapshot is taken
/// *after* step 5 (directory flush) so it matches exactly what the CPU
/// handed to `queue.write_buffer` on this frame's submit.
#[derive(Clone)]
pub struct FrameRecord {
    pub frame_index:            FrameIndex,
    /// CPU-authored directory, one entry per `directory_index`. Length
    /// equals the directory capacity configured by the owning
    /// `WorldView`. Cloned in full each frame — the divergence reporter
    /// needs random-access comparison, and the capacity is bounded by
    /// `SUBCHUNK_MAX_CANDIDATES` so the clone cost is a fixed small
    /// byte total.
    pub directory_snapshot:     Vec<DirEntry>,
    /// `MaterialAllocator::stats().active` at the end of the update.
    /// A standalone scalar rather than the full allocator snapshot —
    /// the directory entries already carry the per-slot state.
    pub allocator_active:       u32,
    /// Prep requests issued on this frame.
    pub prep_requests_issued:   Vec<PrepRequestRecord>,
    /// Retired dirty-list entries that landed this frame, with the
    /// CPU-side transition the retirement applied.
    pub dirty_entries_retired:  Vec<RetiredDirtyEntry>,
    /// Patch copies the retirement emitted into the graph this frame.
    pub patch_copies_issued:    Vec<PatchCopy>,
    /// Residency inserts (`complete_prep` success) by (level, coord,
    /// directory_index).
    pub residency_inserts:      Vec<ResidencyInsert>,
    /// Residency evictions (rolled out of the shell) by (level, coord,
    /// directory_index).
    pub residency_evicts:       Vec<ResidencyEvict>,
    /// Retirement entries whose shader-emitted `directory_index`
    /// disagreed with the CPU's recomputation from
    /// `accepted_coords[idx]`. Non-empty iff the coord↔dir_idx binding
    /// drifted somewhere along the prep → readback → retirement path;
    /// empty under a clean run.
    pub retirement_inconsistencies: Vec<RetirementInconsistency>,
}

impl FrameRecord {
    /// Fresh record for `frame_index` with empty event lists.
    pub fn new(frame_index: FrameIndex) -> Self {
        Self {
            frame_index,
            directory_snapshot:         Vec::new(),
            allocator_active:           0,
            prep_requests_issued:       Vec::new(),
            dirty_entries_retired:      Vec::new(),
            patch_copies_issued:        Vec::new(),
            residency_inserts:          Vec::new(),
            residency_evicts:           Vec::new(),
            retirement_inconsistencies: Vec::new(),
        }
    }
}

/// A prep request the CPU dispatched this frame.
#[derive(Clone, Copy, Debug)]
pub struct PrepRequestRecord {
    pub coord:           SubchunkCoord,
    pub level:           Level,
    pub directory_index: u32,
}

/// A residency insert (successful `complete_prep`) this frame.
#[derive(Clone, Copy, Debug)]
pub struct ResidencyInsert {
    pub level:           Level,
    pub coord:           SubchunkCoord,
    pub directory_index: u32,
}

/// A residency eviction (coord rolled out) this frame.
#[derive(Clone, Copy, Debug)]
pub struct ResidencyEvict {
    pub level:           Level,
    pub coord:           SubchunkCoord,
    pub directory_index: u32,
}

/// One retired dirty entry + the CPU-side transition the retirement
/// applied. Captured after `apply_dirty_entries` runs so the "what the
/// CPU changed" view is exact.
///
/// [`DirEntry`] is an FFI-layout type that does not derive `Debug`; this
/// struct doesn't either to avoid coupling the diagnostic API to manual
/// `Debug` boilerplate that doesn't serve the divergence reporter.
#[derive(Clone, Copy)]
pub struct RetiredDirtyEntry {
    pub directory_index:     u32,
    pub coord:               [i32; 3],
    pub new_bits_partial:    u32,
    /// Staging-buffer index the prep shader wrote this sub-chunk's
    /// occupancy to. Captured on the `RetiredDirtyEntry` so the
    /// per-frame log's `[dirty]` line can be cross-referenced with the
    /// `[patch-copy]` line that consumed the same staging slot.
    pub staging_request_idx: u32,
    pub transition:          TransitionKind,
    /// `Some(slot)` when the retirement ended with a resident directory
    /// entry (sparse classification). `None` for uniform-empty retires.
    pub allocated_slot:      Option<u32>,
    /// The final `DirEntry` the retirement wrote into the CPU-authored
    /// directory at `directory_index`.
    pub new_entry:           DirEntry,
}

// --- RetirementInconsistency ---

/// A retirement where the shader-emitted `directory_index` disagrees with
/// what the CPU recomputes from the coord stored at that index in
/// `accepted_coords`.
///
/// Load-bearing invariant (see module-level doc on the retirement
/// instrumentation in `world_view.rs`): for every dirty entry processed
/// at retirement, the coord CPU filed at `entry.directory_index` must
/// independently resolve back to `entry.directory_index` via the canonical
/// [`cpu_compute_directory_index`](crate::world::pool::cpu_compute_directory_index)
/// formula. Any violation means the retirement pipeline applied correct
/// computed data to an incorrect `directory_index` target — "right
/// values, wrong slots" cross-wiring.
///
/// `dirty` is the raw shader-emitted entry (carries its `directory_index`
/// and `new_bits_partial`). `cpu_coord` / `cpu_level` are what
/// `accepted_coords[dirty.directory_index]` resolved to on the CPU side;
/// `cpu_recomputed_dir_idx` is what applying the CPU formula to that
/// `(coord, level)` pair produced. `shader_dir_idx == dirty.directory_index`
/// is stored explicitly so the log format doesn't have to chase the
/// aliased field through `dirty.directory_index`.
#[derive(Clone, Copy)]
pub struct RetirementInconsistency {
    pub dirty:                  DirtyEntry,
    pub cpu_coord:              SubchunkCoord,
    pub cpu_level:              Level,
    pub cpu_recomputed_dir_idx: u32,
    pub shader_dir_idx:         u32,
}

/// Classification of a dirty-entry retirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionKind {
    /// Previously non-resident, now non-resident. No material slot
    /// allocated or freed.
    UniformFirstTime,
    /// Previously non-resident, now sparse-resident. A material slot was
    /// allocated.
    UniformToSparse,
    /// Previously sparse-resident, still sparse-resident. Material slot
    /// reused in-place.
    SparseUpdate,
    /// Previously sparse-resident, now non-resident. Material slot freed.
    SparseToUniform,
}

// --- StateHistory ---

/// Bounded ring of [`FrameRecord`]s plus a per-frame builder for the
/// in-progress record.
///
/// The builder is finalized and pushed by
/// [`StateHistory::finalize_frame`], which also evicts the oldest entry
/// if the ring is at capacity. Lookup by frame index is linear in the
/// ring length; the ring is small (default 32), so a scan is faster and
/// simpler than a map.
pub struct StateHistory {
    records:                VecDeque<FrameRecord>,
    capacity:               usize,
    current_frame_builder:  Option<FrameRecord>,
    /// Append-only log writer: every finalized `FrameRecord` gets a
    /// human-readable summary appended here. `None` means the writer
    /// could not be opened (failure logged once to stderr, then silent)
    /// — the ring still works and the divergence path still fires.
    log_writer:             Option<BufWriter<File>>,
    log_path:               Option<PathBuf>,
}

impl StateHistory {
    /// Build a history ring of at most `capacity` records. No per-frame
    /// log file is opened — callers who want the file output must call
    /// [`StateHistory::new_with_log`] instead.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`.
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "StateHistory capacity must be > 0; got {capacity}",
        );
        Self {
            records:               VecDeque::with_capacity(capacity),
            capacity,
            current_frame_builder: None,
            log_writer:            None,
            log_path:              None,
        }
    }

    /// Build a history ring that also appends per-frame summaries to the
    /// file at `log_path`. Any existing file at that path is truncated so
    /// a new run starts from an empty log.
    ///
    /// If the file cannot be opened (permission denied, parent directory
    /// missing, etc.), the error is logged once to `stderr` and the
    /// history falls back to in-memory-only behaviour — the ring and
    /// divergence reporter continue to work.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`.
    pub fn new_with_log(capacity: usize, log_path: PathBuf) -> Self {
        assert!(
            capacity > 0,
            "StateHistory capacity must be > 0; got {capacity}",
        );
        let log_writer = match File::create(&log_path) {
            Ok(f)  => Some(BufWriter::new(f)),
            Err(e) => {
                eprintln!(
                    "[state_history] failed to open log file {}: {e} \
                     — continuing without the per-frame log",
                    log_path.display(),
                );
                None
            }
        };
        Self {
            records:               VecDeque::with_capacity(capacity),
            capacity,
            current_frame_builder: None,
            log_writer,
            log_path:              Some(log_path),
        }
    }

    /// Path of the per-frame log file, if any was configured.
    pub fn log_path(&self) -> Option<&std::path::Path> {
        self.log_path.as_deref()
    }

    /// Ring capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of records currently in the ring.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// `true` if the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Start a new builder for `frame`. Overwrites any in-progress
    /// builder from a prior frame that was not finalized — a caller-side
    /// programmer error that is logged but not fatal, since dropping the
    /// incomplete record loses only debug data.
    pub fn begin_frame(&mut self, frame: FrameIndex) {
        if let Some(old) = self.current_frame_builder.take() {
            eprintln!(
                "[state_history] begin_frame({:?}) dropped an unfinalized \
                 builder for frame {:?} — finalize_frame was not called",
                frame, old.frame_index,
            );
        }
        self.current_frame_builder = Some(FrameRecord::new(frame));
    }

    /// Append `record`, evicting the oldest entry if the ring is full.
    pub fn record_frame(&mut self, record: FrameRecord) {
        if self.records.len() == self.capacity {
            self.records.pop_front();
        }
        self.records.push_back(record);
    }

    /// Finalize the in-progress builder and push it into the ring.
    ///
    /// If `begin_frame` was never called for this update (caller logic
    /// bug), this is a silent no-op — the only loss is one frame of
    /// debug coverage.
    ///
    /// When a per-frame log file is configured (see
    /// [`StateHistory::new_with_log`]), a human-readable summary of the
    /// just-finalized record is appended before the ring push. I/O
    /// errors on the writer are logged to `stderr` and then silenced for
    /// the remainder of the run — one error is a signal; N are noise.
    pub fn finalize_frame(&mut self) {
        if let Some(rec) = self.current_frame_builder.take() {
            if let Some(writer) = self.log_writer.as_mut()
                && let Err(e) = write_frame_summary(writer, &rec)
            {
                eprintln!(
                    "[state_history] write to per-frame log failed: {e} \
                     — disabling further log writes for this run",
                );
                self.log_writer = None;
            }
            self.record_frame(rec);
        }
    }

    /// Mutable access to the in-progress frame builder. Returns `None`
    /// if `begin_frame` was not called.
    pub fn builder_mut(&mut self) -> Option<&mut FrameRecord> {
        self.current_frame_builder.as_mut()
    }

    /// Look up the recorded snapshot for `frame`.
    pub fn get(&self, frame: FrameIndex) -> Option<&FrameRecord> {
        self.records.iter().find(|r| r.frame_index == frame)
    }
}

// --- per-frame log formatter ---

/// Append a human-readable summary of one `FrameRecord` to `out`.
///
/// Each relevant line starts with a consistent `[tag]` prefix so the
/// user can grep the log for a specific event class (`[prep]`,
/// `[evict]`, `[insert]`, `[dirty]`, `[patch-copy]`,
/// `[retire-inconsistency]`, …). One frame per line-block; a blank
/// line separates frames.
///
/// The format is intentionally narrow (fixed-width fields, no JSON)
/// because the consumer is `grep` + human eyeballs, not another tool —
/// this file is meant to be opened, scanned, and used to reproduce a
/// bug report. Field order is stable so a `grep + sort` across a run
/// tells a coherent story.
pub fn write_frame_summary(
    out: &mut impl Write,
    rec: &FrameRecord,
) -> std::io::Result<()> {
    let frame = rec.frame_index.get();

    writeln!(
        out,
        "=== frame {frame} (prep_req={}, insert={}, evict={}, \
         dirty={}, patch={}, inconsistencies={}, allocator_active={}) ===",
        rec.prep_requests_issued.len(),
        rec.residency_inserts.len(),
        rec.residency_evicts.len(),
        rec.dirty_entries_retired.len(),
        rec.patch_copies_issued.len(),
        rec.retirement_inconsistencies.len(),
        rec.allocator_active,
    )?;

    for p in &rec.prep_requests_issued {
        writeln!(
            out,
            "[prep] frame {frame}: coord=({},{},{}) L{} dir_idx={}",
            p.coord.x, p.coord.y, p.coord.z,
            p.level.0, p.directory_index,
        )?;
    }

    for i in &rec.residency_inserts {
        writeln!(
            out,
            "[insert] frame {frame}: L{} coord=({},{},{}) dir_idx={}",
            i.level.0, i.coord.x, i.coord.y, i.coord.z, i.directory_index,
        )?;
    }

    for e in &rec.residency_evicts {
        writeln!(
            out,
            "[evict] frame {frame}: L{} coord=({},{},{}) dir_idx={}",
            e.level.0, e.coord.x, e.coord.y, e.coord.z, e.directory_index,
        )?;
    }

    for d in &rec.dirty_entries_retired {
        writeln!(
            out,
            "[dirty] frame {frame}: shader_dir_idx={} new_bits=0x{:08x} \
             staging_req_idx={} transition={:?} allocated_slot={} \
             coord=({},{},{})",
            d.directory_index,
            d.new_bits_partial,
            d.staging_request_idx,
            d.transition,
            match d.allocated_slot { Some(s) => format!("{s}"), None => "-".to_string() },
            d.coord[0], d.coord[1], d.coord[2],
        )?;
    }

    for p in &rec.patch_copies_issued {
        writeln!(
            out,
            "[patch-copy] frame {frame}: staging[req_idx={}] \
             -> material_pool[slot={}]",
            p.staging_request_idx, p.dst_material_slot,
        )?;
    }

    for r in &rec.retirement_inconsistencies {
        writeln!(
            out,
            "[retire-inconsistency] frame {frame}: shader dir_idx={} \
             but accepted_coords[{}] = coord ({},{},{}) L{} \
             recomputes to dir_idx={} (new_bits=0x{:08x}, staging_req_idx={})",
            r.shader_dir_idx,
            r.shader_dir_idx,
            r.cpu_coord.x, r.cpu_coord.y, r.cpu_coord.z,
            r.cpu_level.0,
            r.cpu_recomputed_dir_idx,
            r.dirty.new_bits_partial,
            r.dirty.staging_request_idx,
        )?;
    }

    writeln!(out)?;
    out.flush()?;
    Ok(())
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_frame(n: u64) -> FrameIndex {
        let mut f = FrameIndex::default();
        for _ in 0..n {
            f.advance();
        }
        f
    }

    #[test]
    fn new_enforces_positive_capacity() {
        let h = StateHistory::new(4);
        assert_eq!(h.capacity(), 4);
        assert_eq!(h.len(),      0);
        assert!(h.is_empty());
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn zero_capacity_panics() {
        let _ = StateHistory::new(0);
    }

    #[test]
    fn record_and_lookup_roundtrip() {
        let mut h = StateHistory::new(4);
        let f = mk_frame(7);
        let mut rec = FrameRecord::new(f);
        rec.allocator_active = 42;
        h.record_frame(rec);

        let got = h.get(f).expect("lookup should find the just-recorded frame");
        assert_eq!(got.frame_index, f);
        assert_eq!(got.allocator_active, 42);
    }

    #[test]
    fn ring_eviction_drops_oldest_when_full() {
        let mut h = StateHistory::new(3);
        for n in 0..5 {
            h.record_frame(FrameRecord::new(mk_frame(n)));
        }
        // Capacity=3, recorded 5 ⇒ oldest two (frames 0, 1) evicted.
        assert_eq!(h.len(), 3);
        assert!(h.get(mk_frame(0)).is_none(), "frame 0 should have been evicted");
        assert!(h.get(mk_frame(1)).is_none(), "frame 1 should have been evicted");
        for n in 2..5 {
            assert!(h.get(mk_frame(n)).is_some(), "frame {n} should be present");
        }
    }

    #[test]
    fn begin_finalize_round_trip_pushes_exactly_one() {
        let mut h = StateHistory::new(4);
        let f = mk_frame(3);
        h.begin_frame(f);
        // Mutate through the builder to simulate a typical update.
        let b = h.builder_mut().expect("builder should exist after begin_frame");
        b.allocator_active = 9;
        b.residency_inserts.push(ResidencyInsert {
            level:           Level(1),
            coord:           SubchunkCoord::new(0, 0, 0),
            directory_index: 0,
        });
        h.finalize_frame();

        assert_eq!(h.len(), 1);
        let got = h.get(f).expect("finalized frame should be retrievable");
        assert_eq!(got.allocator_active, 9);
        assert_eq!(got.residency_inserts.len(), 1);
    }

    #[test]
    fn finalize_without_begin_is_noop() {
        let mut h = StateHistory::new(4);
        h.finalize_frame();
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn begin_overwrites_unfinalized_builder() {
        let mut h = StateHistory::new(4);
        h.begin_frame(mk_frame(1));
        // Begin the next frame without finalizing — the prior builder is
        // dropped on the floor (behaviour documented on the method).
        h.begin_frame(mk_frame(2));
        h.finalize_frame();
        assert_eq!(h.len(), 1);
        assert!(h.get(mk_frame(2)).is_some());
        assert!(h.get(mk_frame(1)).is_none());
    }

    #[test]
    fn get_returns_none_for_absent_frame() {
        let mut h = StateHistory::new(4);
        h.record_frame(FrameRecord::new(mk_frame(0)));
        assert!(h.get(mk_frame(999)).is_none());
    }

    // -- per-frame log formatter --
    //
    // `write_frame_summary` is the single entry point into the file log.
    // We want to pin its output format down to the tags the task spec
    // promises (`[prep]`, `[insert]`, `[evict]`, `[dirty]`,
    // `[patch-copy]`, `[retire-inconsistency]`) so a future code change
    // that drops a tag or renames one fails here — anything in this log
    // is user-visible and grep-targeted.

    fn sample_dirty_entry(dir_idx: u32, req_idx: u32) -> DirtyEntry {
        DirtyEntry {
            directory_index:     dir_idx,
            new_bits_partial:    0x0000_0080, // resident bit set
            staging_request_idx: req_idx,
            _pad:                0,
        }
    }

    #[test]
    fn write_frame_summary_emits_all_tagged_lines() {
        let mut rec = FrameRecord::new(mk_frame(1247));
        rec.allocator_active = 3;

        rec.prep_requests_issued.push(PrepRequestRecord {
            coord:           SubchunkCoord::new(0, 0, -1),
            level:           Level(1),
            directory_index: 112,
        });
        rec.residency_inserts.push(ResidencyInsert {
            level:           Level(1),
            coord:           SubchunkCoord::new(0, 0, -1),
            directory_index: 112,
        });
        rec.residency_evicts.push(ResidencyEvict {
            level:           Level(1),
            coord:           SubchunkCoord::new(-2, 0, -1),
            directory_index: 118,
        });
        rec.dirty_entries_retired.push(RetiredDirtyEntry {
            directory_index:     80,
            coord:               [0, 0, -3],
            new_bits_partial:    0x0000_00BF,
            staging_request_idx: 2,
            transition:          TransitionKind::UniformToSparse,
            allocated_slot:      Some(80),
            new_entry:           DirEntry::empty(),
        });
        rec.patch_copies_issued.push(PatchCopy {
            staging_request_idx: 2,
            dst_material_slot:   80,
        });
        rec.retirement_inconsistencies.push(RetirementInconsistency {
            dirty:                  sample_dirty_entry(112, 5),
            cpu_coord:              SubchunkCoord::new(0, 0, -1),
            cpu_level:              Level(1),
            cpu_recomputed_dir_idx: 70,
            shader_dir_idx:         112,
        });

        let mut buf = Vec::<u8>::new();
        write_frame_summary(&mut buf, &rec).expect("write must succeed to a Vec");
        let out = String::from_utf8(buf).expect("log output is valid UTF-8");

        // Header.
        assert!(out.contains("=== frame 1247"),
            "header missing frame number: {out}");

        // One line per tagged event class.
        assert!(out.contains("[prep] frame 1247: coord=(0,0,-1) L1 dir_idx=112"),
            "missing [prep] line: {out}");
        assert!(out.contains("[insert] frame 1247: L1 coord=(0,0,-1) dir_idx=112"),
            "missing [insert] line: {out}");
        assert!(out.contains("[evict] frame 1247: L1 coord=(-2,0,-1) dir_idx=118"),
            "missing [evict] line: {out}");
        assert!(out.contains("[dirty] frame 1247: shader_dir_idx=80"),
            "missing [dirty] line: {out}");
        assert!(out.contains("staging_req_idx=2"),
            "missing staging_req_idx on [dirty] line: {out}");
        assert!(out.contains("[patch-copy] frame 1247: staging[req_idx=2] -> material_pool[slot=80]"),
            "missing [patch-copy] line: {out}");
        assert!(out.contains("[retire-inconsistency] frame 1247"),
            "missing [retire-inconsistency] line: {out}");
        assert!(out.contains("recomputes to dir_idx=70"),
            "missing mismatch detail on [retire-inconsistency]: {out}");
    }

    #[test]
    fn write_frame_summary_clean_frame_is_just_header() {
        // A frame with no events at all still writes the header so grep
        // by frame number finds every frame uniformly.
        let rec = FrameRecord::new(mk_frame(9));
        let mut buf = Vec::<u8>::new();
        write_frame_summary(&mut buf, &rec).expect("write must succeed to a Vec");
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("=== frame 9"),
            "clean-frame header missing: {out}");
        // No event tags should appear.
        for tag in &[
            "[prep]", "[insert]", "[evict]", "[dirty]", "[patch-copy]",
            "[retire-inconsistency]",
        ] {
            assert!(!out.contains(tag),
                "clean frame should not emit {tag} lines: {out}");
        }
    }

    #[test]
    fn write_frame_summary_marks_uniform_first_time_with_dash_slot() {
        // Uniform-empty retires have `allocated_slot: None` — the log
        // renders that as a `-` so a grep for `allocated_slot=-` picks
        // out the uniform cases specifically.
        let mut rec = FrameRecord::new(mk_frame(4));
        rec.dirty_entries_retired.push(RetiredDirtyEntry {
            directory_index:     0,
            coord:               [0, 0, 0],
            new_bits_partial:    0x0000_0080, // resident; exposure=0
            staging_request_idx: 0,
            transition:          TransitionKind::UniformFirstTime,
            allocated_slot:      None,
            new_entry:           DirEntry::empty(),
        });

        let mut buf = Vec::<u8>::new();
        write_frame_summary(&mut buf, &rec).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("allocated_slot=-"),
            "uniform retire should render allocated_slot as '-': {out}");
    }
}
