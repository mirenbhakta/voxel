//! Divergence reporter for the `debug-state-history` feature.
//!
//! Given a CPU-authored directory snapshot (from
//! [`crate::world::state_history::FrameRecord`]) and a GPU-observed
//! snapshot of the same frame's `slot_directory_buf`, classify the
//! per-entry differences so the user can tell *why* the two disagree.
//!
//! The kind classification is the payoff: a `CoordDiffer` points at a
//! slot-compute bug (directory_index → coord mapping differs between
//! CPU and GPU), a `MaterialSlotDiffer` points at an allocator / CPU
//! patch-scheduling bug, a `ResidencyDiffer` usually means a
//! write-ordering bug (the GPU observed an older generation than the
//! CPU authored), and a `BitsDiffer` on identical coord / material_slot
//! means the shader-observed exposure or is_solid flags drifted from
//! what retirement wrote.
//!
//! All types here are feature-gated. See `state_history.rs` for the
//! CPU-side ledger this comparator consumes.

#![cfg(feature = "debug-state-history")]

use std::io::Write;

use renderer::{BITS_EXPOSURE_MASK, BITS_IS_SOLID, DirEntry, FrameIndex};

// --- DivergenceKind ---

/// Why a directory entry's CPU and GPU versions disagree.
///
/// A single entry can disagree in multiple ways (e.g. both coord and
/// bits differ); the classifier reports the highest-priority kind
/// first, on the assumption that the higher-priority class is closer
/// to the root cause. Priority (highest first):
///
/// 1. `ResidencyDiffer` — the resident bit disagrees. Usually a
///    write-ordering bug (CPU flushed a write the GPU hasn't seen yet,
///    or the GPU has state from before the CPU's intended clear).
/// 2. `CoordDiffer` — the entry's coord field differs. Points at the
///    CPU↔GPU slot-resolution formula mismatch (see
///    `failure-resolve-coord-to-slot-diverges-from-cpu-pool`).
/// 3. `MaterialSlotDiffer` — material_slot field in bits differs. The
///    identity allocator shouldn't allow this unless the renderer's
///    `write_directory_entries` raced the patch pass, or the allocator
///    invariant was broken (see `failure-allocator-identity-drift`).
/// 4. `BitsDiffer` — the remaining bits (exposure + is_solid +
///    content_version / last_synth_version) disagree. Typically a prep
///    shader output drift or a retirement-path bit-packing bug.
// The common `Differ` suffix is load-bearing for clarity at the log
// call site (`reason=CoordDiffer`); silence the lint explicitly rather
// than dropping the suffix.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DivergenceKind {
    ResidencyDiffer,
    CoordDiffer,
    MaterialSlotDiffer,
    BitsDiffer,
}

// --- DivergenceSample ---

/// One divergent directory entry, with both CPU and GPU views.
///
/// `DirEntry` doesn't derive `Debug`; this struct doesn't either, since
/// the canonical human-readable formatter is [`format_sample`] (written
/// into the divergence log). Callers needing a quick debug print can
/// format via that helper or hand-format the fields.
#[derive(Clone, Copy)]
pub struct DivergenceSample {
    pub directory_index: u32,
    pub cpu:             DirEntry,
    pub gpu:             DirEntry,
    pub reason:          DivergenceKind,
}

// --- DivergenceReport ---

/// Aggregate result of one CPU↔GPU directory comparison.
///
/// `total` is the entry count of the smaller of the two inputs (the
/// comparator truncates); `divergent` counts how many of those entries
/// disagree. `samples` holds the first `sample_limit` divergences; the
/// default through [`compare_directory_snapshots`] is 8.
#[derive(Clone)]
pub struct DivergenceReport {
    pub frame:     FrameIndex,
    pub total:     usize,
    pub divergent: usize,
    pub samples:   Vec<DivergenceSample>,
}

impl DivergenceReport {
    /// `true` if every compared entry matched.
    pub fn is_clean(&self) -> bool {
        self.divergent == 0
    }

    /// Fingerprint the report by the set of divergent `directory_index`
    /// values. Two frames with the same set of divergent indices return
    /// the same fingerprint, which lets the caller collapse runs of
    /// identical divergences into a single log line. O(samples.len()).
    ///
    /// The fingerprint is a FNV-1a hash of the sorted indices — cheap
    /// to compute and, given that `samples.len()` is capped at ~8, the
    /// theoretical collision risk is far below the noise floor of the
    /// debug path.
    pub fn fingerprint(&self) -> u64 {
        let mut idxs: Vec<u32> = self.samples.iter().map(|s| s.directory_index).collect();
        idxs.sort_unstable();
        let mut h: u64 = 0xcbf29ce484222325;
        for i in &idxs {
            for b in i.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }
}

// --- comparison ---

/// Maximum number of divergence samples emitted per report. Keeps log
/// noise bounded when a wide pattern of entries disagrees; the
/// `divergent` count in the summary tells the user how many were
/// elided.
pub const DEFAULT_SAMPLE_LIMIT: usize = 8;

/// Compare two directory snapshots entry-for-entry, classify each
/// divergence, and write a human-readable summary to `log`.
///
/// Writes nothing to `log` if the snapshots match. When they disagree,
/// writes one header line + one line per sample (up to
/// [`DEFAULT_SAMPLE_LIMIT`]) in the format documented on the module.
///
/// The returned [`DivergenceReport`] is always returned — the write to
/// `log` is a side effect for the interactive / console path; callers
/// that want to suppress or redirect output pass any `impl Write` (e.g.
/// `std::io::sink()` for silent comparison, or a `Vec<u8>` for
/// capture).
pub fn compare_directory_snapshots(
    frame: FrameIndex,
    cpu:   &[DirEntry],
    gpu:   &[DirEntry],
    log:   &mut impl Write,
) -> DivergenceReport {
    compare_directory_snapshots_with_limit(frame, cpu, gpu, DEFAULT_SAMPLE_LIMIT, log)
}

/// Variant of [`compare_directory_snapshots`] with a caller-supplied
/// sample limit. Separated so tests can pin a limit without sharing
/// global state with production callers.
pub fn compare_directory_snapshots_with_limit(
    frame:        FrameIndex,
    cpu:          &[DirEntry],
    gpu:          &[DirEntry],
    sample_limit: usize,
    log:          &mut impl Write,
) -> DivergenceReport {
    let total = cpu.len().min(gpu.len());
    let mut samples = Vec::<DivergenceSample>::new();
    let mut divergent = 0usize;

    for i in 0..total {
        let c = cpu[i];
        let g = gpu[i];
        if let Some(kind) = classify(&c, &g) {
            divergent += 1;
            if samples.len() < sample_limit {
                samples.push(DivergenceSample {
                    directory_index: i as u32,
                    cpu:             c,
                    gpu:             g,
                    reason:          kind,
                });
            }
        }
    }

    let report = DivergenceReport { frame, total, divergent, samples };

    if report.divergent > 0 {
        // Header. Errors are swallowed — the log is an opportunistic
        // side-channel, not a contract.
        let _ = writeln!(
            log,
            "[divergence] frame {}: {}/{} entries disagree",
            frame.get(), report.divergent, report.total,
        );
        for s in &report.samples {
            let _ = writeln!(log, "  {}", format_sample(s));
        }
        if report.divergent > report.samples.len() {
            let _ = writeln!(
                log,
                "  ... {} more divergent entries elided (sample_limit = {})",
                report.divergent - report.samples.len(),
                sample_limit,
            );
        }
    }

    report
}

/// Inspect a pair of entries. Return the highest-priority
/// [`DivergenceKind`] if they disagree, or `None` when they are
/// identical.
fn classify(cpu: &DirEntry, gpu: &DirEntry) -> Option<DivergenceKind> {
    if entries_equal(cpu, gpu) {
        return None;
    }

    // Residency is the first thing the cull shader reads; a mismatch
    // here is usually a dropped or reordered queue.write_buffer and is
    // the most actionable finding.
    if cpu.is_resident() != gpu.is_resident() {
        return Some(DivergenceKind::ResidencyDiffer);
    }

    // Coord mismatch points at slot formula divergence.
    if cpu.coord != gpu.coord {
        return Some(DivergenceKind::CoordDiffer);
    }

    // Material slot mismatch in bits[8..31].
    if cpu.material_slot() != gpu.material_slot() {
        return Some(DivergenceKind::MaterialSlotDiffer);
    }

    // Everything else — exposure drift, is_solid drift, content_version
    // drift.
    Some(DivergenceKind::BitsDiffer)
}

fn entries_equal(a: &DirEntry, b: &DirEntry) -> bool {
    a.coord              == b.coord
        && a.bits               == b.bits
        && a.content_version    == b.content_version
        && a.last_synth_version == b.last_synth_version
}

fn format_sample(s: &DivergenceSample) -> String {
    format!(
        "dir_idx={:>4} reason={:<18} cpu={}  gpu={}",
        s.directory_index,
        format!("{:?}", s.reason),
        fmt_entry(&s.cpu),
        fmt_entry(&s.gpu),
    )
}

fn fmt_entry(e: &DirEntry) -> String {
    let exposure = e.bits & BITS_EXPOSURE_MASK;
    let is_solid = (e.bits & BITS_IS_SOLID) != 0;
    format!(
        "{{coord={:>3},{:>3},{:>3} bits=0x{:08x} mat={:>4} res={} solid={} expo=0x{:02x} cv={} lsv={}}}",
        e.coord[0], e.coord[1], e.coord[2],
        e.bits,
        e.material_slot(),
        e.is_resident() as u32,
        is_solid as u32,
        exposure,
        e.content_version,
        e.last_synth_version,
    )
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use renderer::BITS_EXPOSURE_MASK;

    fn frame(n: u64) -> FrameIndex {
        let mut f = FrameIndex::default();
        for _ in 0..n {
            f.advance();
        }
        f
    }

    #[test]
    fn identical_snapshots_report_no_divergence() {
        let cpu = vec![DirEntry::empty([0, 0, 0]), DirEntry::resident([0, 0, 0], 0x3F, false, 0)];
        let gpu = cpu.clone();
        let mut sink = Vec::<u8>::new();
        let report = compare_directory_snapshots(frame(1), &cpu, &gpu, &mut sink);
        assert!(report.is_clean());
        assert_eq!(report.divergent, 0);
        assert!(report.samples.is_empty());
        assert!(sink.is_empty(), "no log output when snapshots match");
    }

    #[test]
    fn residency_differ_outranks_other_kinds() {
        // CPU says resident at [1,2,3] with material_slot 2.
        // GPU says non-resident at [7,8,9] — both residency AND coord AND
        // material_slot differ, but classifier must prefer Residency.
        let cpu = vec![DirEntry::resident([1, 2, 3], 0x3F, false, 2)];
        let gpu = vec![DirEntry::empty([7, 8, 9])];
        let mut sink = Vec::<u8>::new();
        let r = compare_directory_snapshots(frame(0), &cpu, &gpu, &mut sink);
        assert_eq!(r.divergent, 1);
        assert_eq!(r.samples[0].reason, DivergenceKind::ResidencyDiffer);
    }

    #[test]
    fn coord_differ_when_residency_and_slot_agree() {
        // Both resident, same material_slot, differing coord. Priority
        // falls through Residency → Coord.
        let cpu = vec![DirEntry::resident([0, -2, 0], 0x3F, false, 8)];
        let gpu = vec![DirEntry::resident([0, -2, -2], 0x3F, false, 8)];
        let mut sink = Vec::<u8>::new();
        let r = compare_directory_snapshots(frame(0), &cpu, &gpu, &mut sink);
        assert_eq!(r.divergent, 1);
        assert_eq!(r.samples[0].reason, DivergenceKind::CoordDiffer);
    }

    #[test]
    fn material_slot_differ_when_coord_matches() {
        let cpu = vec![DirEntry::resident([1, 1, 1], 0x3F, false, 5)];
        let gpu = vec![DirEntry::resident([1, 1, 1], 0x3F, false, 9)];
        let mut sink = Vec::<u8>::new();
        let r = compare_directory_snapshots(frame(0), &cpu, &gpu, &mut sink);
        assert_eq!(r.divergent, 1);
        assert_eq!(r.samples[0].reason, DivergenceKind::MaterialSlotDiffer);
    }

    #[test]
    fn bits_differ_on_exposure_drift() {
        // Same coord, same material_slot, different exposure.
        let cpu = vec![DirEntry::resident([2, 2, 2], BITS_EXPOSURE_MASK, false, 3)];
        let gpu = vec![DirEntry::resident([2, 2, 2], 0x21,                 false, 3)];
        let mut sink = Vec::<u8>::new();
        let r = compare_directory_snapshots(frame(0), &cpu, &gpu, &mut sink);
        assert_eq!(r.divergent, 1);
        assert_eq!(r.samples[0].reason, DivergenceKind::BitsDiffer);
    }

    #[test]
    fn bits_differ_on_content_version_drift() {
        // Same coord, same bits field, differing content_version ⇒
        // classifier falls through to BitsDiffer.
        let mut a = DirEntry::resident([0, 0, 0], 0x3F, false, 1);
        let mut b = a;
        a.content_version = 1;
        b.content_version = 2;
        let mut sink = Vec::<u8>::new();
        let r = compare_directory_snapshots(frame(0), &[a], &[b], &mut sink);
        assert_eq!(r.divergent, 1);
        assert_eq!(r.samples[0].reason, DivergenceKind::BitsDiffer);
    }

    #[test]
    fn sample_limit_caps_the_list_but_not_the_count() {
        // Build a run of 20 all-different entries, limit samples to 3.
        let cpu: Vec<DirEntry> = (0..20)
            .map(|i| DirEntry::resident([i, 0, 0], 0x3F, false, i as u32))
            .collect();
        let gpu: Vec<DirEntry> = (0..20)
            .map(|i| DirEntry::resident([i + 100, 0, 0], 0x3F, false, i as u32))
            .collect();
        let mut sink = Vec::<u8>::new();
        let r = compare_directory_snapshots_with_limit(frame(0), &cpu, &gpu, 3, &mut sink);
        assert_eq!(r.divergent, 20, "all 20 entries should be counted");
        assert_eq!(r.samples.len(), 3, "sample list is capped by the limit");
        // Log summary must mention the elided count for observability.
        let out = String::from_utf8(sink).unwrap();
        assert!(
            out.contains("more divergent entries elided"),
            "log should flag elided entries; got {out:?}",
        );
    }

    #[test]
    fn mismatched_lengths_compare_over_the_shorter() {
        let cpu = vec![DirEntry::empty([0, 0, 0]), DirEntry::resident([1, 1, 1], 0x3F, false, 1)];
        let gpu = vec![DirEntry::empty([0, 0, 0])]; // shorter
        let mut sink = Vec::<u8>::new();
        let r = compare_directory_snapshots(frame(0), &cpu, &gpu, &mut sink);
        assert_eq!(r.total, 1, "comparison truncates to the shorter slice");
        assert!(r.is_clean(), "only the shared prefix is compared");
    }

    #[test]
    fn fingerprint_matches_for_same_divergent_indices() {
        // Two reports with the same divergent indices must fingerprint
        // identically; changing which indices disagree must change the
        // fingerprint. This is what the console-collapse logic keys on.
        let cpu = vec![DirEntry::empty([0, 0, 0]); 4];
        let mut gpu_a = cpu.clone();
        let mut gpu_b = cpu.clone();

        // Both reports disagree at indices 1 and 3; but the actual
        // differences are different. Fingerprint should still match.
        gpu_a[1] = DirEntry::resident([1, 0, 0], 0x3F, false, 1);
        gpu_a[3] = DirEntry::resident([3, 0, 0], 0x3F, false, 3);
        gpu_b[1] = DirEntry::resident([9, 9, 9], 0x21, true,  7);
        gpu_b[3] = DirEntry::resident([8, 8, 8], 0x01, false, 5);

        let mut sink = Vec::<u8>::new();
        let ra = compare_directory_snapshots(frame(0), &cpu, &gpu_a, &mut sink);
        sink.clear();
        let rb = compare_directory_snapshots(frame(1), &cpu, &gpu_b, &mut sink);

        assert_eq!(ra.fingerprint(), rb.fingerprint());

        // Divergent at a different index → different fingerprint.
        let mut gpu_c = cpu.clone();
        gpu_c[0] = DirEntry::resident([0, 0, 0], 0x3F, false, 0);
        gpu_c[2] = DirEntry::resident([2, 0, 0], 0x3F, false, 2);
        sink.clear();
        let rc = compare_directory_snapshots(frame(2), &cpu, &gpu_c, &mut sink);
        assert_ne!(ra.fingerprint(), rc.fingerprint());
    }

    #[test]
    fn log_output_is_empty_when_clean() {
        let cpu = vec![DirEntry::empty([0, 0, 0]); 3];
        let gpu = cpu.clone();
        let mut sink = Vec::<u8>::new();
        let _ = compare_directory_snapshots(frame(0), &cpu, &gpu, &mut sink);
        assert!(sink.is_empty());
    }

    #[test]
    fn log_output_mentions_frame_and_counts() {
        let cpu = vec![DirEntry::resident([0, 0, 0], 0x3F, false, 0)];
        let gpu = vec![DirEntry::empty([0, 0, 0])];
        let mut sink = Vec::<u8>::new();
        let r = compare_directory_snapshots(frame(1247), &cpu, &gpu, &mut sink);
        assert_eq!(r.divergent, 1);
        let out = String::from_utf8(sink).unwrap();
        assert!(out.contains("frame 1247"), "log should include the frame number, got: {out:?}");
        assert!(out.contains("1/1"),        "log should include divergent/total, got: {out:?}");
    }
}
