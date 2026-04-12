//! [`RingBookkeeping`] — the non-wgpu-touching core shared by `UploadRing`
//! and `ReadbackChannel`.
//!
//! Manages slot state, command-index assignment, GPU watermark tracking,
//! and overflow-policy decisions without holding any GPU handles. This
//! type exists to make the rings unit-testable without real hardware:
//! the wgpu wrappers in `upload.rs` and `readback.rs` delegate to it and
//! add the buffer/encoder plumbing on top.
//!
//! See `.local/renderer_plan.md` §3.4, §3.5, and §10.1 for the design
//! rationale and test surface.
//!
//! ## Model
//!
//! - `current_slot` is the CPU's active slot. It rotates forward modulo
//!   `frame_count` on each successful [`RingBookkeeping::rotate_frame`].
//! - `slot_watermarks[slot]` is the highest [`CommandIndex`] pushed into
//!   `slot` that has not yet been retired.
//!   [`CommandWatermark::default`] (zero) means "empty / retired."
//! - `slot_fills[slot]` is the number of messages currently in `slot`,
//!   enforced against `capacity_per_frame` by [`RingBookkeeping::push`].
//! - `last_observed_gpu_watermark` is the highest GPU-reported watermark
//!   seen by [`RingBookkeeping::observe_gpu_watermark`]; slots whose
//!   highest command index is `<= last_observed_gpu_watermark` are
//!   retired.
//!
//! ## TimeoutCrash bookkeeping
//!
//! `rotations_without_progress` counts [`RingBookkeeping::rotate_frame`]
//! *attempts*, including those that return
//! [`PolicyError::RotateWouldBlock`] in a poll-retry loop. The counter
//! resets to zero whenever [`RingBookkeeping::observe_gpu_watermark`]
//! strictly advances `last_observed_gpu_watermark`. Under
//! [`OverflowPolicy::TimeoutCrash { frames }`] the method panics with a
//! full diagnostic when the counter exceeds `frames` — i.e. on the
//! `(frames + 1)`-th consecutive no-progress attempt.
//!
//! [`OverflowPolicy::TimeoutCrash { frames }`]: OverflowPolicy::TimeoutCrash

// Increment 8 (`UploadRing`) is now the first non-test consumer of
// `RingBookkeeping`. The file-level `#![allow(dead_code)]` is removed;
// individual inspection methods whose first non-test consumer lands in
// a later increment carry per-method allows below.

use crate::frame::FrameCount;

use super::policy::{OverflowPolicy, PolicyError};
use super::watermark::{CommandIndex, CommandRange, CommandWatermark};

/// Pure-logic slot-and-watermark bookkeeping shared by the CPU-write and
/// GPU-write ring primitives.
///
/// See the module documentation for the full state model. All operations
/// are O(frame_count) or better; there are no allocations on the hot path
/// except the one-time `Vec` allocations in [`Self::new`] and the
/// transient diagnostic `Vec` inside a TimeoutCrash panic.
pub(crate) struct RingBookkeeping {
    /// Human-readable label, used only in diagnostic messages (the
    /// `TimeoutCrash` panic in particular).
    label: String,

    capacity_per_frame : u32,
    frame_count        : FrameCount,
    policy             : OverflowPolicy,

    /// Next [`CommandIndex`] value to assign. Starts at `1` so the zero
    /// [`CommandWatermark`] sentinel unambiguously means "empty slot /
    /// nothing processed yet."
    next_command_index: u64,

    /// Per-slot: highest command index pushed into the slot that is still
    /// pending. `CommandWatermark::default()` means the slot is empty.
    slot_watermarks: Vec<CommandWatermark>,

    /// Per-slot: number of messages currently in the slot.
    slot_fills: Vec<u32>,

    /// Highest watermark the GPU has reported as processed.
    last_observed_gpu_watermark: CommandWatermark,

    /// Active CPU slot (`0..frame_count`).
    current_slot: u32,

    /// Number of [`Self::rotate_frame`] attempts since the GPU watermark
    /// last advanced. Reset by [`Self::observe_gpu_watermark`] on forward
    /// progress. Drives [`OverflowPolicy::TimeoutCrash`].
    rotations_without_progress: u32,
}

// --- RingBookkeeping ---

impl RingBookkeeping {
    /// Construct a fresh bookkeeping record. All slots start empty, the
    /// next command index is `1`, and the no-progress counter is `0`.
    ///
    /// Panics if `capacity_per_frame == 0` — a zero-capacity ring cannot
    /// accept any push, which is a construction-time programmer error.
    pub(crate) fn new(
        label              : impl Into<String>,
        capacity_per_frame : u32,
        frame_count        : FrameCount,
        policy             : OverflowPolicy,
    )
        -> Self
    {
        assert!(
            capacity_per_frame > 0,
            "RingBookkeeping::new: capacity_per_frame must be > 0",
        );
        let n_slots = frame_count.get() as usize;
        Self {
            label              : label.into(),
            capacity_per_frame ,
            frame_count        ,
            policy             ,
            next_command_index          : 1,
            slot_watermarks             : vec![CommandWatermark::default(); n_slots],
            slot_fills                  : vec![0; n_slots],
            last_observed_gpu_watermark : CommandWatermark::default(),
            current_slot                : 0,
            rotations_without_progress  : 0,
        }
    }

    // --- Inspection ---

    pub(crate) fn label(&self) -> &str {
        &self.label
    }

    pub(crate) fn capacity_per_frame(&self) -> u32 {
        self.capacity_per_frame
    }

    pub(crate) fn frame_count(&self) -> FrameCount {
        self.frame_count
    }

    pub(crate) fn policy(&self) -> OverflowPolicy {
        self.policy
    }

    pub(crate) fn current_slot(&self) -> u32 {
        self.current_slot
    }

    #[allow(dead_code)] // First non-test caller: ReadbackChannel (Increment 9).
    pub(crate) fn last_observed_gpu_watermark(&self) -> CommandWatermark {
        self.last_observed_gpu_watermark
    }

    #[allow(dead_code)] // First non-test caller: ReadbackChannel (Increment 9).
    pub(crate) fn slot_fill(&self, slot: u32) -> u32 {
        self.slot_fills[slot as usize]
    }

    #[allow(dead_code)] // First non-test caller: ReadbackChannel (Increment 9).
    pub(crate) fn slot_watermark(&self, slot: u32) -> CommandWatermark {
        self.slot_watermarks[slot as usize]
    }

    #[allow(dead_code)] // First non-test caller: validation harness (Increment 10).
    pub(crate) fn rotations_without_progress(&self) -> u32 {
        self.rotations_without_progress
    }

    /// `true` iff `slot` has any command still pending (non-zero
    /// watermark).
    pub(crate) fn is_slot_pending(&self, slot: u32) -> bool {
        self.slot_watermarks[slot as usize].get() > 0
    }

    // --- Mutation ---

    /// Assign `count` consecutive command indices to the current slot.
    ///
    /// Returns the assigned [`CommandRange`] on success. Returns
    /// [`PolicyError::PushWouldBlock`] if the slot's fill plus `count`
    /// would exceed `capacity_per_frame`.
    ///
    /// Panics if `count == 0` (caller bug — a zero-push has no meaningful
    /// range to return), or if the command-index counter would reach
    /// `u32::MAX` (first-pass shader-side watermark limit, see
    /// `.local/renderer_plan.md` §3.6).
    pub(crate) fn push(
        &mut self,
        count: u32,
    )
        -> Result<CommandRange, PolicyError>
    {
        assert!(
            count > 0,
            "RingBookkeeping::push: count must be > 0 (caller bug)",
        );

        let slot = self.current_slot as usize;
        let fill = self.slot_fills[slot];
        let new_fill = fill.checked_add(count).expect(
            "RingBookkeeping::push: slot fill overflowed u32 (caller \
             pushed more than u32::MAX messages in a single slot)",
        );
        if new_fill > self.capacity_per_frame {
            return Err(PolicyError::PushWouldBlock {
                slot         : self.current_slot,
                current_fill : fill,
                requested    : count,
                capacity     : self.capacity_per_frame,
            });
        }

        // First-pass shader-side watermark is u32; CPU-side indices must
        // stay within u32::MAX. Plan §3.6. Hard panic — a follow-up pass
        // is the correct fix, not a workaround.
        let first_u64 = self.next_command_index;
        let last_u64  = first_u64 + (count as u64) - 1;
        if last_u64 >= u32::MAX as u64 {
            panic!(
                "ring `{}`: CommandIndex {} would reach u32::MAX; the \
                 GPU-side watermark is u32 and cannot track further. \
                 Upgrade to u64 atomics in a follow-up pass (plan §3.6).",
                self.label, last_u64,
            );
        }

        self.next_command_index = last_u64 + 1;
        self.slot_fills[slot] = new_fill;
        self.slot_watermarks[slot] = CommandWatermark(last_u64);
        Ok(CommandRange {
            first : CommandIndex(first_u64),
            last  : CommandIndex(last_u64),
        })
    }

    /// Observe the GPU's reported progress watermark.
    ///
    /// If `w` strictly advances `last_observed_gpu_watermark`, the new
    /// value is recorded and [`Self::rotations_without_progress`] is
    /// reset to zero. Any slot whose highest command index is `<= w` is
    /// then retired (watermark zeroed, fill zeroed).
    ///
    /// Stale or backward-moving observations are ignored — the method is
    /// idempotent for watermarks that don't advance progress.
    pub(crate) fn observe_gpu_watermark(&mut self, w: CommandWatermark) {
        if w > self.last_observed_gpu_watermark {
            self.last_observed_gpu_watermark = w;
            self.rotations_without_progress = 0;
        }
        let observed = self.last_observed_gpu_watermark;
        for slot_idx in 0..self.slot_watermarks.len() {
            let wm = self.slot_watermarks[slot_idx];
            if wm.get() > 0 && wm <= observed {
                self.slot_watermarks[slot_idx] = CommandWatermark::default();
                self.slot_fills[slot_idx] = 0;
            }
        }
    }

    /// Rotate `current_slot` forward by one, modulo `frame_count`.
    ///
    /// Under [`OverflowPolicy::Block`] and [`OverflowPolicy::TimeoutCrash`]
    /// this returns [`PolicyError::RotateWouldBlock`] without side effects
    /// on the slot state if the target slot still has a pending watermark
    /// above the last observed GPU watermark — the wgpu wrapper is
    /// expected to poll the GPU, feed the new watermark in via
    /// [`Self::observe_gpu_watermark`], and retry.
    ///
    /// Under [`OverflowPolicy::Drop`] rotation is infallible: the target
    /// slot is cleared unconditionally, overwriting any pending state.
    /// This is the pathological path the plan calls out in §3.4 — under
    /// normal Drop-policy operation the slot has already retired by the
    /// time we rotate back to it.
    ///
    /// Under [`OverflowPolicy::TimeoutCrash { frames }`] the method also
    /// increments [`Self::rotations_without_progress`] on every attempt
    /// (successful *and* `RotateWouldBlock`) and panics with full
    /// diagnostic context once the counter exceeds `frames`. The panic
    /// fires *before* any state change so that diagnostic state reflects
    /// the situation at the moment of failure.
    pub(crate) fn rotate_frame(&mut self) -> Result<(), PolicyError> {
        let next_slot = (self.current_slot + 1) % self.frame_count.get();

        // Every rotate_frame attempt — successful or RotateWouldBlock —
        // counts against the TimeoutCrash grace period. Reset by
        // observe_gpu_watermark when forward progress is seen. Saturating
        // add so a runaway loop can never silently wrap to zero.
        self.rotations_without_progress =
            self.rotations_without_progress.saturating_add(1);

        // TimeoutCrash panic check, BEFORE mutating slot state so the
        // panic diagnostic still shows the pending slot's watermark.
        if let OverflowPolicy::TimeoutCrash { frames } = self.policy
            && self.rotations_without_progress > frames
        {
            self.timeout_crash_panic(frames, next_slot);
            // unreachable
        }

        let target_pending = self.is_slot_pending(next_slot);
        if target_pending
            && matches!(
                self.policy,
                OverflowPolicy::Block | OverflowPolicy::TimeoutCrash { .. }
            )
        {
            return Err(PolicyError::RotateWouldBlock {
                next_slot ,
                pending   : self.slot_watermarks[next_slot as usize].get(),
                observed  : self.last_observed_gpu_watermark.get(),
            });
        }

        // Commit the rotation. Clearing the target slot is a no-op under
        // Block/TimeoutCrash (already empty at this point) and an
        // intentional overwrite under Drop.
        self.current_slot = next_slot;
        self.slot_watermarks[next_slot as usize] = CommandWatermark::default();
        self.slot_fills[next_slot as usize] = 0;
        Ok(())
    }

    fn timeout_crash_panic(
        &self,
        frames    : u32,
        next_slot : u32,
    )
        -> !
    {
        let pending: Vec<(u32, u64)> = self
            .slot_watermarks
            .iter()
            .enumerate()
            .filter_map(|(i, wm)| {
                if wm.get() > 0 {
                    Some((i as u32, wm.get()))
                }
                else {
                    None
                }
            })
            .collect();

        panic!(
            "ring `{label}`: TimeoutCrash — GPU watermark has not advanced \
             in {count} consecutive rotate_frame attempts (limit {limit})\n  \
             observed GPU watermark: {observed}\n  \
             current slot: {current} -> attempted next slot: {next}\n  \
             frame_count: {frame_count}, capacity_per_frame: {capacity}\n  \
             pending slots (slot, highest command index): {pending:?}",
            label        = self.label,
            count        = self.rotations_without_progress,
            limit        = frames,
            observed     = self.last_observed_gpu_watermark.get(),
            current      = self.current_slot,
            next         = next_slot,
            frame_count  = self.frame_count.get(),
            capacity     = self.capacity_per_frame,
        );
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn new_bk(
        cap    : u32,
        frames : u32,
        policy : OverflowPolicy,
    )
        -> RingBookkeeping
    {
        RingBookkeeping::new(
            "test-ring",
            cap,
            FrameCount::new(frames).unwrap(),
            policy,
        )
    }

    // --- Construction ---

    #[test]
    fn new_starts_with_empty_slots_and_index_one() {
        let bk = new_bk(4, 2, OverflowPolicy::Block);
        assert_eq!(bk.capacity_per_frame(), 4);
        assert_eq!(bk.frame_count().get(), 2);
        assert_eq!(bk.current_slot(), 0);
        assert_eq!(bk.last_observed_gpu_watermark().get(), 0);
        assert_eq!(bk.rotations_without_progress(), 0);
        for slot in 0..bk.frame_count().get() {
            assert_eq!(bk.slot_fill(slot), 0);
            assert_eq!(bk.slot_watermark(slot).get(), 0);
            assert!(!bk.is_slot_pending(slot));
        }
    }

    #[test]
    #[should_panic(expected = "capacity_per_frame must be > 0")]
    fn new_panics_on_zero_capacity() {
        let _ = new_bk(0, 2, OverflowPolicy::Block);
    }

    // --- Push: command-index assignment ---

    #[test]
    fn push_assigns_monotonic_indices_starting_at_one() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Block);

        let r0 = bk.push(1).unwrap();
        assert_eq!(r0.first, CommandIndex(1));
        assert_eq!(r0.last, CommandIndex(1));
        assert_eq!(r0.count(), 1);

        let r1 = bk.push(1).unwrap();
        assert_eq!(r1.first, CommandIndex(2));
        assert_eq!(r1.last, CommandIndex(2));

        let r2 = bk.push(1).unwrap();
        assert_eq!(r2.first, CommandIndex(3));
    }

    #[test]
    fn push_many_returns_contiguous_inclusive_range() {
        let mut bk = new_bk(10, 2, OverflowPolicy::Block);

        let range = bk.push(3).unwrap();
        assert_eq!(range.first, CommandIndex(1));
        assert_eq!(range.last, CommandIndex(3));
        assert_eq!(range.count(), 3);

        let range = bk.push(4).unwrap();
        assert_eq!(range.first, CommandIndex(4));
        assert_eq!(range.last, CommandIndex(7));
        assert_eq!(range.count(), 4);
    }

    #[test]
    fn push_updates_slot_fill_and_watermark() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Block);

        let _ = bk.push(2).unwrap();
        assert_eq!(bk.slot_fill(0), 2);
        assert_eq!(bk.slot_watermark(0).get(), 2); // highest CI = 2
        assert!(bk.is_slot_pending(0));
        // Other slot untouched.
        assert_eq!(bk.slot_fill(1), 0);
        assert!(!bk.is_slot_pending(1));
    }

    // --- Push: capacity ---

    #[test]
    fn push_fills_slot_to_exact_capacity_ok() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Block);
        assert!(bk.push(4).is_ok());
        assert_eq!(bk.slot_fill(0), 4);
    }

    #[test]
    fn push_beyond_capacity_returns_push_would_block() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Block);
        bk.push(3).unwrap();
        let err = bk.push(2).unwrap_err();
        assert_eq!(
            err,
            PolicyError::PushWouldBlock {
                slot         : 0,
                current_fill : 3,
                requested    : 2,
                capacity     : 4,
            },
        );
        // Rejected push is side-effect free.
        assert_eq!(bk.slot_fill(0), 3);
        assert_eq!(bk.slot_watermark(0).get(), 3);
    }

    #[test]
    fn push_beyond_capacity_from_empty_slot_is_rejected() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Drop);
        let err = bk.push(5).unwrap_err();
        assert!(matches!(err, PolicyError::PushWouldBlock { .. }));
        // Nothing committed.
        assert_eq!(bk.slot_fill(0), 0);
        assert_eq!(bk.slot_watermark(0).get(), 0);
    }

    #[test]
    #[should_panic(expected = "count must be > 0")]
    fn push_panics_on_zero_count() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Block);
        let _ = bk.push(0);
    }

    // --- observe_gpu_watermark ---

    #[test]
    fn observe_retires_slots_below_and_equal_to_watermark() {
        let mut bk = new_bk(4, 4, OverflowPolicy::Block);

        // Push one message into each of slots 0..3, rotating between.
        bk.push(1).unwrap();           // slot 0 -> CI 1
        bk.rotate_frame().unwrap();
        bk.push(1).unwrap();           // slot 1 -> CI 2
        bk.rotate_frame().unwrap();
        bk.push(1).unwrap();           // slot 2 -> CI 3
        bk.rotate_frame().unwrap();
        bk.push(1).unwrap();           // slot 3 -> CI 4
        assert_eq!(bk.current_slot(), 3);

        // GPU reports through CI 2 — slots 0 and 1 should retire.
        bk.observe_gpu_watermark(CommandWatermark(2));
        assert!(!bk.is_slot_pending(0));
        assert!(!bk.is_slot_pending(1));
        assert!(bk.is_slot_pending(2));
        assert!(bk.is_slot_pending(3));
        assert_eq!(bk.slot_fill(0), 0);
        assert_eq!(bk.slot_fill(1), 0);
    }

    #[test]
    fn observe_leaves_slots_above_watermark_pending() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Block);

        // Two pushes in slot 0 -> CIs 1, 2. Highest watermark for slot = 2.
        bk.push(2).unwrap();
        bk.rotate_frame().unwrap();
        bk.push(2).unwrap(); // slot 1 -> CIs 3, 4. Highest = 4.

        // Observe watermark 3 — slot 0 retires (2 <= 3), slot 1 stays
        // pending because its highest is 4.
        bk.observe_gpu_watermark(CommandWatermark(3));
        assert!(!bk.is_slot_pending(0));
        assert!(bk.is_slot_pending(1));
        assert_eq!(bk.slot_watermark(1).get(), 4);
    }

    #[test]
    fn observe_below_current_is_ignored() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Block);
        bk.push(1).unwrap();
        bk.observe_gpu_watermark(CommandWatermark(5));
        assert_eq!(bk.last_observed_gpu_watermark().get(), 5);

        // Observing a stale watermark must not move the counter back.
        bk.observe_gpu_watermark(CommandWatermark(3));
        assert_eq!(bk.last_observed_gpu_watermark().get(), 5);
    }

    #[test]
    fn observe_resets_rotations_without_progress_on_advance() {
        let mut bk = new_bk(4, 4, OverflowPolicy::Block);
        bk.push(1).unwrap(); // slot 0 -> CI 1

        bk.rotate_frame().unwrap();
        bk.rotate_frame().unwrap();
        assert_eq!(bk.rotations_without_progress(), 2);

        bk.observe_gpu_watermark(CommandWatermark(1));
        assert_eq!(bk.rotations_without_progress(), 0);
    }

    #[test]
    fn observe_without_advance_does_not_reset_counter() {
        let mut bk = new_bk(4, 4, OverflowPolicy::Block);
        bk.rotate_frame().unwrap();
        bk.rotate_frame().unwrap();
        assert_eq!(bk.rotations_without_progress(), 2);

        // Re-observing the same (zero) watermark must not reset.
        bk.observe_gpu_watermark(CommandWatermark(0));
        assert_eq!(bk.rotations_without_progress(), 2);
    }

    // --- rotate_frame: Block policy ---

    #[test]
    fn rotate_block_with_free_target_succeeds() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Block);
        // Slot 1 is initially free.
        assert!(bk.rotate_frame().is_ok());
        assert_eq!(bk.current_slot(), 1);
    }

    #[test]
    fn rotate_block_wraps_around_frame_count() {
        let mut bk = new_bk(4, 3, OverflowPolicy::Block);
        bk.rotate_frame().unwrap(); // 0 -> 1
        bk.rotate_frame().unwrap(); // 1 -> 2
        bk.rotate_frame().unwrap(); // 2 -> 0
        assert_eq!(bk.current_slot(), 0);
    }

    #[test]
    fn rotate_block_with_pending_target_returns_would_block() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Block);
        // Fill slot 0 and slot 1 with pending commands.
        bk.push(1).unwrap();           // slot 0 pending
        bk.rotate_frame().unwrap();    // -> slot 1
        bk.push(1).unwrap();           // slot 1 pending

        // Next rotation targets slot 0 which is still pending.
        let err = bk.rotate_frame().unwrap_err();
        assert_eq!(
            err,
            PolicyError::RotateWouldBlock {
                next_slot : 0,
                pending   : 1,
                observed  : 0,
            },
        );

        // State is unchanged by the failed rotation.
        assert_eq!(bk.current_slot(), 1);
        assert!(bk.is_slot_pending(0));
        assert!(bk.is_slot_pending(1));
    }

    #[test]
    fn rotate_block_retry_succeeds_after_observe() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Block);
        bk.push(1).unwrap();           // slot 0 CI 1
        bk.rotate_frame().unwrap();
        bk.push(1).unwrap();           // slot 1 CI 2

        // First attempt to rotate 1 -> 0 blocks.
        assert!(bk.rotate_frame().is_err());

        // GPU retires slot 0.
        bk.observe_gpu_watermark(CommandWatermark(1));
        assert!(!bk.is_slot_pending(0));

        // Retry succeeds.
        bk.rotate_frame().unwrap();
        assert_eq!(bk.current_slot(), 0);
        assert!(!bk.is_slot_pending(0));
    }

    // --- rotate_frame: Drop policy ---

    #[test]
    fn rotate_drop_never_errors_even_on_pending_target() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Drop);
        bk.push(1).unwrap();           // slot 0 pending
        bk.rotate_frame().unwrap();    // -> slot 1
        bk.push(1).unwrap();           // slot 1 pending

        // Rotating back to slot 0 is infallible under Drop; the pending
        // state there is overwritten.
        bk.rotate_frame().unwrap();
        assert_eq!(bk.current_slot(), 0);
        assert!(!bk.is_slot_pending(0));
        assert_eq!(bk.slot_fill(0), 0);
    }

    #[test]
    fn rotate_drop_clears_only_the_target_slot() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Drop);
        bk.push(1).unwrap();           // slot 0 -> CI 1
        bk.rotate_frame().unwrap();
        bk.push(2).unwrap();           // slot 1 -> CIs 2, 3

        // Rotate back to slot 0. Slot 1's pending state must be untouched.
        bk.rotate_frame().unwrap();
        assert_eq!(bk.current_slot(), 0);
        assert!(bk.is_slot_pending(1));
        assert_eq!(bk.slot_watermark(1).get(), 3);
    }

    // --- rotate_frame: TimeoutCrash policy ---

    #[test]
    fn rotate_timeout_crash_within_grace_does_not_panic() {
        let mut bk = new_bk(4, 4, OverflowPolicy::TimeoutCrash { frames: 3 });
        bk.rotate_frame().unwrap(); // pwp = 1
        bk.rotate_frame().unwrap(); // pwp = 2
        bk.rotate_frame().unwrap(); // pwp = 3
        assert_eq!(bk.rotations_without_progress(), 3);
        // Three no-progress rotations are tolerated; no panic so far.
    }

    #[test]
    #[should_panic(expected = "TimeoutCrash")]
    fn rotate_timeout_crash_panics_on_fourth_attempt_with_frames_three() {
        let mut bk = new_bk(4, 4, OverflowPolicy::TimeoutCrash { frames: 3 });
        bk.push(1).unwrap(); // leave a pending command so the panic
                             // diagnostic has something to print.
        bk.rotate_frame().unwrap(); // pwp = 1
        bk.rotate_frame().unwrap(); // pwp = 2
        bk.rotate_frame().unwrap(); // pwp = 3
        let _ = bk.rotate_frame();  // pwp = 4 > 3 -> PANIC
    }

    #[test]
    fn rotate_timeout_crash_reset_by_observe_watermark() {
        let mut bk = new_bk(4, 4, OverflowPolicy::TimeoutCrash { frames: 2 });
        bk.push(1).unwrap(); // slot 0 -> CI 1

        for _ in 0..10 {
            bk.rotate_frame().unwrap();
            // Every rotation is followed by an observe that advances the
            // watermark, which must reset rotations_without_progress.
            // Use a watermark that strictly advances every iteration.
            let w = bk.last_observed_gpu_watermark().get() + 1;
            bk.observe_gpu_watermark(CommandWatermark(w));
            assert_eq!(bk.rotations_without_progress(), 0);
        }
    }

    #[test]
    fn rotate_timeout_crash_counts_would_block_retries() {
        // Verifies that RotateWouldBlock retries still count toward the
        // TimeoutCrash grace period — a stuck poll-retry loop must
        // eventually panic.
        let mut bk = new_bk(4, 2, OverflowPolicy::TimeoutCrash { frames: 3 });
        bk.push(1).unwrap();           // slot 0 pending
        bk.rotate_frame().unwrap();    // -> slot 1 (pwp = 1)
        bk.push(1).unwrap();           // slot 1 pending

        // Every subsequent rotate_frame returns RotateWouldBlock because
        // the GPU never progresses. Each attempt still increments pwp.
        assert!(bk.rotate_frame().is_err()); // pwp = 2
        assert_eq!(bk.rotations_without_progress(), 2);

        assert!(bk.rotate_frame().is_err()); // pwp = 3
        assert_eq!(bk.rotations_without_progress(), 3);

        // Fourth attempt: pwp = 4 > 3 -> panic.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = bk.rotate_frame();
        }));
        assert!(result.is_err(), "expected TimeoutCrash panic");
    }

    #[test]
    fn rotate_timeout_crash_panic_message_includes_pending_state() {
        let mut bk = new_bk(4, 4, OverflowPolicy::TimeoutCrash { frames: 1 });
        bk.push(2).unwrap(); // slot 0 -> CIs 1, 2. Highest watermark = 2.
        bk.rotate_frame().unwrap(); // pwp = 1

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // pwp = 2 > 1 -> panic.
            let _ = bk.rotate_frame();
        }));

        let Err(payload) = result
        else {
            panic!("expected TimeoutCrash panic, got Ok");
        };

        let msg: String = if let Some(s) = payload.downcast_ref::<&'static str>() {
            (*s).to_string()
        }
        else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        }
        else {
            panic!("panic payload was neither &str nor String");
        };

        assert!(msg.contains("TimeoutCrash"));
        assert!(msg.contains("test-ring"));
        // The diagnostic lists pending slots; slot 0 still holds watermark 2.
        assert!(msg.contains("pending slots"));
        assert!(msg.contains("(0, 2)"));
    }

    // --- Full lifecycle ---

    #[test]
    fn full_lifecycle_push_rotate_observe_retires() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Block);

        // Frame 0: push two messages.
        let r = bk.push(2).unwrap();
        assert_eq!(r.first, CommandIndex(1));
        assert_eq!(r.last, CommandIndex(2));
        assert!(bk.is_slot_pending(0));

        // Rotate to frame 1.
        bk.rotate_frame().unwrap();
        assert_eq!(bk.current_slot(), 1);

        // Frame 1: push one message.
        let r = bk.push(1).unwrap();
        assert_eq!(r.first, CommandIndex(3));
        assert!(bk.is_slot_pending(1));

        // GPU reports through CI 2 — slot 0 retires, slot 1 stays pending.
        bk.observe_gpu_watermark(CommandWatermark(2));
        assert!(!bk.is_slot_pending(0));
        assert!(bk.is_slot_pending(1));

        // Now rotating back to slot 0 succeeds.
        bk.rotate_frame().unwrap();
        assert_eq!(bk.current_slot(), 0);

        // Frame 2 push reuses slot 0 with a fresh fill.
        let r = bk.push(4).unwrap();
        assert_eq!(r.first, CommandIndex(4));
        assert_eq!(r.last, CommandIndex(7));
        assert_eq!(bk.slot_fill(0), 4);
    }

    #[test]
    fn push_after_retirement_uses_full_slot_capacity_again() {
        let mut bk = new_bk(4, 2, OverflowPolicy::Block);
        bk.push(4).unwrap(); // slot 0 full
        bk.rotate_frame().unwrap();
        bk.observe_gpu_watermark(CommandWatermark(4)); // retire slot 0

        // Rotating back should clear slot 0 fresh; a full capacity push
        // must succeed.
        bk.rotate_frame().unwrap();
        assert_eq!(bk.current_slot(), 0);
        bk.push(4).unwrap();
        assert_eq!(bk.slot_fill(0), 4);
    }
}
