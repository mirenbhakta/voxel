//! [`ReadbackChannel<T>`] — typed async GPU→CPU readback ring.
//!
//! Wraps a [`MultiBufferRing`] of `COPY_DST | MAP_READ` buffers with a
//! per-slot fence state machine, giving consumers a pull-style API:
//!
//! ```text
//!   // once, at subsystem init
//!   let mut channel = ReadbackChannel::<Report>::new(ctx, pool, "prep",
//!                                                    OverflowPolicy::Panic);
//!
//!   // per frame, in subsystem
//!   ctx.device().poll(wgpu::PollType::Poll).unwrap();   // fires callbacks
//!   for report in channel.take_ready() { /* apply */ }
//!
//!   let dst = channel.reserve(frame);
//!   // record copy_buffer_to_buffer(<gpu_src>, dst) in the graph, with
//!   // a pass that writes `dst_handle = graph.import_buffer(dst.clone())`.
//!
//!   let submission = ctx.end_frame(fe);
//!   channel.commit_submit(frame, submission);
//! ```
//!
//! # Scheduling invariants
//!
//! A slot's lifecycle runs on the FIF cadence:
//!
//! | Phase      | Set by              | Cleared by        |
//! |------------|---------------------|-------------------|
//! | `Empty`    | `take_ready` drain  | —                 |
//! | `Reserved` | `reserve`           | `commit_submit`   |
//! | `InFlight` | `commit_submit`     | `take_ready`      |
//!
//! With `FrameCount = 2` and readback scheduled every frame, slot `K`
//! moves `Empty → Reserved → InFlight → Empty` in two frames. Pressing
//! `reserve` on a slot that's still `InFlight` (previous cycle's
//! readback hasn't retired) triggers the configured [`OverflowPolicy`].
//!
//! # Why the channel owns the state, not the ring
//!
//! [`MultiBufferRing`] is stateless — it only knows about slot rotation.
//! The channel layers fence-state on top: one atomic `ready` flag per
//! slot, set by the `map_async` callback and observed by `take_ready`.
//! This keeps the ring reusable for other multi-frame patterns that
//! don't need CPU-visible retirement (e.g. TAA history).

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytemuck::Pod;

use crate::device::RendererContext;
use crate::frame::FrameIndex;
use crate::graph::{BufferDesc, BufferPool};
use crate::multi_buffer::MultiBufferRing;

// --- OverflowPolicy ---

/// What [`ReadbackChannel::reserve`] does when the current frame's slot
/// is still in flight (callback not yet fired) or still holds undrained
/// data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverflowPolicy {
    /// Panic — the caller is misconfigured: readback cadence and frame
    /// count are out of sync, or `take_ready` is not being called each
    /// frame. Appropriate for subsystems that expect at most one
    /// outstanding readback per slot (e.g. GPU prep dirty lists, shadow
    /// ledger snapshots).
    Panic,

    /// Silently skip — `reserve` returns `None`. The caller's pass that
    /// would have written this slot simply doesn't run. Suitable for
    /// opportunistic readbacks where missing a frame is acceptable.
    Skip,
}

// --- SlotPhase ---

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotPhase {
    Empty,
    Reserved { frame: FrameIndex },
    InFlight { frame: FrameIndex },
}

// --- SlotState ---

struct SlotState {
    /// Shared with the `map_async` callback. Set to `true` when the GPU
    /// fence is reached and the callback fires; cleared to `false` after
    /// `take_ready` drains the slot.
    ready: Arc<AtomicBool>,
    phase: SlotPhase,
}

impl SlotState {
    fn new() -> Self {
        Self {
            ready: Arc::new(AtomicBool::new(false)),
            phase: SlotPhase::Empty,
        }
    }
}

// --- ReadbackChannel ---

/// Typed async GPU→CPU readback ring.
///
/// Each slot holds exactly one `T`. Consumers copy their GPU-side report
/// into the slot using `copy_buffer_to_buffer` in a graph pass, then the
/// channel takes care of mapping, callback dispatch, and typed delivery.
///
/// `T` must be [`Pod`]. The GPU writes must produce a byte-compatible
/// layout for `T`; this is the same contract every shared-struct type in
/// the codebase already follows.
pub struct ReadbackChannel<T: Pod> {
    ring         : MultiBufferRing<T>,
    slot_states  : Vec<SlotState>,
    overflow     : OverflowPolicy,
    _pd          : PhantomData<T>,
}

impl<T: Pod> ReadbackChannel<T> {
    /// Create a channel with one slot per frame-in-flight, each sized
    /// to exactly `size_of::<T>()` bytes and usage
    /// `COPY_DST | MAP_READ`.
    pub fn new(
        ctx      : &RendererContext,
        pool     : &mut BufferPool,
        label    : &str,
        overflow : OverflowPolicy,
    )
        -> Self
    {
        let payload_size = std::mem::size_of::<T>() as u64;
        assert!(
            payload_size > 0,
            "ReadbackChannel::new: T must not be a zero-sized type",
        );

        let desc = BufferDesc {
            size  : payload_size,
            usage : wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        };

        let ring      = MultiBufferRing::<T>::new(ctx, pool, label, desc);
        let n         = ring.len() as usize;
        let slot_states = (0..n).map(|_| SlotState::new()).collect();

        Self { ring, slot_states, overflow, _pd: PhantomData }
    }

    /// Claim this frame's slot as the destination for a GPU→CPU copy.
    ///
    /// The returned [`wgpu::Buffer`] is the slot's `MAP_READ | COPY_DST`
    /// buffer, ready to be imported into the render graph and written by
    /// a `copy_buffer_to_buffer` pass whose source is the caller-owned
    /// GPU buffer holding the report for this frame.
    ///
    /// Must be paired with exactly one
    /// [`commit_submit`](Self::commit_submit) call after the graph
    /// submits — otherwise the slot is stuck in `Reserved` and the next
    /// cycle's `reserve` on the same slot trips overflow.
    ///
    /// # Overflow behavior
    ///
    /// If the target slot is still `InFlight` (callback hasn't fired) or
    /// `Reserved` (commit missing from previous frame):
    /// - [`OverflowPolicy::Panic`] — panics.
    /// - [`OverflowPolicy::Skip`] — returns `None`.
    pub fn reserve(&mut self, frame: FrameIndex) -> Option<&wgpu::Buffer> {
        let slot_idx = frame.slot(self.ring_frame_count()) as usize;
        let phase    = self.slot_states[slot_idx].phase;

        match phase {
            SlotPhase::Empty => {
                self.slot_states[slot_idx].phase = SlotPhase::Reserved { frame };
                Some(self.ring.current(frame))
            }
            SlotPhase::Reserved { .. } | SlotPhase::InFlight { .. } => {
                match self.overflow {
                    OverflowPolicy::Panic => panic!(
                        "ReadbackChannel::reserve: slot {slot_idx} (frame \
                         {}) not drained — previous readback never \
                         retired. Check that take_ready is called each \
                         frame and device.poll is driven.",
                        frame.get(),
                    ),
                    OverflowPolicy::Skip => None,
                }
            }
        }
    }

    /// Arm the `map_async` callback for the slot most recently reserved
    /// for `frame`.
    ///
    /// Records a `SubmissionIndex` is not strictly needed here — wgpu
    /// fires the callback once *all* work preceding the `map_async`
    /// record in submission order has completed, which is guaranteed by
    /// the FIF fence discipline. The `_submission` parameter is accepted
    /// for API symmetry with other fence-aware primitives and future-proofs
    /// the channel against wgpu backends that expose per-submission
    /// waiting.
    ///
    /// # Panics
    ///
    /// Panics if `frame` does not match the slot's current `Reserved`
    /// entry — a signal that `reserve` was called on a different frame
    /// than the one being committed.
    pub fn commit_submit(&mut self, frame: FrameIndex, _submission: wgpu::SubmissionIndex) {
        let slot_idx = frame.slot(self.ring_frame_count()) as usize;
        match self.slot_states[slot_idx].phase {
            SlotPhase::Reserved { frame: reserved_frame } => {
                assert_eq!(
                    reserved_frame, frame,
                    "ReadbackChannel::commit_submit: reserved frame \
                     {} != commit frame {}",
                    reserved_frame.get(),
                    frame.get(),
                );
            }
            other => panic!(
                "ReadbackChannel::commit_submit: slot {slot_idx} is {other:?}, \
                 expected Reserved (did you forget to call reserve?)",
            ),
        }

        let ready = self.slot_states[slot_idx].ready.clone();
        self.ring.current(frame)
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                // A failed map is fatal: the slot will never retire and
                // subsequent frames will trip the overflow policy. Log
                // loudly; the caller's watchdog decides how to recover.
                if let Err(e) = result {
                    eprintln!("ReadbackChannel map_async failed: {e:?}");
                    return;
                }
                ready.store(true, Ordering::Release);
            });

        self.slot_states[slot_idx].phase = SlotPhase::InFlight { frame };
    }

    /// Drain every slot whose `map_async` callback has fired.
    ///
    /// For each ready slot: reads the mapped `T`, unmaps the buffer,
    /// resets the slot to `Empty`, and appends `(frame, value)` to the
    /// returned vec. `frame` is the [`FrameIndex`] the slot was last
    /// reserved on — consumers use it to match against per-frame
    /// bookkeeping (e.g. "which prep requests were dispatched that
    /// frame").
    ///
    /// The caller must have driven `device.poll` (at least once this
    /// frame, typically at the top of the frame) for the callbacks to
    /// have fired.
    pub fn take_ready(&mut self) -> Vec<(FrameIndex, T)> {
        let mut out = Vec::new();

        for slot_idx in 0..self.slot_states.len() {
            let SlotPhase::InFlight { frame } = self.slot_states[slot_idx].phase else {
                continue;
            };
            if !self.slot_states[slot_idx].ready.load(Ordering::Acquire) {
                continue;
            }

            let buffer = self.ring.current(frame);
            let slice  = buffer.slice(..);
            let data   = slice.get_mapped_range();
            let value  = bytemuck::pod_read_unaligned::<T>(&data[..std::mem::size_of::<T>()]);
            drop(data);
            buffer.unmap();

            self.slot_states[slot_idx].ready.store(false, Ordering::Release);
            self.slot_states[slot_idx].phase = SlotPhase::Empty;

            out.push((frame, value));
        }

        out
    }

    /// Return the ring's buffers to `pool` with a delayed-availability
    /// gate so in-flight GPU work doesn't race an unrelated transient
    /// reusing a slot.
    ///
    /// Forward to [`MultiBufferRing::release_into`]. Called when the
    /// channel's owning subsystem is torn down.
    pub fn release_into(self, pool: &mut BufferPool, current_frame: FrameIndex) {
        // Any slot still `InFlight` will complete after our gate, and
        // the pool's `available_at` check handles that.
        //
        // Slots in `Reserved` state (mis-used API) will never fire a
        // callback — we abandon them; the buffer itself is still safe
        // to recycle by the pool after the gate passes.
        for state in &self.slot_states {
            // Best-effort cleanup: clear ready flags so any late callback
            // firing on a now-released Arc is a harmless store.
            state.ready.store(false, Ordering::Release);
        }

        self.ring.release_into(pool, current_frame);
    }

    /// Number of slots. Equals `ctx.frame_count()` at construction.
    pub fn slot_count(&self) -> u32 {
        self.ring.len()
    }

    fn ring_frame_count(&self) -> crate::frame::FrameCount {
        // Ring's len matches ctx.frame_count() at construction; we rebuild
        // a FrameCount from it to reuse FrameIndex::slot's mod arithmetic
        // without threading FrameCount through SlotState.
        crate::frame::FrameCount::new(self.ring.len())
            .expect("ring len is always a valid FrameCount")
    }
}
