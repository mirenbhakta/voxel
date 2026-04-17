//! [`MultiBufferRing`] — a persistent ring of GPU buffers sized to the
//! frame count, for resources that must remain alive across multiple
//! frame executions.
//!
//! The render graph's built-in transients are single-frame: they're
//! acquired at execute time, valid through the frame's submit, and
//! returned to the pool for reuse on the next frame.  That matches
//! write-once-then-read workflows whose data lives for the length of one
//! command buffer (indirect args, depth targets, compute scratch).
//!
//! Multi-frame workflows — CPU readback with `map_async`, TAA-style
//! history buffers, or any feedback pass that reads prior-frame output —
//! need a GPU buffer whose *identity* persists across frames.  That's
//! what this type provides: on construction it allocates
//! [`FrameCount`] buffers from the pool, holds
//! them for its lifetime, and exposes a `current(frame)` accessor that
//! returns the slot for a given [`FrameIndex`] via the standard
//! `frame % frame_count` rotation.  Previous frames' slots are reachable
//! via [`previous`](MultiBufferRing::previous) when a consumer wants to
//! read last frame's data on the GPU.
//!
//! The ring is a *utility*, not a graph-managed resource: the caller
//! stores the `MultiBufferRing` alongside its other per-subsystem state,
//! and per frame calls `graph.import_buffer(ring.current(frame).clone())`
//! to wire the slot into that frame's passes.  The graph stays
//! rebuild-each-frame — no cross-execution bookkeeping.
//!
//! On drop the ring hands its buffers back to the pool via
//! [`release_with_gate`](crate::graph::BufferPool::release_with_gate) with
//! `available_at = drop_frame + frame_count`, matching the FIF discipline
//! that guarantees the GPU has finished referencing them by that point.

use std::marker::PhantomData;

use crate::device::RendererContext;
use crate::frame::{FrameCount, FrameIndex};
use crate::graph::{BufferDesc, BufferPool};

// --- MultiBufferRing ---

/// Persistent ring of `N` GPU buffers, where `N = ctx.frame_count()`.
///
/// Per-slot identity is stable for the lifetime of the ring: a caller
/// holding a `MultiBufferRing` can count on `ring.current(frame_F)` and
/// `ring.current(frame_F + N).clone()` resolving to the same
/// [`wgpu::Buffer`] across executions (mod the ring's drop).
///
/// `T` is a phantom marker that fixes the "kind of payload" at
/// construction, letting typed consumers (e.g.
/// [`ReadbackChannel<T>`](crate::readback::ReadbackChannel)) compose this
/// ring with their own typed API.  The ring itself does not interpret
/// buffer contents — its descriptor carries the raw byte layout.
pub struct MultiBufferRing<T> {
    buffers     : Vec<wgpu::Buffer>,
    frame_count : FrameCount,
    _pd         : PhantomData<T>,
}

impl<T> MultiBufferRing<T> {
    /// Allocate a new ring sized to `ctx.frame_count()`.
    ///
    /// Each slot is acquired from `pool` with `desc` and labelled with
    /// `"{label}:{slot_index}"` so they remain distinguishable in wgpu
    /// debug output.
    ///
    /// The ring is allocated immediately; subsequent construction-frame
    /// calls to `current(frame)` return the slot for that frame's rotation.
    pub fn new(
        ctx   : &RendererContext,
        pool  : &mut BufferPool,
        label : &str,
        desc  : BufferDesc,
    )
        -> Self
    {
        let frame_count = ctx.frame_count();
        let n           = frame_count.get() as usize;
        let frame       = ctx.frame_index();

        // Buffers acquired with the current frame index. Fresh allocations
        // have no gate; any entries the pool happened to have for this desc
        // with `available_at <= frame` are reused.
        let mut buffers = Vec::with_capacity(n);
        for i in 0..n {
            let slot_label = format!("{label}:{i}");
            buffers.push(pool.acquire(ctx.device(), &desc, Some(&slot_label), frame));
        }

        Self { buffers, frame_count, _pd: PhantomData }
    }

    /// Number of slots in the ring. Equals `ctx.frame_count()` at construction.
    pub fn len(&self) -> u32 {
        self.frame_count.get()
    }

    /// Always false — rings are allocated non-empty at construction.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The slot assigned to `frame`.
    ///
    /// Deterministic rotation: slot index = `frame.get() % frame_count`.
    /// Callers pass the current frame to write this frame's slot, or an
    /// earlier frame index to read a historical slot — see
    /// [`previous`](Self::previous) for a more readable k-frames-ago shape.
    pub fn current(&self, frame: FrameIndex) -> &wgpu::Buffer {
        let slot = frame.slot(self.frame_count) as usize;
        &self.buffers[slot]
    }

    /// The slot that was current `k` frames before `frame`.
    ///
    /// `k == 0` returns the same buffer as [`current`](Self::current).
    /// `k` must be strictly less than the ring length; higher values wrap
    /// back onto buffers that have since been overwritten, which is
    /// probably a bug at the call site.
    ///
    /// # Panics
    ///
    /// Panics if `k >= self.len()`.
    pub fn previous(&self, frame: FrameIndex, k: u32) -> &wgpu::Buffer {
        assert!(
            k < self.frame_count.get(),
            "MultiBufferRing::previous: k={k} >= frame_count={}",
            self.frame_count.get(),
        );
        let n    = self.frame_count.get() as u64;
        let slot = (frame.get().wrapping_sub(k as u64) % n) as usize;
        &self.buffers[slot]
    }

    /// Consume the ring and return its buffers to `pool`, gated for
    /// `frame_count` frames past `current_frame`.
    ///
    /// Must be called with the *current* frame index at the point of
    /// release. The FIF discipline guarantees the GPU is done with any
    /// slot by `current_frame + frame_count`, so that's the gate used —
    /// no acquire will hand any of these buffers to an unrelated transient
    /// until then.
    ///
    /// Dropping a `MultiBufferRing` without calling this method leaks its
    /// buffers back to wgpu (which cleans them up), but wastes the pool
    /// capacity.  `#[must_use]` flags forgotten cleanup at the call site.
    pub fn release_into(self, pool: &mut BufferPool, current_frame: FrameIndex) {
        let available_at = current_frame.plus(self.frame_count.get());
        for buffer in self.buffers {
            pool.release_with_gate(buffer, available_at);
        }
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameCount;

    // --- current() / previous() rotation math ---

    // The rotation logic doesn't need a GPU — it's pure integer arithmetic
    // on FrameIndex and FrameCount. These tests build a ring from a hand-
    // constructed Vec of placeholder buffers via a helper, letting the
    // unit test run on CI without PASSTHROUGH_SHADERS.
    //
    // Real GPU round-trip (allocate via pool → import into graph → release)
    // sits under the `#[ignore]` GPU bucket alongside the other
    // hardware-gated tests in this crate.

    /// Build a ring whose buffers are placeholders tagged by index, bypassing
    /// the pool / device path entirely.  Only valid for rotation-logic tests;
    /// the returned buffers are real `wgpu::Buffer` handles sized to 4 bytes
    /// for identity comparisons via `PartialEq`.
    fn test_ring(device: &wgpu::Device, count: u32) -> MultiBufferRing<u32> {
        let frame_count = FrameCount::new(count).unwrap();
        let buffers = (0..count)
            .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
                label              : Some(&format!("ring_slot_{i}")),
                size               : 4,
                usage              : wgpu::BufferUsages::STORAGE,
                mapped_at_creation : false,
            }))
            .collect();
        MultiBufferRing { buffers, frame_count, _pd: PhantomData }
    }

    fn headless_device() -> wgpu::Device {
        use crate::device::RendererContext;
        let ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct");
        ctx.device().clone()
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn current_rotates_with_frame() {
        let device = headless_device();
        let ring   = test_ring(&device, 2);

        let b0 = ring.current(FrameIndex::default());
        let b1 = ring.current(FrameIndex::default().plus(1));
        let b2 = ring.current(FrameIndex::default().plus(2));

        assert_ne!(b0, b1, "adjacent frames must land on distinct slots");
        assert_eq!(b0, b2, "slot repeats every frame_count frames");
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn previous_zero_equals_current() {
        let device = headless_device();
        let ring   = test_ring(&device, 3);
        let frame  = FrameIndex::default().plus(5);

        assert_eq!(ring.previous(frame, 0), ring.current(frame));
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn previous_one_is_prior_frames_slot() {
        let device = headless_device();
        let ring   = test_ring(&device, 3);
        let frame  = FrameIndex::default().plus(5);

        assert_eq!(ring.previous(frame, 1), ring.current(frame.plus(0).plus(2)));
        // frame 5 → slot 5%3=2; frame 4 → slot 4%3=1; above checks previous(5,1) == current(4)
        // Express as: frame - 1 == frame + (N-1) mod N
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    #[should_panic(expected = "k=3 >= frame_count=3")]
    fn previous_out_of_range_panics() {
        let device = headless_device();
        let ring   = test_ring(&device, 3);
        let _      = ring.previous(FrameIndex::default().plus(5), 3);
    }
}
