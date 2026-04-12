//! `UploadRing<T>`: CPU-write, GPU-read ring with per-frame-in-flight slots.
//!
//! Wraps [`RingBookkeeping`] with a single `wgpu::Buffer` and a CPU-side
//! staging `Vec<T>`. The buffer holds `N * capacity_per_frame` elements of
//! `T` (one contiguous region per slot); the shader accesses the active
//! slot at `buffer[consts.upload_slot * consts.upload_capacity + i]`.
//!
//! See `.local/renderer_plan.md` §3.4 and principle 1 in
//! `docs/renderer_rewrite_principles.md`.
//!
//! ## Buffer layout
//!
//! ```text
//! [ slot_0: capacity * sizeof(T) | slot_1: capacity * sizeof(T) | ... ]
//! ```
//!
//! `queue.write_buffer` writes the staging area into the active slot's byte
//! range on [`Self::flush`]. The active slot is surfaced through
//! [`GpuConstsData::upload_slot`] so the shader never needs to know the
//! buffer's internal layout.
//!
//! ## Overflow policy
//!
//! The pure-logic `RingBookkeeping` returns [`PolicyError`] uniformly for
//! all three policies; the wgpu wrapper here does **not** contain a
//! poll-and-retry loop for [`OverflowPolicy::Block`]. The caller (or a
//! higher-level frame loop) is responsible for calling
//! `device.poll(Wait)`, feeding back the GPU watermark via
//! [`Self::observe_gpu_watermark`], and retrying. This keeps the ring
//! primitive independent of the readback channel it does not own.
//!
//! [`RingBookkeeping`]: super::bookkeeping::RingBookkeeping
//! [`GpuConstsData::upload_slot`]: crate::gpu_consts::GpuConstsData::upload_slot
//! [`PolicyError`]: super::policy::PolicyError

use crate::device::RendererContext;
use crate::frame::FrameCount;
use crate::pipeline::binding::{BindEntry, BindKind};

use super::bookkeeping::RingBookkeeping;
use super::policy::{OverflowPolicy, PolicyError};
use super::watermark::{CommandIndex, CommandRange, CommandWatermark};

/// CPU-write, GPU-read ring buffer with per-frame-in-flight slot rotation.
///
/// `T` must be [`bytemuck::Pod`] — the standard bound for GPU-visible POD in
/// the Rust/wgpu ecosystem. `bytemuck::Pod` implies `Copy`, `'static`, and
/// `#[repr(C)]` (or equivalent transparent/packed repr), which is exactly
/// what `queue.write_buffer` and HLSL `StructuredBuffer<T>` need.
///
/// See the module documentation for buffer layout, flush semantics, and
/// overflow-policy behavior.
pub struct UploadRing<T: bytemuck::Pod> {
    bookkeeping: RingBookkeeping,
    buffer: wgpu::Buffer,
    /// CPU-side write area for the current slot. Length tracks
    /// `bookkeeping.slot_fill(current_slot)` — they are always in sync.
    staging: Vec<T>,
    /// Byte offset of a single slot within the buffer:
    /// `capacity_per_frame * size_of::<T>()`.
    slot_stride_bytes: u64,
}

impl<T: bytemuck::Pod> UploadRing<T> {
    /// Construct a new upload ring.
    ///
    /// `capacity_per_frame` is the maximum number of `T` messages that can be
    /// pushed between frame rotations. The GPU buffer is allocated as
    /// `frame_count * capacity_per_frame * size_of::<T>()` bytes with
    /// `STORAGE | COPY_DST` usage.
    ///
    /// # Panics
    ///
    /// Panics if `capacity_per_frame == 0` (delegated to `RingBookkeeping::new`).
    /// Panics if `size_of::<T>() == 0` — zero-sized types have no meaningful
    /// GPU representation.
    pub fn new(
        ctx: &RendererContext,
        label: &str,
        capacity_per_frame: u32,
        policy: OverflowPolicy,
    ) -> Self {
        let t_size = std::mem::size_of::<T>();
        assert!(
            t_size > 0,
            "UploadRing::new: T must not be a zero-sized type \
             (size_of::<T>() == 0 has no GPU representation)",
        );

        let frame_count = ctx.frame_count();
        let bookkeeping = RingBookkeeping::new(
            label,
            capacity_per_frame,
            frame_count,
            policy,
        );

        let slot_stride_bytes = capacity_per_frame as u64 * t_size as u64;
        let total_bytes = frame_count.get() as u64 * slot_stride_bytes;

        let buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            bookkeeping,
            buffer,
            staging: Vec::with_capacity(capacity_per_frame as usize),
            slot_stride_bytes,
        }
    }

    // --- Inspection ---

    /// Human-readable label, forwarded from construction.
    pub fn label(&self) -> &str {
        self.bookkeeping.label()
    }

    /// Maximum messages per frame slot.
    pub fn capacity_per_frame(&self) -> u32 {
        self.bookkeeping.capacity_per_frame()
    }

    /// The frame count this ring was constructed for.
    pub fn frame_count(&self) -> FrameCount {
        self.bookkeeping.frame_count()
    }

    /// The overflow policy this ring was constructed with.
    pub fn policy(&self) -> OverflowPolicy {
        self.bookkeeping.policy()
    }

    /// The active CPU slot index (`0..frame_count`).
    pub fn current_slot(&self) -> u32 {
        self.bookkeeping.current_slot()
    }

    /// Number of messages currently staged for the active slot.
    pub fn current_fill(&self) -> u32 {
        self.staging.len() as u32
    }

    // --- Push ---

    /// Push a single message into the current slot's staging area.
    ///
    /// Returns the [`CommandIndex`] assigned to the message on success.
    /// Returns [`PolicyError::PushWouldBlock`] if the slot is full.
    pub fn push(&mut self, msg: T) -> Result<CommandIndex, PolicyError> {
        let range = self.bookkeeping.push(1)?;
        self.staging.push(msg);
        Ok(range.first)
    }

    /// Push multiple messages. All-or-nothing: either all are accepted
    /// (and assigned a contiguous [`CommandRange`]) or none are.
    pub fn push_many(&mut self, msgs: &[T]) -> Result<CommandRange, PolicyError> {
        assert!(
            !msgs.is_empty(),
            "UploadRing::push_many: msgs must not be empty",
        );
        let range = self.bookkeeping.push(msgs.len() as u32)?;
        self.staging.extend_from_slice(msgs);
        Ok(range)
    }

    // --- Watermark ---

    /// Observe the GPU's reported progress watermark. Retires any slots
    /// whose highest command index is `<=` the observed watermark.
    ///
    /// Called by the subsystem owner after reading a watermark from a
    /// `ReadbackChannel` or equivalent CPU-visible signal.
    pub fn observe_gpu_watermark(&mut self, w: CommandWatermark) {
        self.bookkeeping.observe_gpu_watermark(w);
    }

    // --- Flush / Rotate ---

    /// Flush the current slot's staging area to the GPU buffer via
    /// `queue.write_buffer`.
    ///
    /// Writes `staging.len() * size_of::<T>()` bytes into the buffer at
    /// the current slot's byte offset. A no-op if the staging area is
    /// empty (nothing was pushed since the last flush or rotation).
    ///
    /// Exposed publicly for the rare case where a subsystem wants to flush
    /// mid-frame. Normally called automatically by [`Self::rotate_frame`].
    pub fn flush(&mut self, ctx: &RendererContext) {
        if self.staging.is_empty() {
            return;
        }
        let offset = self.bookkeeping.current_slot() as u64 * self.slot_stride_bytes;
        ctx.queue().write_buffer(
            &self.buffer,
            offset,
            bytemuck::cast_slice(&self.staging),
        );
    }

    /// Rotate to the next frame slot.
    ///
    /// Flushes the current slot's staging area, then delegates to
    /// `RingBookkeeping::rotate_frame` for slot-state management and
    /// overflow-policy enforcement. On success, the staging area is
    /// cleared for the new slot.
    ///
    /// Returns [`PolicyError::RotateWouldBlock`] if the next slot has
    /// pending commands that the GPU has not yet retired — the caller
    /// should poll the device, feed back the GPU watermark via
    /// [`Self::observe_gpu_watermark`], and retry.
    ///
    /// Under [`OverflowPolicy::TimeoutCrash`], panics with full diagnostic
    /// context if the GPU watermark has not advanced within the configured
    /// grace period.
    ///
    /// Under [`OverflowPolicy::Drop`], rotation is infallible — the
    /// target slot is cleared unconditionally.
    #[allow(dead_code)] // First non-test caller: RendererContext::begin_frame (later increment).
    pub(crate) fn rotate_frame(
        &mut self,
        ctx: &RendererContext,
    ) -> Result<(), PolicyError> {
        self.flush(ctx);
        self.bookkeeping.rotate_frame()?;
        self.staging.clear();
        Ok(())
    }

    // --- Binding ---

    /// Return the [`BindEntry`] descriptor for this ring's GPU buffer.
    ///
    /// The buffer is bound as a read-only storage buffer (the GPU reads
    /// messages; only the CPU writes via `queue.write_buffer`). The
    /// `size` field covers the full N-slot buffer so the bind group layout
    /// matches regardless of which slot is active.
    pub fn bind_entry(&self, binding: u32) -> BindEntry {
        BindEntry {
            binding,
            kind: BindKind::StorageBufferReadOnly {
                size: self.buffer.size(),
            },
            visibility: wgpu::ShaderStages::COMPUTE,
        }
    }

    /// Direct access to the underlying wgpu buffer. Available within the
    /// crate for test readback verification and bind-group construction.
    #[allow(dead_code)] // First non-test caller: bind-group construction in validation binary (Increment 10).
    pub(crate) fn raw_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameCount;

    // --- CPU tests (no GPU needed) ---
    //
    // These tests verify the UploadRing's delegation to RingBookkeeping
    // and staging-area management. The bookkeeping's own invariants are
    // exhaustively tested in bookkeeping.rs; here we test the integration
    // layer.

    /// Helper: construct a headless context. Panics (test failure) if no
    /// GPU is available — all tests using this are `#[ignore]`-gated.
    fn headless_ctx(frames: u32) -> RendererContext {
        pollster::block_on(RendererContext::new_headless(
            FrameCount::new(frames).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine")
    }

    // --- Construction ---

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn new_creates_buffer_of_correct_size() {
        let ctx = headless_ctx(2);
        let ring: UploadRing<[u32; 4]> = UploadRing::new(
            &ctx, "test_ring", 8, OverflowPolicy::Block,
        );

        // 2 slots * 8 messages * 16 bytes = 256
        assert_eq!(ring.raw_buffer().size(), 256);
        assert_eq!(ring.capacity_per_frame(), 8);
        assert_eq!(ring.frame_count().get(), 2);
        assert_eq!(ring.current_slot(), 0);
        assert_eq!(ring.current_fill(), 0);
        assert_eq!(ring.label(), "test_ring");
        assert_eq!(ring.policy(), OverflowPolicy::Block);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    #[should_panic(expected = "zero-sized type")]
    fn new_panics_on_zst() {
        let ctx = headless_ctx(2);
        let _: UploadRing<()> = UploadRing::new(
            &ctx, "zst_ring", 4, OverflowPolicy::Block,
        );
    }

    // --- Push ---

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn push_increments_fill_and_returns_monotonic_indices() {
        let ctx = headless_ctx(2);
        let mut ring: UploadRing<u32> = UploadRing::new(
            &ctx, "test", 4, OverflowPolicy::Block,
        );

        let ci0 = ring.push(100).unwrap();
        assert_eq!(ci0, CommandIndex(1));
        assert_eq!(ring.current_fill(), 1);

        let ci1 = ring.push(200).unwrap();
        assert_eq!(ci1, CommandIndex(2));
        assert_eq!(ring.current_fill(), 2);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn push_many_stages_all_messages() {
        let ctx = headless_ctx(2);
        let mut ring: UploadRing<u32> = UploadRing::new(
            &ctx, "test", 8, OverflowPolicy::Block,
        );

        let range = ring.push_many(&[10, 20, 30]).unwrap();
        assert_eq!(range.first, CommandIndex(1));
        assert_eq!(range.last, CommandIndex(3));
        assert_eq!(ring.current_fill(), 3);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn push_beyond_capacity_returns_error() {
        let ctx = headless_ctx(2);
        let mut ring: UploadRing<u32> = UploadRing::new(
            &ctx, "test", 4, OverflowPolicy::Drop,
        );

        ring.push_many(&[1, 2, 3, 4]).unwrap();
        let err = ring.push(5).unwrap_err();
        assert!(matches!(err, PolicyError::PushWouldBlock { .. }));
        // Staging unchanged after rejected push.
        assert_eq!(ring.current_fill(), 4);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    #[should_panic(expected = "must not be empty")]
    fn push_many_panics_on_empty_slice() {
        let ctx = headless_ctx(2);
        let mut ring: UploadRing<u32> = UploadRing::new(
            &ctx, "test", 4, OverflowPolicy::Block,
        );
        let _ = ring.push_many(&[]);
    }

    // --- Flush / Rotate ---

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn rotate_clears_staging_for_next_slot() {
        let ctx = headless_ctx(2);
        let mut ring: UploadRing<u32> = UploadRing::new(
            &ctx, "test", 4, OverflowPolicy::Drop,
        );

        ring.push_many(&[1, 2]).unwrap();
        assert_eq!(ring.current_fill(), 2);

        ring.rotate_frame(&ctx).unwrap();
        assert_eq!(ring.current_slot(), 1);
        assert_eq!(ring.current_fill(), 0);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn rotate_block_returns_would_block_on_pending_slot() {
        let ctx = headless_ctx(2);
        let mut ring: UploadRing<u32> = UploadRing::new(
            &ctx, "test", 4, OverflowPolicy::Block,
        );

        ring.push(1).unwrap();
        ring.rotate_frame(&ctx).unwrap(); // slot 0 -> 1
        ring.push(2).unwrap();

        // Rotate 1 -> 0: slot 0 is still pending (no GPU progress).
        let err = ring.rotate_frame(&ctx).unwrap_err();
        assert!(matches!(err, PolicyError::RotateWouldBlock { .. }));
        // Still on slot 1 after failed rotation.
        assert_eq!(ring.current_slot(), 1);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn rotate_succeeds_after_observe_retires_target() {
        let ctx = headless_ctx(2);
        let mut ring: UploadRing<u32> = UploadRing::new(
            &ctx, "test", 4, OverflowPolicy::Block,
        );

        ring.push(1).unwrap(); // slot 0, CI 1
        ring.rotate_frame(&ctx).unwrap();
        ring.push(2).unwrap(); // slot 1, CI 2

        // Slot 0 is pending. Observe watermark that retires it.
        ring.observe_gpu_watermark(CommandWatermark(1));

        // Now rotation 1 -> 0 succeeds.
        ring.rotate_frame(&ctx).unwrap();
        assert_eq!(ring.current_slot(), 0);
        assert_eq!(ring.current_fill(), 0);
    }

    // --- Binding ---

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn bind_entry_produces_correct_descriptor() {
        let ctx = headless_ctx(2);
        let ring: UploadRing<[u32; 4]> = UploadRing::new(
            &ctx, "test", 8, OverflowPolicy::Block,
        );

        let entry = ring.bind_entry(1);
        assert_eq!(entry.binding, 1);
        assert_eq!(entry.visibility, wgpu::ShaderStages::COMPUTE);
        match entry.kind {
            BindKind::StorageBufferReadOnly { size } => {
                assert_eq!(size, 256); // 2 * 8 * 16
            }
            other => panic!("expected StorageBufferReadOnly, got {other:?}"),
        }
    }

    // --- GPU integration: flush + readback verification ---

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn flush_writes_staging_to_correct_buffer_offset() {
        let ctx = headless_ctx(2);
        let mut ring: UploadRing<u32> = UploadRing::new(
            &ctx, "test", 4, OverflowPolicy::Block,
        );

        // Push two messages into slot 0 and flush.
        ring.push(0xDEAD_BEEFu32).unwrap();
        ring.push(0xCAFE_BABEu32).unwrap();
        ring.flush(&ctx);

        // Read back the buffer contents via a staging buffer + map.
        let readback = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: ring.raw_buffer().size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        encoder.copy_buffer_to_buffer(
            ring.raw_buffer(), 0,
            &readback, 0,
            ring.raw_buffer().size(),
        );
        ctx.queue().submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        ctx.device().poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);

        // Slot 0 starts at offset 0. First two words are our messages.
        assert_eq!(words[0], 0xDEAD_BEEF);
        assert_eq!(words[1], 0xCAFE_BABE);

        drop(data);
        readback.unmap();
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn flush_writes_to_slot_1_after_rotation() {
        let ctx = headless_ctx(2);
        let mut ring: UploadRing<u32> = UploadRing::new(
            &ctx, "test", 4, OverflowPolicy::Drop,
        );

        // Slot 0: push and rotate (flushes slot 0).
        ring.push(0xAAAA_AAAAu32).unwrap();
        ring.rotate_frame(&ctx).unwrap();

        // Slot 1: push and flush.
        ring.push(0xBBBB_BBBBu32).unwrap();
        ring.flush(&ctx);

        // Read back the entire buffer.
        let readback = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: ring.raw_buffer().size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        encoder.copy_buffer_to_buffer(
            ring.raw_buffer(), 0,
            &readback, 0,
            ring.raw_buffer().size(),
        );
        ctx.queue().submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        ctx.device().poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);

        // Slot 0 at offset 0: first word is 0xAAAA_AAAA.
        assert_eq!(words[0], 0xAAAA_AAAA);
        // Slot 1 at offset 4 (capacity_per_frame=4 u32s): first word is 0xBBBB_BBBB.
        assert_eq!(words[4], 0xBBBB_BBBB);

        drop(data);
        readback.unmap();
    }
}
