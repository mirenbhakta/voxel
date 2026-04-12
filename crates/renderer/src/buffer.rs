//! [`StagedBuffer<T>`] — typed GPU buffer with CPU-side staging.
//!
//! The fundamental building block for CPU-write, GPU-read data flow in render
//! graph passes. Push typed values into the CPU staging area, then [`flush`]
//! to transfer them to the GPU buffer via `queue.write_buffer`.
//!
//! This type handles only the buffer and staging lifecycle. Scheduling
//! (when to flush, buffer reuse, pool management) is the render graph's
//! concern.
//!
//! [`flush`]: StagedBuffer::flush

use crate::device::RendererContext;

/// A typed GPU buffer with CPU-side staging.
///
/// `T` must be [`bytemuck::Pod`] — the standard bound for GPU-visible POD in
/// the Rust/wgpu ecosystem. `bytemuck::Pod` implies `Copy`, `'static`, and
/// `#[repr(C)]` (or equivalent transparent/packed repr), which is exactly
/// what `queue.write_buffer` and HLSL `StructuredBuffer<T>` need.
///
/// ## Lifecycle
///
/// 1. [`push`](Self::push) / [`push_many`](Self::push_many) — stage values
///    on the CPU side.
/// 2. [`flush`](Self::flush) — transfer staged data to the GPU buffer and
///    clear the staging area.
/// 3. Downstream passes bind [`buffer()`](Self::buffer) in their bind groups
///    and read the data on the GPU.
///
/// [`clear`](Self::clear) discards staged data without flushing.
pub struct StagedBuffer<T: bytemuck::Pod> {
    buffer: wgpu::Buffer,
    staging: Vec<T>,
    capacity: u32,
    label: String,
}

impl<T: bytemuck::Pod> StagedBuffer<T> {
    /// Create a new staged buffer.
    ///
    /// `capacity` is the maximum number of `T` values the buffer can hold.
    /// The GPU buffer is allocated as `capacity * size_of::<T>()` bytes with
    /// `STORAGE | COPY_DST` usage.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0` or if `size_of::<T>() == 0` — zero-sized
    /// types have no meaningful GPU representation.
    pub fn new(ctx: &RendererContext, label: &str, capacity: u32) -> Self {
        let t_size = std::mem::size_of::<T>();
        assert!(
            t_size > 0,
            "StagedBuffer::new: T must not be a zero-sized type \
             (size_of::<T>() == 0 has no GPU representation)",
        );
        assert!(
            capacity > 0,
            "StagedBuffer::new: capacity must be > 0",
        );

        let total_bytes = capacity as u64 * t_size as u64;

        let buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            staging: Vec::with_capacity(capacity as usize),
            capacity,
            label: label.to_string(),
        }
    }

    // --- Inspection ---

    /// Human-readable label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Maximum number of `T` values the buffer can hold.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Number of values currently staged (not yet flushed).
    pub fn len(&self) -> u32 {
        self.staging.len() as u32
    }

    /// Whether the staging area is empty.
    pub fn is_empty(&self) -> bool {
        self.staging.is_empty()
    }

    // --- Push ---

    /// Stage a single value for upload.
    ///
    /// # Panics
    ///
    /// Panics if the staging area is at capacity.
    pub fn push(&mut self, value: T) {
        assert!(
            (self.staging.len() as u32) < self.capacity,
            "StagedBuffer `{}`: push at capacity ({}) — flush before pushing more",
            self.label,
            self.capacity,
        );
        self.staging.push(value);
    }

    /// Stage multiple values for upload.
    ///
    /// # Panics
    ///
    /// Panics if `values` is empty or if staging would exceed capacity.
    pub fn push_many(&mut self, values: &[T]) {
        assert!(
            !values.is_empty(),
            "StagedBuffer::push_many: values must not be empty",
        );
        assert!(
            (self.staging.len() + values.len()) as u32 <= self.capacity,
            "StagedBuffer `{}`: push_many of {} at fill {} would exceed capacity {}",
            self.label,
            values.len(),
            self.staging.len(),
            self.capacity,
        );
        self.staging.extend_from_slice(values);
    }

    // --- Flush / Clear ---

    /// Flush the staging area to the GPU buffer via `queue.write_buffer`.
    ///
    /// Writes `staging.len() * size_of::<T>()` bytes to offset 0 of the
    /// buffer, then clears the staging area. A no-op if nothing was staged.
    pub fn flush(&mut self, ctx: &RendererContext) {
        if self.staging.is_empty() {
            return;
        }
        ctx.queue().write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&self.staging),
        );
        self.staging.clear();
    }

    /// Discard all staged data without flushing to the GPU.
    pub fn clear(&mut self) {
        self.staging.clear();
    }

    // --- Buffer access ---

    /// The underlying `wgpu::Buffer`.
    ///
    /// Use this for bind-group construction and render-graph import.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameCount;

    fn headless_ctx() -> RendererContext {
        pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine")
    }

    // --- Construction ---

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn new_creates_buffer_of_correct_size() {
        let ctx = headless_ctx();
        let buf: StagedBuffer<[u32; 4]> = StagedBuffer::new(&ctx, "test_buf", 8);

        // 8 values * 16 bytes = 128
        assert_eq!(buf.buffer().size(), 128);
        assert_eq!(buf.capacity(), 8);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert_eq!(buf.label(), "test_buf");
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    #[should_panic(expected = "zero-sized type")]
    fn new_panics_on_zst() {
        let ctx = headless_ctx();
        let _: StagedBuffer<()> = StagedBuffer::new(&ctx, "zst", 4);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    #[should_panic(expected = "capacity must be > 0")]
    fn new_panics_on_zero_capacity() {
        let ctx = headless_ctx();
        let _: StagedBuffer<u32> = StagedBuffer::new(&ctx, "test", 0);
    }

    // --- Push ---

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn push_increments_len() {
        let ctx = headless_ctx();
        let mut buf: StagedBuffer<u32> = StagedBuffer::new(&ctx, "test", 4);

        buf.push(100);
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());

        buf.push(200);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn push_many_stages_all() {
        let ctx = headless_ctx();
        let mut buf: StagedBuffer<u32> = StagedBuffer::new(&ctx, "test", 8);

        buf.push_many(&[10, 20, 30]);
        assert_eq!(buf.len(), 3);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    #[should_panic(expected = "push at capacity")]
    fn push_beyond_capacity_panics() {
        let ctx = headless_ctx();
        let mut buf: StagedBuffer<u32> = StagedBuffer::new(&ctx, "test", 4);

        buf.push_many(&[1, 2, 3, 4]);
        buf.push(5);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    #[should_panic(expected = "must not be empty")]
    fn push_many_panics_on_empty_slice() {
        let ctx = headless_ctx();
        let mut buf: StagedBuffer<u32> = StagedBuffer::new(&ctx, "test", 4);
        buf.push_many(&[]);
    }

    // --- Flush / Clear ---

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn flush_clears_staging() {
        let ctx = headless_ctx();
        let mut buf: StagedBuffer<u32> = StagedBuffer::new(&ctx, "test", 4);

        buf.push_many(&[1, 2]);
        assert_eq!(buf.len(), 2);

        buf.flush(&ctx);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn clear_discards_without_flushing() {
        let ctx = headless_ctx();
        let mut buf: StagedBuffer<u32> = StagedBuffer::new(&ctx, "test", 4);

        buf.push_many(&[1, 2, 3]);
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn flush_writes_data_to_buffer() {
        let ctx = headless_ctx();
        let mut buf: StagedBuffer<u32> = StagedBuffer::new(&ctx, "test", 4);

        buf.push(0xDEAD_BEEFu32);
        buf.push(0xCAFE_BABEu32);
        buf.flush(&ctx);

        let readback = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: buf.buffer().size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        encoder.copy_buffer_to_buffer(
            buf.buffer(), 0,
            &readback, 0,
            buf.buffer().size(),
        );
        ctx.queue().submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        ctx.device().poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);

        assert_eq!(words[0], 0xDEAD_BEEF);
        assert_eq!(words[1], 0xCAFE_BABE);

        drop(data);
        readback.unmap();
    }
}
