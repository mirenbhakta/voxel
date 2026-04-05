//! GPU timestamp query support for per-frame timing measurement.
//!
//! Wraps a wgpu query set with async readback to measure GPU execution
//! time of render passes without stalling the pipeline. Results arrive
//! with one frame of latency.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device,
    QuerySet, QuerySetDescriptor, QueryType, Queue,
    RenderPassTimestampWrites,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Voxel render pass begin timestamp slot.
const RENDER_BEGIN: u32 = 0;

/// Voxel render pass end timestamp slot.
const RENDER_END: u32 = 1;

/// Total timestamp query slots.
const QUERY_COUNT: u32 = 2;

/// Bytes needed for resolved timestamps (one u64 per slot).
const RESOLVE_BYTES: u64 = QUERY_COUNT as u64 * 8;

/// EMA smoothing factor for GPU timing values.
const SMOOTHING: f32 = 0.05;

// ---------------------------------------------------------------------------
// TimestampQueries
// ---------------------------------------------------------------------------

/// GPU timestamp query manager.
///
/// Measures render pass execution time using hardware timestamp queries.
/// Results are read back asynchronously with one frame of latency to
/// avoid pipeline stalls.
///
/// Per-frame usage:
///
/// 1. [`begin_frame`](Self::begin_frame) -- poll for previous results.
/// 2. [`render_pass_timestamps`](Self::render_pass_timestamps) -- attach
///    to the voxel render pass descriptor.
/// 3. [`resolve`](Self::resolve) -- resolve queries after all passes.
/// 4. [`request_readback`](Self::request_readback) -- initiate async map
///    after queue submission.
pub struct TimestampQueries {
    /// The hardware query set.
    query_set        : QuerySet,
    /// Intermediate buffer for resolve_query_set output.
    resolve_buf      : Buffer,
    /// CPU-readable staging buffer for async readback.
    staging_buf      : Buffer,
    /// Nanoseconds per timestamp tick.
    period           : f32,
    /// Set by the map callback when staging data is available.
    readback_ready   : Arc<AtomicBool>,
    /// Whether a map request is currently in flight.
    readback_pending : bool,
    /// Whether resolve was recorded this frame.
    resolved         : bool,
    /// Smoothed GPU render pass time in milliseconds.
    render_ms        : f32,
}

// --- TimestampQueries ---

impl TimestampQueries {
    /// Create a new timestamp query manager.
    ///
    /// The device must have been created with `Features::TIMESTAMP_QUERY`.
    pub fn new(device: &Device, queue: &Queue) -> Self {
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label : Some("timestamps"),
            ty    : QueryType::Timestamp,
            count : QUERY_COUNT,
        });

        let resolve_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("ts_resolve"),
            size               : RESOLVE_BYTES,
            usage              : BufferUsages::QUERY_RESOLVE
                               | BufferUsages::COPY_SRC,
            mapped_at_creation : false,
        });

        let staging_buf = device.create_buffer(&BufferDescriptor {
            label              : Some("ts_staging"),
            size               : RESOLVE_BYTES,
            usage              : BufferUsages::COPY_DST
                               | BufferUsages::MAP_READ,
            mapped_at_creation : false,
        });

        let period = queue.get_timestamp_period();

        TimestampQueries {
            query_set        : query_set,
            resolve_buf      : resolve_buf,
            staging_buf      : staging_buf,
            period           : period,
            readback_ready   : Arc::new(AtomicBool::new(false)),
            readback_pending : false,
            resolved         : false,
            render_ms        : 0.0,
        }
    }

    /// Poll for completed timestamp readback from the previous frame.
    ///
    /// Call at the start of each frame before recording new commands.
    /// Updates the smoothed render time if results are available.
    pub fn begin_frame(&mut self, device: &Device) {
        if !self.readback_pending {
            return;
        }

        // Non-blocking poll to process map callbacks.
        let _ = device.poll(wgpu::PollType::Poll);

        if self.readback_ready.load(Ordering::Acquire) {
            self.consume_readback();
        }
    }

    /// Timestamp writes configuration for the voxel render pass.
    ///
    /// Attach the returned value to the render pass descriptor's
    /// `timestamp_writes` field.
    pub fn render_pass_timestamps(
        &self,
    ) -> RenderPassTimestampWrites<'_>
    {
        RenderPassTimestampWrites {
            query_set                     : &self.query_set,
            beginning_of_pass_write_index : Some(RENDER_BEGIN),
            end_of_pass_write_index       : Some(RENDER_END),
        }
    }

    /// Resolve timestamp queries and copy to the staging buffer.
    ///
    /// If a previous readback completed since [`begin_frame`], it is
    /// consumed first to ensure the staging buffer is unmapped. If the
    /// previous readback is still in flight, the resolve is skipped
    /// for this frame.
    ///
    /// Call after all timestamped passes, before `encoder.finish()`.
    pub fn resolve(&mut self, encoder: &mut CommandEncoder) {
        self.resolved = false;

        // The staging buffer must not be mapped when we copy to it.
        if self.readback_pending {
            if self.readback_ready.load(Ordering::Acquire) {
                self.consume_readback();
            }
            else {
                // Map still in flight. Skip this frame's resolve.
                return;
            }
        }

        encoder.resolve_query_set(
            &self.query_set,
            0..QUERY_COUNT,
            &self.resolve_buf,
            0,
        );

        encoder.copy_buffer_to_buffer(
            &self.resolve_buf, 0,
            &self.staging_buf, 0,
            RESOLVE_BYTES,
        );

        self.resolved = true;
    }

    /// Request async readback of resolved timestamps.
    ///
    /// Call after `queue.submit()`. Results become available in the
    /// next frame's [`begin_frame`](Self::begin_frame).
    pub fn request_readback(&mut self) {
        if !self.resolved {
            return;
        }

        self.resolved         = false;
        self.readback_pending = true;
        self.readback_ready.store(false, Ordering::Release);

        let ready = self.readback_ready.clone();

        self.staging_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                result.unwrap();
                ready.store(true, Ordering::Release);
            });
    }

    /// Smoothed GPU render pass time in milliseconds.
    pub fn render_ms(&self) -> f32 {
        self.render_ms
    }

    /// Read mapped staging data and update the smoothed render time.
    fn consume_readback(&mut self) {
        let slice = self.staging_buf.slice(..);
        let data  = slice.get_mapped_range();

        let begin = u64::from_le_bytes(
            data[0..8].try_into().unwrap(),
        );

        let end = u64::from_le_bytes(
            data[8..16].try_into().unwrap(),
        );

        drop(data);
        self.staging_buf.unmap();

        self.readback_pending = false;
        self.readback_ready.store(false, Ordering::Release);

        // Convert ticks to milliseconds.
        let elapsed_ms = (end.wrapping_sub(begin)) as f64
                       * self.period as f64
                       / 1_000_000.0;

        // Exponential moving average.
        if self.render_ms == 0.0 {
            self.render_ms = elapsed_ms as f32;
        }
        else {
            self.render_ms +=
                (elapsed_ms as f32 - self.render_ms) * SMOOTHING;
        }
    }
}
