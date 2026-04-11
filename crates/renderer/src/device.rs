//! Device and queue context.
//!
//! Principle 3: the `wgpu::Device` and `wgpu::Queue` are owned here and
//! nowhere else. Every other module that needs GPU handles receives them
//! through ring or pipeline primitives that themselves go through this type.
//! See `.local/renderer_plan.md` §3.2.

use crate::error::RendererError;
use crate::frame::{FrameCount, FrameIndex};

/// Owns the wgpu device and queue plus the renderer's frame counters. The
/// single place any other module in this crate receives GPU handles from.
///
/// Increment 3 exposes only headless construction and frame getters; the
/// surface-based constructor, `begin_frame`/`end_frame`, and the `pub(crate)`
/// device/queue accessors land as later increments need them.
pub struct RendererContext {
    // Held but unused in Increment 3. Accessors land in Increment 4 when
    // `GpuConsts` first needs to allocate a uniform buffer.
    #[allow(dead_code)]
    device: wgpu::Device,
    #[allow(dead_code)]
    queue: wgpu::Queue,
    frame_count: FrameCount,
    frame_index: FrameIndex,
}

impl RendererContext {
    /// Construct a headless GPU context. No surface, no swapchain, no window.
    ///
    /// `frame_count` is the number of frames-in-flight the ring primitives
    /// will be sized against. Headless callers pass it explicitly because
    /// there is no swapchain to read it from.
    ///
    /// Requests `Features::empty()` — this increment only verifies that a
    /// device opens. Later increments widen the feature set as they need
    /// specific capabilities (`PASSTHROUGH_SHADERS`, etc.).
    pub async fn new_headless(frame_count: FrameCount) -> Result<Self, RendererError> {
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle(),
        );

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|_| RendererError::NoCompatibleAdapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("renderer_headless_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| RendererError::DeviceCreationFailed(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            frame_count,
            frame_index: FrameIndex::default(),
        })
    }

    /// The number of frames-in-flight this context was configured for.
    pub fn frame_count(&self) -> FrameCount {
        self.frame_count
    }

    /// The current monotonic frame index. Starts at zero at construction and
    /// advances as `begin_frame`/`end_frame` land in later increments.
    pub fn frame_index(&self) -> FrameIndex {
        self.frame_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exit criterion for Increment 3: a wgpu adapter opens on real hardware
    /// through this crate's public API.
    ///
    /// Gated with `#[ignore]` because it requires a working Vulkan stack;
    /// non-GPU CI runs `cargo test --workspace` and does not exercise this
    /// test. Run locally with `cargo test -p renderer -- --ignored`.
    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn headless_context_constructs_with_frame_count_2() {
        let ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine");

        assert_eq!(ctx.frame_count().get(), 2);
        assert_eq!(ctx.frame_index().get(), 0);
    }
}
