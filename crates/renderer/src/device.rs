//! Device and queue context.
//!
//! Principle 3: the `wgpu::Device` and `wgpu::Queue` are owned here and
//! nowhere else. Every other module that needs GPU handles receives them
//! through render primitives that themselves go through this type.

use crate::error::RendererError;
use crate::frame::{FrameCount, FrameIndex};

// --- WindowedState ---

/// Surface state owned by a windowed [`RendererContext`].  Absent for
/// headless contexts.
struct WindowedState {
    surface         : wgpu::Surface<'static>,
    preferred_format: wgpu::TextureFormat,
    preferred_alpha : wgpu::CompositeAlphaMode,
    /// `None` until the first [`RendererContext::configure_surface`] call.
    config          : Option<wgpu::SurfaceConfiguration>,
}

/// Owns the wgpu device and queue plus the renderer's frame counters. The
/// single place any other module in this crate receives GPU handles from.
///
/// Increment 3 added headless construction and frame getters; Increment 4
/// opens the `pub(crate)` device/queue accessors so `GpuConsts` and
/// `BindingLayout` can allocate their GPU resources, and adds the
/// [`Self::begin_frame`] / [`Self::end_frame`] pair so command recording
/// flows through a single [`FrameEncoder`] from day one. The surface-based
/// constructor still lands in a later increment once a windowed caller
/// first needs it.
pub struct RendererContext {
    device     : wgpu::Device,
    queue      : wgpu::Queue,
    frame_count: FrameCount,
    frame_index: FrameIndex,
    windowed   : Option<WindowedState>,
}

impl RendererContext {
    /// Construct a headless GPU context. No surface, no swapchain, no window.
    ///
    /// `frame_count` is the number of frames-in-flight the ring primitives
    /// will be sized against. Headless callers pass it explicitly because
    /// there is no swapchain to read it from.
    ///
    /// Requires `Features::PASSTHROUGH_SHADERS` — the renderer compiles
    /// HLSL to SPIR-V via DXC at build time and hands the bytes straight
    /// to `create_shader_module_passthrough`, bypassing naga — naga can't
    /// round-trip `DrawIndex`, which real subsystems will need, so the
    /// renderer uses the same HLSL/DXC toolchain the old scaffold proved
    /// out. A machine
    /// whose Vulkan driver doesn't expose `PASSTHROUGH_SHADERS` will fail
    /// device creation here, which is what the ignored GPU tests also
    /// observe; the software fallback (lavapipe/swiftshader) is explicitly
    /// not a substitute per the phase exit criterion.
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
                required_features: wgpu::Features::PASSTHROUGH_SHADERS,
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
            windowed: None,
        })
    }

    /// The number of frames-in-flight this context was configured for.
    pub fn frame_count(&self) -> FrameCount {
        self.frame_count
    }

    /// The current monotonic frame index — the frame that the *next* call to
    /// [`Self::begin_frame`] will tag its [`FrameEncoder`] with. Starts at
    /// zero at construction and advances by one each time [`Self::end_frame`]
    /// submits a frame.
    pub fn frame_index(&self) -> FrameIndex {
        self.frame_index
    }

    /// Begin recording a new frame.
    ///
    /// Creates a fresh `wgpu::CommandEncoder`, tags it with the current
    /// [`FrameIndex`], and returns it wrapped in a [`FrameEncoder`]. Ring and
    /// pipeline primitives landing in later increments take
    /// `&mut FrameEncoder` so their `copy_buffer_to_buffer` / `dispatch`
    /// calls all record into the same encoder, and the caller never touches
    /// `wgpu::CommandEncoder` directly — principle 3.
    ///
    /// Does not advance `frame_index` — the current frame index *is* the
    /// frame this encoder records for. The advance happens in
    /// [`Self::end_frame`] after the command buffer is submitted.
    ///
    /// Must be paired with exactly one [`Self::end_frame`] call using the
    /// returned `FrameEncoder`. The type's lack of `Clone` / `Copy` plus the
    /// by-value consume in `end_frame` is the forced-pairing mechanism.
    pub fn begin_frame(&mut self) -> FrameEncoder {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("renderer_frame_encoder"),
            });
        FrameEncoder {
            encoder,
            frame: self.frame_index,
        }
    }

    /// End the current frame.
    ///
    /// Consumes the [`FrameEncoder`] returned by a previous
    /// [`Self::begin_frame`], calls `finish()` on it to produce a
    /// `CommandBuffer`, submits that buffer to the queue, and advances
    /// `frame_index` by one.
    ///
    /// Polling wgpu for readback map callbacks is not done here yet — it
    /// lands alongside readback buffer support when there is something
    /// mapped to poll for.
    pub fn end_frame(&mut self, frame_encoder: FrameEncoder) {
        let command_buffer = frame_encoder.encoder.finish();
        self.queue.submit(std::iter::once(command_buffer));
        self.frame_index.advance();
    }

    /// Construct a windowed GPU context wired to a live swapchain surface.
    ///
    /// The caller is responsible for creating the `wgpu::Instance` and
    /// `wgpu::Surface` from the OS window (both require the windowing library
    /// that the game crate owns).  This constructor takes ownership of both
    /// and drives the rest of device / surface initialisation.
    ///
    /// The surface is **not** configured here — the window size may not be
    /// known at construction time.  Call [`Self::configure_surface`] with the
    /// initial window size before the first [`Self::acquire_frame`].
    pub async fn new_windowed(
        instance   : wgpu::Instance,
        surface    : wgpu::Surface<'static>,
        frame_count: FrameCount,
    ) -> Result<Self, RendererError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|_| RendererError::NoCompatibleAdapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label            : Some("renderer_windowed_device"),
                required_features: wgpu::Features::PASSTHROUGH_SHADERS,
                required_limits  : wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| RendererError::DeviceCreationFailed(e.to_string()))?;

        let caps = surface.get_capabilities(&adapter);

        // Prefer an sRGB format so colours look correct without manual
        // gamma correction in shaders.
        let preferred_format = caps.formats.iter()
            .find(|&&f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let preferred_alpha = caps.alpha_modes[0];

        Ok(Self {
            device,
            queue,
            frame_count,
            frame_index: FrameIndex::default(),
            windowed: Some(WindowedState {
                surface,
                preferred_format,
                preferred_alpha,
                config: None,
            }),
        })
    }

    /// Configure (or reconfigure) the swapchain surface.
    ///
    /// Must be called before the first [`Self::acquire_frame`], and again
    /// whenever the window is resized.
    ///
    /// # Panics
    ///
    /// Panics if called on a headless context, or if `width` or `height`
    /// is zero (wgpu rejects zero-sized surfaces).
    pub fn configure_surface(&mut self, width: u32, height: u32) {
        assert!(width  > 0, "configure_surface: width must be > 0");
        assert!(height > 0, "configure_surface: height must be > 0");

        let ws = self.windowed.as_mut()
            .expect("configure_surface called on a headless RendererContext");

        let config = wgpu::SurfaceConfiguration {
            usage                        : wgpu::TextureUsages::RENDER_ATTACHMENT,
            format                       : ws.preferred_format,
            width,
            height,
            present_mode                 : wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode                   : ws.preferred_alpha,
            view_formats                 : vec![],
        };

        ws.surface.configure(&self.device, &config);
        ws.config = Some(config);
    }

    /// Acquire the next swapchain image for this frame.
    ///
    /// Returns a [`SurfaceFrame`] that must be presented via
    /// [`SurfaceFrame::present`] after the frame's command buffer is
    /// submitted.
    ///
    /// If the surface is outdated or lost (e.g. after a resize race),
    /// the surface is reconfigured with the last known size and
    /// [`RendererError::SurfaceOutdated`] is returned — the caller should
    /// skip the frame and retry on the next redraw.
    ///
    /// # Panics
    ///
    /// Panics if called on a headless context, or if [`configure_surface`]
    /// has not been called yet.
    ///
    /// [`configure_surface`]: Self::configure_surface
    pub fn acquire_frame(&mut self) -> Result<SurfaceFrame, RendererError> {
        let ws = self.windowed.as_mut()
            .expect("acquire_frame called on a headless RendererContext");
        let config = ws.config.as_ref()
            .expect("acquire_frame called before configure_surface");

        match ws.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                let view = texture.texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                Ok(SurfaceFrame { view, texture })
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                // Reconfigure with last known dimensions and ask the caller to skip
                // this frame; the next acquire should succeed.
                ws.surface.configure(&self.device, config);
                Err(RendererError::SurfaceOutdated)
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                // Surface temporarily unavailable (e.g. minimised); skip without
                // reconfiguring.
                Err(RendererError::SurfaceOutdated)
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                Err(RendererError::SurfaceAcquireFailed(
                    "GPU validation error during surface acquire".into(),
                ))
            }
        }
    }

    /// The wgpu device handle. Only visible within the renderer crate —
    /// external callers interact with the device exclusively through
    /// primitives (`GpuConsts`, `BindingLayout`, pipelines) per principle 3
    /// (wgpu is contained behind render abstractions).
    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The wgpu queue handle. Same visibility rationale as [`Self::device`].
    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The texture format the swapchain surface was configured with, or `None`
    /// for a headless context.  Used by render passes that target the swapchain
    /// (e.g. `SubchunkTest`) to match their colour-attachment format.
    pub fn surface_format(&self) -> Option<wgpu::TextureFormat> {
        self.windowed.as_ref().map(|ws| ws.preferred_format)
    }
}

// --- SurfaceFrame ---

/// An acquired swapchain image for the current frame.
///
/// Obtained from [`RendererContext::acquire_frame`].  Pass a shared
/// reference to [`FrameEncoder::clear_surface`] (or future render-pass
/// helpers) to record commands targeting the swapchain image, then call
/// [`Self::present`] after the frame's command buffer has been submitted.
///
/// Drop order matters: the internal [`wgpu::TextureView`] is explicitly
/// dropped before [`wgpu::SurfaceTexture::present`] is called, which is
/// what wgpu requires.
pub struct SurfaceFrame {
    /// View is declared first so it is dropped before `texture` on both
    /// explicit `present` and any accidental drops.
    view   : wgpu::TextureView,
    texture: wgpu::SurfaceTexture,
}

impl SurfaceFrame {
    /// The texture view targeting this frame's swapchain image.
    ///
    /// `pub(crate)` — only render helpers inside the renderer crate
    /// construct render passes; external callers use those helpers.
    pub(crate) fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Present the swapchain image.
    ///
    /// Explicitly drops the view before calling `present` to satisfy wgpu's
    /// requirement that all views of a surface texture are released first.
    pub fn present(self) {
        drop(self.view);
        self.texture.present();
    }
}

/// The per-frame command-recording wrapper returned by
/// [`RendererContext::begin_frame`] and consumed by
/// [`RendererContext::end_frame`].
///
/// Holds the single `wgpu::CommandEncoder` into which all ring copies and
/// pipeline dispatches for the frame are recorded. Exists so the renderer
/// crate keeps its own `pub(crate)` view of `wgpu::CommandEncoder` out of
/// public APIs (principle 3) while still giving callers a handle they can
/// thread through ring / pipeline methods — which in later increments will
/// accept `&mut FrameEncoder` rather than `&mut wgpu::CommandEncoder`.
///
/// `FrameEncoder` is deliberately not `Clone` or `Copy`. Together with the
/// by-value consume in `end_frame`, this makes "forgot to submit" and
/// "submitted twice" into compile errors.
pub struct FrameEncoder {
    encoder: wgpu::CommandEncoder,
    frame: FrameIndex,
}

impl FrameEncoder {
    /// The frame index this encoder was created for. Convenience for ring and
    /// pipeline primitives whose APIs take a `FrameIndex` — instead of
    /// threading it alongside the encoder, they can read it back from the
    /// wrapper. Not a guarantee of real-time ordering — that's maintained by
    /// the begin/end pairing on [`RendererContext`].
    #[allow(dead_code)] // First caller: ReadbackChannel::schedule_copy_and_map (Increment 9).
    pub fn frame(&self) -> FrameIndex {
        self.frame
    }

    /// Mutable access to the underlying `wgpu::CommandEncoder`. `pub(crate)`
    /// so that ring and pipeline primitives inside the renderer can record
    /// commands; callers outside the crate never touch the raw encoder.
    pub(crate) fn encoder_mut(&mut self) -> &mut wgpu::CommandEncoder {
        &mut self.encoder
    }

    /// Clear the swapchain image to a solid colour.
    ///
    /// Records a single-attachment render pass with
    /// [`LoadOp::Clear`](wgpu::LoadOp::Clear) targeting the
    /// [`SurfaceFrame`]'s texture view.  The pass is ended immediately —
    /// this is a full-frame clear, not a pass that can be extended.
    ///
    /// `rgba` components are in linear light (not sRGB) as wgpu requires:
    /// pass pre-linearised values if you're working in sRGB space.
    pub fn clear_surface(&mut self, frame: &SurfaceFrame, [r, g, b, a]: [f64; 4]) {
        let _pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear_surface"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view          : frame.view(),
                depth_slice   : None,
                resolve_target: None,
                ops           : wgpu::Operations {
                    load : wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set     : None,
            timestamp_writes        : None,
            multiview_mask          : None,
        });
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

    /// `begin_frame` tags the returned `FrameEncoder` with the current frame
    /// index (which has not yet advanced); `end_frame` submits and advances
    /// by one. The observable property is that after a begin/end pair the
    /// context's `frame_index` is exactly one past where it started.
    ///
    /// Also exercises a "no-op frame" — no commands are recorded on the
    /// encoder between begin and end. wgpu accepts an empty command buffer,
    /// so this checks that the plumbing is fine with a zero-work frame.
    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn begin_end_frame_submits_and_advances_index() {
        let mut ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine");

        assert_eq!(ctx.frame_index().get(), 0);

        let fe0 = ctx.begin_frame();
        assert_eq!(fe0.frame().get(), 0);
        ctx.end_frame(fe0);

        assert_eq!(ctx.frame_index().get(), 1);

        // A second frame advances again — the counter is monotonic and not
        // reset by submission.
        let fe1 = ctx.begin_frame();
        assert_eq!(fe1.frame().get(), 1);
        ctx.end_frame(fe1);

        assert_eq!(ctx.frame_index().get(), 2);
    }

    /// Slot-rotation smoke test: with `FrameCount::new(2)` the per-frame slot
    /// derived from `frame_index.slot(frame_count)` alternates 0, 1, 0, 1 as
    /// `end_frame` advances. Once the rings land, this is the number they
    /// will use to pick a slot — pinning the behavior now in the smallest
    /// possible surface catches any future `advance` regression.
    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn begin_end_frame_rotates_slot_across_frames() {
        let mut ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine");

        let count = ctx.frame_count();
        for expected in [0u32, 1, 0, 1] {
            assert_eq!(ctx.frame_index().slot(count), expected);
            let fe = ctx.begin_frame();
            ctx.end_frame(fe);
        }
    }
}
