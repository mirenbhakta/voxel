//! Game binary — windowed renderer entry point.
//!
//! Uses winit 0.30's `ApplicationHandler` pattern for cross-platform event
//! loop management.  The renderer is initialised lazily on the first
//! `resumed` event so that the surface is always created from a live window.
//!
//! # Validation feature
//!
//! `cargo build --features validation` (or `cargo run --features validation`)
//! runs a GPU validation suite against real hardware and exits, rather than
//! opening a window.  Useful for bug reports and CI on GPU-capable machines.

// Windowed-path imports are only needed when the validation feature is off.
// Gating them suppresses dead_code / unused_imports warnings in validation
// builds.
#[cfg(not(feature = "validation"))]
use std::sync::Arc;

#[cfg(not(feature = "validation"))]
use renderer::{FrameCount, RendererContext, RendererError};
#[cfg(not(feature = "validation"))]
use renderer::{SubchunkTest, TestCamera, sphere_occupancy};
#[cfg(not(feature = "validation"))]
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

// ---------------------------------------------------------------------------

fn main() {
    #[cfg(feature = "validation")]
    validation::run();

    #[cfg(not(feature = "validation"))]
    {
        let event_loop = EventLoop::new().expect("failed to create event loop");
        let mut app = App::default();
        event_loop.run_app(&mut app).expect("event loop error");
    }
}

// ---------------------------------------------------------------------------

/// Application state.  Fields are populated on the first `resumed` event
/// and remain `Some` for the rest of the session.
#[cfg(not(feature = "validation"))]
#[derive(Default)]
struct App {
    window:        Option<Arc<Window>>,
    ctx:           Option<RendererContext>,
    subchunk_test: Option<SubchunkTest>,
}

#[cfg(not(feature = "validation"))]
impl ApplicationHandler for App {
    /// Called when the OS grants the application an active event loop.
    ///
    /// On desktop this fires once at startup.  On mobile / web it can fire
    /// multiple times (suspend/resume cycles), so the guard at the top
    /// prevents re-initialisation.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("voxel"))
                .expect("failed to create window"),
        );

        // Vulkan instance — no display handle needed for Vulkan on Linux.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("failed to create wgpu surface");

        let mut ctx = pollster::block_on(RendererContext::new_windowed(
            instance,
            surface,
            FrameCount::new(2).unwrap(),
        ))
        .expect("failed to create windowed renderer context");

        let size = window.inner_size();
        ctx.configure_surface(size.width.max(1), size.height.max(1));

        let occ  = sphere_occupancy();
        let test = SubchunkTest::new(&ctx, &occ);
        self.subchunk_test = Some(test);

        self.ctx = Some(ctx);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            // Reconfigure the swapchain whenever the window is resized.
            // `max(1)` guards against the zero-size case on minimise.
            WindowEvent::Resized(size) => {
                if let Some(ctx) = &mut self.ctx {
                    ctx.configure_surface(size.width.max(1), size.height.max(1));
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                let (Some(ctx), Some(window), Some(test)) =
                    (&mut self.ctx, &self.window, &self.subchunk_test) else { return; };

                let size = window.inner_size();
                let aspect = size.width as f32 / size.height.max(1) as f32;

                let camera = TestCamera {
                    pos:     [4.0, 4.0, -5.0],
                    fov_y:   std::f32::consts::FRAC_PI_3,
                    forward: [0.0, 0.0,  1.0],
                    aspect,
                    right:   [1.0, 0.0,  0.0],
                    _pad0:   0.0,
                    up:      [0.0, 1.0,  0.0],
                    _pad1:   0.0,
                };

                match ctx.acquire_frame() {
                    Ok(frame) => {
                        let mut fe = ctx.begin_frame();
                        test.draw(ctx, &mut fe, &frame, &camera);
                        ctx.end_frame(fe);
                        frame.present();
                    }
                    Err(RendererError::SurfaceOutdated) => {}
                    Err(e) => {
                        eprintln!("fatal frame error: {e}");
                        event_loop.exit();
                    }
                }

                window.request_redraw();
            }

            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(feature = "validation")]
mod validation;
