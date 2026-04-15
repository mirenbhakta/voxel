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
use renderer::{SUBCHUNK_DEPTH_FORMAT, SubchunkTest, TestCamera, nodes, sphere_occupancy};
#[cfg(not(feature = "validation"))]
use renderer::graph::{BufferPool, RenderGraph, TextureDesc, TexturePool};
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
    subchunk_test: Option<Arc<SubchunkTest>>,
    buf_pool:      BufferPool,
    tex_pool:      TexturePool,
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
        let w = size.width.max(1);
        let h = size.height.max(1);
        ctx.configure_surface(w, h);

        let occ  = sphere_occupancy();
        let test = Arc::new(SubchunkTest::new(&ctx, &occ));
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
                let w = size.width.max(1);
                let h = size.height.max(1);
                if let Some(ctx) = &mut self.ctx {
                    ctx.configure_surface(w, h);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                let (Some(ctx), Some(window), Some(test)) = (
                    self.ctx.as_mut(),
                    self.window.as_ref(),
                    self.subchunk_test.as_ref(),
                ) else {
                    return;
                };

                let size   = window.inner_size();
                let w      = size.width.max(1);
                let h      = size.height.max(1);
                let aspect = w as f32 / h as f32;

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

                test.write_camera(ctx, &camera);

                let surface_frame = match ctx.acquire_frame() {
                    Ok(f) => f,
                    Err(RendererError::SurfaceOutdated) => {
                        window.request_redraw();
                        return;
                    }
                    Err(e) => {
                        eprintln!("fatal frame error: {e}");
                        event_loop.exit();
                        return;
                    }
                };

                let mut graph = RenderGraph::new();
                let color = graph.import_texture();
                let depth = graph.create_texture(
                    "subchunk_depth",
                    TextureDesc::new_2d(
                        w, h,
                        SUBCHUNK_DEPTH_FORMAT,
                        wgpu::TextureUsages::RENDER_ATTACHMENT,
                    ),
                );
                let (color_v, _depth_v) = nodes::subchunk_test(&mut graph, test, color, depth);
                graph.present(color_v);

                let mut compiled = graph.compile().expect("render graph compile");
                compiled.bind_texture(color, surface_frame.texture_clone());
                compiled.allocate_transients(
                    &mut self.buf_pool, &mut self.tex_pool, ctx.device(),
                );

                let mut fe = ctx.begin_frame();
                let frame  = ctx.frame_index();
                let pending = compiled.execute(&mut fe, frame);
                ctx.end_frame(fe);

                surface_frame.present();

                pending.release(&mut self.buf_pool, &mut self.tex_pool);

                window.request_redraw();
            }

            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(feature = "validation")]
mod validation;
