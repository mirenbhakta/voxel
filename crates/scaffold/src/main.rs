//! Visual testing scaffold for the voxel library.
//!
//! Opens a window with a wgpu device and render loop. Provides a minimal
//! starting point for exercising library code with real GPU output.

use std::sync::Arc;

use pollster::FutureExt as _;
use wgpu::{
    Color, CommandEncoderDescriptor, CurrentSurfaceTexture, Device,
    DeviceDescriptor, Instance, InstanceDescriptor, LoadOp, Operations, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RequestAdapterOptions,
    StoreOp, Surface, SurfaceConfiguration, TextureViewDescriptor,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

/// Top-level application state for the scaffold window.
struct App {
    /// The GPU state, initialized on resume.
    gpu : Option<Gpu>,
}

/// Initialized GPU resources tied to a window surface.
struct Gpu {
    /// The window handle, kept alive for the surface.
    window  : Arc<Window>,
    /// The wgpu rendering surface.
    surface : Surface<'static>,
    /// The surface configuration (format, size, present mode).
    config  : SurfaceConfiguration,
    /// The logical device.
    device  : Device,
    /// The command submission queue.
    queue   : Queue,
}

// --- App ---

impl App {
    /// Create a new application with no GPU state.
    fn new() -> Self {
        Self { gpu: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Build the window.
        let attrs = WindowAttributes::default()
            .with_title("voxel scaffold")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));

        let window = Arc::new(
            event_loop.create_window(attrs)
                .expect("failed to create window"),
        );

        // Create the wgpu instance and surface.
        let instance = Instance::new(
            InstanceDescriptor::new_without_display_handle(),
        );

        let surface = instance.create_surface(window.clone())
            .expect("failed to create surface");

        // Request an adapter compatible with the surface.
        let adapter = instance.request_adapter(&RequestAdapterOptions {
            compatible_surface : Some(&surface),
            ..Default::default()
        })
            .block_on()
            .expect("no compatible adapter found");

        // Request the device and queue.
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default())
            .block_on()
            .expect("failed to create device");

        // Configure the surface at the window's current size.
        let size   = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by adapter");

        surface.configure(&device, &config);

        self.gpu = Some(Gpu { window, surface, config, device, queue });

        // Start the render loop.
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn window_event(
        &mut self,
        event_loop : &ActiveEventLoop,
        _id        : WindowId,
        event      : WindowEvent,
    ) {
        let Some(gpu) = self.gpu.as_mut()
        else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                gpu.config.width  = size.width.max(1);
                gpu.config.height = size.height.max(1);
                gpu.surface.configure(&gpu.device, &gpu.config);
                gpu.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                // Acquire the next frame.
                let frame = match gpu.surface.get_current_texture() {
                    CurrentSurfaceTexture::Success(t)
                    | CurrentSurfaceTexture::Suboptimal(t) => t,
                    CurrentSurfaceTexture::Outdated
                    | CurrentSurfaceTexture::Lost => {
                        gpu.surface.configure(&gpu.device, &gpu.config);
                        gpu.window.request_redraw();
                        return;
                    }
                    CurrentSurfaceTexture::Timeout
                    | CurrentSurfaceTexture::Occluded => return,
                    other => panic!("surface error: {other:?}"),
                };

                let view = frame.texture
                    .create_view(&TextureViewDescriptor::default());

                let mut encoder = gpu.device
                    .create_command_encoder(&CommandEncoderDescriptor::default());

                // Clear to dark blue-gray.
                {
                    let _pass = encoder.begin_render_pass(&RenderPassDescriptor {
                        label                    : Some("clear"),
                        color_attachments        : &[Some(RenderPassColorAttachment {
                            view          : &view,
                            depth_slice   : None,
                            resolve_target: None,
                            ops           : Operations {
                                load  : LoadOp::Clear(Color {
                                    r : 0.1,
                                    g : 0.1,
                                    b : 0.15,
                                    a : 1.0,
                                }),
                                store : StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment : None,
                        ..Default::default()
                    });
                }

                gpu.queue.submit(Some(encoder.finish()));
                frame.present();

                gpu.window.request_redraw();
            }

            _ => {}
        }
    }
}

// --- Entry ---

fn main() {
    let event_loop = EventLoop::new()
        .expect("failed to create event loop");

    let mut app = App::new();

    event_loop.run_app(&mut app)
        .expect("event loop error");
}
