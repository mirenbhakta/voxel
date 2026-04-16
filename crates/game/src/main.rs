//! Game binary — windowed renderer entry point.
//!
//! Uses winit 0.30's `ApplicationHandler` pattern for cross-platform event
//! loop management.  The renderer is initialised lazily on the first
//! `resumed` event so that the surface is always created from a live window.
//!
//! A minimal fly-camera controller (WASD + mouse look + Space/Shift for
//! vertical) is wired in so the scene can be navigated for visual debugging.
//! Cursor is grabbed on window creation; Escape releases it and exits.
//!
//! # Validation mode
//!
//! Run `game --validate` (e.g. `cargo run -- --validate`) to execute a GPU
//! validation suite against real hardware and exit, rather than opening a
//! window.  Useful for bug reports and CI on GPU-capable machines.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use renderer::{FrameCount, RendererContext, RendererError};
use renderer::{
    SUBCHUNK_DEPTH_FORMAT, SUBCHUNK_MAX_CANDIDATES,
    SubchunkInstance, SubchunkTest, TestCamera, nodes,
    occupancy_exposure, sphere_occupancy,
};
use renderer::graph::{BufferPool, RenderGraph, TextureDesc, TexturePool};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

mod validation;

// ---------------------------------------------------------------------------

fn main() {
    if std::env::args().any(|a| a == "--validate") {
        validation::run();
        return;
    }

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop error");
}

// ---------------------------------------------------------------------------

/// Camera tuning. Units are world voxels (sub-chunks are 8 voxels wide).
const MOVE_SPEED:      f32 = 15.0;
const MOVE_SPEED_FAST: f32 = 45.0;
const MOUSE_SENS:      f32 = 0.0015;
/// Clamp pitch just shy of ±90° so `right` (built from world-up × forward)
/// never degenerates.
const PITCH_CLAMP:     f32 = 1.553; // ~89°

/// Application state.  Fields are populated on the first `resumed` event
/// and remain `Some` for the rest of the session.
struct App {
    window:        Option<Arc<Window>>,
    ctx:           Option<RendererContext>,
    subchunk_test: Option<Arc<SubchunkTest>>,
    buf_pool:      BufferPool,
    tex_pool:      TexturePool,

    // --- Camera state ---
    pos:    [f32; 3],
    yaw:    f32,
    pitch:  f32,
    keys:   HashSet<KeyCode>,
    last_t: Option<Instant>,
}

impl App {
    fn new() -> Self {
        Self {
            window:        None,
            ctx:           None,
            subchunk_test: None,
            buf_pool:      BufferPool::default(),
            tex_pool:      TexturePool::default(),
            // Starting pose: outside the 4×4×4 grid [0,32]^3, facing +Z.
            pos:    [16.0, 16.0, -40.0],
            yaw:    0.0,
            pitch:  0.0,
            keys:   HashSet::new(),
            last_t: None,
        }
    }
}

/// Build an orthonormal camera basis `(forward, right, up)` from yaw + pitch.
///
/// Convention: `yaw = 0, pitch = 0` ⇒ `forward = (0, 0, 1)`, `right = (1, 0, 0)`,
/// `up = (0, 1, 0)` — matches the shader's left-handed basis where
/// `cross(right, up) = forward`. Yaw rotates around world-up (+Y); pitch tilts
/// around the camera's right axis.
fn camera_basis(yaw: f32, pitch: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();

    let forward = [cp * sy, sp, cp * cy];

    // right = normalize(cross(world_up, forward)) with world_up = (0, 1, 0).
    // cross((0,1,0), (fx,fy,fz)) = (fz, 0, -fx). Normalize in the XZ plane —
    // pitch is clamped so the length is never zero.
    let rx  = forward[2];
    let rz  = -forward[0];
    let len = (rx * rx + rz * rz).sqrt();
    let right = [rx / len, 0.0, rz / len];

    // up = cross(forward, right).
    let up = [
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
    ];

    (forward, right, up)
}

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

        // Try to lock the cursor for continuous mouse-look. Locked is
        // preferred (Wayland native, some X11 compositors). Fall back to
        // Confined on platforms that reject Locked. If both fail, mouse
        // motion still works but the cursor escapes the window — acceptable
        // for a debugging binary.
        if window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
            let _ = window.set_cursor_grab(CursorGrabMode::Confined);
        }
        window.set_cursor_visible(false);

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

        // Build a 4×4×4 grid of 8³ sub-chunks spanning world-space [0,32]^3.
        // Each candidate gets a sphere occupancy; the 6-bit directional
        // exposure mask is derived from the occupancy itself (for a sphere,
        // all six bits are set — the mask contributes no rejections here).
        let sphere    = sphere_occupancy();
        let sphere_ex = occupancy_exposure(&sphere);
        let mut instances = [SubchunkInstance::new([0, 0, 0], 0, 0); SUBCHUNK_MAX_CANDIDATES];
        let mut occs      = [sphere; SUBCHUNK_MAX_CANDIDATES];
        for z in 0u32..4 {
            for y in 0u32..4 {
                for x in 0u32..4 {
                    let idx = (z * 16 + y * 4 + x) as usize;
                    instances[idx] = SubchunkInstance::new(
                        [(x * 8) as i32, (y * 8) as i32, (z * 8) as i32],
                        idx as u32,
                        sphere_ex,
                    );
                    occs[idx] = sphere;
                }
            }
        }

        let test = Arc::new(SubchunkTest::new(&ctx, &instances, &occs));
        self.subchunk_test = Some(test);

        self.ctx = Some(ctx);
        self.window = Some(window);
    }

    /// Raw device events — used for continuous relative mouse motion.
    ///
    /// `WindowEvent::CursorMoved` is absolute position and doesn't work once
    /// the cursor is locked; `DeviceEvent::MouseMotion` gives deltas that
    /// keep arriving even when the cursor is pinned.
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.yaw   += dx as f32 * MOUSE_SENS;
            self.pitch -= dy as f32 * MOUSE_SENS;
            self.pitch  = self.pitch.clamp(-PITCH_CLAMP, PITCH_CLAMP);
        }
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

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state,
                    repeat,
                    ..
                },
                ..
            } => {
                // Escape releases the cursor and exits — single keystroke to
                // recover from a locked cursor is worth more than the
                // "escape-to-release-then-click-to-re-grab" dance for a
                // debugging binary.
                if code == KeyCode::Escape && state == ElementState::Pressed {
                    if let Some(window) = &self.window {
                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                        window.set_cursor_visible(true);
                    }
                    event_loop.exit();
                    return;
                }

                // Track held-key state for per-frame integration. Ignore
                // auto-repeat — only press/release edges matter.
                if !repeat {
                    match state {
                        ElementState::Pressed  => { self.keys.insert(code); }
                        ElementState::Released => { self.keys.remove(&code); }
                    }
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

                // Delta time. On the first frame `last_t` is None → dt = 0,
                // which leaves the camera where resumed() placed it.
                let now = Instant::now();
                let dt  = match self.last_t {
                    Some(t) => (now - t).as_secs_f32(),
                    None    => 0.0,
                };
                self.last_t = Some(now);

                // Integrate input into camera state.
                let (forward, right, up) = camera_basis(self.yaw, self.pitch);
                let speed = if self.keys.contains(&KeyCode::ShiftLeft)
                    || self.keys.contains(&KeyCode::ShiftRight)
                {
                    MOVE_SPEED_FAST
                } else {
                    MOVE_SPEED
                };
                let step = speed * dt;

                let mut mv = [0.0f32; 3];
                let mut add = |v: [f32; 3], s: f32| {
                    mv[0] += v[0] * s;
                    mv[1] += v[1] * s;
                    mv[2] += v[2] * s;
                };
                if self.keys.contains(&KeyCode::KeyW)        { add(forward,  step); }
                if self.keys.contains(&KeyCode::KeyS)        { add(forward, -step); }
                if self.keys.contains(&KeyCode::KeyD)        { add(right,    step); }
                if self.keys.contains(&KeyCode::KeyA)        { add(right,   -step); }
                if self.keys.contains(&KeyCode::Space)       { add([0.0, 1.0, 0.0],  step); }
                if self.keys.contains(&KeyCode::ControlLeft) { add([0.0, 1.0, 0.0], -step); }
                self.pos[0] += mv[0];
                self.pos[1] += mv[1];
                self.pos[2] += mv[2];

                let size   = window.inner_size();
                let w      = size.width.max(1);
                let h      = size.height.max(1);
                let aspect = w as f32 / h as f32;

                let camera = TestCamera {
                    pos:     self.pos,
                    fov_y:   std::f32::consts::FRAC_PI_3,
                    forward,
                    aspect,
                    right,
                    _pad0:   0.0,
                    up,
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
                let color = graph.present(surface_frame);
                let depth = graph.create_texture(
                    "subchunk_depth",
                    TextureDesc::new_2d(
                        w, h,
                        SUBCHUNK_DEPTH_FORMAT,
                        wgpu::TextureUsages::RENDER_ATTACHMENT,
                    ),
                );
                nodes::subchunk_test(&mut graph, test, color, depth);

                let mut fe = ctx.begin_frame();
                let frame  = ctx.frame_index();
                let (pending, present_token) = graph.compile()
                    .expect("render graph compile")
                    .execute(&mut fe, frame, &mut self.buf_pool, &mut self.tex_pool, ctx.device());
                ctx.end_frame(fe);
                present_token.present();

                pending.release(&mut self.buf_pool, &mut self.tex_pool);

                window.request_redraw();
            }

            _ => {}
        }
    }
}
