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
use renderer::{SUBCHUNK_DEPTH_FORMAT, SubchunkCamera, nodes};
use renderer::graph::{BufferPool, RenderGraph, TextureDesc, TexturePool};

use crate::world::coord::Level;
use crate::world::residency::LevelConfig;
use crate::world_view::WorldView;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

mod validation;
mod world;
mod world_view;

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
    window:     Option<Arc<Window>>,
    ctx:        Option<RendererContext>,
    world_view: Option<WorldView>,
    buf_pool:   BufferPool,
    tex_pool:   TexturePool,

    // --- Camera state ---
    pos:    [f32; 3],
    yaw:    f32,
    pitch:  f32,
    keys:   HashSet<KeyCode>,
    last_t: Option<Instant>,

    // --- Debug toggles ---
    /// When `true`, `WorldView::update` is skipped so residency (and the
    /// resulting GPU state) stays frozen. The camera still moves, which
    /// lets you orbit around the frozen content to inspect LOD artefacts
    /// from multiple angles. Toggled by `L`.
    lod_frozen: bool,
}

impl App {
    fn new() -> Self {
        Self {
            window:     None,
            ctx:        None,
            world_view: None,
            buf_pool:   BufferPool::default(),
            tex_pool:   TexturePool::default(),
            // Starting pose: camera outside the initial L0 shell,
            // looking toward +Z. At radius 2 the L0 shell world AABB is
            // `[-16, 16]³` centered on the origin anchor vertex.
            pos:    [4.0, 4.0, -40.0],
            yaw:    0.0,
            pitch:  0.0,
            keys:   HashSet::new(),
            last_t: None,
            lod_frozen: false,
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

        // Bring up the residency-driven world view. Populates the GPU's
        // instance and occupancy buffers from the initial multi-level
        // shells around the camera before the first render.
        //
        // Three levels at radius 2 → 3 × 64 = 192 resident sub-chunks
        // (fits in the 256 MAX_CANDIDATES budget). Coverage (world
        // widths, centered on the shared anchor vertex): L0 32 m,
        // L1 64 m, L2 128 m. Radii must be even so each level's shell
        // aligns to an integer cluster of next-coarser sub-chunks —
        // that's what lets the cull's "fully inside finer shell" test
        // cull an exact 2×2×2 = 8 sub-chunks per coarser level instead
        // of straddling boundaries.
        let configs = [
            LevelConfig { level: Level::ZERO, radius: [2, 2, 2] },
            LevelConfig { level: Level(1),    radius: [2, 2, 2] },
            LevelConfig { level: Level(2),    radius: [2, 2, 2] },
        ];
        let world_view = WorldView::new(&ctx, &configs, self.pos);
        self.world_view = Some(world_view);

        self.ctx    = Some(ctx);
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

                // L toggles LOD-streaming freeze. Residency stops updating
                // so the camera can orbit without mutating what's loaded.
                if code == KeyCode::KeyL && state == ElementState::Pressed && !repeat {
                    self.lod_frozen = !self.lod_frozen;
                    eprintln!(
                        "lod streaming {}",
                        if self.lod_frozen { "frozen" } else { "resumed" },
                    );
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
                let (Some(ctx), Some(window), Some(world_view)) = (
                    self.ctx.as_mut(),
                    self.window.as_ref(),
                    self.world_view.as_mut(),
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

                let camera = SubchunkCamera {
                    pos:     self.pos,
                    fov_y:   std::f32::consts::FRAC_PI_3,
                    forward,
                    aspect,
                    right,
                    _pad0:   0.0,
                    up,
                    _pad1:   0.0,
                };

                // Drive residency off the current camera position before
                // uploading per-frame GPU state. `update` handles evictions,
                // prep synthesis, slot uploads, and instance-list rebuild.
                // When LOD is frozen, skip the residency update — camera
                // still moves for rendering, but sub-chunk contents stay
                // put so the hole / artefact being inspected does not roll
                // away under the camera.
                if !self.lod_frozen {
                    world_view.update(ctx, self.pos);
                    if world_view.evicted_last_update() > 0 {
                        eprintln!(
                            "evicted {} sub-chunks (total {})",
                            world_view.evicted_last_update(),
                            world_view.evicted_total(),
                        );
                    }
                }
                world_view.renderer().write_camera(ctx, &camera);

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
                nodes::subchunk_world(&mut graph, world_view.renderer(), color, depth);

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
