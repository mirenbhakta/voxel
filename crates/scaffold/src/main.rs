//! Visual testing scaffold for the voxel library.
//!
//! Opens a window with a wgpu device and render loop. Renders voxel face
//! quads using instanced drawing from packed quad descriptors. Includes an
//! FPS camera with WASD + mouse look controls.

mod camera;

use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use pollster::FutureExt as _;
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages,
    Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor,
    CompareFunction, CurrentSurfaceTexture, DepthBiasState, DepthStencilState,
    Device, DeviceDescriptor, Extent3d, Face, FragmentState, FrontFace,
    Instance, InstanceDescriptor, LoadOp, MultisampleState, Operations,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode,
    PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StencilState, StoreOp, Surface,
    SurfaceConfiguration, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor, VertexState,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use voxel::render::{Direction, FaceMasks, FaceNeighbors};

use camera::Camera;

// ---------------------------------------------------------------------------
// GPU types
// ---------------------------------------------------------------------------

/// Camera uniform data uploaded to the GPU.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    /// Column-major view-projection matrix.
    view_proj : [f32; 16],
}

// ---------------------------------------------------------------------------
// Input state
// ---------------------------------------------------------------------------

/// Tracks which movement keys are currently held.
struct InputState {
    /// W key (forward).
    forward  : bool,
    /// S key (backward).
    backward : bool,
    /// A key (strafe left).
    left     : bool,
    /// D key (strafe right).
    right    : bool,
    /// Space key (ascend).
    up       : bool,
    /// Left shift key (descend).
    down     : bool,
}

// --- InputState ---

impl InputState {
    /// Create an input state with no keys pressed.
    fn new() -> Self {
        InputState {
            forward  : false,
            backward : false,
            left     : false,
            right    : false,
            up       : false,
            down     : false,
        }
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

/// Top-level application state for the scaffold window.
struct App {
    /// The GPU state, initialized on resume.
    gpu        : Option<Gpu>,
    /// The first-person camera.
    camera     : Camera,
    /// Currently held movement keys.
    input      : InputState,
    /// Whether the cursor is grabbed for mouse look.
    grabbed    : bool,
    /// Timestamp of the previous frame for delta time.
    last_frame : Instant,
}

/// Initialized GPU resources tied to a window surface.
struct Gpu {
    /// The window handle, kept alive for the surface.
    window     : Arc<Window>,
    /// The wgpu rendering surface.
    surface    : Surface<'static>,
    /// The surface configuration (format, size, present mode).
    config     : SurfaceConfiguration,
    /// The logical device.
    device     : Device,
    /// The command submission queue.
    queue      : Queue,
    /// The voxel rendering pipeline.
    pipeline   : RenderPipeline,
    /// The bind group containing camera and quad buffers.
    bind_group : BindGroup,
    /// The camera uniform buffer.
    camera_buf : Buffer,
    /// The depth buffer texture view.
    depth_view : TextureView,
    /// The number of quad instances to draw.
    quad_count : u32,
}

// --- App ---

impl App {
    /// Create a new application with no GPU state.
    fn new() -> Self {
        Self {
            gpu        : None,
            camera     : Camera::new(),
            input      : InputState::new(),
            grabbed    : false,
            last_frame : Instant::now(),
        }
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

        // Update camera aspect ratio to match the window.
        self.camera.aspect = config.width as f32 / config.height as f32;

        // Generate test quad data from a chunk's face masks.
        let quads      = generate_test_quads();
        let quad_count = quads.len() as u32;

        let quad_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label    : Some("quad_buf"),
            contents : bytemuck::cast_slice(&quads),
            usage    : BufferUsages::STORAGE,
        });

        // Camera uniform buffer.
        let camera_uniform = CameraUniform {
            view_proj : self.camera.view_proj().to_cols_array(),
        };

        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label    : Some("camera_buf"),
            contents : bytemuck::bytes_of(&camera_uniform),
            usage    : BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Bind group layout: camera uniform + quad storage.
        let bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                label   : Some("main_bgl"),
                entries : &[
                    BindGroupLayoutEntry {
                        binding    : 0,
                        visibility : ShaderStages::VERTEX,
                        ty         : BindingType::Buffer {
                            ty                 : BufferBindingType::Uniform,
                            has_dynamic_offset : false,
                            min_binding_size   : None,
                        },
                        count : None,
                    },
                    BindGroupLayoutEntry {
                        binding    : 1,
                        visibility : ShaderStages::VERTEX,
                        ty         : BindingType::Buffer {
                            ty                 : BufferBindingType::Storage {
                                read_only : true,
                            },
                            has_dynamic_offset : false,
                            min_binding_size   : None,
                        },
                        count : None,
                    },
                ],
            },
        );

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label   : Some("main_bg"),
            layout  : &bind_group_layout,
            entries : &[
                BindGroupEntry {
                    binding  : 0,
                    resource : camera_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding  : 1,
                    resource : quad_buf.as_entire_binding(),
                },
            ],
        });

        // Load the shader.
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label  : Some("voxel_shader"),
            source : ShaderSource::Wgsl(
                include_str!("shaders/voxel.wgsl").into(),
            ),
        });

        // Pipeline layout.
        let pipeline_layout = device.create_pipeline_layout(
            &PipelineLayoutDescriptor {
                label                : Some("main_pl"),
                bind_group_layouts : &[Some(&bind_group_layout)],
                immediate_size     : 0,
            },
        );

        // Render pipeline.
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label    : Some("voxel_pipeline"),
            layout   : Some(&pipeline_layout),
            vertex   : VertexState {
                module              : &shader,
                entry_point         : Some("vs_main"),
                buffers             : &[],
                compilation_options : PipelineCompilationOptions::default(),
            },
            fragment : Some(FragmentState {
                module              : &shader,
                entry_point         : Some("fs_main"),
                targets             : &[Some(ColorTargetState {
                    format     : config.format,
                    blend      : None,
                    write_mask : ColorWrites::ALL,
                })],
                compilation_options : PipelineCompilationOptions::default(),
            }),
            primitive : PrimitiveState {
                topology   : PrimitiveTopology::TriangleList,
                front_face : FrontFace::Ccw,
                cull_mode  : Some(Face::Back),
                polygon_mode : PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil : Some(DepthStencilState {
                format              : TextureFormat::Depth32Float,
                depth_write_enabled : Some(true),
                depth_compare       : Some(CompareFunction::Less),
                stencil             : StencilState::default(),
                bias                : DepthBiasState::default(),
            }),
            multisample : MultisampleState::default(),
            multiview_mask : None,
            cache       : None,
        });

        // Depth texture.
        let depth_view = create_depth_texture(
            &device,
            config.width,
            config.height,
        );

        self.gpu = Some(Gpu {
            window, surface, config, device, queue,
            pipeline, bind_group, camera_buf, depth_view, quad_count,
        });

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

                gpu.depth_view = create_depth_texture(
                    &gpu.device,
                    gpu.config.width,
                    gpu.config.height,
                );

                self.camera.aspect =
                    gpu.config.width as f32 / gpu.config.height as f32;

                gpu.window.request_redraw();
            }

            // -- Keyboard input --

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key : PhysicalKey::Code(code),
                    state,
                    ..
                },
                ..
            } => {
                let pressed = state == ElementState::Pressed;

                match code {
                    KeyCode::KeyW      => self.input.forward  = pressed,
                    KeyCode::KeyS      => self.input.backward = pressed,
                    KeyCode::KeyA      => self.input.left     = pressed,
                    KeyCode::KeyD      => self.input.right    = pressed,
                    KeyCode::Space     => self.input.up       = pressed,
                    KeyCode::ShiftLeft => self.input.down     = pressed,

                    // Release cursor grab on Escape.
                    KeyCode::Escape if pressed => {
                        self.grabbed = false;
                        let _ = gpu.window
                            .set_cursor_grab(CursorGrabMode::None);
                        gpu.window.set_cursor_visible(true);
                    }

                    _ => {}
                }
            }

            // -- Mouse click to grab cursor --

            WindowEvent::MouseInput {
                button : MouseButton::Left,
                state  : ElementState::Pressed,
                ..
            } if !self.grabbed => {
                self.grabbed = true;
                let _ = gpu.window
                    .set_cursor_grab(CursorGrabMode::Confined)
                    .or_else(|_| {
                        gpu.window.set_cursor_grab(CursorGrabMode::Locked)
                    });
                gpu.window.set_cursor_visible(false);
            }

            // -- Frame --

            WindowEvent::RedrawRequested => {
                // Compute delta time.
                let now = Instant::now();
                let dt  = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                // Update camera position from keyboard input.
                update_camera_movement(&mut self.camera, &self.input, dt);

                // Write camera uniform to the GPU buffer.
                let uniform = CameraUniform {
                    view_proj : self.camera.view_proj().to_cols_array(),
                };

                gpu.queue.write_buffer(
                    &gpu.camera_buf,
                    0,
                    bytemuck::bytes_of(&uniform),
                );

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

                let mut encoder = gpu.device.create_command_encoder(
                    &CommandEncoderDescriptor::default(),
                );

                // Render pass: clear + draw quads.
                {
                    let mut pass = encoder.begin_render_pass(
                        &RenderPassDescriptor {
                            label             : Some("main"),
                            color_attachments : &[Some(
                                RenderPassColorAttachment {
                                    view           : &view,
                                    depth_slice    : None,
                                    resolve_target : None,
                                    ops            : Operations {
                                        load  : LoadOp::Clear(Color {
                                            r : 0.1,
                                            g : 0.1,
                                            b : 0.15,
                                            a : 1.0,
                                        }),
                                        store : StoreOp::Store,
                                    },
                                },
                            )],
                            depth_stencil_attachment : Some(
                                RenderPassDepthStencilAttachment {
                                    view        : &gpu.depth_view,
                                    depth_ops   : Some(Operations {
                                        load  : LoadOp::Clear(1.0),
                                        store : StoreOp::Store,
                                    }),
                                    stencil_ops : None,
                                },
                            ),
                            ..Default::default()
                        },
                    );

                    pass.set_pipeline(&gpu.pipeline);
                    pass.set_bind_group(0, &gpu.bind_group, &[]);
                    pass.draw(0..6, 0..gpu.quad_count);
                }

                gpu.queue.submit(Some(encoder.finish()));
                frame.present();

                gpu.window.request_redraw();
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop : &ActiveEventLoop,
        _device_id  : DeviceId,
        event       : DeviceEvent,
    ) {
        // Only process mouse motion when the cursor is grabbed.
        if !self.grabbed {
            return;
        }

        if let DeviceEvent::MouseMotion { delta } = event {
            let sensitivity = 0.003;
            self.camera.yaw   += delta.0 as f32 * sensitivity;
            self.camera.pitch -= delta.1 as f32 * sensitivity;

            // Clamp pitch to prevent gimbal lock.
            let limit = std::f32::consts::FRAC_PI_2 - 0.01;
            self.camera.pitch = self.camera.pitch.clamp(-limit, limit);
        }
    }
}

// ---------------------------------------------------------------------------
// Camera movement
// ---------------------------------------------------------------------------

/// Apply keyboard movement to the camera for one frame.
///
/// Movement is on the XZ plane relative to yaw (forward/back/strafe),
/// with vertical movement along world Y (space/shift).
fn update_camera_movement(
    camera : &mut Camera,
    input  : &InputState,
    dt     : f32,
) {
    let speed = 20.0;

    let forward = camera.forward();
    let right   = camera.right();

    // Project forward onto the XZ plane for ground-relative movement.
    let forward_xz = Vec3::new(forward.x, 0.0, forward.z)
        .normalize_or_zero();

    let mut velocity = Vec3::ZERO;

    if input.forward  { velocity += forward_xz; }
    if input.backward { velocity -= forward_xz; }
    if input.right    { velocity += right; }
    if input.left     { velocity -= right; }
    if input.up       { velocity += Vec3::Y; }
    if input.down     { velocity -= Vec3::Y; }

    if velocity.length_squared() > 0.0 {
        camera.position += velocity.normalize() * speed * dt;
    }
}

// ---------------------------------------------------------------------------
// Depth texture
// ---------------------------------------------------------------------------

/// Create a depth texture view for the given dimensions.
fn create_depth_texture(
    device : &Device,
    width  : u32,
    height : u32,
) -> TextureView
{
    let texture = device.create_texture(&TextureDescriptor {
        label           : Some("depth"),
        size            : Extent3d {
            width,
            height,
            depth_or_array_layers : 1,
        },
        mip_level_count : 1,
        sample_count    : 1,
        dimension       : TextureDimension::D2,
        format          : TextureFormat::Depth32Float,
        usage           : TextureUsages::RENDER_ATTACHMENT,
        view_formats    : &[],
    });

    texture.create_view(&TextureViewDescriptor::default())
}

// ---------------------------------------------------------------------------
// Test data generation
// ---------------------------------------------------------------------------

/// Generate packed quad descriptors from a test chunk.
///
/// Fills a 32x8x32 platform at y=0..8, derives face masks, and emits
/// one 1x1 quad per visible face. Direction is packed into bits 25-27.
fn generate_test_quads() -> Vec<u32> {
    // Build occupancy directly. Layout: occ[z * 32 + y], bit x.
    // Fill y=0..8 for all x and z (a solid platform).
    let mut occ = vec![0u32; 1024];

    for z in 0..32 {
        for y in 0..8 {
            // All 32 x-bits set in this word.
            occ[z * 32 + y] = !0u32;
        }
    }

    // Derive face masks with no neighbors (boundary faces exposed).
    let neighbors = FaceNeighbors::none();
    let faces     = FaceMasks::from_occupancy(&occ, &neighbors);

    // Emit one packed u32 per visible face bit.
    let mut quads = Vec::new();

    for &dir in &Direction::ALL {
        for layer in 0..32usize {
            for row in 0..32usize {
                let word = faces.word(dir, layer, row);
                let mut bits = word;

                while bits != 0 {
                    let col = bits.trailing_zeros();
                    bits &= bits - 1;

                    // Pack: col(5) | row(5) | layer(5) | 0(5) | 0(5) | dir(3)
                    let packed = col
                        | ((row as u32) << 5)
                        | ((layer as u32) << 10)
                        | ((dir as u32) << 25);

                    quads.push(packed);
                }
            }
        }
    }

    quads
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

fn main() {
    let event_loop = EventLoop::new()
        .expect("failed to create event loop");

    let mut app = App::new();

    event_loop.run_app(&mut app)
        .expect("event loop error");
}
