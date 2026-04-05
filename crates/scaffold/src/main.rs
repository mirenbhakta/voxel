//! Visual testing scaffold for the voxel library.
//!
//! Opens a window with a wgpu device and render loop. Renders voxel face
//! quads using instanced drawing from packed quad descriptors. Includes an
//! FPS camera with WASD + mouse look controls.

mod build;
mod camera;
mod world;

use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use pollster::FutureExt as _;
use voxel::block::{BlockId, BlockRegistry, Material};
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer,
    BufferBindingType, BufferUsages, Color, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, CompareFunction, CurrentSurfaceTexture,
    DepthBiasState, DepthStencilState, Device, DeviceDescriptor, Extent3d,
    Face, Features, FragmentState, FrontFace, Instance, InstanceDescriptor,
    Limits, LoadOp, MultisampleState, Operations,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode,
    PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState,
    StoreOp, Surface, SurfaceConfiguration, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureView, TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use camera::Camera;
use voxel::chunk::ChunkPos;
use world::{build_material_tables, GpuWorld};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Width and height of each texture layer in pixels.
const TEX_SIZE: u32 = 16;

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
    /// The camera uniform buffer.
    camera_buf : Buffer,
    /// The depth buffer texture view.
    depth_view : TextureView,
    /// The GPU world manager (chunks, build pipeline, draw state).
    world      : GpuWorld,
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

        // Request the device and queue with immediates support.
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label             : Some("scaffold_device"),
                required_features : Features::IMMEDIATES
                                  | Features::INDIRECT_FIRST_INSTANCE,
                required_limits   : Limits {
                    max_immediate_size : 4,
                    ..Limits::default()
                },
                ..Default::default()
            })
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

        // Camera uniform buffer.
        let camera_uniform = CameraUniform {
            view_proj : self.camera.view_proj().to_cols_array(),
        };

        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label    : Some("camera_buf"),
            contents : bytemuck::bytes_of(&camera_uniform),
            usage    : BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Register block types.
        let mut registry = BlockRegistry::new();
        let stone = registry.register(
            "stone", Material::from_rgb(128, 128, 128).with_texture(1),
        );
        let dirt = registry.register(
            "dirt", Material::from_rgb(139, 90, 43).with_texture(2),
        );
        let grass = registry.register(
            "grass",
            Material::from_rgb(255, 255, 255)
                .with_top_bottom_side(3, 2, 4),
        );

        // Build GPU material tables from the registry.
        let (materials, face_tex) = build_material_tables(&registry);

        // Generate procedural block textures.
        let texture_pixels = generate_textures();
        let texture_layers = 5u32;

        // Bind group layout: camera, quad pool, page table, chunk offsets,
        // material volume, material table, texture array, sampler.
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
                    BindGroupLayoutEntry {
                        binding    : 2,
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
                    BindGroupLayoutEntry {
                        binding    : 3,
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
                    BindGroupLayoutEntry {
                        binding    : 4,
                        visibility : ShaderStages::FRAGMENT,
                        ty         : BindingType::Buffer {
                            ty                 : BufferBindingType::Storage {
                                read_only : true,
                            },
                            has_dynamic_offset : false,
                            min_binding_size   : None,
                        },
                        count : None,
                    },
                    BindGroupLayoutEntry {
                        binding    : 5,
                        visibility : ShaderStages::FRAGMENT,
                        ty         : BindingType::Buffer {
                            ty                 : BufferBindingType::Storage {
                                read_only : true,
                            },
                            has_dynamic_offset : false,
                            min_binding_size   : None,
                        },
                        count : None,
                    },
                    BindGroupLayoutEntry {
                        binding    : 6,
                        visibility : ShaderStages::FRAGMENT,
                        ty         : BindingType::Buffer {
                            ty                 : BufferBindingType::Storage {
                                read_only : true,
                            },
                            has_dynamic_offset : false,
                            min_binding_size   : None,
                        },
                        count : None,
                    },
                    BindGroupLayoutEntry {
                        binding    : 7,
                        visibility : ShaderStages::FRAGMENT,
                        ty         : BindingType::Texture {
                            sample_type    : TextureSampleType::Float {
                                filterable : true,
                            },
                            view_dimension : TextureViewDimension::D2Array,
                            multisampled   : false,
                        },
                        count : None,
                    },
                    BindGroupLayoutEntry {
                        binding    : 8,
                        visibility : ShaderStages::FRAGMENT,
                        ty         : BindingType::Sampler(
                            SamplerBindingType::Filtering,
                        ),
                        count : None,
                    },
                ],
            },
        );

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
                label              : Some("main_pl"),
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

        // GPU world manager.
        let mut world = GpuWorld::new(
            &device,
            &queue,
            bind_group_layout,
            camera_buf.clone(),
            &materials,
            &face_tex,
            &texture_pixels,
            TEX_SIZE,
            texture_layers,
        );

        for cx in 0..4i32 {
            for cz in 0..4i32 {
                let (occ, mat) = generate_test_terrain(
                    cx, cz, stone, dirt, grass,
                );
                world.insert(
                    &device, &queue, ChunkPos::new(cx, 0, cz), &occ, &mat,
                );
            }
        }

        world.rebuild(&device, &queue);

        self.gpu = Some(Gpu {
            window, surface, config, device, queue,
            pipeline, camera_buf, depth_view, world,
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
                    gpu.world.draw(&mut pass);
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

/// Generate terrain occupancy and material data for a test chunk.
///
/// Produces layered terrain with stone at the bottom, dirt in the middle,
/// and grass on top. Fill height varies by chunk position so adjacent
/// chunks are visually distinguishable.
///
/// # Arguments
///
/// * `cx`    - Chunk X coordinate (for height variation).
/// * `cz`    - Chunk Z coordinate (for height variation).
/// * `stone` - Block ID for stone layers.
/// * `dirt`  - Block ID for dirt layers.
/// * `grass` - Block ID for the top grass layer.
fn generate_test_terrain(
    cx    : i32,
    cz    : i32,
    stone : BlockId,
    dirt  : BlockId,
    grass : BlockId,
) -> ([u32; 1024], Box<[u8; 32768]>)
{
    let mut occ = [0u32; 1024];
    let mut mat = Box::new([0u8; 32768]);

    let height = (4 + ((cx * 3 + cz * 7).unsigned_abs() % 8)) as usize;

    for z in 0..32usize {
        for y in 0..height.min(32) {
            occ[z * 32 + y] = !0u32;

            // Assign block type based on depth from the surface.
            let block = if y == height - 1 {
                grass
            }
            else if y + 3 >= height {
                dirt
            }
            else {
                stone
            };

            let block_id = block.raw() as u8;
            for x in 0..32usize {
                mat[z * 1024 + y * 32 + x] = block_id;
            }
        }
    }

    (occ, mat)
}

// ---------------------------------------------------------------------------
// Procedural textures
// ---------------------------------------------------------------------------

/// Generate all block texture layers as raw RGBA pixel data.
///
/// Returns a byte vector containing all layers packed sequentially.
/// Layer 0 is a solid white fallback. Layers 1-3 are noisy variations
/// of stone, dirt, and grass.
fn generate_textures() -> Vec<u8> {
    let mut pixels = Vec::new();
    let layer_bytes = (TEX_SIZE * TEX_SIZE * 4) as usize;

    // Layer 0: solid white fallback.
    pixels.extend(std::iter::repeat(255u8).take(layer_bytes));

    // Layer 1: stone (gray, speckled).
    pixels.extend(generate_noise_texture(128, 128, 128, 20, 1));

    // Layer 2: dirt (brown).
    pixels.extend(generate_noise_texture(139, 90, 43, 15, 2));

    // Layer 3: grass top (green).
    pixels.extend(generate_noise_texture(80, 160, 50, 20, 3));

    // Layer 4: grass side (brownish-green).
    pixels.extend(generate_noise_texture(100, 120, 60, 15, 4));

    pixels
}

/// Generate a single noise texture layer.
///
/// Produces a TEX_SIZE x TEX_SIZE RGBA image with per-pixel noise
/// applied to the base color. The noise is deterministic for a given
/// seed.
///
/// # Arguments
///
/// * `base_r` - Base red channel value.
/// * `base_g` - Base green channel value.
/// * `base_b` - Base blue channel value.
/// * `noise`  - Maximum per-channel deviation from the base color.
/// * `seed`   - Hash seed for deterministic noise generation.
fn generate_noise_texture(
    base_r : u8,
    base_g : u8,
    base_b : u8,
    noise  : u8,
    seed   : u32,
) -> Vec<u8>
{
    let mut pixels = Vec::with_capacity((TEX_SIZE * TEX_SIZE * 4) as usize);

    for y in 0..TEX_SIZE {
        for x in 0..TEX_SIZE {
            let h     = tex_hash(x, y, seed);
            let delta = (h % (noise as u32 * 2 + 1)) as i16 - noise as i16;
            let r     = (base_r as i16 + delta).clamp(0, 255) as u8;
            let g     = (base_g as i16 + delta).clamp(0, 255) as u8;
            let b     = (base_b as i16 + delta).clamp(0, 255) as u8;
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
    }

    pixels
}

/// Simple integer hash for deterministic texture noise.
fn tex_hash(x: u32, y: u32, seed: u32) -> u32 {
    let mut h = x.wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(seed.wrapping_mul(2654435761));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^ (h >> 16)
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
