//! First-person camera with perspective projection.

use glam::{Mat4, Vec3, Vec4};

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/// A first-person camera with perspective projection.
///
/// Controlled by yaw and pitch for look direction, with an explicit
/// world-space position. The coordinate system is right-handed:
/// +X right, +Y up, -Z forward at yaw = 0.
pub struct Camera {
    /// World-space position.
    pub position : Vec3,
    /// Horizontal rotation in radians. Zero looks along -Z.
    pub yaw      : f32,
    /// Vertical rotation in radians. Clamped to (-PI/2, PI/2).
    pub pitch    : f32,
    /// Vertical field of view in radians.
    pub fov_y    : f32,
    /// Viewport aspect ratio (width / height).
    pub aspect   : f32,
    /// Near clip plane distance.
    pub near     : f32,
    /// Far clip plane distance.
    pub far      : f32,
}

// --- Camera ---

impl Camera {
    /// Create a camera with default settings.
    ///
    /// Positioned above and behind the chunk origin, looking toward the
    /// center. 60 degree vertical FOV, 16:9 aspect.
    pub fn new() -> Self {
        Camera {
            position : Vec3::new(16.0, 20.0, -10.0),
            yaw      : 0.0,
            pitch    : -0.4,
            fov_y    : 60.0_f32.to_radians(),
            aspect   : 16.0 / 9.0,
            near     : 0.1,
            far      : 500.0,
        }
    }

    /// Returns the forward direction vector (normalized).
    ///
    /// Derived from yaw and pitch. At yaw=0, pitch=0, forward is -Z.
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, -cy * cp)
    }

    /// Returns the right direction vector (normalized, always horizontal).
    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy, 0.0, sy)
    }

    /// Returns the view matrix (world to camera space).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }

    /// Returns the perspective projection matrix.
    ///
    /// Right-handed with depth range [0, 1] for wgpu/Vulkan convention.
    pub fn projection_matrix(&self) -> Mat4 {
        let f  = 1.0 / (self.fov_y * 0.5).tan();
        let nf = 1.0 / (self.near - self.far);

        Mat4::from_cols(
            Vec4::new(f / self.aspect, 0.0, 0.0,                       0.0),
            Vec4::new(0.0,             f,   0.0,                       0.0),
            Vec4::new(0.0,             0.0, self.far * nf,            -1.0),
            Vec4::new(0.0,             0.0, self.near * self.far * nf, 0.0),
        )
    }

    /// Returns the combined view-projection matrix.
    pub fn view_proj(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }
}
