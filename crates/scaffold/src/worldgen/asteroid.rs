//! 3D noise asteroid field generator.
//!
//! Produces floating blobs of rock distributed uniformly in all three
//! dimensions. There is no preferred axis and no height constraint --
//! the field extends infinitely in every direction.

use eden_math::Vector3;
use noise::{Fbm, NoiseFn, Perlin};
use voxel::block::BlockId;
use voxel::chunk::{Chunk, ChunkPos, CHUNK_SIZE};
use voxel::world::ChunkProvider;

// ---------------------------------------------------------------------------
// AsteroidField
// ---------------------------------------------------------------------------

/// A 3D noise-based asteroid field generator.
///
/// Evaluates FBM noise at every voxel position. Where the noise value
/// exceeds a threshold, a solid block is placed. The threshold controls
/// density: higher values produce smaller, sparser asteroids; lower
/// values produce denser, more connected masses.
pub struct AsteroidField {
    /// The 3D FBM noise function.
    noise     : Fbm<Perlin>,
    /// Noise values above this threshold produce solid voxels.
    /// Typical range: 0.1 (dense) to 0.5 (sparse).
    threshold : f64,
    /// Coordinate scale. Larger values produce bigger features.
    scale     : f64,
    /// Block ID for solid asteroid voxels.
    stone     : BlockId,
}

impl AsteroidField {
    /// Create an asteroid field generator with sensible defaults.
    ///
    /// # Arguments
    ///
    /// * `seed`  - Noise seed for reproducibility.
    /// * `stone` - Block ID for solid asteroid voxels.
    pub fn new(seed: u32, stone: BlockId) -> Self {
        let mut noise     = Fbm::<Perlin>::new(seed);
        noise.octaves     = 4;
        noise.frequency   = 1.0;
        noise.lacunarity  = 2.0;
        noise.persistence = 0.5;

        AsteroidField {
            noise     : noise,
            threshold : 0.3,
            scale     : 32.0,
            stone     : stone,
        }
    }

    /// Set the density threshold. Higher values produce sparser asteroids.
    pub fn with_threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }

    /// Set the feature scale. Larger values produce bigger asteroids.
    pub fn with_scale(mut self, s: f64) -> Self {
        self.scale = s;
        self
    }
}

impl ChunkProvider for AsteroidField {
    fn generate(&self, pos: ChunkPos) -> Option<Chunk> {
        let cs        = CHUNK_SIZE as i32;
        let base_x    = pos.x * cs;
        let base_y    = pos.y * cs;
        let base_z    = pos.z * cs;
        let inv_scale = 1.0 / self.scale;

        let mut chunk     = Chunk::new();
        let mut any_solid = false;

        for lz in 0..CHUNK_SIZE as u8 {
            let wz = (base_z + lz as i32) as f64 * inv_scale;

            for lx in 0..CHUNK_SIZE as u8 {
                let wx = (base_x + lx as i32) as f64 * inv_scale;

                for ly in 0..CHUNK_SIZE as u8 {
                    let wy = (base_y + ly as i32) as f64 * inv_scale;

                    let val = self.noise.get([wx, wy, wz]);

                    if val > self.threshold {
                        chunk.set_block(
                            &Vector3::new(lx, ly, lz),
                            self.stone,
                        );
                        any_solid = true;
                    }
                }
            }
        }

        if any_solid { Some(chunk) } else { None }
    }
}
