//! Heightmap-based surface terrain generator.
//!
//! Produces a classic overworld surface: rolling hills with grass on top,
//! a few layers of dirt, and stone below. The Y axis is constrained by
//! the terrain height -- chunks entirely above the surface are skipped.

use eden_math::Vector3;
use noise::{Fbm, NoiseFn, Perlin};
use voxel::block::BlockId;
use voxel::chunk::{Chunk, ChunkPos, CHUNK_SIZE};
use voxel::world::ChunkProvider;

// ---------------------------------------------------------------------------
// SurfaceTerrain
// ---------------------------------------------------------------------------

/// A heightmap-driven terrain generator.
///
/// Uses 2D fractional Brownian motion (FBM) noise to produce a height
/// value for each (x, z) column. Voxels below the surface are filled
/// with layered block types: grass at the top, dirt for a few layers
/// below, and stone for everything deeper.
pub struct SurfaceTerrain {
    /// The 2D FBM noise function for the heightmap.
    heightmap   : Fbm<Perlin>,
    /// Base surface height in world-space Y coordinates.
    sea_level   : i32,
    /// Amplitude of the noise in voxels. The surface oscillates
    /// `sea_level +/- amplitude` depending on noise output.
    amplitude   : f64,
    /// Horizontal scale factor. Larger values produce broader features.
    scale       : f64,
    /// Block ID for the topmost surface layer.
    grass       : BlockId,
    /// Block ID for the subsurface layers (typically 3 deep).
    dirt        : BlockId,
    /// Block ID for the deep interior.
    stone       : BlockId,
    /// Number of dirt layers below the grass surface.
    dirt_depth  : i32,
}

impl SurfaceTerrain {
    /// Create a surface terrain generator with sensible defaults.
    ///
    /// # Arguments
    ///
    /// * `seed`  - Noise seed for reproducibility.
    /// * `stone` - Block ID for deep stone layers.
    /// * `dirt`  - Block ID for subsurface dirt layers.
    /// * `grass` - Block ID for the grass surface.
    pub fn new(
        seed  : u32,
        stone : BlockId,
        dirt  : BlockId,
        grass : BlockId,
    ) -> Self
    {
        let mut heightmap     = Fbm::<Perlin>::new(seed);
        heightmap.octaves     = 6;
        heightmap.frequency   = 1.0;
        heightmap.lacunarity  = 2.0;
        heightmap.persistence = 0.5;

        SurfaceTerrain {
            heightmap   : heightmap,
            sea_level   : 32,
            amplitude   : 48.0,
            scale       : 256.0,
            grass       : grass,
            dirt        : dirt,
            stone       : stone,
            dirt_depth  : 3,
        }
    }

    /// Set the base surface height in world-space Y.
    pub fn with_sea_level(mut self, y: i32) -> Self {
        self.sea_level = y;
        self
    }

    /// Set the noise amplitude in voxels.
    pub fn with_amplitude(mut self, a: f64) -> Self {
        self.amplitude = a;
        self
    }

    /// Set the horizontal feature scale.
    pub fn with_scale(mut self, s: f64) -> Self {
        self.scale = s;
        self
    }

    /// Set the number of dirt layers below the grass surface.
    pub fn with_dirt_depth(mut self, d: i32) -> Self {
        self.dirt_depth = d;
        self
    }

    /// Sample the terrain height at a world-space (x, z) column.
    fn surface_height(&self, wx: i32, wz: i32) -> i32 {
        let nx  = wx as f64 / self.scale;
        let nz  = wz as f64 / self.scale;
        let val = self.heightmap.get([nx, nz]);

        self.sea_level + (val * self.amplitude) as i32
    }
}

impl ChunkProvider for SurfaceTerrain {
    fn generate(&self, pos: ChunkPos) -> Option<Chunk> {
        let cs       = CHUNK_SIZE as i32;
        let chunk_y0 = pos.y * cs;
        let chunk_y1 = chunk_y0 + cs - 1;

        // Quick reject: compute the min/max surface height across
        // this chunk's XZ footprint.  If the chunk is entirely above
        // the tallest column, it is all air.
        let base_x = pos.x * cs;
        let base_z = pos.z * cs;

        let mut min_h = i32::MAX;
        let mut max_h = i32::MIN;

        // Sample corners and midpoints for the reject test.  This is
        // approximate but catches the common case cheaply.
        for &sx in &[0, cs / 2, cs - 1] {
            for &sz in &[0, cs / 2, cs - 1] {
                let h = self.surface_height(base_x + sx, base_z + sz);
                min_h = min_h.min(h);
                max_h = max_h.max(h);
            }
        }

        // If the chunk is entirely above the highest sampled surface,
        // it is probably all air. Allow one chunk of margin for noise
        // peaks between sample points.
        if chunk_y0 > max_h + cs {
            return None;
        }

        let mut chunk     = Chunk::new();
        let mut any_solid = false;

        for lz in 0..CHUNK_SIZE as u8 {
            for lx in 0..CHUNK_SIZE as u8 {
                let wx = base_x + lx as i32;
                let wz = base_z + lz as i32;
                let sh = self.surface_height(wx, wz);

                for ly in 0..CHUNK_SIZE as u8 {
                    let wy = chunk_y0 + ly as i32;

                    if wy > sh {
                        continue;
                    }

                    // Depth below the surface.
                    let depth = sh - wy;
                    let block = if depth == 0 {
                        self.grass
                    }
                    else if depth <= self.dirt_depth {
                        self.dirt
                    }
                    else {
                        self.stone
                    };

                    chunk.set_block(&Vector3::new(lx, ly, lz), block);
                    any_solid = true;
                }
            }
        }

        if any_solid { Some(chunk) } else { None }
    }
}
