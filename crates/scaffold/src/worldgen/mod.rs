//! Procedural world generators.
//!
//! Each generator implements [`ChunkProvider`] and produces chunk data
//! from coherent noise. Two styles are provided:
//!
//! - [`SurfaceTerrain`] -- heightmap-based terrain with a constrained Y
//!   axis. Grass/dirt/stone layers, rolling hills, suitable for classic
//!   overworld surfaces.
//! - [`AsteroidField`] -- 3D noise blobs uniform in every direction.
//!   No preferred axis, infinite in all dimensions.

mod asteroid;
mod surface;

pub use asteroid::AsteroidField;
pub use surface::SurfaceTerrain;
