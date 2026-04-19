//! CPU-side world model for the voxel streaming system.
//!
//! Implements the data structures described in `docs/world_streaming.md`:
//! sub-chunk primitives, typed coordinates, and (in later slices) the
//! clipmap residency control plane that drives the renderer.
//!
//! No wgpu or GPU dependencies; this module is pure CPU infrastructure.

pub mod coord;
pub mod lod;
pub mod material_data_pool;
pub mod material_pool;
pub mod pool;
pub mod residency;
pub mod shell;
pub mod slot_directory;
pub mod subchunk;

#[cfg(feature = "debug-state-history")]
pub mod state_history;

#[cfg(feature = "debug-state-history")]
pub mod divergence;
