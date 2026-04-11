//! Rendering pipeline data structures.
//!
//! Derived geometry representations for rasterization and ray traversal.
//! All types are CPU-side with no GPU dependencies. The rendering module
//! reads from chunk source data (occupancy bitmask, material array) but
//! the chunk and world modules have no knowledge of rendering.

pub mod direction;
pub mod face;
pub mod quad;

pub use direction::Direction;
pub use face::{FaceMasks, FaceNeighbors};
pub use quad::{LayerOccupancy, QuadDescriptor};
