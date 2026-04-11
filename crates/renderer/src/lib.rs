//! Renderer crate — primitives layer.
//!
//! See `docs/renderer_rewrite_principles.md` for the design principles this
//! crate enforces, and `.local/renderer_plan.md` for the implementation plan
//! of the first rewrite pass.

pub mod error;
pub mod frame;
pub mod gpu_consts;
pub mod pipeline;
pub mod ring;
pub mod shader;

mod device;

pub use device::RendererContext;
pub use error::RendererError;
pub use frame::{FrameCount, FrameIndex};
