//! Graph pass registration helpers.
//!
//! Each function here registers one or more render-graph passes against a
//! [`RenderGraph`](crate::graph::RenderGraph), wiring its inputs and outputs
//! through SSA-versioned resource handles. Passes record their GPU work
//! through [`Commands`](crate::commands::Commands) inside the execute closure;
//! handles captured by the closure are `Copy` and do not borrow the graph
//! builder.

mod cull;
mod indirect_args;
mod mdi_draw;
mod present_blit;

pub use cull::{CullArgs, cull};
pub use indirect_args::IndirectArgs;
pub use mdi_draw::{ColorTarget, DrawArgs, mdi_draw};
pub use present_blit::present_blit;

pub use crate::subchunk::{
    subchunk_exposure, subchunk_material_patch, subchunk_patch, subchunk_prep, subchunk_world,
};
