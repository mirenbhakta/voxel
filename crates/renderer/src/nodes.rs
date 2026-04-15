//! Graph pass registration helpers.
//!
//! Each function here registers one render-graph pass against a
//! [`RenderGraph`](crate::graph::RenderGraph), wiring its inputs and outputs
//! through SSA-versioned resource handles. Passes record their GPU work
//! through [`Commands`](crate::commands::Commands) inside the execute closure;
//! handles captured by the closure are `Copy` and do not borrow the graph
//! builder.

pub use crate::subchunk_test::subchunk_test;
