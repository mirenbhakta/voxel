//! Error types for the renderer crate.

use thiserror::Error;

/// Top-level error type returned by renderer APIs.
///
/// Variants are added as each increment of the first rewrite pass introduces
/// new failure modes. See `.local/renderer_plan.md` §12.
#[derive(Debug, Error)]
pub enum RendererError {}
