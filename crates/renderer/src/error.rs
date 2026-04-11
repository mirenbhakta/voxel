//! Error types for the renderer crate.

use thiserror::Error;

/// Top-level error type returned by renderer APIs.
///
/// Variants are added as each increment of the first rewrite pass introduces
/// new failure modes. See `.local/renderer_plan.md` §12.
#[derive(Debug, Error)]
pub enum RendererError {
    /// [`FrameCount::new`](crate::frame::FrameCount::new) was called with a
    /// value outside `[FrameCount::MIN, FrameCount::MAX]`.
    #[error("invalid frame count {requested}: must be in {min}..={max}")]
    InvalidFrameCount {
        requested: u32,
        min: u32,
        max: u32,
    },

    /// `wgpu::Instance::request_adapter` returned no compatible adapter. On
    /// headless Linux this typically means Vulkan is not available on the
    /// machine (install vulkan drivers / vulkan-sdk).
    #[error("no wgpu adapter found (is Vulkan available on this machine?)")]
    NoCompatibleAdapter,

    /// `wgpu::Adapter::request_device` failed. The inner message is whatever
    /// wgpu produced; there is no structured form because wgpu's
    /// `RequestDeviceError` is a thin opaque type.
    #[error("failed to create wgpu device: {0}")]
    DeviceCreationFailed(String),

    /// SPIR-V reflection encountered a structural error in the module.
    ///
    /// Covers bugs in the reflection surface itself — malformed SPV that the
    /// parser rejected, a module with no entry point matching the requested
    /// name, or a missing `LocalSize` execution mode that is required for
    /// workgroup-size reflection. These are build-time programming errors (bad
    /// shader, wrong entry-point name, unsupported execution-mode form) rather
    /// than recoverable runtime conditions; see `.local/renderer_plan.md` §7.
    ///
    /// `LocalSizeId` (spec-constant workgroup sizes) produces this variant
    /// with a message advising use of literal `numthreads` instead — not yet
    /// supported in the first rewrite pass.
    #[error("shader reflection failed: {0}")]
    ShaderReflectionFailed(String),
}
