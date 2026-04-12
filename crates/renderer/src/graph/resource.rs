//! Resource handles and access types for the render graph.

// --- BufferDesc ---

/// Describes a transient buffer's requirements for pool allocation.
#[derive(Clone, Copy, Debug)]
pub struct BufferDesc {
    /// Buffer size in bytes.
    pub size  : u64,
    /// Required wgpu usage flags.
    pub usage : wgpu::BufferUsages,
}

// --- BufferHandle ---

/// Opaque handle to a buffer resource in the render graph.
///
/// Handles are [`Copy`] so pass execute closures can capture them without
/// borrowing the graph builder.  The inner index is meaningful only to the
/// [`RenderGraph`](super::RenderGraph) that issued it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BufferHandle(pub(super) u32);

// --- Access ---

/// How a pass accesses a buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Access {
    /// The pass reads the buffer.
    Read,
    /// The pass writes (or overwrites) the buffer.
    Write,
}
