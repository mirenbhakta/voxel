//! Indirect draw arguments — the shared input/output type for cull and
//! draw nodes.
//!
//! Carries the versioned handles for an MDI indirect-args buffer and its
//! matching GPU-sourced draw-count buffer, plus the capacity bound.  The
//! outer pipeline allocates once via [`IndirectArgs::new`] and threads the
//! value through cull → any intermediate passes → draw; each node that
//! writes returns an updated [`IndirectArgs`] carrying the new versions.

use crate::graph::{BufferDesc, BufferHandle, RenderGraph};

/// Versioned handles for an MDI indirect-args + count buffer pair.
///
/// `indirect` is the `DrawIndirectArgs[]` buffer, `count` is the single
/// `u32` GPU-sourced draw count consumed by
/// [`multi_draw_indirect_count`](crate::commands::RasterPass::multi_draw_indirect_count).
/// `max_draws` is the capacity of `indirect` — the `max_count` ceiling at
/// the draw callsite.
#[derive(Clone, Copy)]
pub struct IndirectArgs {
    /// Indirect-args buffer.  Written by cull, read by draw.
    pub indirect  : BufferHandle,
    /// Single-`u32` draw count.  Written by cull, read by draw.
    pub count     : BufferHandle,
    /// Capacity of `indirect` in draw entries.  Doubles as the `max_count`
    /// ceiling at the draw callsite.
    pub max_draws : u32,
}

impl IndirectArgs {
    /// Allocate both transient buffers for an indirect-args pair.
    ///
    /// `max_draws` sets the indirect buffer's capacity (each entry is 16
    /// bytes — four `u32`s, the `DrawIndirectArgs` layout).  The count
    /// buffer is a single `u32`; its `COPY_DST` usage is included so the
    /// caller (or the cull pass itself) can clear it before the atomic
    /// increment.
    pub fn new(graph: &mut RenderGraph, max_draws: u32) -> Self {
        const INDIRECT_STRIDE : u64 = 16;

        let indirect = graph.create_buffer("indirect_args", BufferDesc {
            size  : INDIRECT_STRIDE * max_draws as u64,
            usage : wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
        });

        let count = graph.create_buffer("indirect_count", BufferDesc {
            size  : 4,
            usage : wgpu::BufferUsages::INDIRECT
                  | wgpu::BufferUsages::STORAGE
                  | wgpu::BufferUsages::COPY_DST,
        });

        Self { indirect, count, max_draws }
    }
}
