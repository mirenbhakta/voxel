//! MDI draw node: register a raster pass that issues a multi-draw-indirect
//! with a GPU-sourced draw count, reading the [`IndirectArgs`] produced by
//! an earlier pass (typically [`cull`](super::cull::cull)).
//!
//! The graph tracks bind-group resource accesses automatically (reads,
//! since every draw-side bind group entry is read-only), plus the
//! indirect-args and count buffers (reads, bound outside any bind group
//! as `INDIRECT` usage) and the colour / depth attachments (writes).
//!
//! # MRT
//!
//! The colour side takes a slice so callers can drive an MRT pipeline.
//! This is the minimal change that keeps the node a plain "issue one MDI
//! draw into N colour attachments + 1 depth" primitive and leaves the MRT
//! specifics (which slot carries what) to the pipeline the caller passes
//! in. The alternative — a specialised `subchunk_draw` wrapper — was
//! rejected because the MRT logic here is zero node-specific behaviour:
//! the slice is forwarded untouched to [`RasterPassDesc::color`], which
//! already takes `&[ColorAttachment]`. Forking a second node to hide a
//! slice length would grow the node count without adding expressive
//! power. Callers with a single colour target pass a one-element slice.

use std::sync::Arc;

use crate::commands::{ColorAttachment, DepthAttachment, RasterPassDesc};
use crate::graph::{BindGroupHandle, RenderGraph, TextureHandle};
use crate::pipeline::RenderPipeline;

use super::indirect_args::IndirectArgs;

// --- DrawArgs ---

/// Configuration for a [`mdi_draw`] invocation.
///
/// Grows as the draw pass acquires clear overrides, viewport scissors,
/// pipeline selection, etc.
#[derive(Default)]
pub struct DrawArgs {}

// --- ColorTarget ---

/// A single colour MRT slot for a [`mdi_draw`] invocation: the texture to
/// render into and the clear value to apply at pass start.
///
/// Slot ordinals correspond 1:1 with the pipeline's `color_targets` —
/// `colors[0]` maps to `SV_Target0`, `colors[1]` to `SV_Target1`, etc.
///
/// The clear value is interpreted against the target's texel format by
/// wgpu: RGBA floats for float / UNORM / SRGB targets; bitwise integer
/// cast for integer targets (e.g. pack an `0xFFFFFFFF` sentinel for an
/// R32_UINT target by passing `[4294967295.0, 0.0, 0.0, 0.0]`).
#[derive(Clone, Copy)]
pub struct ColorTarget {
    pub texture : TextureHandle,
    pub clear   : [f64; 4],
}

impl ColorTarget {
    /// Black-clear helper for RGBA targets.
    pub fn black(texture: TextureHandle) -> Self {
        Self { texture, clear: [0.0, 0.0, 0.0, 1.0] }
    }
}

// --- mdi_draw ---

/// Register an MDI raster pass into `graph`.
///
/// Declares resource accesses automatically from `bind_group`'s pipeline
/// bind entries (typically all read-only entries for the draw side) plus
/// explicit reads of `indirect.indirect` and `indirect.count` (these
/// travel with `INDIRECT` usage and are not bound through the layout),
/// and writes of each `colors` entry and `depth` (all cleared at pass
/// start).  Dispatches via `multi_draw_indirect_count` — the draw count
/// is read from `indirect.count` on the GPU, capped at
/// `indirect.max_draws`.
///
/// Returns the versioned handles of the written attachments: one per
/// entry in `colors`, in input order, plus the depth handle.
#[allow(clippy::too_many_arguments)]
pub fn mdi_draw(
    graph      : &mut RenderGraph,
    pipeline   : &Arc<RenderPipeline>,
    bind_group : BindGroupHandle,
    indirect   : &IndirectArgs,
    _args      : &DrawArgs,
    colors     : &[ColorTarget],
    depth      : TextureHandle,
)
    -> (Vec<TextureHandle>, TextureHandle)
{
    let indirect_h = indirect.indirect;
    let count_h    = indirect.count;
    let max_draws  = indirect.max_draws;

    // Capture the per-slot clear values in an owned vec so the execute
    // closure can rebuild `ColorAttachment` entries (which borrow from the
    // texture-view store) without borrowing from the outer graph-build
    // stack.
    let clears: Vec<[f64; 4]> = colors.iter().map(|c| c.clear).collect();

    graph.add_pass("mdi_draw", |pass| {
        pass.use_bind_group(bind_group);
        pass.read_buffer(indirect_h);
        pass.read_buffer(count_h);

        let color_vs: Vec<TextureHandle> =
            colors.iter().map(|c| pass.write_texture(c.texture)).collect();
        let depth_v = pass.write_texture(depth);

        let pipeline = Arc::clone(pipeline);
        // Move the per-slot clear values + versioned writes into the
        // execute closure; they are needed at record time to build the
        // `ColorAttachment` slice.
        let color_vs_for_exec = color_vs.clone();

        pass.execute(move |ctx| {
            let bg           = ctx.resources.bind_group(bind_group);
            let indirect_buf = ctx.resources.buffer(indirect_h);
            let count_buf    = ctx.resources.buffer(count_h);
            let depth_view   = ctx.resources.texture_view(depth_v);

            // Fresh per-invocation vec — at most a handful of targets, so
            // the heap hit is negligible. Kept local so the borrow of each
            // texture view lives exactly for the raster pass.
            let color_views: Vec<&wgpu::TextureView> = color_vs_for_exec
                .iter()
                .map(|&h| ctx.resources.texture_view(h))
                .collect();

            let color_attachments: Vec<ColorAttachment<'_>> = color_views
                .iter()
                .zip(clears.iter())
                .map(|(view, clear)| ColorAttachment::clear(view, *clear))
                .collect();

            ctx.commands.raster_pass(
                &RasterPassDesc {
                    label : "mdi_draw",
                    color : &color_attachments,
                    depth : Some(DepthAttachment::clear(depth_view, 1.0)),
                },
                |rp| {
                    rp.multi_draw_indirect_count(
                        &pipeline, &[bg],
                        indirect_buf, 0,
                        count_buf,    0,
                        max_draws,
                    );
                },
            );
        });

        (color_vs, depth_v)
    })
}
