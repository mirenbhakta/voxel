//! MDI draw node: register a raster pass that issues a multi-draw-indirect
//! with a GPU-sourced draw count, reading the [`IndirectArgs`] produced by
//! an earlier pass (typically [`cull`](super::cull::cull)).
//!
//! The graph tracks bind-group resource accesses automatically (reads,
//! since every draw-side bind group entry is read-only), plus the
//! indirect-args and count buffers (reads, bound outside any bind group
//! as `INDIRECT` usage) and the colour / depth attachments (writes).

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

// --- mdi_draw ---

/// Register an MDI raster pass into `graph`.
///
/// Declares resource accesses automatically from `bind_group`'s pipeline
/// bind entries (typically all read-only entries for the draw side) plus
/// explicit reads of
/// `indirect.indirect` and `indirect.count` (these travel with `INDIRECT`
/// usage and are not bound through the layout), and writes of `color`
/// and `depth` (both cleared at pass start).  Returns the new versioned
/// attachment handles.  Dispatches via `multi_draw_indirect_count` — the
/// draw count is read from `indirect.count` on the GPU, capped at
/// `indirect.max_draws`.
pub fn mdi_draw(
    graph      : &mut RenderGraph,
    pipeline   : &Arc<RenderPipeline>,
    bind_group : BindGroupHandle,
    indirect   : &IndirectArgs,
    _args      : &DrawArgs,
    color      : TextureHandle,
    depth      : TextureHandle,
)
    -> (TextureHandle, TextureHandle)
{
    let indirect_h = indirect.indirect;
    let count_h    = indirect.count;
    let max_draws  = indirect.max_draws;

    graph.add_pass("mdi_draw", |pass| {
        pass.use_bind_group(bind_group);
        pass.read_buffer(indirect_h);
        pass.read_buffer(count_h);
        let color_v = pass.write_texture(color);
        let depth_v = pass.write_texture(depth);

        let pipeline = Arc::clone(pipeline);

        pass.execute(move |ctx| {
            let bg           = ctx.resources.bind_group(bind_group);
            let indirect_buf = ctx.resources.buffer(indirect_h);
            let count_buf    = ctx.resources.buffer(count_h);
            let color_view   = ctx.resources.texture_view(color_v);
            let depth_view   = ctx.resources.texture_view(depth_v);

            ctx.commands.raster_pass(
                &RasterPassDesc {
                    label : "mdi_draw",
                    color : &[ColorAttachment::clear(color_view, [0.0, 0.0, 0.0, 1.0])],
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

        (color_v, depth_v)
    })
}
