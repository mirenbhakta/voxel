//! Present-blit node: register a raster pass that samples a source
//! texture and writes it into a destination texture (typically the
//! swapchain).
//!
//! The pass dispatches three vertices against the fullscreen-triangle VS
//! at `shaders/present_blit.vs.hlsl`; the PS at
//! `shaders/present_blit.ps.hlsl` `.Load`s from the source texture with
//! no sampler (the source and destination extents match 1:1). The
//! destination is cleared to opaque black before the draw so uninitialised
//! regions — if any exist at task boundaries — do not leak stale content;
//! the PS overwrites every pixel under the 1:1 assumption.
//!
//! This is the terminal pass of phase-1 sub-chunk rendering: the shade
//! compute writes an `Rgba8Unorm` linear transient, and this blit gets
//! those pixels into the `Bgra8UnormSrgb` swapchain with the hardware
//! performing the linear→sRGB encode.

use std::sync::Arc;

use crate::commands::{ColorAttachment, RasterPassDesc};
use crate::graph::{RenderGraph, TextureHandle};
use crate::pipeline::RenderPipeline;

// --- present_blit ---

/// Register a fullscreen-triangle blit pass that samples `src` and writes
/// to `dst`.
///
/// The bind group is built inside this node from the pipeline's reflected
/// layout, wiring `src` at slot 0. The pass declares `src` as a read
/// (implicit via the bind group's `SampledTexture` entry) and `dst` as a
/// write (via the colour attachment); the graph handles barriers.
///
/// `pipeline` must be the phase-1.5 present-blit render pipeline — see
/// [`WorldRenderer::present_blit_pipeline`](crate::WorldRenderer::present_blit_pipeline).
/// The pipeline's sole colour target must match `dst`'s format or wgpu
/// rejects the raster pass at record time.
///
/// Returns the post-write versioned handle for `dst` — callers thread
/// this out if a downstream pass needs to observe the blit's output (for
/// the swapchain case the graph's `present()` seeds the output version
/// automatically at compile time, so callers typically discard the
/// return).
pub fn present_blit(
    graph    : &mut RenderGraph,
    pipeline : &Arc<RenderPipeline>,
    src      : TextureHandle,
    dst      : TextureHandle,
)
    -> TextureHandle
{
    let bg = graph.create_bind_group(
        "present_blit_bg",
        pipeline.as_ref(),
        None,
        &[(0, src.into())],
    );

    graph.add_pass("present_blit", |pass| {
        // Read of `src` is auto-declared via the `SampledTexture` entry in
        // the bind group. `dst` is the colour attachment and must be
        // declared as an explicit write here — colour attachments are not
        // part of the bind group's reflected accesses.
        pass.use_bind_group(bg);
        let dst_v   = pass.write_texture(dst);
        let pipeline = Arc::clone(pipeline);

        pass.execute(move |ctx| {
            let bg_ref   = ctx.resources.bind_group(bg);
            let dst_view = ctx.resources.texture_view(dst_v);

            ctx.commands.raster_pass(
                &RasterPassDesc {
                    label : "present_blit",
                    color : &[ColorAttachment::clear(dst_view, [0.0, 0.0, 0.0, 1.0])],
                    depth : None,
                },
                |rp| {
                    rp.draw(&pipeline, &[bg_ref], 0..3, 0..1);
                },
            );
        });

        dst_v
    })
}
