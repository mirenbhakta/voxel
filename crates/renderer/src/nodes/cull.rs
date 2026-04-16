//! Cull node: register a compute pass that populates an
//! [`IndirectArgs`] pair.
//!
//! Takes the pre-allocated [`IndirectArgs`] as input and returns the
//! updated (versioned) [`IndirectArgs`] for the next pass to consume.
//! Cull does not own allocation — the outer pipeline calls
//! [`IndirectArgs::new`] once and threads the value through.

use std::sync::Arc;

use crate::graph::{BindGroupHandle, RenderGraph};
use crate::pipeline::ComputePipeline;

use super::indirect_args::IndirectArgs;

// --- CullArgs ---

/// Configuration for a [`cull`] invocation.
///
/// Grows as cull acquires flags (frustum toggle, direction masks, etc.).
#[derive(Default)]
pub struct CullArgs {
    /// Dispatch grid handed to `ctx.commands.dispatch`.
    pub workgroups : [u32; 3],
}

// --- cull ---

/// Register a cull compute pass into `graph`.
///
/// Declares resource accesses automatically from `bind_group`'s pipeline
/// bind entries — every read-only binding is registered as a read at its
/// current version, and every read-write binding is registered as a write at
/// a freshly-minted version.  Callers therefore do not pass separate
/// read/write lists.
///
/// The cull node owns the indirect-buffer binding internally via a set-1
/// bind group built from the pipeline's reflected set-1 layout — callers do
/// not include the indirect buffer in their `bind_group`.
///
/// `indirect.count` is passed through unchanged: when the count buffer
/// is a fixed persistent value that the cull shader does not touch, it
/// should not appear in `bind_group` at all, avoiding a spurious write.
///
/// The cull shader is responsible for initialising the indirect args
/// before its atomic increment — transient indirect buffers hand back
/// undefined contents from the pool.
pub fn cull(
    graph      : &mut RenderGraph,
    pipeline   : &Arc<ComputePipeline>,
    bind_group : BindGroupHandle,
    args       : &CullArgs,
    indirect   : IndirectArgs,
)
    -> IndirectArgs
{
    let internal_bg = graph.create_internal_bind_group(
        "cull_internal",
        pipeline.as_ref(),
        &[(0, indirect.indirect.into())],
    );

    graph.add_pass("cull", |pass| {
        pass.use_bind_group(bind_group);
        let indirect_v = pass.write_buffer(indirect.indirect);
        let pipeline   = Arc::clone(pipeline);
        let workgroups = args.workgroups;

        pass.execute(move |ctx| {
            let ext_bg = ctx.resources.bind_group(bind_group);
            let int_bg = ctx.resources.bind_group(internal_bg);
            ctx.commands.dispatch(&pipeline, &[ext_bg, int_bg], workgroups, &[]);
        });

        IndirectArgs {
            indirect  : indirect_v,
            count     : indirect.count,
            max_draws : indirect.max_draws,
        }
    })
}
