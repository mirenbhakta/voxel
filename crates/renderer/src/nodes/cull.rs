//! Cull node: register a compute pass that populates an
//! [`IndirectArgs`] pair.
//!
//! Takes the pre-allocated [`IndirectArgs`] as input, writes both of its
//! buffers, and returns the updated (versioned) [`IndirectArgs`] for the
//! next pass to consume.  Cull does not own allocation — the outer
//! pipeline calls [`IndirectArgs::new`] once and threads the value
//! through.

use std::sync::Arc;

use crate::graph::RenderGraph;
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
/// Declares writes on both buffers in `indirect`, dispatches the supplied
/// compute pipeline, and returns an updated [`IndirectArgs`] carrying the
/// new versioned handles for downstream passes.
///
/// The cull shader is responsible for zeroing / initialising the count
/// before its atomic increment — the pool hands back undefined contents.
pub fn cull(
    graph      : &mut RenderGraph,
    pipeline   : &Arc<ComputePipeline>,
    bind_group : &wgpu::BindGroup,
    args       : &CullArgs,
    indirect   : IndirectArgs,
)
    -> IndirectArgs
{
    graph.add_pass("cull", |pass| {
        let indirect_v = pass.write_buffer(indirect.indirect);
        let count_v    = pass.write_buffer(indirect.count);
        let pipeline   = Arc::clone(pipeline);
        let bind_group = bind_group.clone();
        let workgroups = args.workgroups;

        pass.execute(move |ctx| {
            ctx.commands.dispatch(&pipeline, &bind_group, workgroups, &[]);
        });

        IndirectArgs {
            indirect  : indirect_v,
            count     : count_v,
            max_draws : indirect.max_draws,
        }
    })
}
