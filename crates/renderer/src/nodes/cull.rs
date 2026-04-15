//! Cull node: register a compute pass that populates an
//! [`IndirectArgs`] pair.
//!
//! Takes the pre-allocated [`IndirectArgs`] as input, writes both of its
//! buffers, and returns the updated (versioned) [`IndirectArgs`] for the
//! next pass to consume.  Cull does not own allocation — the outer
//! pipeline calls [`IndirectArgs::new`] once and threads the value
//! through.

use std::sync::Arc;

use crate::graph::{BindGroupHandle, BufferHandle, RenderGraph};
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
/// Declares writes on `indirect.indirect` and on every handle in
/// `extra_writes`, then dispatches the supplied compute pipeline.  Returns
/// an updated [`IndirectArgs`] carrying the new versioned `indirect`
/// handle for downstream passes.  `indirect.count` is passed through
/// unchanged — when the count buffer is a fixed persistent value that the
/// cull shader does not touch, it should not be declared as a write.
///
/// `extra_writes` covers cull outputs that aren't part of `IndirectArgs`
/// itself — the visibility list is the canonical example.  Callers receive
/// the versioned write handles back via the returned vector so downstream
/// passes can declare reads against the produced version.
///
/// The cull shader is responsible for initialising the indirect args
/// before its atomic increment — transient indirect buffers hand back
/// undefined contents from the pool.
pub fn cull(
    graph        : &mut RenderGraph,
    pipeline     : &Arc<ComputePipeline>,
    bind_group   : BindGroupHandle,
    args         : &CullArgs,
    indirect     : IndirectArgs,
    extra_writes : &[BufferHandle],
)
    -> (IndirectArgs, Vec<BufferHandle>)
{
    graph.add_pass("cull", |pass| {
        let indirect_v = pass.write_buffer(indirect.indirect);
        let extra_v: Vec<BufferHandle> = extra_writes
            .iter()
            .map(|&h| pass.write_buffer(h))
            .collect();
        let pipeline   = Arc::clone(pipeline);
        let workgroups = args.workgroups;

        pass.execute(move |ctx| {
            let bg = ctx.resources.bind_group(bind_group);
            ctx.commands.dispatch(&pipeline, bg, workgroups, &[]);
        });

        let out = IndirectArgs {
            indirect  : indirect_v,
            count     : indirect.count,
            max_draws : indirect.max_draws,
        };
        (out, extra_v)
    })
}
