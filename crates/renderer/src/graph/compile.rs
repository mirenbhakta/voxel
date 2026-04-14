//! Render graph compilation: topological sort, dead-pass culling, barrier
//! derivation.
//!
//! All logic here is pure — no wgpu types, no GPU interaction.  The
//! [`compile`] function takes the graph's declared passes and resource
//! access patterns, and produces an execution order with barriers and
//! culling information.

use std::collections::VecDeque;

use super::resource::{Access, ResourceId};

// --- Barrier ---

/// A resource transition that the runtime must enforce between two passes.
///
/// Indices in [`after`](Self::after) and [`before`](Self::before) refer to
/// positions in the compiled execution order (not the original declaration
/// order).
#[derive(Clone, Debug)]
pub struct Barrier {
    /// The resource that transitions.
    pub resource : ResourceId,
    /// Access kind in the earlier pass.
    pub src      : Access,
    /// Access kind in the later pass.
    pub dst      : Access,
    /// Position in execution order of the pass whose access precedes the
    /// transition.
    pub after    : usize,
    /// Position in execution order of the pass whose access follows the
    /// transition.
    pub before   : usize,
}

// --- CompileResult ---

/// Internal output of graph compilation, consumed by
/// [`RenderGraph::compile`](super::RenderGraph::compile).
pub(super) struct CompileResult {
    /// Original pass indices in execution order (after culling).
    pub execution_order : Vec<usize>,
    /// Barriers between adjacent accesses in execution order.
    pub barriers        : Vec<Barrier>,
    /// Number of passes culled.
    pub culled_count    : usize,
}

// --- CompileError ---

/// Error produced when render graph compilation fails.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CompileError {
    /// The dependency graph contains a cycle, preventing topological sort.
    ///
    /// The builder API prevents cycles by construction (handles are issued
    /// sequentially and cannot refer forward), so this variant exists only
    /// as a defensive check in the compile algorithm.
    #[error("render graph contains a dependency cycle")]
    Cycle,
}

// --- compile ---

/// Compile the render graph.
///
/// Accepts per-pass resource declarations and output-resource markers.
/// Returns an execution order (after dead-pass culling), barrier metadata,
/// and the number of culled passes.
///
/// ## Algorithm
///
/// 1. **Dependency DAG** — built from resource access declarations.  For
///    each resource, the function tracks the last writer and all readers
///    since the last write.  Edges encode RAW, WAW, and WAR hazards.
/// 2. **Topological sort** — Kahn's algorithm.  Fails with
///    [`CompileError::Cycle`] if the graph contains a cycle.
/// 3. **Dead-pass culling** — backward reachability from output-resource
///    writers.  A pass is live if it transitively contributes to an output.
///    Note: WAW-chained passes are kept alive conservatively (the
///    superseded writer is not culled even though its output is
///    overwritten).
/// 4. **Barrier derivation** — walks execution order and emits a barrier
///    whenever a resource transitions between incompatible access kinds
///    (any transition except Read → Read).
pub(super) fn compile(
    pass_accesses    : &[Vec<(u32, Access)>],
    output_resources : &[ResourceId],
    resource_count   : usize,
)
    -> Result<CompileResult, CompileError>
{
    let pass_count = pass_accesses.len();

    // -- 1. Build dependency DAG --

    let mut edges     : Vec<Vec<usize>> = vec![vec![]; pass_count];
    let mut in_degree : Vec<usize>      = vec![0; pass_count];

    // Per-resource tracking: last writer and readers since last write.
    let mut last_writer        : Vec<Option<usize>> = vec![None; resource_count];
    let mut readers_since_write: Vec<Vec<usize>>    = vec![vec![]; resource_count];

    for pass in 0..pass_count {
        // Explicit read/write declarations.
        for &(idx, access) in &pass_accesses[pass] {
            let b = idx as usize;

            match access {
                Access::Read => {
                    // RAW: depend on last writer.
                    if let Some(writer) = last_writer[b] {
                        add_edge(&mut edges, &mut in_degree, writer, pass);
                    }

                    readers_since_write[b].push(pass);
                }

                Access::Write => {
                    // WAW: depend on previous writer.
                    if let Some(prev) = last_writer[b] {
                        add_edge(&mut edges, &mut in_degree, prev, pass);
                    }

                    // WAR: depend on all readers since last write.
                    for &reader in &readers_since_write[b] {
                        add_edge(&mut edges, &mut in_degree, reader, pass);
                    }

                    last_writer[b] = Some(pass);
                    readers_since_write[b].clear();
                }
            }
        }
    }

    // -- 2. Topological sort (Kahn's algorithm) --

    let mut queue = VecDeque::new();

    for i in 0..pass_count {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut topo_order = Vec::with_capacity(pass_count);

    while let Some(node) = queue.pop_front() {
        topo_order.push(node);

        for &neighbor in &edges[node] {
            in_degree[neighbor] -= 1;

            if in_degree[neighbor] == 0 {
                queue.push_back(neighbor);
            }
        }
    }

    if topo_order.len() != pass_count {
        return Err(CompileError::Cycle);
    }

    // -- 3. Dead-pass culling --

    let mut live = vec![false; pass_count];

    // Seed: last writer of each output resource is live.
    for res in output_resources {
        if let Some(writer) = last_writer[res.0 as usize] {
            live[writer] = true;
        }
    }

    // Build reverse edges for backward reachability.
    let mut reverse_edges: Vec<Vec<usize>> = vec![vec![]; pass_count];

    for (src, dsts) in edges.iter().enumerate() {
        for &dst in dsts {
            reverse_edges[dst].push(src);
        }
    }

    // Walk topo order in reverse: live passes make their dependencies live.
    for &pass in topo_order.iter().rev() {
        if live[pass] {
            for &dep in &reverse_edges[pass] {
                live[dep] = true;
            }
        }
    }

    let culled_count = live.iter().filter(|&&l| !l).count();

    let execution_order: Vec<usize> = topo_order.into_iter()
        .filter(|&i| live[i])
        .collect();

    // -- 4. Barrier derivation --

    // Per-resource: last (access kind, execution-order position).
    let mut buf_last: Vec<Option<(Access, usize)>> = vec![None; resource_count];
    let mut barriers = Vec::new();

    for (pos, &pass) in execution_order.iter().enumerate() {
        // Collect the effective access per resource for this pass.
        // Write dominates Read when both appear on the same resource.
        let mut effective: Vec<(usize, Access)> = Vec::new();

        for &(idx, access) in &pass_accesses[pass] {
            let b = idx as usize;

            if let Some(entry) = effective.iter_mut().find(|(i, _)| *i == b) {
                if access == Access::Write {
                    entry.1 = Access::Write;
                }
            }
            else {
                effective.push((b, access));
            }
        }

        for &(b, access) in &effective {
            if let Some((prev_access, prev_pos)) = buf_last[b] {
                // Read → Read needs no barrier.
                if !(prev_access == Access::Read && access == Access::Read) {
                    barriers.push(Barrier {
                        resource : ResourceId(b as u32),
                        src      : prev_access,
                        dst      : access,
                        after    : prev_pos,
                        before   : pos,
                    });
                }
            }

            buf_last[b] = Some((access, pos));
        }
    }

    Ok(CompileResult { execution_order, barriers, culled_count })
}

/// Add a directed edge, deduplicating.
fn add_edge(
    edges     : &mut [Vec<usize>],
    in_degree : &mut [usize],
    from      : usize,
    to        : usize,
) {
    if from == to {
        return;
    }

    if !edges[from].contains(&to) {
        edges[from].push(to);
        in_degree[to] += 1;
    }
}
