//! Render graph compilation: topological sort, dead-pass culling, barrier
//! derivation.
//!
//! All logic here is pure — no wgpu types, no GPU interaction.  The
//! [`compile`] function takes the graph's declared passes and resource
//! access patterns, and produces an execution order with barriers and
//! culling information.

use std::collections::VecDeque;

use super::resource::{Access, ResourceHandle, ResourceId};

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
    /// The builder API prevents cycles by construction (write handles are
    /// returned sequentially and cannot refer forward), so this variant exists
    /// only as a defensive check in the compile algorithm.
    #[error("render graph contains a dependency cycle")]
    Cycle,
}

// --- PassAccess ---

/// A single (handle, access) pair recorded by a pass.
#[derive(Clone, Copy)]
pub(super) struct PassAccess {
    pub handle : ResourceHandle,
    pub access : Access,
}

// --- compile ---

/// Compile the render graph.
///
/// Accepts per-pass resource declarations and the set of live (resource, version)
/// pairs that are graph outputs.  Returns an execution order (after dead-pass
/// culling), barrier metadata, and the number of culled passes.
///
/// ## Algorithm
///
/// 1. **Producer map** — maps `(resource, version)` to the pass that wrote it.
///    Version 0 (initial import / creation) has no producer; reads of version 0
///    add no edge.
/// 2. **Dependency DAG** — for each read of (R, V): add edge from producer(R, V)
///    to the reading pass (RAW).  For each write of (R, V_new): add edges from
///    all passes that accessed (R, V_new-1) into the writer (WAR + WAW) — this
///    enforces physical-resource ordering even though versions are logically
///    independent.
/// 3. **Topological sort** — Kahn's algorithm.  Fails with
///    [`CompileError::Cycle`] if the graph contains a cycle.
/// 4. **Dead-pass culling** — backward reachability from the passes that
///    produced the output (resource, version) pairs.
/// 5. **Barrier derivation** — walks execution order and emits a barrier
///    whenever a resource (by identity, ignoring version) transitions between
///    incompatible access kinds (any transition except Read → Read).
pub(super) fn compile(
    pass_accesses    : &[Vec<PassAccess>],
    output_versions  : &[ResourceHandle],
    resource_count   : usize,
)
    -> Result<CompileResult, CompileError>
{
    let pass_count = pass_accesses.len();

    // -- 1. Build producer map and per-resource access tracking --

    // Maps a versioned handle → producing pass index.
    // Version 0 is never in this map (no producer).
    let mut producer: std::collections::HashMap<ResourceHandle, usize> =
        std::collections::HashMap::new();

    // Per-resource: last writer pass and all passes that accessed the previous
    // version (for WAR + WAW dependency insertion).
    let mut last_writer         : Vec<Option<usize>> = vec![None; resource_count];
    let mut readers_since_write : Vec<Vec<usize>>    = vec![vec![]; resource_count];

    for (pass, accesses) in pass_accesses.iter().enumerate() {
        for &pa in accesses {
            if pa.access == Access::Write {
                producer.insert(pa.handle, pass);
            }
        }
    }

    // -- 2. Build dependency DAG --

    let mut edges     : Vec<Vec<usize>> = vec![vec![]; pass_count];
    let mut in_degree : Vec<usize>      = vec![0; pass_count];

    for (pass, accesses) in pass_accesses.iter().enumerate() {
        for &pa in accesses {
            let r = pa.handle.resource as usize;

            match pa.access {
                Access::Read => {
                    // RAW: depend on the pass that produced this handle.
                    // Version 0 has no producer — no edge.
                    if pa.handle.version > 0
                        && let Some(&writer) = producer.get(&pa.handle)
                    {
                        add_edge(&mut edges, &mut in_degree, writer, pass);
                    }

                    readers_since_write[r].push(pass);
                }

                Access::Write => {
                    // WAW: depend on the previous writer of this physical resource.
                    if let Some(prev_writer) = last_writer[r] {
                        add_edge(&mut edges, &mut in_degree, prev_writer, pass);
                    }

                    // WAR: depend on all readers of the previous version.
                    for &reader in &readers_since_write[r] {
                        add_edge(&mut edges, &mut in_degree, reader, pass);
                    }

                    last_writer[r] = Some(pass);
                    readers_since_write[r].clear();
                }
            }
        }
    }

    // -- 3. Topological sort (Kahn's algorithm) --

    let mut queue = VecDeque::new();

    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
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

    // -- 4. Dead-pass culling --

    let mut live = vec![false; pass_count];

    // Seed: the pass that produced each output handle is live.
    for &handle in output_versions {
        if handle.version == 0 {
            // Version 0 has no producer; no pass is seeded.
            continue;
        }

        if let Some(&writer) = producer.get(&handle) {
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

    // -- 5. Barrier derivation --

    // Per-resource (by identity, version-insensitive): last (access kind, execution-order position).
    let mut res_last: Vec<Option<(Access, usize)>> = vec![None; resource_count];
    let mut barriers = Vec::new();

    for (pos, &pass) in execution_order.iter().enumerate() {
        // Collect the effective access per resource for this pass.
        // Write dominates Read when both appear on the same resource.
        let mut effective: Vec<(usize, Access)> = Vec::new();

        for &pa in &pass_accesses[pass] {
            let r = pa.handle.resource as usize;

            if let Some(entry) = effective.iter_mut().find(|(i, _)| *i == r) {
                if pa.access == Access::Write {
                    entry.1 = Access::Write;
                }
            }
            else {
                effective.push((r, pa.access));
            }
        }

        for &(r, access) in &effective {
            if let Some((prev_access, prev_pos)) = res_last[r] {
                // Read → Read needs no barrier.
                if !(prev_access == Access::Read && access == Access::Read) {
                    barriers.push(Barrier {
                        resource : ResourceId(r as u32),
                        src      : prev_access,
                        dst      : access,
                        after    : prev_pos,
                        before   : pos,
                    });
                }
            }

            res_last[r] = Some((access, pos));
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
