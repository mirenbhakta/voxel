//! Render graph — three-phase (build / compile / execute) scheduling for
//! GPU work.
//!
//! # Phases
//!
//! 1. **Build** — declare passes and their resource usage via
//!    [`RenderGraph::add_pass`].  No GPU work happens.  Closures capture
//!    only [`Copy`] resource handles, not application state.
//! 2. **Compile** — [`RenderGraph::compile`] resolves execution order
//!    (topological sort), inserts barriers, and culls dead passes.
//! 3. **Execute** — [`CompiledGraph::execute`] walks the resolved order and
//!    calls each pass's closure sequentially with a [`PassContext`].
//!
//! The graph is rebuilt from scratch each frame.

mod compile;
mod pool;
mod resource;

pub use compile::{Barrier, CompileError};
pub use pool::{BufferPool, PendingRelease, TexturePool};
pub use resource::{Access, BufferDesc, BufferHandle, ResourceId, TextureDesc, TextureHandle};

use crate::commands::Commands;
use crate::device::FrameEncoder;
use crate::frame::FrameIndex;

/// Type alias for the per-pass execute closure stored in [`PassData`].
///
/// Factored out to avoid triggering the `clippy::type_complexity` lint on
/// every field declaration.
type ExecuteFn = Option<Box<dyn FnOnce(&mut PassContext<'_>) + 'static>>;

// --- ResourceEntry ---

enum ResourceEntry {
    Buffer(wgpu::Buffer),
    Texture(wgpu::Texture, wgpu::TextureView),
}

// --- ResourceMap ---

/// Resolved resource handles available during pass execution.
///
/// Maps render-graph handles to their backing wgpu resources.
/// Passed to each pass's execute closure via [`PassContext`] so it can look
/// up actual GPU resources by handle rather than capturing raw wgpu types.
pub struct ResourceMap {
    entries: Vec<Option<ResourceEntry>>,
}

impl ResourceMap {
    /// Look up the backing GPU buffer for a render graph handle.
    ///
    /// Version is ignored — lookup is by resource identity.
    ///
    /// # Panics
    ///
    /// Panics if the handle was not bound via [`CompiledGraph::bind`].
    /// This is a programmer error (invariant: bind all imported handles
    /// before execute).
    pub fn buffer(&self, handle: BufferHandle) -> &wgpu::Buffer {
        match self.entries[handle.resource as usize]
            .as_ref()
            .unwrap_or_else(|| panic!(
                "buffer handle (resource={}) not bound — \
                 call CompiledGraph::bind() before execute",
                handle.resource,
            ))
        {
            ResourceEntry::Buffer(b) => b,
            _ => panic!("handle (resource={}) is a texture, not a buffer", handle.resource),
        }
    }

    /// Look up the backing GPU texture for a render graph handle.
    ///
    /// Version is ignored — lookup is by resource identity.
    ///
    /// # Panics
    ///
    /// Panics if the handle was not bound via [`CompiledGraph::bind_texture`].
    pub fn texture(&self, handle: TextureHandle) -> &wgpu::Texture {
        match self.entries[handle.resource as usize]
            .as_ref()
            .unwrap_or_else(|| panic!(
                "texture handle (resource={}) not bound — \
                 call CompiledGraph::bind_texture() before execute",
                handle.resource,
            ))
        {
            ResourceEntry::Texture(t, _) => t,
            _ => panic!("handle (resource={}) is a buffer, not a texture", handle.resource),
        }
    }

    /// Look up the default [`wgpu::TextureView`] for a render graph handle.
    ///
    /// The view is created at bind time via
    /// [`TextureViewDescriptor::default`](wgpu::TextureViewDescriptor).
    /// For custom views (e.g. per-mip or per-layer), use
    /// [`texture`](Self::texture) and call
    /// [`create_view`](wgpu::Texture::create_view) directly.
    ///
    /// Version is ignored — lookup is by resource identity.
    ///
    /// # Panics
    ///
    /// Panics if the handle was not bound via [`CompiledGraph::bind_texture`].
    pub fn texture_view(&self, handle: TextureHandle) -> &wgpu::TextureView {
        match self.entries[handle.resource as usize]
            .as_ref()
            .unwrap_or_else(|| panic!(
                "texture handle (resource={}) not bound — \
                 call CompiledGraph::bind_texture() before execute",
                handle.resource,
            ))
        {
            ResourceEntry::Texture(_, v) => v,
            _ => panic!("handle (resource={}) is a buffer, not a texture", handle.resource),
        }
    }
}

// --- PassContext ---

/// Context passed to each pass's execute closure.
///
/// Fields are `pub` for disjoint-borrow ergonomics — a closure that writes
/// through `commands` and reads from `resources` simultaneously does not need
/// to work around borrow-checker interference on a single `&mut self`.
pub struct PassContext<'a> {
    /// Command recorder for this pass.
    pub commands  : Commands<'a>,
    /// Resolved resource map for this frame.
    pub resources : &'a ResourceMap,
    /// The frame index this graph is executing for.
    pub frame     : FrameIndex,
}

// --- PassData ---

/// Internal storage for a declared pass.
struct PassData {
    name       : String,
    accesses   : Vec<compile::PassAccess>,
    execute_fn : ExecuteFn,
}

// --- RenderGraph ---

/// A per-frame render graph.
///
/// Build a graph by calling [`import_buffer`](Self::import_buffer),
/// [`create_buffer`](Self::create_buffer),
/// [`create_texture`](Self::create_texture),
/// [`add_pass`](Self::add_pass), and [`mark_output`](Self::mark_output),
/// then [`compile`](Self::compile) it into a [`CompiledGraph`].
pub struct RenderGraph {
    passes                  : Vec<PassData>,
    resource_count          : u32,
    /// Per-resource current version — incremented each time a pass writes.
    resource_versions       : Vec<u32>,
    /// Output (resource, version) pairs seeded by mark_output / present.
    output_versions         : Vec<(u32, u32)>,
    transient_buffer_descs  : Vec<(BufferHandle, BufferDesc, String)>,
    transient_texture_descs : Vec<(TextureHandle, TextureDesc, String)>,
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderGraph {
    /// Create a new empty render graph.
    pub fn new() -> Self {
        Self {
            passes                  : Vec::new(),
            resource_count          : 0,
            resource_versions       : Vec::new(),
            output_versions         : Vec::new(),
            transient_buffer_descs  : Vec::new(),
            transient_texture_descs : Vec::new(),
        }
    }

    /// Import a persistent (application-owned) buffer into the graph.
    ///
    /// Returns a version-0 handle.  Passes that write to it receive a new
    /// version from [`PassBuilder::write_buffer`].
    pub fn import_buffer(&mut self) -> BufferHandle {
        self.alloc_buffer_handle()
    }

    /// Import a persistent (application-owned) texture into the graph.
    ///
    /// Returns a version-0 handle.  Passes that write to it receive a new
    /// version from [`PassBuilder::write_texture`].
    pub fn import_texture(&mut self) -> TextureHandle {
        self.alloc_texture_handle()
    }

    /// Create a transient buffer that will be allocated from the pool.
    ///
    /// Returns a version-0 handle.  Passes that write to it receive a new
    /// version from [`PassBuilder::write_buffer`].
    ///
    /// `name` is used as the wgpu debug label on the allocated buffer.
    pub fn create_buffer(&mut self, name: &str, desc: BufferDesc) -> BufferHandle {
        let handle = self.alloc_buffer_handle();
        self.transient_buffer_descs.push((handle, desc, name.to_string()));
        handle
    }

    /// Create a transient texture that will be allocated from the pool.
    ///
    /// Returns a version-0 handle.  Passes that write to it receive a new
    /// version from [`PassBuilder::write_texture`].
    ///
    /// `name` is used as the wgpu debug label on the allocated texture.
    pub fn create_texture(&mut self, name: &str, desc: TextureDesc) -> TextureHandle {
        let handle = self.alloc_texture_handle();
        self.transient_texture_descs.push((handle, desc, name.to_string()));
        handle
    }

    /// Mark a buffer version as a graph output.
    ///
    /// The pass that wrote `handle`'s version — and all passes it
    /// transitively depends on — will not be culled during compilation.
    pub fn mark_output(&mut self, handle: BufferHandle) {
        self.output_versions.push((handle.resource, handle.version));
    }

    /// Mark a texture version as the frame output (swapchain present target).
    ///
    /// Semantically distinct from [`mark_output`](Self::mark_output): this
    /// signals which texture the frame loop should present after submit.
    /// Mechanically it seeds dead-pass culling, just as `mark_output` does.
    pub fn present(&mut self, handle: TextureHandle) {
        self.output_versions.push((handle.resource, handle.version));
    }

    /// Declare a new pass.
    ///
    /// The closure receives a [`PassBuilder`] for declaring resource usage
    /// (reads, writes) and setting the execute closure.  Write declarations
    /// return new versioned handles; read declarations pin a dependency on the
    /// version supplied.  The return value of `f` is forwarded to the caller —
    /// use it to thread versioned write handles out of the pass.
    pub fn add_pass<R>(
        &mut self,
        name : &str,
        f    : impl FnOnce(&mut PassBuilder<'_>) -> R,
    )
        -> R
    {
        let pass_index = self.passes.len();

        self.passes.push(PassData {
            name       : name.to_string(),
            accesses   : Vec::new(),
            execute_fn : None,
        });

        let mut builder = PassBuilder { graph: self, pass_index };
        f(&mut builder)
    }

    /// Compile the graph: resolve execution order, cull dead passes,
    /// derive barriers.
    ///
    /// Consumes the graph and returns a [`CompiledGraph`] ready for
    /// execution.
    pub fn compile(self) -> Result<CompiledGraph, CompileError> {
        let pass_accesses: Vec<Vec<compile::PassAccess>> = self.passes.iter()
            .map(|p| p.accesses.clone())
            .collect();

        let result = compile::compile(
            &pass_accesses,
            &self.output_versions,
            self.resource_count as usize,
        )?;

        // A transient handle is live if any live pass declares a Write on it.
        let live_writes: std::collections::HashSet<u32> = result.execution_order.iter()
            .flat_map(|&i| {
                pass_accesses[i].iter()
                    .filter(|pa| pa.access == Access::Write)
                    .map(|pa| pa.resource)
            })
            .collect();

        let transient_buffer_descs: Vec<(BufferHandle, BufferDesc, String)> =
            self.transient_buffer_descs.into_iter()
                .filter(|(h, _, _)| live_writes.contains(&h.resource))
                .collect();

        let transient_texture_descs: Vec<(TextureHandle, TextureDesc, String)> =
            self.transient_texture_descs.into_iter()
                .filter(|(h, _, _)| live_writes.contains(&h.resource))
                .collect();

        // Reorder passes into execution order, dropping culled passes.
        let mut slots: Vec<Option<PassData>> = self.passes
            .into_iter()
            .map(Some)
            .collect();

        let passes = result.execution_order.iter()
            .map(|&i| slots[i].take().expect("pass used twice in execution order"))
            .collect();

        let entry_count = self.resource_count as usize;

        Ok(CompiledGraph {
            passes,
            barriers               : result.barriers,
            culled_count           : result.culled_count,
            entries                : (0..entry_count).map(|_| None).collect(),
            transient_buffer_descs,
            transient_texture_descs,
        })
    }
}

impl RenderGraph {
    fn alloc_resource(&mut self) -> u32 {
        let idx = self.resource_count;
        self.resource_count += 1;
        self.resource_versions.push(0);
        idx
    }

    fn alloc_buffer_handle(&mut self) -> BufferHandle {
        let resource = self.alloc_resource();
        BufferHandle { resource, version: 0 }
    }

    fn alloc_texture_handle(&mut self) -> TextureHandle {
        let resource = self.alloc_resource();
        TextureHandle { resource, version: 0 }
    }
}

// --- PassBuilder ---

/// Builder for declaring a single pass's resource usage.
///
/// Obtained from the closure passed to [`RenderGraph::add_pass`].
pub struct PassBuilder<'g> {
    graph      : &'g mut RenderGraph,
    pass_index : usize,
}

impl<'g> PassBuilder<'g> {
    /// Declare that this pass reads the specific buffer version in `h`.
    ///
    /// Pins a dependency on the pass that produced this version.
    pub fn read_buffer(&mut self, h: BufferHandle) {
        self.record_access(h.resource, h.version, Access::Read);
    }

    /// Declare that this pass writes a buffer, minting a new version.
    ///
    /// Returns the new versioned handle.  Downstream passes that want to
    /// observe this write must use the returned handle.
    pub fn write_buffer(&mut self, h: BufferHandle) -> BufferHandle {
        let new_version = self.next_version(h.resource);
        self.record_access(h.resource, new_version, Access::Write);
        BufferHandle { resource: h.resource, version: new_version }
    }

    /// Declare that this pass reads the specific texture version in `h`.
    pub fn read_texture(&mut self, h: TextureHandle) {
        self.record_access(h.resource, h.version, Access::Read);
    }

    /// Declare that this pass writes a texture, minting a new version.
    ///
    /// Returns the new versioned handle.
    pub fn write_texture(&mut self, h: TextureHandle) -> TextureHandle {
        let new_version = self.next_version(h.resource);
        self.record_access(h.resource, new_version, Access::Write);
        TextureHandle { resource: h.resource, version: new_version }
    }

    /// Set the execute closure for this pass.
    ///
    /// The closure is called during [`CompiledGraph::execute`] with a
    /// [`PassContext`] that provides a [`Commands`] recorder, the resolved
    /// [`ResourceMap`], and the current [`FrameIndex`].
    ///
    /// Closures should capture only [`Copy`] resource handles.
    pub fn execute(
        &mut self,
        f: impl FnOnce(&mut PassContext<'_>) + 'static,
    ) {
        self.graph.passes[self.pass_index].execute_fn = Some(Box::new(f));
    }

    fn record_access(&mut self, resource: u32, version: u32, access: Access) {
        self.graph.passes[self.pass_index].accesses.push(compile::PassAccess {
            resource,
            version,
            access,
        });
    }

    fn next_version(&mut self, resource: u32) -> u32 {
        let v = &mut self.graph.resource_versions[resource as usize];
        *v += 1;
        *v
    }
}

// --- CompiledGraph ---

/// A compiled render graph ready for execution.
///
/// Contains passes in execution order (dead passes already culled) and
/// barrier metadata.  Obtain by calling [`RenderGraph::compile`].
pub struct CompiledGraph {
    passes                  : Vec<PassData>,
    barriers                : Vec<Barrier>,
    culled_count            : usize,
    entries                 : Vec<Option<ResourceEntry>>,
    transient_buffer_descs  : Vec<(BufferHandle, BufferDesc, String)>,
    transient_texture_descs : Vec<(TextureHandle, TextureDesc, String)>,
}

impl CompiledGraph {
    /// Pass names in execution order.
    pub fn pass_names(&self) -> Vec<&str> {
        self.passes.iter().map(|p| p.name.as_str()).collect()
    }

    /// Barriers between passes in execution order.
    pub fn barriers(&self) -> &[Barrier] {
        &self.barriers
    }

    /// Number of passes culled during compilation.
    pub fn culled_count(&self) -> usize {
        self.culled_count
    }

    /// Bind an imported buffer handle to its backing GPU buffer.
    ///
    /// Call once per imported handle that any live pass references,
    /// before calling [`execute`](Self::execute).
    ///
    /// # Panics
    ///
    /// Panics if the handle index is out of range (programmer error).
    pub fn bind(&mut self, handle: BufferHandle, buffer: wgpu::Buffer) {
        self.entries[handle.resource as usize] = Some(ResourceEntry::Buffer(buffer));
    }

    /// Bind an imported texture handle to its backing GPU texture.
    ///
    /// Creates a default [`wgpu::TextureView`] for the texture immediately.
    /// Call once per imported texture handle before [`execute`](Self::execute).
    ///
    /// # Panics
    ///
    /// Panics if the handle index is out of range (programmer error).
    pub fn bind_texture(&mut self, handle: TextureHandle, texture: wgpu::Texture) {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.entries[handle.resource as usize] = Some(ResourceEntry::Texture(texture, view));
    }

    /// Allocate transient buffers and textures from their respective pools.
    ///
    /// Call after binding all imported handles and before
    /// [`execute`](Self::execute).  Each transient resource created during
    /// the build phase (and not culled) is allocated from the appropriate
    /// pool.
    pub fn allocate_transients(
        &mut self,
        buf_pool : &mut BufferPool,
        tex_pool : &mut TexturePool,
        device   : &wgpu::Device,
    ) {
        for &(handle, ref desc, _) in &self.transient_buffer_descs {
            let buffer = buf_pool.acquire(device, desc);
            self.entries[handle.resource as usize] = Some(ResourceEntry::Buffer(buffer));
        }

        for &(handle, ref desc, _) in &self.transient_texture_descs {
            let texture = tex_pool.acquire(device, desc);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.entries[handle.resource as usize] = Some(ResourceEntry::Texture(texture, view));
        }
    }

    /// Execute all passes in compiled order.
    ///
    /// Constructs a [`Commands`] recorder from `encoder`, then calls each
    /// pass's execute closure sequentially with a [`PassContext`].
    ///
    /// Returns a [`PendingRelease`] holding the transient resources.
    /// The caller must hold it until the GPU has completed the submit
    /// containing this frame's commands, then call
    /// [`PendingRelease::release`] to return the resources to the pools.
    pub fn execute(self, encoder: &mut FrameEncoder, frame: FrameIndex) -> PendingRelease {
        let Self {
            passes,
            transient_buffer_descs,
            transient_texture_descs,
            entries,
            barriers,
            ..
        } = self;

        // Clone transient resource arcs before ResourceMap takes ownership.
        let transient_buffers: Vec<wgpu::Buffer> = transient_buffer_descs.iter()
            .filter_map(|&(handle, _, _)| {
                if let Some(ResourceEntry::Buffer(b)) = entries[handle.resource as usize].as_ref() {
                    Some(b.clone())
                }
                else {
                    None
                }
            })
            .collect();

        let transient_textures: Vec<wgpu::Texture> = transient_texture_descs.iter()
            .filter_map(|&(handle, _, _)| {
                if let Some(ResourceEntry::Texture(t, _)) = entries[handle.resource as usize].as_ref() {
                    Some(t.clone())
                }
                else {
                    None
                }
            })
            .collect();

        let resources = ResourceMap { entries };

        // `barriers` is sorted non-decreasingly by `before` (compilation
        // invariant: emitted in a single forward pass over execution order).
        // A single cursor advancing in lockstep with the pass loop is O(P + B).
        let mut barrier_cursor = 0usize;

        for (i, pass) in passes.into_iter().enumerate() {
            // Emit a debug marker for each barrier that fires before pass i.
            while barrier_cursor < barriers.len()
                && barriers[barrier_cursor].before == i
            {
                let b = &barriers[barrier_cursor];
                encoder.encoder_mut().insert_debug_marker(&format!(
                    "barrier: res{} {:?}->{:?} (after={}, before={})",
                    b.resource.0, b.src, b.dst, b.after, b.before,
                ));
                barrier_cursor += 1;
            }

            if let Some(f) = pass.execute_fn {
                let commands = Commands { encoder: encoder.encoder_mut() };
                let mut ctx = PassContext { commands, resources: &resources, frame };
                f(&mut ctx);
            }
        }

        // All barriers must have been consumed; an orphan would mean a barrier
        // was emitted with `before >= passes.len()`, which is a compiler bug.
        debug_assert_eq!(
            barrier_cursor,
            barriers.len(),
            "orphaned barriers with before >= passes.len(); compiler bug",
        );

        PendingRelease { buffers: transient_buffers, textures: transient_textures }
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn desc() -> BufferDesc {
        BufferDesc { size: 256, usage: wgpu::BufferUsages::STORAGE }
    }

    #[test]
    fn empty_graph_compiles() {
        let graph = RenderGraph::new();
        let compiled = graph.compile().unwrap();

        assert!(compiled.pass_names().is_empty());
        assert!(compiled.barriers().is_empty());
        assert_eq!(compiled.culled_count(), 0);
    }

    #[test]
    fn default_graph_compiles() {
        let graph = RenderGraph::default();
        let compiled = graph.compile().unwrap();

        assert!(compiled.pass_names().is_empty());
    }

    #[test]
    fn single_live_pass() {
        let mut graph = RenderGraph::new();

        let buf = graph.create_buffer("A_buf", desc());
        let v1 = graph.add_pass("A", |pass| pass.write_buffer(buf));
        graph.mark_output(v1);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["A"]);
        assert_eq!(compiled.culled_count(), 0);
    }

    #[test]
    fn single_dead_pass_is_culled() {
        let mut graph = RenderGraph::new();

        let buf = graph.create_buffer("A_buf", desc());
        graph.add_pass("A", |pass| { pass.write_buffer(buf); });

        let compiled = graph.compile().unwrap();
        assert!(compiled.pass_names().is_empty());
        assert_eq!(compiled.culled_count(), 1);
    }

    #[test]
    fn linear_chain_preserves_order() {
        let mut graph = RenderGraph::new();

        let a_buf = graph.create_buffer("a_out", desc());
        let a_v1 = graph.add_pass("A", |pass| pass.write_buffer(a_buf));

        let b_buf = graph.create_buffer("b_out", desc());
        let b_v1 = graph.add_pass("B", |pass| {
            pass.read_buffer(a_v1);
            pass.write_buffer(b_buf)
        });

        let c_buf = graph.create_buffer("c_out", desc());
        let c_v1 = graph.add_pass("C", |pass| {
            pass.read_buffer(b_v1);
            pass.write_buffer(c_buf)
        });

        graph.mark_output(c_v1);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["A", "B", "C"]);
        assert_eq!(compiled.culled_count(), 0);
    }

    #[test]
    fn diamond_dag_valid_order() {
        let mut graph = RenderGraph::new();

        let a_buf = graph.create_buffer("a_out", desc());
        let a_v1 = graph.add_pass("A", |pass| pass.write_buffer(a_buf));

        let b_buf = graph.create_buffer("b_out", desc());
        let b_v1 = graph.add_pass("B", |pass| {
            pass.read_buffer(a_v1);
            pass.write_buffer(b_buf)
        });

        let c_buf = graph.create_buffer("c_out", desc());
        let c_v1 = graph.add_pass("C", |pass| {
            pass.read_buffer(a_v1);
            pass.write_buffer(c_buf)
        });

        let d_buf = graph.create_buffer("d_out", desc());
        let d_v1 = graph.add_pass("D", |pass| {
            pass.read_buffer(b_v1);
            pass.read_buffer(c_v1);
            pass.write_buffer(d_buf)
        });

        graph.mark_output(d_v1);

        let compiled = graph.compile().unwrap();
        let names = compiled.pass_names();

        // A must come first, D must come last. B and C can be either order.
        assert_eq!(names.len(), 4);
        assert_eq!(names[0], "A");
        assert_eq!(names[3], "D");
        assert!(names[1..3].contains(&"B"));
        assert!(names[1..3].contains(&"C"));
        assert_eq!(compiled.culled_count(), 0);
    }

    #[test]
    fn dead_branch_is_culled() {
        let mut graph = RenderGraph::new();

        let shared = graph.create_buffer("shared", desc());
        let shared_v1 = graph.add_pass("root", |pass| pass.write_buffer(shared));

        let live_buf = graph.create_buffer("live_out", desc());
        let live_v1 = graph.add_pass("live_branch", |pass| {
            pass.read_buffer(shared_v1);
            pass.write_buffer(live_buf)
        });

        let dead_buf = graph.create_buffer("dead_out", desc());
        graph.add_pass("dead_branch", |pass| {
            pass.read_buffer(shared_v1);
            pass.write_buffer(dead_buf);
        });

        graph.mark_output(live_v1);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["root", "live_branch"]);
        assert_eq!(compiled.culled_count(), 1);
    }

    #[test]
    fn no_outputs_culls_all_passes() {
        let mut graph = RenderGraph::new();

        let a_buf = graph.create_buffer("a_out", desc());
        let a_v1 = graph.add_pass("A", |pass| pass.write_buffer(a_buf));

        let b_buf = graph.create_buffer("b_out", desc());
        graph.add_pass("B", |pass| {
            pass.read_buffer(a_v1);
            pass.write_buffer(b_buf);
        });

        let compiled = graph.compile().unwrap();
        assert!(compiled.pass_names().is_empty());
        assert_eq!(compiled.culled_count(), 2);
    }

    #[test]
    fn output_buffer_with_no_writer_culls_all() {
        let mut graph = RenderGraph::new();
        let imported = graph.import_buffer();

        let _buf = graph.create_buffer("unrelated_buf", desc());
        graph.add_pass("unrelated", |pass| {
            pass.write_buffer(_buf);
        });

        // Nobody writes imported (version 0 has no producer), so no live pass
        // is seeded.
        graph.mark_output(imported);

        let compiled = graph.compile().unwrap();
        assert!(compiled.pass_names().is_empty());
        assert_eq!(compiled.culled_count(), 1);
    }

    #[test]
    fn pass_live_when_any_created_buffer_is_output() {
        let mut graph = RenderGraph::new();

        let a = graph.create_buffer("a", desc());
        let b = graph.create_buffer("b", desc());
        let (a_v1, _b_v1) = graph.add_pass("multi_output", |pass| {
            let av = pass.write_buffer(a);
            let bv = pass.write_buffer(b);
            (av, bv)
        });

        graph.mark_output(a_v1);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["multi_output"]);
        assert_eq!(compiled.culled_count(), 0);
    }

    // -- Barrier tests --

    #[test]
    fn write_then_read_produces_barrier() {
        let mut graph = RenderGraph::new();

        let buf = graph.create_buffer("buf", desc());
        let buf_v1 = graph.add_pass("writer", |pass| pass.write_buffer(buf));

        let out = graph.create_buffer("out", desc());
        let out_v1 = graph.add_pass("reader", |pass| {
            pass.read_buffer(buf_v1);
            pass.write_buffer(out)
        });

        graph.mark_output(out_v1);

        let compiled = graph.compile().unwrap();
        let barriers = compiled.barriers();

        // Barrier on buf: Write → Read.
        let buf_barriers: Vec<_> = barriers.iter()
            .filter(|b| b.resource == ResourceId::from(buf_v1))
            .collect();

        assert_eq!(buf_barriers.len(), 1);
        assert_eq!(buf_barriers[0].src, Access::Write);
        assert_eq!(buf_barriers[0].dst, Access::Read);
        assert_eq!(buf_barriers[0].after, 0);
        assert_eq!(buf_barriers[0].before, 1);
    }

    #[test]
    fn consecutive_reads_produce_no_barrier() {
        let mut graph = RenderGraph::new();
        let imported = graph.import_buffer();

        let out_a = graph.create_buffer("out_a", desc());
        let out_a_v1 = graph.add_pass("reader_a", |pass| {
            pass.read_buffer(imported);
            pass.write_buffer(out_a)
        });

        let out_b = graph.create_buffer("out_b", desc());
        let out_b_v1 = graph.add_pass("reader_b", |pass| {
            pass.read_buffer(imported);
            pass.write_buffer(out_b)
        });

        graph.mark_output(out_a_v1);
        graph.mark_output(out_b_v1);

        let compiled = graph.compile().unwrap();

        let buf_barriers: Vec<_> = compiled.barriers().iter()
            .filter(|b| b.resource == ResourceId::from(imported))
            .collect();

        assert!(buf_barriers.is_empty());
    }

    #[test]
    fn consecutive_writes_produce_barrier() {
        let mut graph = RenderGraph::new();
        let imported = graph.import_buffer();

        let imported_v1 = graph.add_pass("writer_a", |pass| pass.write_buffer(imported));

        let out = graph.create_buffer("out", desc());
        let out_v1 = graph.add_pass("writer_b", |pass| {
            let v2 = pass.write_buffer(imported_v1);
            let ov = pass.write_buffer(out);
            // We need writer_b to depend on imported_v1 via the write chain.
            // write_buffer(imported_v1) returns imported_v2, but we don't use it
            // here — the WAW dependency is encoded in the graph.
            let _ = v2;
            ov
        });

        graph.mark_output(out_v1);

        let compiled = graph.compile().unwrap();

        // Both passes live (WAW dependency keeps writer_a alive).
        assert_eq!(compiled.pass_names(), ["writer_a", "writer_b"]);

        let buf_barriers: Vec<_> = compiled.barriers().iter()
            .filter(|b| b.resource == ResourceId::from(imported))
            .collect();

        assert_eq!(buf_barriers.len(), 1);
        assert_eq!(buf_barriers[0].src, Access::Write);
        assert_eq!(buf_barriers[0].dst, Access::Write);
    }

    #[test]
    fn imported_buffer_write_read_chain() {
        let mut graph = RenderGraph::new();
        let imported = graph.import_buffer();

        let imported_v1 = graph.add_pass("upload", |pass| pass.write_buffer(imported));

        let out = graph.create_buffer("out", desc());
        let out_v1 = graph.add_pass("consume", |pass| {
            pass.read_buffer(imported_v1);
            pass.write_buffer(out)
        });

        graph.mark_output(out_v1);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["upload", "consume"]);

        let buf_barriers: Vec<_> = compiled.barriers().iter()
            .filter(|b| b.resource == ResourceId::from(imported))
            .collect();

        assert_eq!(buf_barriers.len(), 1);
        assert_eq!(buf_barriers[0].src, Access::Write);
        assert_eq!(buf_barriers[0].dst, Access::Read);
    }

    // -- SSA versioning tests --

    #[test]
    fn two_writes_produce_different_handles() {
        let mut graph = RenderGraph::new();

        let buf = graph.import_buffer();
        let v1 = graph.add_pass("writer_a", |pass| pass.write_buffer(buf));
        let v2 = graph.add_pass("writer_b", |pass| pass.write_buffer(v1));

        // Different versions, same resource.
        assert_eq!(v1.resource, buf.resource);
        assert_eq!(v2.resource, buf.resource);
        assert_ne!(v1.version, v2.version);
        assert_ne!(v1.version, buf.version);
    }

    #[test]
    fn reader_of_first_version_culls_second_writer() {
        let mut graph = RenderGraph::new();

        let buf = graph.import_buffer();

        // writer_a writes v1.
        let v1 = graph.add_pass("writer_a", |pass| pass.write_buffer(buf));

        // writer_b writes v2 (depends on v1 via WAW).
        graph.add_pass("writer_b", |pass| { pass.write_buffer(v1); });

        // reader reads v1 — only writer_a is needed, writer_b is dead.
        let out = graph.create_buffer("out", desc());
        let out_v1 = graph.add_pass("reader", |pass| {
            pass.read_buffer(v1);
            pass.write_buffer(out)
        });

        graph.mark_output(out_v1);

        let compiled = graph.compile().unwrap();

        // writer_b is dead (reader only needs v1, which writer_a produces).
        // writer_a and reader are live.
        let names = compiled.pass_names();
        assert!(names.contains(&"writer_a"), "writer_a must be live");
        assert!(names.contains(&"reader"),   "reader must be live");
        assert!(!names.contains(&"writer_b"), "writer_b must be culled");
    }

    #[test]
    fn reader_of_second_version_keeps_both_writers_live() {
        let mut graph = RenderGraph::new();

        let buf = graph.import_buffer();

        let v1 = graph.add_pass("writer_a", |pass| pass.write_buffer(buf));
        let v2 = graph.add_pass("writer_b", |pass| pass.write_buffer(v1));

        let out = graph.create_buffer("out", desc());
        let out_v1 = graph.add_pass("reader", |pass| {
            pass.read_buffer(v2);
            pass.write_buffer(out)
        });

        graph.mark_output(out_v1);

        let compiled = graph.compile().unwrap();
        let names = compiled.pass_names();

        // Both writers and the reader are live.
        assert!(names.contains(&"writer_a"), "writer_a must be live");
        assert!(names.contains(&"writer_b"), "writer_b must be live");
        assert!(names.contains(&"reader"),   "reader must be live");
        assert_eq!(compiled.culled_count(), 0);
    }

    // -- GPU test (requires Vulkan) --

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn execute_resolves_bound_buffer() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use crate::device::RendererContext;
        use crate::frame::FrameCount;

        let mut ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine");

        let gpu_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_bind_resolve"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mut graph = RenderGraph::new();
        let h = graph.import_buffer();

        let expected = gpu_buf.clone();
        let ran = std::sync::Arc::new(AtomicBool::new(false));
        let ran_clone = ran.clone();

        let out = graph.create_buffer("out", desc());
        let h_v1 = graph.add_pass("resolve_test", |pass| {
            pass.read_buffer(h);
            let v = pass.write_buffer(out);
            pass.execute(move |ctx| {
                assert_eq!(
                    ctx.resources.buffer(h), &expected,
                    "resolved buffer should be the bound GPU buffer",
                );
                ran_clone.store(true, Ordering::SeqCst);
            });
            v
        });
        graph.mark_output(h_v1);

        let mut compiled = graph.compile().unwrap();
        compiled.bind(h, gpu_buf);

        let mut encoder = ctx.begin_frame();
        let frame = ctx.frame_index();
        let _pending = compiled.execute(&mut encoder, frame);
        ctx.end_frame(encoder);

        assert!(ran.load(Ordering::SeqCst), "execute closure must have run");
    }

    // -- Transient / pool tests --

    #[test]
    fn culled_pass_transient_not_in_compiled() {
        let mut graph = RenderGraph::new();

        let live = graph.create_buffer("live", BufferDesc {
            size: 512,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let live_v1 = graph.add_pass("live", |pass| pass.write_buffer(live));

        let dead = graph.create_buffer("dead", BufferDesc {
            size: 1024,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        graph.add_pass("dead", |pass| { pass.write_buffer(dead); });

        graph.mark_output(live_v1);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["live"]);
        assert_eq!(compiled.transient_buffer_descs.len(), 1);
        assert_eq!(compiled.transient_buffer_descs[0].1.size, 512);
    }

    #[test]
    fn no_transients_yields_empty_descs() {
        let mut graph = RenderGraph::new();
        let imported = graph.import_buffer();

        let imported_v1 = graph.add_pass("writer", |pass| pass.write_buffer(imported));
        graph.mark_output(imported_v1);

        let compiled = graph.compile().unwrap();
        assert!(compiled.transient_buffer_descs.is_empty());
        assert!(compiled.transient_texture_descs.is_empty());
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn pool_acquire_creates_and_release_recycles() {
        use crate::device::RendererContext;
        use crate::frame::FrameCount;

        let ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context");

        let mut pool = BufferPool::new();
        let d = BufferDesc { size: 128, usage: wgpu::BufferUsages::STORAGE };

        // First acquire creates a fresh buffer.
        let buf = pool.acquire(ctx.device(), &d);
        assert_eq!(buf.size(), 128);

        // Release and re-acquire: same buffer returned (Arc identity).
        let reference = buf.clone();
        pool.release(buf);
        let buf2 = pool.acquire(ctx.device(), &d);
        assert_eq!(buf2, reference);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn pool_different_desc_not_shared() {
        use crate::device::RendererContext;
        use crate::frame::FrameCount;

        let ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context");

        let mut pool = BufferPool::new();
        let d_small = BufferDesc { size: 64, usage: wgpu::BufferUsages::STORAGE };
        let d_large = BufferDesc { size: 256, usage: wgpu::BufferUsages::STORAGE };

        let buf_small = pool.acquire(ctx.device(), &d_small);
        let ref_small = buf_small.clone();
        pool.release(buf_small);

        // Different size — should get a new buffer, not the recycled one.
        let buf_large = pool.acquire(ctx.device(), &d_large);
        assert_ne!(buf_large, ref_small);
        assert_eq!(buf_large.size(), 256);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn allocate_transients_and_pending_release() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use crate::device::RendererContext;
        use crate::frame::FrameCount;

        let mut ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context");

        let mut pool = BufferPool::new();
        let mut tex_pool = TexturePool::new();

        let mut graph = RenderGraph::new();
        let imported = graph.import_buffer();

        let ran = std::sync::Arc::new(AtomicBool::new(false));
        let ran_clone = ran.clone();

        let transient = graph.create_buffer("transient", BufferDesc {
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let transient_v1 = graph.add_pass("compute", |pass| {
            pass.read_buffer(imported);
            let v = pass.write_buffer(transient);
            pass.execute(move |ctx| {
                // Both handles should resolve.
                let _imp = ctx.resources.buffer(imported);
                let t = ctx.resources.buffer(transient);
                assert_eq!(t.size(), 1024);
                ran_clone.store(true, Ordering::SeqCst);
            });
            v
        });
        graph.mark_output(transient_v1);

        let mut compiled = graph.compile().unwrap();

        // Bind imported buffer.
        let gpu_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_imported"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        compiled.bind(imported, gpu_buf);

        // Allocate transients from the pools.
        compiled.allocate_transients(&mut pool, &mut tex_pool, ctx.device());

        let mut encoder = ctx.begin_frame();
        let frame = ctx.frame_index();
        let pending = compiled.execute(&mut encoder, frame);
        ctx.end_frame(encoder);

        assert!(ran.load(Ordering::SeqCst), "pass closure must have run");
        assert_eq!(pending.len(), 1);

        // Release back to pool — pool should now have a recyclable buffer.
        pending.release(&mut pool, &mut tex_pool);
        let recycled = pool.acquire(ctx.device(), &BufferDesc {
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
        });
        assert_eq!(recycled.size(), 1024);
    }

    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn execute_emits_barriers_in_order() {
        use std::sync::{Arc, Mutex};
        use crate::device::RendererContext;
        use crate::frame::FrameCount;

        let mut ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context");

        let mut pool = BufferPool::new();
        let mut tex_pool = TexturePool::new();
        let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));

        let mut graph = RenderGraph::new();

        // Pass 0: writes buf. Pass 1: reads buf.
        // Compilation must produce exactly one Write->Read barrier at before=1.
        let buf = graph.create_buffer("buf", desc());
        let buf_v1 = graph.add_pass("writer", |pass| {
            let v = pass.write_buffer(buf);
            let o = order.clone();
            pass.execute(move |_| { o.lock().unwrap().push("writer"); });
            v
        });

        let out = graph.create_buffer("out", desc());
        let out_v1 = graph.add_pass("reader", |pass| {
            pass.read_buffer(buf_v1);
            let v = pass.write_buffer(out);
            let o = order.clone();
            pass.execute(move |_| { o.lock().unwrap().push("reader"); });
            v
        });

        graph.mark_output(out_v1);

        let mut compiled = graph.compile().unwrap();
        assert_eq!(compiled.barriers().len(), 1);
        assert_eq!(compiled.barriers()[0].src, Access::Write);
        assert_eq!(compiled.barriers()[0].dst, Access::Read);
        assert_eq!(compiled.barriers()[0].before, 1);

        compiled.allocate_transients(&mut pool, &mut tex_pool, ctx.device());

        let mut encoder = ctx.begin_frame();
        let frame = ctx.frame_index();
        let pending = compiled.execute(&mut encoder, frame);
        ctx.end_frame(encoder);

        // Both closures must have run in the declared order; barrier emission
        // must not have skipped or reordered any passes.
        assert_eq!(*order.lock().unwrap(), ["writer", "reader"]);

        pending.release(&mut pool, &mut tex_pool);
    }
}
