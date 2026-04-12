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
//!    calls each pass's closure sequentially with `&mut T` user context.
//!
//! The graph is rebuilt from scratch each frame.

mod compile;
mod pool;
mod resource;

pub use compile::{Barrier, CompileError};
pub use pool::{BufferPool, PendingRelease};
pub use resource::{Access, BufferDesc, BufferHandle};

use crate::device::FrameEncoder;

// --- ResourceMap ---

/// Resolved resource handles available during pass execution.
///
/// Maps render-graph [`BufferHandle`]s to their backing wgpu resources.
/// Passed to each pass's execute closure so it can look up actual GPU
/// buffers by handle rather than capturing raw wgpu types.
pub struct ResourceMap {
    buffers: Vec<Option<wgpu::Buffer>>,
}

impl ResourceMap {
    /// Look up the backing GPU buffer for a render graph handle.
    ///
    /// # Panics
    ///
    /// Panics if the handle was not bound via [`CompiledGraph::bind`].
    /// This is a programmer error (invariant: bind all imported handles
    /// before execute).
    pub fn buffer(&self, handle: BufferHandle) -> &wgpu::Buffer {
        self.buffers[handle.0 as usize]
            .as_ref()
            .unwrap_or_else(|| panic!(
                "buffer handle {} not bound — \
                 call CompiledGraph::bind() before execute",
                handle.0,
            ))
    }
}

// --- PassData ---

/// Internal storage for a declared pass.
struct PassData<T> {
    name       : String,
    accesses   : Vec<(BufferHandle, Access)>,
    creates    : Vec<BufferHandle>,
    execute_fn : Option<Box<dyn FnOnce(&mut T, &mut FrameEncoder, &ResourceMap)>>,
}

// --- RenderGraph ---

/// A per-frame render graph.  Generic over user context `T` which is
/// passed to each pass's execute closure as `&mut T`.
///
/// Build a graph by calling [`import_buffer`](Self::import_buffer),
/// [`add_pass`](Self::add_pass), and [`mark_output`](Self::mark_output),
/// then [`compile`](Self::compile) it into a [`CompiledGraph`].
pub struct RenderGraph<T> {
    passes          : Vec<PassData<T>>,
    buf_count       : u32,
    outputs         : Vec<BufferHandle>,
    transient_descs : Vec<(BufferHandle, BufferDesc)>,
}

impl<T> Default for RenderGraph<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> RenderGraph<T> {
    /// Create a new empty render graph.
    pub fn new() -> Self {
        Self {
            passes          : Vec::new(),
            buf_count       : 0,
            outputs         : Vec::new(),
            transient_descs : Vec::new(),
        }
    }

    /// Import a persistent (application-owned) buffer into the graph.
    ///
    /// Returns a handle that passes can declare reads or writes against.
    /// The graph tracks usage for ordering and barriers but does not manage
    /// the buffer's lifetime.
    pub fn import_buffer(&mut self) -> BufferHandle {
        self.alloc_handle()
    }

    /// Mark a buffer as a graph output.
    ///
    /// Passes that contribute to an output buffer — directly or
    /// transitively — will not be culled during compilation.  Passes whose
    /// outputs are never read by a live pass (and are not themselves writing
    /// to an output) are dead and will be removed.
    pub fn mark_output(&mut self, handle: BufferHandle) {
        self.outputs.push(handle);
    }

    /// Declare a new pass.
    ///
    /// The closure receives a [`PassBuilder`] for declaring resource usage
    /// (reads, writes, creates) and setting the execute closure.  The
    /// return value of the closure is forwarded to the caller — use this to
    /// return [`BufferHandle`]s created inside the pass so other passes can
    /// reference them.
    ///
    /// ```ignore
    /// let output = graph.add_pass("compute", |pass| {
    ///     let buf = pass.create_buffer(BufferDesc { size: 1024, usage: wgpu::BufferUsages::STORAGE });
    ///     pass.read(input);
    ///     pass.execute(move |ctx, encoder, res| { /* ... */ });
    ///     buf
    /// });
    /// ```
    pub fn add_pass<R>(
        &mut self,
        name : &str,
        f    : impl FnOnce(&mut PassBuilder<'_, T>) -> R,
    )
        -> R
    {
        let pass_index = self.passes.len();

        self.passes.push(PassData {
            name       : name.to_string(),
            accesses   : Vec::new(),
            creates    : Vec::new(),
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
    pub fn compile(self) -> Result<CompiledGraph<T>, CompileError> {
        let pass_accesses: Vec<Vec<(BufferHandle, Access)>> = self.passes.iter()
            .map(|p| p.accesses.clone())
            .collect();

        let pass_creates: Vec<Vec<BufferHandle>> = self.passes.iter()
            .map(|p| p.creates.clone())
            .collect();

        let result = compile::compile(
            &pass_accesses,
            &pass_creates,
            &self.outputs,
            self.buf_count as usize,
        )?;

        // Determine which transient handles survive culling.
        let live_creates: std::collections::HashSet<BufferHandle> =
            result.execution_order.iter()
                .flat_map(|&i| pass_creates[i].iter().copied())
                .collect();

        let transient_descs: Vec<(BufferHandle, BufferDesc)> =
            self.transient_descs.into_iter()
                .filter(|(h, _)| live_creates.contains(h))
                .collect();

        // Reorder passes into execution order, dropping culled passes.
        let mut slots: Vec<Option<PassData<T>>> = self.passes
            .into_iter()
            .map(Some)
            .collect();

        let ordered = result.execution_order.iter()
            .map(|&i| slots[i].take().expect("pass used twice in execution order"))
            .collect();

        Ok(CompiledGraph {
            passes          : ordered,
            barriers        : result.barriers,
            culled_count    : result.culled_count,
            bindings        : vec![None; self.buf_count as usize],
            transient_descs,
        })
    }
}

impl<T> RenderGraph<T> {
    fn alloc_handle(&mut self) -> BufferHandle {
        let handle = BufferHandle(self.buf_count);
        self.buf_count += 1;
        handle
    }
}

// --- PassBuilder ---

/// Builder for declaring a single pass's resource usage.
///
/// Obtained from the closure passed to [`RenderGraph::add_pass`].
pub struct PassBuilder<'g, T> {
    graph      : &'g mut RenderGraph<T>,
    pass_index : usize,
}

impl<'g, T> PassBuilder<'g, T> {
    /// Create a new transient buffer owned by this pass.
    ///
    /// The pass is implicitly the first writer.  Returns a handle that
    /// other passes can reference via [`read`](Self::read) or
    /// [`write`](Self::write).
    ///
    /// The [`BufferDesc`] specifies the size and usage flags for pool
    /// allocation.  The actual GPU buffer is allocated later via
    /// [`CompiledGraph::allocate_transients`].
    pub fn create_buffer(&mut self, desc: BufferDesc) -> BufferHandle {
        let handle = self.graph.alloc_handle();
        self.graph.passes[self.pass_index].creates.push(handle);
        self.graph.transient_descs.push((handle, desc));
        handle
    }

    /// Declare that this pass reads a buffer.
    pub fn read(&mut self, handle: BufferHandle) {
        self.graph.passes[self.pass_index]
            .accesses
            .push((handle, Access::Read));
    }

    /// Declare that this pass writes a buffer.
    pub fn write(&mut self, handle: BufferHandle) {
        self.graph.passes[self.pass_index]
            .accesses
            .push((handle, Access::Write));
    }

    /// Set the execute closure for this pass.
    ///
    /// The closure is called during [`CompiledGraph::execute`] with the
    /// user context, a [`FrameEncoder`] for recording GPU commands, and a
    /// [`ResourceMap`] for resolving buffer handles to wgpu resources.
    ///
    /// Closures should capture only [`Copy`] resource handles — application
    /// state flows through the `&mut T` parameter.
    pub fn execute(
        &mut self,
        f: impl FnOnce(&mut T, &mut FrameEncoder, &ResourceMap) + 'static,
    ) {
        self.graph.passes[self.pass_index].execute_fn = Some(Box::new(f));
    }
}

// --- CompiledGraph ---

/// A compiled render graph ready for execution.
///
/// Contains passes in execution order (dead passes already culled) and
/// barrier metadata.  Obtain by calling [`RenderGraph::compile`].
pub struct CompiledGraph<T> {
    passes          : Vec<PassData<T>>,
    barriers        : Vec<Barrier>,
    culled_count    : usize,
    bindings        : Vec<Option<wgpu::Buffer>>,
    transient_descs : Vec<(BufferHandle, BufferDesc)>,
}

impl<T> CompiledGraph<T> {
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
        self.bindings[handle.0 as usize] = Some(buffer);
    }

    /// Allocate transient buffers from the pool.
    ///
    /// Call after [`bind`](Self::bind)ing all imported handles and before
    /// [`execute`](Self::execute).  Each transient buffer created during
    /// the build phase (and not culled) is allocated from `pool`.
    pub fn allocate_transients(
        &mut self,
        pool   : &mut BufferPool,
        device : &wgpu::Device,
    ) {
        for &(handle, ref desc) in &self.transient_descs {
            let buffer = pool.acquire(device, desc);
            self.bindings[handle.0 as usize] = Some(buffer);
        }
    }

    /// Execute all passes in compiled order.
    ///
    /// Calls each pass's execute closure sequentially with the user
    /// context.  Only one `&mut T` borrow exists at a time.
    ///
    /// Each pass's closure receives a [`ResourceMap`] built from the
    /// bindings established via [`bind`](Self::bind) and
    /// [`allocate_transients`](Self::allocate_transients).
    ///
    /// Returns a [`PendingRelease`] holding the transient buffers.
    /// The caller must hold it until the GPU has completed the submit
    /// containing this frame's commands, then call
    /// [`PendingRelease::release`] to return the buffers to the pool.
    pub fn execute(self, ctx: &mut T, encoder: &mut FrameEncoder) -> PendingRelease {
        let Self { passes, transient_descs, bindings, .. } = self;

        // Clone transient buffer arcs before ResourceMap takes ownership.
        let transient_buffers: Vec<wgpu::Buffer> = transient_descs.iter()
            .filter_map(|&(handle, _)| bindings[handle.0 as usize].clone())
            .collect();

        let resources = ResourceMap { buffers: bindings };

        for pass in passes {
            if let Some(f) = pass.execute_fn {
                f(ctx, encoder, &resources);
            }
        }

        PendingRelease { buffers: transient_buffers }
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
        let graph = RenderGraph::<()>::new();
        let compiled = graph.compile().unwrap();

        assert!(compiled.pass_names().is_empty());
        assert!(compiled.barriers().is_empty());
        assert_eq!(compiled.culled_count(), 0);
    }

    #[test]
    fn default_graph_compiles() {
        let graph = RenderGraph::<()>::default();
        let compiled = graph.compile().unwrap();

        assert!(compiled.pass_names().is_empty());
    }

    #[test]
    fn single_live_pass() {
        let mut graph = RenderGraph::<()>::new();

        let buf = graph.add_pass("A", |pass| pass.create_buffer(desc()));
        graph.mark_output(buf);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["A"]);
        assert_eq!(compiled.culled_count(), 0);
    }

    #[test]
    fn single_dead_pass_is_culled() {
        let mut graph = RenderGraph::<()>::new();
        graph.add_pass("A", |pass| {
            pass.create_buffer(desc());
        });

        let compiled = graph.compile().unwrap();
        assert!(compiled.pass_names().is_empty());
        assert_eq!(compiled.culled_count(), 1);
    }

    #[test]
    fn linear_chain_preserves_order() {
        let mut graph = RenderGraph::<()>::new();

        let a_out = graph.add_pass("A", |pass| pass.create_buffer(desc()));

        let b_out = graph.add_pass("B", |pass| {
            pass.read(a_out);
            pass.create_buffer(desc())
        });

        let c_out = graph.add_pass("C", |pass| {
            pass.read(b_out);
            pass.create_buffer(desc())
        });

        graph.mark_output(c_out);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["A", "B", "C"]);
        assert_eq!(compiled.culled_count(), 0);
    }

    #[test]
    fn diamond_dag_valid_order() {
        let mut graph = RenderGraph::<()>::new();

        let a_out = graph.add_pass("A", |pass| pass.create_buffer(desc()));

        let b_out = graph.add_pass("B", |pass| {
            pass.read(a_out);
            pass.create_buffer(desc())
        });

        let c_out = graph.add_pass("C", |pass| {
            pass.read(a_out);
            pass.create_buffer(desc())
        });

        let d_out = graph.add_pass("D", |pass| {
            pass.read(b_out);
            pass.read(c_out);
            pass.create_buffer(desc())
        });

        graph.mark_output(d_out);

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
        let mut graph = RenderGraph::<()>::new();

        let shared = graph.add_pass("root", |pass| pass.create_buffer(desc()));

        let live_out = graph.add_pass("live_branch", |pass| {
            pass.read(shared);
            pass.create_buffer(desc())
        });

        // Dead: reads shared but output is never consumed.
        graph.add_pass("dead_branch", |pass| {
            pass.read(shared);
            pass.create_buffer(desc());
        });

        graph.mark_output(live_out);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["root", "live_branch"]);
        assert_eq!(compiled.culled_count(), 1);
    }

    #[test]
    fn no_outputs_culls_all_passes() {
        let mut graph = RenderGraph::<()>::new();

        let a_out = graph.add_pass("A", |pass| pass.create_buffer(desc()));

        graph.add_pass("B", |pass| {
            pass.read(a_out);
            pass.create_buffer(desc());
        });

        let compiled = graph.compile().unwrap();
        assert!(compiled.pass_names().is_empty());
        assert_eq!(compiled.culled_count(), 2);
    }

    #[test]
    fn output_buffer_with_no_writer_culls_all() {
        let mut graph = RenderGraph::<()>::new();
        let imported = graph.import_buffer();

        graph.add_pass("unrelated", |pass| {
            pass.create_buffer(desc());
        });

        // Nobody writes imported, so no live pass is seeded.
        graph.mark_output(imported);

        let compiled = graph.compile().unwrap();
        assert!(compiled.pass_names().is_empty());
        assert_eq!(compiled.culled_count(), 1);
    }

    #[test]
    fn pass_live_when_any_created_buffer_is_output() {
        let mut graph = RenderGraph::<()>::new();

        let (live, _dead) = graph.add_pass("multi_output", |pass| {
            let a = pass.create_buffer(desc());
            let b = pass.create_buffer(desc());
            (a, b)
        });

        graph.mark_output(live);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["multi_output"]);
        assert_eq!(compiled.culled_count(), 0);
    }

    // -- Barrier tests --

    #[test]
    fn write_then_read_produces_barrier() {
        let mut graph = RenderGraph::<()>::new();

        let buf = graph.add_pass("writer", |pass| pass.create_buffer(desc()));

        let out = graph.add_pass("reader", |pass| {
            pass.read(buf);
            pass.create_buffer(desc())
        });

        graph.mark_output(out);

        let compiled = graph.compile().unwrap();
        let barriers = compiled.barriers();

        // Barrier on buf: Write (create) → Read.
        let buf_barriers: Vec<_> = barriers.iter()
            .filter(|b| b.resource == buf)
            .collect();

        assert_eq!(buf_barriers.len(), 1);
        assert_eq!(buf_barriers[0].src, Access::Write);
        assert_eq!(buf_barriers[0].dst, Access::Read);
        assert_eq!(buf_barriers[0].after, 0);
        assert_eq!(buf_barriers[0].before, 1);
    }

    #[test]
    fn consecutive_reads_produce_no_barrier() {
        let mut graph = RenderGraph::<()>::new();
        let imported = graph.import_buffer();

        let out_a = graph.add_pass("reader_a", |pass| {
            pass.read(imported);
            pass.create_buffer(desc())
        });

        let out_b = graph.add_pass("reader_b", |pass| {
            pass.read(imported);
            pass.create_buffer(desc())
        });

        graph.mark_output(out_a);
        graph.mark_output(out_b);

        let compiled = graph.compile().unwrap();

        let buf_barriers: Vec<_> = compiled.barriers().iter()
            .filter(|b| b.resource == imported)
            .collect();

        assert!(buf_barriers.is_empty());
    }

    #[test]
    fn consecutive_writes_produce_barrier() {
        let mut graph = RenderGraph::<()>::new();
        let imported = graph.import_buffer();

        graph.add_pass("writer_a", |pass| {
            pass.write(imported);
        });

        let out = graph.add_pass("writer_b", |pass| {
            pass.write(imported);
            pass.create_buffer(desc())
        });

        graph.mark_output(out);

        let compiled = graph.compile().unwrap();

        // Both passes live (WAW dependency keeps writer_a alive).
        assert_eq!(compiled.pass_names(), ["writer_a", "writer_b"]);

        let buf_barriers: Vec<_> = compiled.barriers().iter()
            .filter(|b| b.resource == imported)
            .collect();

        assert_eq!(buf_barriers.len(), 1);
        assert_eq!(buf_barriers[0].src, Access::Write);
        assert_eq!(buf_barriers[0].dst, Access::Write);
    }

    #[test]
    fn imported_buffer_write_read_chain() {
        let mut graph = RenderGraph::<()>::new();
        let imported = graph.import_buffer();

        graph.add_pass("upload", |pass| {
            pass.write(imported);
        });

        let out = graph.add_pass("consume", |pass| {
            pass.read(imported);
            pass.create_buffer(desc())
        });

        graph.mark_output(out);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["upload", "consume"]);

        let buf_barriers: Vec<_> = compiled.barriers().iter()
            .filter(|b| b.resource == imported)
            .collect();

        assert_eq!(buf_barriers.len(), 1);
        assert_eq!(buf_barriers[0].src, Access::Write);
        assert_eq!(buf_barriers[0].dst, Access::Read);
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

        let mut graph = RenderGraph::<()>::new();
        let h = graph.import_buffer();

        let expected = gpu_buf.clone();
        let ran = std::sync::Arc::new(AtomicBool::new(false));
        let ran_clone = ran.clone();

        let out = graph.add_pass("resolve_test", |pass| {
            pass.read(h);
            let out = pass.create_buffer(desc());
            pass.execute(move |_ctx, _enc, resources| {
                assert_eq!(
                    resources.buffer(h), &expected,
                    "resolved buffer should be the bound GPU buffer",
                );
                ran_clone.store(true, Ordering::SeqCst);
            });
            out
        });
        graph.mark_output(out);

        let mut compiled = graph.compile().unwrap();
        compiled.bind(h, gpu_buf);

        let mut encoder = ctx.begin_frame();
        let _pending = compiled.execute(&mut (), &mut encoder);
        ctx.end_frame(encoder);

        assert!(ran.load(Ordering::SeqCst), "execute closure must have run");
    }

    // -- Transient / pool tests --

    #[test]
    fn culled_pass_transient_not_in_compiled() {
        let mut graph = RenderGraph::<()>::new();

        let live = graph.add_pass("live", |pass| {
            pass.create_buffer(BufferDesc {
                size: 512,
                usage: wgpu::BufferUsages::STORAGE,
            })
        });

        // Dead pass — its transient should be filtered out.
        graph.add_pass("dead", |pass| {
            pass.create_buffer(BufferDesc {
                size: 1024,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        });

        graph.mark_output(live);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["live"]);
        assert_eq!(compiled.transient_descs.len(), 1);
        assert_eq!(compiled.transient_descs[0].1.size, 512);
    }

    #[test]
    fn no_transients_yields_empty_descs() {
        let mut graph = RenderGraph::<()>::new();
        let imported = graph.import_buffer();

        graph.add_pass("writer", |pass| {
            pass.write(imported);
        });

        graph.mark_output(imported);

        let compiled = graph.compile().unwrap();
        assert!(compiled.transient_descs.is_empty());
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

        let mut graph = RenderGraph::<()>::new();
        let imported = graph.import_buffer();

        let ran = std::sync::Arc::new(AtomicBool::new(false));
        let ran_clone = ran.clone();

        let out = graph.add_pass("compute", |pass| {
            let transient = pass.create_buffer(BufferDesc {
                size: 1024,
                usage: wgpu::BufferUsages::STORAGE,
            });
            pass.read(imported);
            pass.execute(move |_ctx, _enc, resources| {
                // Both handles should resolve.
                let _imp = resources.buffer(imported);
                let t = resources.buffer(transient);
                assert_eq!(t.size(), 1024);
                ran_clone.store(true, Ordering::SeqCst);
            });
            transient
        });
        graph.mark_output(out);

        let mut compiled = graph.compile().unwrap();

        // Bind imported buffer.
        let gpu_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_imported"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        compiled.bind(imported, gpu_buf);

        // Allocate transients from the pool.
        compiled.allocate_transients(&mut pool, ctx.device());

        let mut encoder = ctx.begin_frame();
        let pending = compiled.execute(&mut (), &mut encoder);
        ctx.end_frame(encoder);

        assert!(ran.load(Ordering::SeqCst), "pass closure must have run");
        assert_eq!(pending.len(), 1);

        // Release back to pool — pool should now have a recyclable buffer.
        pending.release(&mut pool);
        let recycled = pool.acquire(ctx.device(), &BufferDesc {
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
        });
        assert_eq!(recycled.size(), 1024);
    }
}
