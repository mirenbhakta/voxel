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
pub use resource::{
    Access, BindGroupHandle, BufferDesc, BufferHandle, ResourceId, TextureDesc, TextureHandle,
};
use resource::ResourceHandle;

use crate::commands::Commands;
use crate::device::{FrameEncoder, SurfaceFrame};
use crate::frame::FrameIndex;
use crate::gpu_consts::GpuConsts;
use crate::pipeline::PipelineBindLayout;
use crate::pipeline::binding::{BindEntry, BindKind};

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

// --- BindGroupTemplate ---

/// Deferred bind-group description registered at graph-build time.
///
/// Records the pipeline's reflected bind entries, the slot-0 GpuConsts buffer
/// (if present), and the (binding, resource) pairs for user slots.  The actual
/// [`wgpu::BindGroup`] is produced during
/// [`CompiledGraph::resolve_bind_groups`] once every referenced transient
/// buffer has been allocated.
struct BindGroupTemplate {
    label          : String,
    bg_layout      : wgpu::BindGroupLayout,
    bind_entries   : Vec<BindEntry>,
    /// `Some` when the graph auto-wires a [`GpuConsts`] buffer into the
    /// reserved set-0 slot for externally-composed bind groups.  `None` when
    /// the caller supplies every slot — notably for set-N bind groups (N > 0)
    /// owned by a node.
    gpu_consts_buf : Option<wgpu::Buffer>,
    entries        : Vec<(u32, ResourceId)>,
}

// --- ResourceMap ---

/// Resolved resource handles available during pass execution.
///
/// Maps render-graph handles to their backing wgpu resources.
/// Passed to each pass's execute closure via [`PassContext`] so it can look
/// up actual GPU resources by handle rather than capturing raw wgpu types.
pub struct ResourceMap {
    entries     : Vec<Option<ResourceEntry>>,
    bind_groups : Vec<Option<wgpu::BindGroup>>,
}

impl ResourceMap {
    /// Look up the backing GPU buffer for a render graph handle.
    ///
    /// Version is ignored — lookup is by resource identity.
    ///
    /// # Panics
    ///
    /// Panics if the handle has no backing buffer.  An imported handle is
    /// bound at build time by [`RenderGraph::import_buffer`]; a transient
    /// is bound by [`CompiledGraph::allocate_transients`].  A miss here is
    /// a programmer error.
    pub fn buffer(&self, handle: BufferHandle) -> &wgpu::Buffer {
        match self.entries[handle.0.resource as usize]
            .as_ref()
            .unwrap_or_else(|| panic!(
                "buffer handle (resource={}) not bound — import via \
                 RenderGraph::import_buffer or allocate via \
                 CompiledGraph::allocate_transients",
                handle.0.resource,
            ))
        {
            ResourceEntry::Buffer(b) => b,
            _ => panic!("handle (resource={}) is a texture, not a buffer", handle.0.resource),
        }
    }

    /// Look up the backing GPU texture for a render graph handle.
    ///
    /// Version is ignored — lookup is by resource identity.
    ///
    /// # Panics
    ///
    /// Panics if the handle has no backing texture.  An imported handle
    /// is bound at build time by [`RenderGraph::import_texture`]; a
    /// transient is bound by [`CompiledGraph::allocate_transients`].  A
    /// miss here is a programmer error.
    pub fn texture(&self, handle: TextureHandle) -> &wgpu::Texture {
        match self.entries[handle.0.resource as usize]
            .as_ref()
            .unwrap_or_else(|| panic!(
                "texture handle (resource={}) not bound — import via \
                 RenderGraph::import_texture or allocate via \
                 CompiledGraph::allocate_transients",
                handle.0.resource,
            ))
        {
            ResourceEntry::Texture(t, _) => t,
            _ => panic!("handle (resource={}) is a buffer, not a texture", handle.0.resource),
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
    /// Panics if the handle has no backing texture.  An imported handle
    /// is bound at build time by [`RenderGraph::import_texture`]; a
    /// transient is bound by [`CompiledGraph::allocate_transients`].  A
    /// miss here is a programmer error.
    pub fn texture_view(&self, handle: TextureHandle) -> &wgpu::TextureView {
        match self.entries[handle.0.resource as usize]
            .as_ref()
            .unwrap_or_else(|| panic!(
                "texture handle (resource={}) not bound — import via \
                 RenderGraph::import_texture or allocate via \
                 CompiledGraph::allocate_transients",
                handle.0.resource,
            ))
        {
            ResourceEntry::Texture(_, v) => v,
            _ => panic!("handle (resource={}) is a buffer, not a texture", handle.0.resource),
        }
    }

    /// Look up the resolved [`wgpu::BindGroup`] for a render-graph handle.
    ///
    /// # Panics
    ///
    /// Panics if the bind group has not been resolved.  Call
    /// [`CompiledGraph::resolve_bind_groups`] after
    /// [`allocate_transients`](CompiledGraph::allocate_transients) and
    /// before [`execute`](CompiledGraph::execute) to populate them.
    pub fn bind_group(&self, handle: BindGroupHandle) -> &wgpu::BindGroup {
        self.bind_groups[handle.0 as usize]
            .as_ref()
            .unwrap_or_else(|| panic!(
                "bind group handle ({}) not resolved — call \
                 CompiledGraph::resolve_bind_groups before execute",
                handle.0,
            ))
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
/// Build a graph by calling [`import_buffer`](Self::import_buffer) /
/// [`import_texture`](Self::import_texture) (for external, caller-owned
/// resources, which are bound at build time),
/// [`create_buffer`](Self::create_buffer) /
/// [`create_texture`](Self::create_texture) (for pool-backed transients),
/// [`add_pass`](Self::add_pass), and [`mark_output`](Self::mark_output),
/// then [`compile`](Self::compile) it into a [`CompiledGraph`].
pub struct RenderGraph {
    passes                  : Vec<PassData>,
    resource_count          : u32,
    /// Per-resource current version — incremented each time a pass writes.
    resource_versions       : Vec<u32>,
    /// Output resource versions seeded by mark_output / present.
    output_versions         : Vec<ResourceHandle>,
    transient_buffer_descs  : Vec<(BufferHandle, BufferDesc, String)>,
    transient_texture_descs : Vec<(TextureHandle, TextureDesc, String)>,
    /// Imported buffers bound at build time; written into the compiled
    /// [`ResourceMap`] by [`compile`](Self::compile).
    imported_buffers        : Vec<(u32, wgpu::Buffer)>,
    /// Imported textures bound at build time; written into the compiled
    /// [`ResourceMap`] by [`compile`](Self::compile).
    imported_textures       : Vec<(u32, wgpu::Texture)>,
    /// Bind-group templates registered via [`create_bind_group`] and
    /// resolved into [`wgpu::BindGroup`]s during
    /// [`CompiledGraph::resolve_bind_groups`].
    bind_group_templates    : Vec<BindGroupTemplate>,
    /// Swapchain image for this frame, owned by the graph so that
    /// [`CompiledGraph::execute`] can hand back a token that presents it
    /// after the caller submits.  `Some` after [`present`](Self::present) is
    /// called; `None` for validation/test/headless paths.  Paired with the
    /// resource id of the imported texture so [`compile`](Self::compile) can
    /// seed `output_versions` with the current version at compile time.
    present_target          : Option<(u32, SurfaceFrame)>,
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
            imported_buffers        : Vec::new(),
            imported_textures       : Vec::new(),
            bind_group_templates    : Vec::new(),
            present_target          : None,
        }
    }

    /// Import a persistent (application-owned) buffer into the graph,
    /// binding it to a version-0 handle at build time.
    ///
    /// Passes that write to the returned handle receive a new version from
    /// [`PassBuilder::write_buffer`].  The buffer is resolved directly at
    /// execute time via [`ResourceMap::buffer`] — no post-compile bind step.
    pub fn import_buffer(&mut self, buffer: wgpu::Buffer) -> BufferHandle {
        let handle = self.alloc_buffer_handle();
        self.imported_buffers.push((handle.0.resource, buffer));
        handle
    }

    /// Import a persistent (application-owned) texture into the graph,
    /// binding it to a version-0 handle at build time.
    ///
    /// Passes that write to the returned handle receive a new version from
    /// [`PassBuilder::write_texture`].  A default [`wgpu::TextureView`] is
    /// created during [`compile`](Self::compile) and resolved at execute
    /// time via [`ResourceMap::texture_view`].
    pub fn import_texture(&mut self, texture: wgpu::Texture) -> TextureHandle {
        let handle = self.alloc_texture_handle();
        self.imported_textures.push((handle.0.resource, texture));
        handle
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

    /// Register a deferred bind group against `pipeline`'s reflected layout.
    ///
    /// When `gpu_consts` is `Some`, the referenced buffer is bound at
    /// [`GpuConsts::SLOT`] and the caller must omit that slot from `entries`.
    /// When `None`, slot 0 (if the pipeline declares it) must be supplied by
    /// the caller through `entries` like any other binding.
    ///
    /// Returns an opaque [`BindGroupHandle`].  The actual wgpu bind group
    /// is produced during [`CompiledGraph::resolve_bind_groups`] — the
    /// call must happen *after*
    /// [`allocate_transients`](CompiledGraph::allocate_transients) so all
    /// referenced resources have backing GPU handles, and *before*
    /// [`execute`](CompiledGraph::execute).  Inside execute closures,
    /// look up the resolved group via
    /// [`ResourceMap::bind_group`](ResourceMap::bind_group).
    ///
    /// `ResourceId` may come from either a [`BufferHandle`] or a
    /// [`TextureHandle`] (via [`From`]); the resolver branches on the
    /// underlying [`ResourceEntry`] kind.
    pub fn create_bind_group(
        &mut self,
        label      : &str,
        pipeline   : &dyn PipelineBindLayout,
        gpu_consts : Option<&GpuConsts>,
        entries    : &[(u32, ResourceId)],
    )
        -> BindGroupHandle
    {
        let handle = BindGroupHandle(self.bind_group_templates.len() as u32);
        self.bind_group_templates.push(BindGroupTemplate {
            label          : label.to_string(),
            bg_layout      : pipeline.bg_layout().clone(),
            bind_entries   : pipeline.bind_entries().to_vec(),
            gpu_consts_buf : gpu_consts.map(|c| c.buffer().clone()),
            entries        : entries.to_vec(),
        });
        handle
    }

    /// Register a bind group against `pipeline`'s set-1 reflected layout.
    ///
    /// For render-graph nodes that own a dedicated descriptor set in their
    /// pipeline (e.g. cull's indirect-buffer binding, or a raster node's
    /// per-draw storage).  Set-1 bind groups are fully caller-composed —
    /// there is no reserved GpuConsts slot.
    ///
    /// # Panics
    ///
    /// Panics if `pipeline` does not declare a set-1 layout — check that the
    /// shader has `[[vk::binding(N, 1)]]` entries.
    pub fn create_bind_group_set1(
        &mut self,
        label    : &str,
        pipeline : &dyn PipelineBindLayout,
        entries  : &[(u32, ResourceId)],
    )
        -> BindGroupHandle
    {
        let handle = BindGroupHandle(self.bind_group_templates.len() as u32);
        self.bind_group_templates.push(BindGroupTemplate {
            label          : label.to_string(),
            bg_layout      : pipeline.bg_layout_set1()
                .expect(
                    "create_bind_group_set1: pipeline does not declare set 1 \
                     — check shader has [[vk::binding(N, 1)]] entries",
                )
                .clone(),
            bind_entries   : pipeline.bind_entries_set1().to_vec(),
            gpu_consts_buf : None,
            entries        : entries.to_vec(),
        });
        handle
    }

    /// Mark a buffer version as a graph output.
    ///
    /// The pass that wrote `handle`'s version — and all passes it
    /// transitively depends on — will not be culled during compilation.
    pub fn mark_output(&mut self, handle: BufferHandle) {
        self.output_versions.push(handle.0);
    }

    /// Mark a texture version as a graph output.
    ///
    /// Same semantics as [`mark_output`](Self::mark_output), but accepts
    /// a [`TextureHandle`].  Used by producers whose consumer has not
    /// landed yet (e.g. phase-1.4's shaded-colour transient, which the
    /// phase-1.5 blit pass will read in a later task) so the upstream
    /// passes are not dead-culled during compile.
    pub fn mark_texture_output(&mut self, handle: TextureHandle) {
        self.output_versions.push(handle.0);
    }

    /// Hand the frame's swapchain image to the graph as the frame output.
    ///
    /// Imports the surface texture, stashes the [`SurfaceFrame`] for
    /// post-submit presentation, and returns a [`TextureHandle`] that passes
    /// render into.  The version of this resource at [`compile`](Self::compile)
    /// time — i.e. whatever the final pass wrote — is automatically seeded
    /// into `output_versions`, so callers do not make a second "mark as
    /// output" call.  The paired [`SurfacePresent`] token returned from
    /// [`CompiledGraph::execute`] must be presented after the caller's queue
    /// submit.
    ///
    /// # Panics
    ///
    /// Panics if called twice on the same graph — a frame has exactly one
    /// swapchain output.
    pub fn present(&mut self, surface_frame: SurfaceFrame) -> TextureHandle {
        assert!(
            self.present_target.is_none(),
            "RenderGraph::present called twice — a frame has exactly one \
             swapchain output",
        );
        let handle = self.alloc_texture_handle();
        self.imported_textures.push((handle.0.resource, surface_frame.texture_clone()));
        self.present_target = Some((handle.0.resource, surface_frame));
        handle
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
    pub fn compile(mut self) -> Result<CompiledGraph, CompileError> {
        // Seed the present-target output with the resource's current version
        // (= the final pass's write).  Done here rather than in `present()`
        // because the handle returned there is version 0 — no passes have
        // run yet at build time.
        if let Some((resource, _)) = &self.present_target {
            let version = self.resource_versions[*resource as usize];
            self.output_versions.push(ResourceHandle { resource: *resource, version });
        }

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
                    .map(|pa| pa.handle.resource)
            })
            .collect();

        let transient_buffer_descs: Vec<(BufferHandle, BufferDesc, String)> =
            self.transient_buffer_descs.into_iter()
                .filter(|(h, _, _)| live_writes.contains(&h.0.resource))
                .collect();

        let transient_texture_descs: Vec<(TextureHandle, TextureDesc, String)> =
            self.transient_texture_descs.into_iter()
                .filter(|(h, _, _)| live_writes.contains(&h.0.resource))
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
        let mut entries: Vec<Option<ResourceEntry>> = (0..entry_count)
            .map(|_| None)
            .collect();

        // Pre-fill entries with build-time-bound imports.  Transient entries
        // are filled later by [`CompiledGraph::allocate_transients`].
        for (resource, buffer) in self.imported_buffers {
            entries[resource as usize] = Some(ResourceEntry::Buffer(buffer));
        }
        for (resource, texture) in self.imported_textures {
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            entries[resource as usize] = Some(ResourceEntry::Texture(texture, view));
        }

        let bind_group_count = self.bind_group_templates.len();
        let bind_groups: Vec<Option<wgpu::BindGroup>> =
            (0..bind_group_count).map(|_| None).collect();

        Ok(CompiledGraph {
            passes,
            barriers               : result.barriers,
            culled_count           : result.culled_count,
            entries,
            bind_groups,
            transient_buffer_descs,
            transient_texture_descs,
            bind_group_templates   : self.bind_group_templates,
            present_target         : self.present_target.map(|(_, frame)| frame),
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
        BufferHandle(ResourceHandle { resource, version: 0 })
    }

    fn alloc_texture_handle(&mut self) -> TextureHandle {
        let resource = self.alloc_resource();
        TextureHandle(ResourceHandle { resource, version: 0 })
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
        self.record_read(h.0);
    }

    /// Declare that this pass writes a buffer, minting a new version.
    ///
    /// Returns the new versioned handle.  Downstream passes that want to
    /// observe this write must use the returned handle.
    pub fn write_buffer(&mut self, h: BufferHandle) -> BufferHandle {
        BufferHandle(self.record_write(h.0))
    }

    /// Declare that this pass reads the specific texture version in `h`.
    pub fn read_texture(&mut self, h: TextureHandle) {
        self.record_read(h.0);
    }

    /// Declare that this pass writes a texture, minting a new version.
    ///
    /// Returns the new versioned handle.
    pub fn write_texture(&mut self, h: TextureHandle) -> TextureHandle {
        TextureHandle(self.record_write(h.0))
    }

    /// Auto-record pass accesses from a bind group's pipeline entries.
    ///
    /// Walks each entry of the bind group's template, pairs it with the
    /// matching [`BindEntry`](crate::pipeline::binding::BindEntry) from the
    /// pipeline's reflected layout, and records one access per entry:
    ///
    /// - [`BindKind::UniformBuffer`] / [`BindKind::StorageBufferReadOnly`]:
    ///   [`Access::Read`] at the resource's current version.
    /// - [`BindKind::StorageBufferReadWrite`]: [`Access::Write`] at a
    ///   newly-minted version.
    ///
    /// The returned [`BindGroupWrites`] exposes the post-write versioned
    /// handle for each read-write binding — needed when downstream passes
    /// read the resource by an explicit handle (e.g. an indirect-args
    /// buffer also bound read-write in this bind group).  Reads do not
    /// produce entries in [`BindGroupWrites`]; callers that need the
    /// post-read handle simply use the original.
    ///
    /// The slot-0 `GpuConsts` entry is never part of the template and is
    /// skipped here — it is resolved internally at bind-group creation
    /// and never requires barriers.
    pub fn use_bind_group(&mut self, bg: BindGroupHandle) -> BindGroupWrites {
        // One borrow over the template to build a work list, then a second
        // pass to record accesses without holding the borrow.
        let work: Vec<(ResourceId, BindKind)> = {
            let tpl = &self.graph.bind_group_templates[bg.0 as usize];
            tpl.entries.iter().map(|&(binding, res_id)| {
                let entry = tpl.bind_entries.iter()
                    .find(|e| e.binding == binding)
                    .unwrap_or_else(|| panic!(
                        "bind group template entry at slot {binding} has no \
                         matching layout entry — layout/template mismatch is \
                         a programmer error",
                    ));
                (res_id, entry.kind)
            }).collect()
        };

        let mut writes: Vec<(u32, u32)> = Vec::new();
        for (res_id, kind) in work {
            match kind {
                BindKind::UniformBuffer { .. }
                | BindKind::StorageBufferReadOnly { .. }
                | BindKind::SampledTexture { .. } => {
                    let version = self.graph.resource_versions[res_id.0 as usize];
                    self.record_access(
                        ResourceHandle { resource: res_id.0, version },
                        Access::Read,
                    );
                }
                BindKind::StorageBufferReadWrite { .. }
                | BindKind::StorageTexture { .. } => {
                    let new_version = self.next_version(res_id.0);
                    self.record_access(
                        ResourceHandle { resource: res_id.0, version: new_version },
                        Access::Write,
                    );
                    writes.push((res_id.0, new_version));
                }
            }
        }
        BindGroupWrites { writes }
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

    fn record_read(&mut self, h: ResourceHandle) {
        self.record_access(h, Access::Read);
    }

    fn record_write(&mut self, h: ResourceHandle) -> ResourceHandle {
        let new_version = self.next_version(h.resource);
        let new_handle = ResourceHandle { resource: h.resource, version: new_version };
        self.record_access(new_handle, Access::Write);
        new_handle
    }

    fn record_access(&mut self, handle: ResourceHandle, access: Access) {
        self.graph.passes[self.pass_index].accesses.push(compile::PassAccess {
            handle,
            access,
        });
    }

    fn next_version(&mut self, resource: u32) -> u32 {
        let v = &mut self.graph.resource_versions[resource as usize];
        *v += 1;
        *v
    }
}

// --- BindGroupWrites ---

/// Post-[`use_bind_group`](PassBuilder::use_bind_group) lookup table for
/// versioned write handles.
///
/// For every [`BindKind::StorageBufferReadWrite`] binding in the bind
/// group, `use_bind_group` mints a new resource version and records a
/// [`Write`](Access::Write) access.  Callers retrieve the resulting
/// versioned handle via [`write_of`](Self::write_of) — typically to
/// thread the output to a downstream pass that reads the same resource
/// by an explicit handle (outside the bind group).
///
/// Read-only bindings do not produce entries here: the graph's version
/// counter already resolves their read versions implicitly, and downstream
/// reads of the same resource through another bind group pick up the
/// latest version at pass-build time.
pub struct BindGroupWrites {
    /// `(resource_id, new_version)` pairs in bind-group entry order.
    writes: Vec<(u32, u32)>,
}

impl BindGroupWrites {
    /// Return the versioned [`TextureHandle`] produced by the bind
    /// group's storage-texture binding on `h`'s resource.
    ///
    /// # Panics
    ///
    /// Same as [`write_of`](Self::write_of) — the lookup is by resource
    /// id, the `TextureHandle` wrapper is purely for type safety on the
    /// caller side.
    pub fn write_texture_of(&self, h: TextureHandle) -> TextureHandle {
        let (_, version) = *self.writes.iter()
            .find(|(r, _)| *r == h.0.resource)
            .unwrap_or_else(|| panic!(
                "BindGroupWrites::write_texture_of: resource {} is not a \
                 storage-texture binding in this bind group — check the \
                 layout kind or ensure the correct bind group was passed \
                 to use_bind_group",
                h.0.resource,
            ));
        TextureHandle(ResourceHandle { resource: h.0.resource, version })
    }

    /// Return the versioned handle produced by the bind group's
    /// read-write binding on `h`'s resource.
    ///
    /// # Panics
    ///
    /// Panics if `h.resource` is not bound read-write in the bind group
    /// that produced this `BindGroupWrites`.  A typical cause is looking
    /// up a read-only binding (the graph has no new version for it —
    /// use the original handle), or mismatching bind groups between the
    /// `use_bind_group` call and the `write_of` lookup.
    pub fn write_of(&self, h: BufferHandle) -> BufferHandle {
        let (_, version) = *self.writes.iter()
            .find(|(r, _)| *r == h.0.resource)
            .unwrap_or_else(|| panic!(
                "BindGroupWrites::write_of: resource {} is not a read-write \
                 binding in this bind group — check the layout kind or \
                 ensure the correct bind group was passed to use_bind_group",
                h.0.resource,
            ));
        BufferHandle(ResourceHandle { resource: h.0.resource, version })
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
    bind_groups             : Vec<Option<wgpu::BindGroup>>,
    transient_buffer_descs  : Vec<(BufferHandle, BufferDesc, String)>,
    transient_texture_descs : Vec<(TextureHandle, TextureDesc, String)>,
    bind_group_templates    : Vec<BindGroupTemplate>,
    /// Swapchain frame to hand back as a [`SurfacePresent`] token at the end
    /// of [`execute`](Self::execute).  `None` for graphs that did not call
    /// [`RenderGraph::present`] (validation harness, tests).
    present_target          : Option<SurfaceFrame>,
}

// --- SurfacePresent ---

/// Swapchain-present token returned from [`CompiledGraph::execute`].
///
/// Holds the [`SurfaceFrame`] for the frame the graph rendered into.  The
/// caller must call [`present`](Self::present) *after* the queue submit
/// containing the frame's commands — wgpu requires the submit to happen
/// before the present, and dropping the token without presenting silently
/// drops the frame.
///
/// A no-op (wraps `None`) when the graph was not handed a swapchain image,
/// so headless and validation paths can keep a uniform call shape.
pub struct SurfacePresent {
    frame: Option<SurfaceFrame>,
}

impl SurfacePresent {
    /// Present the frame's swapchain image, or no-op if the graph did not
    /// have a swapchain target.
    ///
    /// Must be called after the queue submit containing this frame's
    /// commands.
    pub fn present(self) {
        if let Some(frame) = self.frame {
            frame.present();
        }
    }
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

    /// Allocate transient buffers and textures from their respective pools.
    ///
    /// Each transient resource created during the build phase (and not culled)
    /// is allocated from the appropriate pool.  Imported resources were bound
    /// at build time by [`RenderGraph::import_buffer`] /
    /// [`RenderGraph::import_texture`] and need no further attention.
    ///
    /// # Panics
    ///
    /// Panics if a transient descriptor references a resource index out of
    /// range, which is a compiler bug.
    fn allocate_transients(
        &mut self,
        buf_pool : &mut BufferPool,
        tex_pool : &mut TexturePool,
        device   : &wgpu::Device,
        frame    : FrameIndex,
    ) {
        for (handle, desc, name) in &self.transient_buffer_descs {
            let buffer = buf_pool.acquire(device, desc, Some(name.as_str()), frame);
            self.entries[handle.0.resource as usize] = Some(ResourceEntry::Buffer(buffer));
        }

        for (handle, desc, name) in &self.transient_texture_descs {
            let texture = tex_pool.acquire(device, desc, Some(name.as_str()));
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.entries[handle.0.resource as usize] = Some(ResourceEntry::Texture(texture, view));
        }
    }

    /// Resolve every registered bind group into a [`wgpu::BindGroup`].
    ///
    /// Called after [`allocate_transients`](Self::allocate_transients) — every
    /// referenced transient must already have its backing buffer/texture
    /// in the entries array.  A template referencing a resource that is still
    /// `None` (e.g. a transient that was culled because nothing writes it)
    /// panics with the offending label and resource id.
    fn resolve_bind_groups(&mut self, device: &wgpu::Device) {
        for (i, tpl) in self.bind_group_templates.iter().enumerate() {
            let has_consts = tpl.gpu_consts_buf.is_some();
            let mut bg_entries: Vec<wgpu::BindGroupEntry> =
                Vec::with_capacity(tpl.entries.len() + usize::from(has_consts));

            // Slot 0: GpuConsts — present for externally-composed set-0 bind
            // groups; absent for caller-composed set-N bind groups.
            if let Some(consts_buf) = &tpl.gpu_consts_buf {
                bg_entries.push(wgpu::BindGroupEntry {
                    binding  : GpuConsts::SLOT,
                    resource : consts_buf.as_entire_binding(),
                });
            }

            // User slots: resolve each ResourceId via entries array.
            for &(binding, res_id) in &tpl.entries {
                let entry = self.entries[res_id.0 as usize]
                    .as_ref()
                    .unwrap_or_else(|| panic!(
                        "bind group '{}' references resource {} which has \
                         no backing GPU resource — import it, call \
                         allocate_transients, or remove the reference",
                        tpl.label, res_id.0,
                    ));

                let resource = match entry {
                    ResourceEntry::Buffer(b)      => b.as_entire_binding(),
                    ResourceEntry::Texture(_, v)  => wgpu::BindingResource::TextureView(v),
                };

                bg_entries.push(wgpu::BindGroupEntry { binding, resource });
            }

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label   : Some(&tpl.label),
                layout  : &tpl.bg_layout,
                entries : &bg_entries,
            });

            self.bind_groups[i] = Some(bg);
        }
    }

    /// Execute all passes in compiled order.
    ///
    /// Allocates transient resources from the pools, resolves bind groups, then
    /// constructs a [`Commands`] recorder from `encoder` and calls each pass's
    /// execute closure sequentially with a [`PassContext`].
    ///
    /// Returns a [`PendingRelease`] holding the transient resources — the
    /// caller must hold it until the GPU has completed the submit containing
    /// this frame's commands, then call [`PendingRelease::release`] to return
    /// the resources to the pools — and a [`SurfacePresent`] token that
    /// presents the swapchain image after submit (a no-op for graphs without
    /// a [`RenderGraph::present`] call).
    pub fn execute(
        mut self,
        encoder  : &mut FrameEncoder,
        frame    : FrameIndex,
        buf_pool : &mut BufferPool,
        tex_pool : &mut TexturePool,
        device   : &wgpu::Device,
    )
        -> (PendingRelease, SurfacePresent)
    {
        self.allocate_transients(buf_pool, tex_pool, device, frame);
        self.resolve_bind_groups(device);

        let Self {
            passes,
            transient_buffer_descs,
            transient_texture_descs,
            entries,
            bind_groups,
            barriers,
            present_target,
            ..
        } = self;

        // Clone transient resource arcs before ResourceMap takes ownership.
        let transient_buffers: Vec<wgpu::Buffer> = transient_buffer_descs.iter()
            .filter_map(|(handle, _, _)| {
                if let Some(ResourceEntry::Buffer(b)) = entries[handle.0.resource as usize].as_ref() {
                    Some(b.clone())
                }
                else {
                    None
                }
            })
            .collect();

        let transient_textures: Vec<wgpu::Texture> = transient_texture_descs.iter()
            .filter_map(|(handle, _, _)| {
                if let Some(ResourceEntry::Texture(t, _)) = entries[handle.0.resource as usize].as_ref() {
                    Some(t.clone())
                }
                else {
                    None
                }
            })
            .collect();

        let resources = ResourceMap { entries, bind_groups };

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

        let pending = PendingRelease {
            buffers  : transient_buffers,
            textures : transient_textures,
        };
        let present = SurfacePresent { frame: present_target };
        (pending, present)
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
        let orphan = graph.create_buffer("orphan", desc());

        let _buf = graph.create_buffer("unrelated_buf", desc());
        graph.add_pass("unrelated", |pass| {
            pass.write_buffer(_buf);
        });

        // Nobody writes orphan (version 0 has no producer), so no live pass
        // is seeded.
        graph.mark_output(orphan);

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
        let shared = graph.create_buffer("shared", desc());

        let out_a = graph.create_buffer("out_a", desc());
        let out_a_v1 = graph.add_pass("reader_a", |pass| {
            pass.read_buffer(shared);
            pass.write_buffer(out_a)
        });

        let out_b = graph.create_buffer("out_b", desc());
        let out_b_v1 = graph.add_pass("reader_b", |pass| {
            pass.read_buffer(shared);
            pass.write_buffer(out_b)
        });

        graph.mark_output(out_a_v1);
        graph.mark_output(out_b_v1);

        let compiled = graph.compile().unwrap();

        let buf_barriers: Vec<_> = compiled.barriers().iter()
            .filter(|b| b.resource == ResourceId::from(shared))
            .collect();

        assert!(buf_barriers.is_empty());
    }

    #[test]
    fn consecutive_writes_produce_barrier() {
        let mut graph = RenderGraph::new();
        let shared = graph.create_buffer("shared", desc());

        let shared_v1 = graph.add_pass("writer_a", |pass| pass.write_buffer(shared));

        let out = graph.create_buffer("out", desc());
        let out_v1 = graph.add_pass("writer_b", |pass| {
            let v2 = pass.write_buffer(shared_v1);
            let ov = pass.write_buffer(out);
            // We need writer_b to depend on shared_v1 via the write chain.
            // write_buffer(shared_v1) returns shared_v2, but we don't use it
            // here — the WAW dependency is encoded in the graph.
            let _ = v2;
            ov
        });

        graph.mark_output(out_v1);

        let compiled = graph.compile().unwrap();

        // Both passes live (WAW dependency keeps writer_a alive).
        assert_eq!(compiled.pass_names(), ["writer_a", "writer_b"]);

        let buf_barriers: Vec<_> = compiled.barriers().iter()
            .filter(|b| b.resource == ResourceId::from(shared))
            .collect();

        assert_eq!(buf_barriers.len(), 1);
        assert_eq!(buf_barriers[0].src, Access::Write);
        assert_eq!(buf_barriers[0].dst, Access::Write);
    }

    #[test]
    fn buffer_write_read_chain() {
        let mut graph = RenderGraph::new();
        let shared = graph.create_buffer("shared", desc());

        let shared_v1 = graph.add_pass("upload", |pass| pass.write_buffer(shared));

        let out = graph.create_buffer("out", desc());
        let out_v1 = graph.add_pass("consume", |pass| {
            pass.read_buffer(shared_v1);
            pass.write_buffer(out)
        });

        graph.mark_output(out_v1);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_names(), ["upload", "consume"]);

        let buf_barriers: Vec<_> = compiled.barriers().iter()
            .filter(|b| b.resource == ResourceId::from(shared))
            .collect();

        assert_eq!(buf_barriers.len(), 1);
        assert_eq!(buf_barriers[0].src, Access::Write);
        assert_eq!(buf_barriers[0].dst, Access::Read);
    }

    // -- SSA versioning tests --

    #[test]
    fn two_writes_produce_different_handles() {
        let mut graph = RenderGraph::new();

        let buf = graph.create_buffer("buf", desc());
        let v1 = graph.add_pass("writer_a", |pass| pass.write_buffer(buf));
        let v2 = graph.add_pass("writer_b", |pass| pass.write_buffer(v1));

        // Different versions, same resource.
        assert_eq!(v1.0.resource, buf.0.resource);
        assert_eq!(v2.0.resource, buf.0.resource);
        assert_ne!(v1.0.version, v2.0.version);
        assert_ne!(v1.0.version, buf.0.version);
    }

    #[test]
    fn reader_of_first_version_culls_second_writer() {
        let mut graph = RenderGraph::new();

        let buf = graph.create_buffer("buf", desc());

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

        let buf = graph.create_buffer("buf", desc());

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

        let expected = gpu_buf.clone();
        let ran = std::sync::Arc::new(AtomicBool::new(false));
        let ran_clone = ran.clone();

        let mut graph = RenderGraph::new();
        let h = graph.import_buffer(gpu_buf);

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

        let compiled = graph.compile().unwrap();

        let mut buf_pool = BufferPool::new();
        let mut tex_pool = TexturePool::new();
        let mut encoder = ctx.begin_frame();
        let frame = ctx.frame_index();
        let _pending = compiled.execute(&mut encoder, frame, &mut buf_pool, &mut tex_pool, ctx.device());
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
        let buf = pool.acquire(ctx.device(), &d, None, FrameIndex::default());
        assert_eq!(buf.size(), 128);

        // Release and re-acquire: same buffer returned (Arc identity).
        let reference = buf.clone();
        pool.release(buf);
        let buf2 = pool.acquire(ctx.device(), &d, None, FrameIndex::default());
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

        let buf_small = pool.acquire(ctx.device(), &d_small, None, FrameIndex::default());
        let ref_small = buf_small.clone();
        pool.release(buf_small);

        // Different size — should get a new buffer, not the recycled one.
        let buf_large = pool.acquire(ctx.device(), &d_large, None, FrameIndex::default());
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

        let gpu_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_imported"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mut graph = RenderGraph::new();
        let imported = graph.import_buffer(gpu_buf);

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

        let compiled = graph.compile().unwrap();

        let mut encoder = ctx.begin_frame();
        let frame = ctx.frame_index();
        let (pending, _present) = compiled.execute(&mut encoder, frame, &mut pool, &mut tex_pool, ctx.device());
        ctx.end_frame(encoder);

        assert!(ran.load(Ordering::SeqCst), "pass closure must have run");
        assert_eq!(pending.len(), 1);

        // Release back to pool — pool should now have a recyclable buffer.
        pending.release(&mut pool, &mut tex_pool);
        let recycled = pool.acquire(ctx.device(), &BufferDesc {
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
        }, None, FrameIndex::default());
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

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.barriers().len(), 1);
        assert_eq!(compiled.barriers()[0].src, Access::Write);
        assert_eq!(compiled.barriers()[0].dst, Access::Read);
        assert_eq!(compiled.barriers()[0].before, 1);

        let mut encoder = ctx.begin_frame();
        let frame = ctx.frame_index();
        let (pending, _present) = compiled.execute(&mut encoder, frame, &mut pool, &mut tex_pool, ctx.device());
        ctx.end_frame(encoder);

        // Both closures must have run in the declared order; barrier emission
        // must not have skipped or reordered any passes.
        assert_eq!(*order.lock().unwrap(), ["writer", "reader"]);

        pending.release(&mut pool, &mut tex_pool);
    }

}
