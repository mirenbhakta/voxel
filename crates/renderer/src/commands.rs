//! Command recorders for GPU work within a render graph pass.
//!
//! [`Commands`] is the primary recorder, obtained from [`PassContext`] during
//! graph execution.  It wraps the frame's `wgpu::CommandEncoder` and exposes
//! typed dispatch and raster-pass entry points.
//!
//! [`RasterPass`] is a scoped raster-pass recorder opened via
//! [`Commands::raster_pass`].  It is dropped (ending the wgpu render pass)
//! when the closure passed to `raster_pass` returns.

use crate::pipeline::compute::ComputePipeline;
use crate::pipeline::render::RenderPipeline;

// --- ColorAttachment ---

/// A single colour attachment for a raster pass.
pub struct ColorAttachment<'a> {
    /// The texture view to render into.
    pub view  : &'a wgpu::TextureView,
    /// `Some(rgba)` = clear to this colour before rendering.  `None` = load
    /// the existing contents.
    pub clear : Option<[f64; 4]>,
    /// Store operation for this attachment after the pass.
    pub store : wgpu::StoreOp,
}

impl<'a> ColorAttachment<'a> {
    /// Clear the attachment to `rgba` at the start of the pass.
    pub fn clear(view: &'a wgpu::TextureView, rgba: [f64; 4]) -> Self {
        Self { view, clear: Some(rgba), store: wgpu::StoreOp::Store }
    }

    /// Load the existing contents of the attachment.
    pub fn load(view: &'a wgpu::TextureView) -> Self {
        Self { view, clear: None, store: wgpu::StoreOp::Store }
    }
}

// --- DepthAttachment ---

/// A depth attachment for a raster pass.
pub struct DepthAttachment<'a> {
    /// The depth texture view.
    pub view  : &'a wgpu::TextureView,
    /// `Some(v)` = clear to depth `v` before rendering.  `None` = load.
    pub clear : Option<f32>,
    /// Store operation for this attachment after the pass.
    pub store : wgpu::StoreOp,
}

impl<'a> DepthAttachment<'a> {
    /// Clear the depth attachment to `v` at the start of the pass.
    pub fn clear(view: &'a wgpu::TextureView, v: f32) -> Self {
        Self { view, clear: Some(v), store: wgpu::StoreOp::Store }
    }

    /// Load the existing depth contents.
    pub fn load(view: &'a wgpu::TextureView) -> Self {
        Self { view, clear: None, store: wgpu::StoreOp::Store }
    }
}

// --- RasterPassDesc ---

/// Parameters for opening a raster pass via [`Commands::raster_pass`].
pub struct RasterPassDesc<'a> {
    /// Debug label forwarded to wgpu.
    pub label : &'a str,
    /// Colour attachments for this pass.
    pub color : &'a [ColorAttachment<'a>],
    /// Optional depth attachment.
    pub depth : Option<DepthAttachment<'a>>,
}

// --- RasterPass ---

/// A scoped raster-pass recorder.
///
/// Constructed inside [`Commands::raster_pass`] and dropped when the closure
/// returns, which ends the underlying wgpu render pass.  Draw calls are
/// recorded through the methods on this type.
pub struct RasterPass<'a> {
    pass: wgpu::RenderPass<'a>,
}

impl RasterPass<'_> {
    /// Record a draw call.
    ///
    /// Sets `pipeline`, binds each group in `bind_groups` at its slice index
    /// (`bind_groups[0]` = set 0, `bind_groups[1]` = set 1, …), and issues a
    /// `draw(vertices, instances)` call.
    pub fn draw(
        &mut self,
        pipeline    : &RenderPipeline,
        bind_groups : &[&wgpu::BindGroup],
        vertices    : std::ops::Range<u32>,
        instances   : std::ops::Range<u32>,
    ) {
        self.pass.set_pipeline(pipeline.inner());
        for (i, bg) in bind_groups.iter().enumerate() {
            self.pass.set_bind_group(i as u32, *bg, &[]);
        }
        self.pass.draw(vertices, instances);
    }

    /// Record an indirect draw call.
    ///
    /// Sets `pipeline`, binds each group in `bind_groups` at its slice index,
    /// then issues `draw_indirect` reading parameters from `indirect_buf` at
    /// `offset`.
    pub fn draw_indirect(
        &mut self,
        pipeline     : &RenderPipeline,
        bind_groups  : &[&wgpu::BindGroup],
        indirect_buf : &wgpu::Buffer,
        offset       : u64,
    ) {
        self.pass.set_pipeline(pipeline.inner());
        for (i, bg) in bind_groups.iter().enumerate() {
            self.pass.set_bind_group(i as u32, *bg, &[]);
        }
        self.pass.draw_indirect(indirect_buf, offset);
    }

    /// Record a multi-draw-indirect call with a CPU-supplied draw count.
    ///
    /// Always available on wgpu 29 — emulated with repeated `draw_indirect`
    /// on backends without native support.  `wgpu::Features::MULTI_DRAW_INDIRECT_COUNT`
    /// (enabled in [`RendererContext`](crate::device::RendererContext)) guarantees
    /// this call is not emulated.  Sets `pipeline`, binds each group in
    /// `bind_groups` at its slice index, then issues `count` indirect draws
    /// reading parameters from `indirect_buf` at `offset`.
    ///
    /// For GPU-sourced draw counts (the matching primitive for GPU culling),
    /// use [`multi_draw_indirect_count`](Self::multi_draw_indirect_count).
    pub fn multi_draw_indirect(
        &mut self,
        pipeline     : &RenderPipeline,
        bind_groups  : &[&wgpu::BindGroup],
        indirect_buf : &wgpu::Buffer,
        offset       : u64,
        count        : u32,
    ) {
        self.pass.set_pipeline(pipeline.inner());
        for (i, bg) in bind_groups.iter().enumerate() {
            self.pass.set_bind_group(i as u32, *bg, &[]);
        }
        self.pass.multi_draw_indirect(indirect_buf, offset, count);
    }

    /// Record a multi-draw-indirect call with a GPU-sourced draw count.
    ///
    /// Requires `wgpu::Features::MULTI_DRAW_INDIRECT_COUNT` on the device
    /// (enabled in [`RendererContext`](crate::device::RendererContext)).
    /// Draws up to `max_count` entries from `indirect_buf` at `indirect_offset`,
    /// stopping at the `u32` value read from `count_buf` at `count_offset`.
    /// This is the matching primitive for GPU culling: the cull shader writes
    /// both the indirect args and the count atomically, and the draw reads
    /// them without a CPU roundtrip.
    ///
    /// Each group in `bind_groups` is bound at its slice index
    /// (`bind_groups[0]` = set 0, `bind_groups[1]` = set 1, …).
    #[allow(clippy::too_many_arguments)]
    pub fn multi_draw_indirect_count(
        &mut self,
        pipeline        : &RenderPipeline,
        bind_groups     : &[&wgpu::BindGroup],
        indirect_buf    : &wgpu::Buffer,
        indirect_offset : u64,
        count_buf       : &wgpu::Buffer,
        count_offset    : u64,
        max_count       : u32,
    ) {
        self.pass.set_pipeline(pipeline.inner());
        for (i, bg) in bind_groups.iter().enumerate() {
            self.pass.set_bind_group(i as u32, *bg, &[]);
        }
        self.pass.multi_draw_indirect_count(
            indirect_buf, indirect_offset,
            count_buf,    count_offset,
            max_count,
        );
    }

    /// Write `data` into the pipeline's immediate-data block at `offset` bytes.
    ///
    /// Applies to all stages declared in the pipeline layout's immediate-data
    /// budget.
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.pass.set_immediates(offset, data);
    }
}

// --- Commands ---

/// Recorder for GPU commands within a single render graph pass.
///
/// Obtained from [`PassContext::commands`](crate::graph::PassContext).  Wraps
/// the frame's `wgpu::CommandEncoder` and exposes typed compute-dispatch and
/// raster-pass entry points.
pub struct Commands<'a> {
    pub(crate) encoder: &'a mut wgpu::CommandEncoder,
}

impl Commands<'_> {
    /// Record a compute dispatch.
    ///
    /// `bind_groups` is bound in order — `bind_groups[0]` is set 0,
    /// `bind_groups[1]` is set 1, and so on.  Callers that only need a
    /// single bind group pass `&[bg]`.
    ///
    /// `workgroups` is the dispatch grid in workgroups — `[x, y, z]`.
    /// `immediates` is the immediate-data payload; pass `&[]` if the pipeline
    /// declares no immediate data.
    pub fn dispatch(
        &mut self,
        pipeline    : &ComputePipeline,
        bind_groups : &[&wgpu::BindGroup],
        workgroups  : [u32; 3],
        immediates  : &[u8],
    ) {
        let mut cpass = self.encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor {
                label: Some(pipeline.label()),
                timestamp_writes: None,
            },
        );
        cpass.set_pipeline(pipeline.inner());
        for (i, bg) in bind_groups.iter().enumerate() {
            cpass.set_bind_group(i as u32, *bg, &[]);
        }
        if !immediates.is_empty() {
            cpass.set_immediates(0, immediates);
        }
        cpass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
    }

    /// Record a 1-D compute dispatch for `count` elements.
    ///
    /// Computes workgroups-X as `count.div_ceil(x)` where `x` is the first
    /// component of [`ComputePipeline::workgroup_size`].  Y and Z are 1.
    ///
    /// `bind_groups` is bound in order — each group is bound at its slice
    /// index, so `[bg0, bg1]` sets group 0 and group 1.
    /// `immediates` is the immediate-data payload; pass `&[]` if the pipeline
    /// declares no immediate data.
    pub fn dispatch_linear(
        &mut self,
        pipeline    : &ComputePipeline,
        bind_groups : &[&wgpu::BindGroup],
        count       : u32,
        immediates  : &[u8],
    ) {
        let wg = pipeline.workgroup_size()[0];
        let x  = count.div_ceil(wg);
        self.dispatch(pipeline, bind_groups, [x, 1, 1], immediates);
    }

    /// Record an indirect compute dispatch.
    ///
    /// `bind_groups` is bound in order — each group is bound at its slice
    /// index, so `[bg0, bg1]` sets group 0 and group 1.
    /// `immediates` is the immediate-data payload; pass `&[]` if the pipeline
    /// declares no immediate data.
    pub fn dispatch_indirect(
        &mut self,
        pipeline     : &ComputePipeline,
        bind_groups  : &[&wgpu::BindGroup],
        indirect_buf : &wgpu::Buffer,
        offset       : u64,
        immediates   : &[u8],
    ) {
        let mut cpass = self.encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor {
                label: Some(pipeline.label()),
                timestamp_writes: None,
            },
        );
        cpass.set_pipeline(pipeline.inner());
        for (i, bg) in bind_groups.iter().enumerate() {
            cpass.set_bind_group(i as u32, *bg, &[]);
        }
        if !immediates.is_empty() {
            cpass.set_immediates(0, immediates);
        }
        cpass.dispatch_workgroups_indirect(indirect_buf, offset);
    }

    /// Record a clear of `size` bytes of `buf` starting at `offset`, or
    /// to the end of the buffer if `size` is `None`.
    ///
    /// Filled with zero bytes. `buf` must carry
    /// [`wgpu::BufferUsages::COPY_DST`] at creation time, and the caller is
    /// responsible for declaring a write on `buf` in the enclosing render-
    /// graph pass so barriers are inserted correctly.
    pub fn clear_buffer(
        &mut self,
        buf    : &wgpu::Buffer,
        offset : u64,
        size   : Option<u64>,
    ) {
        self.encoder.clear_buffer(buf, offset, size);
    }

    /// Record a buffer-to-buffer copy of `size` bytes from `src` at
    /// `src_offset` into `dst` at `dst_offset`.
    ///
    /// The caller is responsible for declaring `src` as read and `dst` as
    /// write on the enclosing render-graph pass so barriers are inserted
    /// correctly — this method only records the copy command. `src` must
    /// carry [`wgpu::BufferUsages::COPY_SRC`] and `dst`
    /// [`wgpu::BufferUsages::COPY_DST`] at creation time.
    pub fn copy_buffer_to_buffer(
        &mut self,
        src        : &wgpu::Buffer,
        src_offset : u64,
        dst        : &wgpu::Buffer,
        dst_offset : u64,
        size       : u64,
    ) {
        self.encoder.copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size);
    }

    /// Open a raster pass, run `f`, then end the pass.
    ///
    /// The [`RasterPass`] is scoped to the closure so its lifetime cannot
    /// escape — the underlying wgpu render pass is ended when `f` returns.
    pub fn raster_pass<R>(
        &mut self,
        desc : &RasterPassDesc<'_>,
        f    : impl FnOnce(&mut RasterPass<'_>) -> R,
    )
        -> R
    {
        let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = desc.color
            .iter()
            .map(|a| {
                Some(wgpu::RenderPassColorAttachment {
                    view          : a.view,
                    depth_slice   : None,
                    resolve_target: None,
                    ops           : wgpu::Operations {
                        load : match a.clear {
                            Some([r, g, b, a]) => wgpu::LoadOp::Clear(
                                wgpu::Color { r, g, b, a },
                            ),
                            None => wgpu::LoadOp::Load,
                        },
                        store: a.store,
                    },
                })
            })
            .collect();

        let depth_attachment =
            desc.depth.as_ref().map(|d| wgpu::RenderPassDepthStencilAttachment {
                view          : d.view,
                depth_ops     : Some(wgpu::Operations {
                    load : match d.clear {
                        Some(v) => wgpu::LoadOp::Clear(v),
                        None    => wgpu::LoadOp::Load,
                    },
                    store: d.store,
                }),
                stencil_ops   : None,
            });

        let pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label                   : Some(desc.label),
            color_attachments       : &color_attachments,
            depth_stencil_attachment: depth_attachment,
            occlusion_query_set     : None,
            timestamp_writes        : None,
            multiview_mask          : None,
        });

        let mut rp = RasterPass { pass };
        f(&mut rp)
    }
}
