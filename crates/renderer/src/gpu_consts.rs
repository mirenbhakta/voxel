//! Shared GPU constants uniform buffer.
//!
//! [`GpuConstsData`] is the single source of truth for constants read by every
//! shader in the renderer. It is bound at slot 0 of every pipeline's bind
//! group layout (asserted by reflection at pipeline construction).
//! See [`GpuConsts::SLOT`].

use crate::device::RendererContext;

/// Shared constants read by all shaders via a uniform buffer bound at
/// [`GpuConsts::SLOT`] of every pipeline's bind group layout.
///
/// Updated rarely — this is a "constants" table, not a per-frame ring. Later
/// passes grow this table as real shared constants appear.
///
/// The `_reserved*` fields hold byte positions formerly used by per-ring slot
/// machinery (`upload_slot`, `readback_slot`, `upload_capacity`) which moved
/// to render-graph-managed resource resolution. They exist to preserve the
/// 32-byte layout agreement with the HLSL mirror and are available for future
/// graph-level constants.
///
/// ## Layout agreement with HLSL
///
/// `size_of::<GpuConstsData>()` must equal `sizeof(GpuConsts{in HLSL})`,
/// field-for-field, against the mirror struct in
/// `shaders/include/gpu_consts.hlsl`. This is enforced at three levels:
///
/// 1. A Rust const assertion: `size_of::<GpuConstsData>() == 32` (below).
///    Any change that breaks the size fails to compile.
/// 2. A runtime SPIR-V reflection check that asserts the HLSL uniform buffer
///    size reflected out of the compiled shader matches the Rust side.
/// 3. Behavioral verification via the validation binary's sentinel roundtrip
///    — any layout disagreement produces a garbage echo which the test
///    detects loudly.
///
/// ## Padding fields
///
/// The first pass has no `vec3`/`vec4` members, so the only `std140` alignment
/// rule in play is the struct-level 16-byte multiple, which `8 * u32 = 32`
/// already satisfies. The `_pad*` fields exist explicitly so future additions
/// default to "16-byte aligned unless someone consciously repacks."
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuConstsData {
    // --- Reserved (formerly ring slot machinery; now graph-managed) ---
    #[doc(hidden)]
    pub _reserved0: u32,
    #[doc(hidden)]
    pub _reserved1: u32,
    /// Frame count (2..=4). Used by shaders that need frame-in-flight context.
    pub frame_count: u32,
    #[doc(hidden)]
    pub _reserved2: u32,

    // --- Validation binary only ---
    /// A sentinel the CPU writes each frame, which the validation shader
    /// echoes into the readback channel for roundtrip verification. Normal
    /// renderer code leaves this at zero.
    pub frame_sentinel: u32,

    // --- std140 padding to a 16-byte multiple (see doc comment above) ---
    #[doc(hidden)]
    pub _pad0: u32,
    #[doc(hidden)]
    pub _pad1: u32,
    #[doc(hidden)]
    pub _pad2: u32,
}

// Load-bearing invariant: this struct is 32 bytes, matching the HLSL mirror.
// A mismatch would produce silent garbage reads on the GPU side; we want the
// loudest possible signal, which is a compile error.
const _: () = assert!(std::mem::size_of::<GpuConstsData>() == 32);

// Alignment must be 4 (the alignment of u32, the largest member under
// #[repr(C)]). A change here means someone added a wider member, which
// would alter padding and break the field-for-field layout agreement with
// the HLSL mirror — force a review.
const _: () = assert!(std::mem::align_of::<GpuConstsData>() == 4);

/// GPU-resident uniform buffer holding a single snapshot of [`GpuConstsData`].
///
/// Owns one persistent `wgpu::Buffer` sized for the Pod struct. Updates flow
/// through a CPU-side shadow with a dirty flag — [`Self::data_mut`] sets the
/// flag, [`Self::upload_if_dirty`] flushes via `queue.write_buffer` and clears
/// it. The typical pattern is a single upload per frame from the
/// `RendererContext::begin_frame` hook landing in a later increment.
///
/// ## Why `GpuConsts` is not a ring
///
/// Principle 1 forbids single-copy buffers mutated on both the CPU and GPU
/// sides. `GpuConsts` is not mutated on both sides: the CPU writes, the GPU
/// reads, there is no GPU write back. The concerning race on the same bytes
/// does not arise. The sub-case of "CPU writes while GPU is mid-read of the
/// old value" is handled by wgpu's implicit ordering: `queue.write_buffer`
/// is sequenced before subsequent command submissions within a queue, so a
/// write in frame `k+1` is visible to shaders in frame `k+1` and invisible
/// to shaders still running from frame `k`. This is effectively a "published
/// immutable snapshot that flips atomically at frame boundaries" — which is
/// exactly the workload uniforms are good at.
pub struct GpuConsts {
    // `buffer` is read back via the `buffer()` accessor, whose first caller
    // lands in a later increment (bind group construction for the validation
    // binary). The allow is the same pattern Increment 3 used for the
    // device/queue fields before GpuConsts itself consumed them.
    #[allow(dead_code)]
    buffer: wgpu::Buffer,
    data: GpuConstsData,
    dirty: bool,
}

impl GpuConsts {
    /// The descriptor binding slot reserved for `GpuConstsData`.
    ///
    /// This is the single number that both the HLSL side
    /// (`[[vk::binding(0, 0)]] ConstantBuffer<GpuConsts>` in
    /// `shaders/include/gpu_consts.hlsl`) and every pipeline constructor
    /// commit to. Pipeline constructors assert slot 0 is a `UniformBuffer`
    /// sized to `GpuConstsData` during reflection.
    pub const SLOT: u32 = 0;

    /// Create a new uniform buffer initialized with the given constants.
    ///
    /// Allocates one `wgpu::Buffer` of `size_of::<GpuConstsData>()` bytes with
    /// `UNIFORM | COPY_DST` usage and performs the initial upload via
    /// `queue.write_buffer`. The returned `GpuConsts` is clean; the next call
    /// to [`Self::data_mut`] sets the dirty flag.
    pub fn new(ctx: &RendererContext, initial: GpuConstsData) -> Self {
        let buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("renderer_gpu_consts"),
            size: std::mem::size_of::<GpuConstsData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        ctx.queue()
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&initial));

        Self {
            buffer,
            data: initial,
            dirty: false,
        }
    }

    /// Immutable view of the CPU-side shadow.
    pub fn data(&self) -> &GpuConstsData {
        &self.data
    }

    /// Mutable view of the CPU-side shadow. Sets the dirty flag
    /// unconditionally — a `&mut` that does not modify still pays the dirty
    /// cost. The correct tradeoff: a missing flush is a silent garbage read;
    /// an extra flush is a cheap `queue.write_buffer` of 32 bytes.
    pub fn data_mut(&mut self) -> &mut GpuConstsData {
        self.dirty = true;
        &mut self.data
    }

    /// If dirty, upload the CPU shadow to the GPU buffer via
    /// `queue.write_buffer` and clear the flag. Intended to be called from
    /// `RendererContext::begin_frame` before any bind group resolution.
    #[allow(dead_code)] // First caller: RendererContext::begin_frame (later increment).
    pub(crate) fn upload_if_dirty(&mut self, ctx: &RendererContext) {
        if !self.dirty {
            return;
        }
        ctx.queue()
            .write_buffer(&self.buffer, 0, bytemuck::bytes_of(&self.data));
        self.dirty = false;
    }

    /// Access the underlying wgpu buffer. Only visible to primitives within
    /// the crate — external callers pass a `&GpuConsts` into
    /// `RenderGraph::create_bind_group`, which resolves slot 0 internally.
    #[allow(dead_code)] // First caller: bind group construction in the validation binary (later increment).
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameCount;

    /// The reserved slot constant is zero — this is the single number that the
    /// shader-side `[[vk::binding(0, 0)]] ConstantBuffer<GpuConsts>` in
    /// `shaders/include/gpu_consts.hlsl` also commits to.
    #[test]
    fn gpu_consts_slot_is_zero() {
        assert_eq!(GpuConsts::SLOT, 0);
    }

    /// Pure-CPU size assertion — mirrors the const assert above, but surfaces
    /// the number in a test report rather than a build error when it breaks.
    #[test]
    fn gpu_consts_data_is_32_bytes() {
        assert_eq!(std::mem::size_of::<GpuConstsData>(), 32);
    }

    /// Pure-CPU default check — `Zeroable` + `Default` derives should produce
    /// an all-zero struct with no surprises.
    #[test]
    fn gpu_consts_data_default_is_zeroed() {
        let d = GpuConstsData::default();
        assert_eq!(d.frame_count, 0);
        assert_eq!(d.frame_sentinel, 0);
    }

    /// GPU smoke test: construct a `GpuConsts`, mutate its data, flush, and
    /// verify the CPU shadow is retained. The "dirty flag actually got
    /// cleared" side of the test is exercised by calling `upload_if_dirty`
    /// twice in a row — the second call is a no-op if the flag was cleared,
    /// which is the observable correctness property from outside the type.
    ///
    /// Gated with `#[ignore]` because it requires a working Vulkan stack;
    /// matches the pattern established in Increment 3.
    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn gpu_consts_roundtrips_through_data_mut_and_upload() {
        let ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine");

        let initial = GpuConstsData {
            frame_count: 2,
            ..Default::default()
        };

        let mut consts = GpuConsts::new(&ctx, initial);
        assert_eq!(consts.data().frame_count, 2);
        assert_eq!(consts.data().frame_sentinel, 0);

        consts.data_mut().frame_sentinel = 42;
        assert_eq!(consts.data().frame_sentinel, 42);

        // First flush: writes the updated shadow to the GPU buffer.
        consts.upload_if_dirty(&ctx);
        // Second flush: should be a no-op because the flag was cleared.
        consts.upload_if_dirty(&ctx);

        // Buffer handle remains valid and readable to primitives.
        assert_eq!(
            consts.buffer().size(),
            std::mem::size_of::<GpuConstsData>() as u64
        );
    }
}
