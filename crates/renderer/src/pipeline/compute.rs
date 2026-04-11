//! `ComputePipeline` with workgroup-size and GpuConsts-size assertions.
//!
//! [`ComputePipeline`] is the only path to dispatching a compute shader in
//! the renderer. It enforces two invariants at pipeline load time (inside
//! [`ComputePipeline::new`]), before any wgpu call, so a CPU-side mismatch
//! panics deterministically without touching the GPU:
//!
//! 1. **Workgroup-size assertion** (`.local/renderer_plan.md` §7.1): if
//!    `ComputePipelineDescriptor::expected_workgroup_size` is `Some(expected)`,
//!    the value is compared against the `LocalSize` execution mode reflected
//!    from the SPIR-V binary. A mismatch is a `panic!` with a message pointing
//!    at both the expected and actual values — the CPU dispatch code and the
//!    shader `[numthreads(...)]` are out of sync.
//!
//! 2. **GpuConsts size assertion**: if the reflected SPIR-V declares a uniform
//!    buffer at descriptor set 0 binding 0 (the slot [`GpuConstsData`] always
//!    occupies), its byte size is asserted equal to `size_of::<GpuConstsData>()`.
//!    A mismatch indicates a Rust/HLSL layout disagreement.
//!
//! Both assertions fire at construction, not at dispatch, because "fires at
//! load" is explicitly called out in the design (principle 5 / §7.1). If
//! invariants were checked at dispatch the first failure would be invisible in
//! tests that never actually dispatch.
//!
//! ## wgpu::BindGroup leakage
//!
//! [`ComputePipeline::dispatch`] and [`ComputePipeline::dispatch_linear`]
//! accept a `&wgpu::BindGroup` directly. This is a deliberate exception to
//! the renderer's containment discipline (principle 3) — a thin wrapper around
//! `wgpu::BindGroup` would duplicate wgpu semantics without adding any safety
//! property. The same exception applies to `wgpu::ShaderStages` in
//! [`BindEntry`](crate::pipeline::binding::BindEntry), documented there.
//!
//! ## push_constant_bytes deferral
//!
//! The plan's §4.2 sketch included a `push_constant_bytes: u8` field on
//! [`ComputePipelineDescriptor`] with a `<= 8` assertion. That field is
//! deliberately absent here: enabling `wgpu::Features::PUSH_CONSTANTS` is a
//! separate device-feature request that no first-pass shader uses. When the
//! first caller needs push constants the field lands alongside the feature
//! flag, and the `<= 8` assertion from §4.2 goes in at that point.
//!
//! [`GpuConstsData`]: crate::gpu_consts::GpuConstsData

use std::mem::size_of;
use std::sync::Arc;

use crate::device::{FrameEncoder, RendererContext};
use crate::error::RendererError;
use crate::gpu_consts::GpuConstsData;
use crate::pipeline::binding::BindingLayout;
use crate::pipeline::reflect;
use crate::shader::{ShaderSource, load_shader};

// --- ComputePipelineDescriptor ---

/// Construction parameters for a [`ComputePipeline`].
///
/// See module-level documentation for `push_constant_bytes` deferral and
/// the `wgpu::BindGroup` leakage rationale. `.local/renderer_plan.md` §4.2.
pub struct ComputePipelineDescriptor<'a> {
    /// Debug label forwarded to wgpu pipeline + pipeline layout.
    pub label: &'a str,
    /// Compiled SPIR-V shader source.
    pub shader: ShaderSource,
    /// Entry point name within the shader (typically `"main"` for DXC output).
    pub entry_point: &'a str,
    /// The binding layout this pipeline was built against. Stored on the
    /// pipeline and available via [`ComputePipeline::layout`].
    pub layout: Arc<BindingLayout>,
    /// If `Some`, the reflected workgroup size must equal this value. Triggers
    /// a `panic!` at construction if it does not, with a message pointing at
    /// both values. Pass `None` to skip the assertion (not recommended — it
    /// exists precisely to catch `[numthreads]` / dispatch-size drift).
    ///
    /// See `.local/renderer_plan.md` §7.1 for the policy.
    pub expected_workgroup_size: Option<[u32; 3]>,
}

// --- ComputePipeline ---

/// A compute pipeline with baked-in binding layout and reflected workgroup
/// size.
///
/// The only path to dispatching compute shaders in the renderer. Workgroup-size
/// and GpuConsts-size assertions fire during [`Self::new`], before any wgpu
/// call, so mismatches are caught at pipeline load rather than at dispatch.
///
/// See `.local/renderer_plan.md` §4.2 and module-level documentation.
pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    layout: Arc<BindingLayout>,
    workgroup_size: [u32; 3],
    label: String,
}

// --- ComputePipeline ---

impl ComputePipeline {
    /// Construct a new `ComputePipeline` from `desc`.
    ///
    /// The sequence is:
    /// 1. SPIR-V reflection (CPU only — no GPU calls yet).
    /// 2. Workgroup-size assertion if `expected_workgroup_size` is `Some`.
    /// 3. GpuConsts size assertion if the shader declares a uniform buffer at
    ///    `(set=0, binding=0)`.
    /// 4. `wgpu` pipeline layout + compute pipeline construction.
    ///
    /// Steps 2–3 deliberately precede step 4 so a CPU-side mismatch panics
    /// without issuing any GPU calls.
    ///
    /// # Errors
    ///
    /// Returns [`RendererError::ShaderReflectionFailed`] if SPIR-V reflection
    /// fails (malformed bytes, missing entry point, `LocalSizeId` mode).
    ///
    /// # Panics
    ///
    /// Panics if `expected_workgroup_size` is `Some` and does not match the
    /// reflected size. Panics if the reflected GpuConsts buffer size does not
    /// equal `size_of::<GpuConstsData>()`.
    pub fn new(
        ctx: &RendererContext,
        desc: ComputePipelineDescriptor<'_>,
    )
        -> Result<Self, RendererError>
    {
        // Step 1: extract the SPIR-V bytes. `ShaderSource::Spirv` holds a
        // `&'static [u8]` which is `Copy`, so we can use it for both
        // reflection and for `load_shader` below without moving the enum.
        let ShaderSource::Spirv(spv_bytes) = desc.shader;

        let reflected = reflect::reflect_spirv(spv_bytes, desc.entry_point)?;

        // Step 2: workgroup-size assertion (fires before any wgpu calls).
        if let Some(expected) = desc.expected_workgroup_size
            && reflected.workgroup_size != expected
        {
            panic!(
                "workgroup size mismatch for pipeline `{}`:\n  \
                 expected {:?}, shader has {:?}\n  \
                 (CPU dispatch code and shader [numthreads(...)] are out of sync)",
                desc.label,
                expected,
                reflected.workgroup_size,
            );
        }

        // Step 3: GpuConsts size assertion (fires before any wgpu calls).
        if let Some(size) = reflected.gpu_consts_byte_size {
            assert_eq!(
                size as usize,
                size_of::<GpuConstsData>(),
                "GpuConsts size mismatch: HLSL sees {} bytes, Rust has {} bytes",
                size,
                size_of::<GpuConstsData>(),
            );
        }

        // Step 4a: load the shader module.
        let module = load_shader(ctx, desc.label, ShaderSource::Spirv(spv_bytes));

        // Step 4b: create the pipeline layout.
        let pipeline_layout =
            ctx.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(desc.label),
                bind_group_layouts: &[Some(desc.layout.wgpu_layout())],
                immediate_size: 0,
            });

        // Step 4c: create the compute pipeline.
        let pipeline =
            ctx.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(desc.label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(desc.entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            pipeline,
            layout: desc.layout,
            workgroup_size: reflected.workgroup_size,
            label: desc.label.to_string(),
        })
    }

    /// The workgroup size reflected from the shader's `LocalSize` execution
    /// mode — `[x, y, z]`, matching `[numthreads(x, y, z)]` in HLSL.
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    /// The binding layout this pipeline was constructed with.
    pub fn layout(&self) -> &Arc<BindingLayout> {
        &self.layout
    }

    /// Record a compute dispatch into `frame`'s command encoder.
    ///
    /// `workgroups` is the dispatch grid in workgroups — `[x, y, z]`. To
    /// dispatch over a flat element count, see [`Self::dispatch_linear`].
    ///
    /// `bind_group` leaks `wgpu::BindGroup` through this API; see module-level
    /// documentation for the rationale.
    pub fn dispatch(
        &self,
        frame: &mut FrameEncoder,
        bind_group: &wgpu::BindGroup,
        workgroups: [u32; 3],
    ) {
        let mut cpass = frame.encoder_mut().begin_compute_pass(
            &wgpu::ComputePassDescriptor {
                label: Some(&self.label),
                timestamp_writes: None,
            },
        );
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
    }

    /// Record a 1-D compute dispatch for `count` elements.
    ///
    /// Computes the number of workgroups along X as `count.div_ceil(x)` where
    /// `x` is the first component of [`Self::workgroup_size`]. Y and Z are
    /// always 1. The shader is responsible for bounds-checking thread IDs
    /// against `count` when `count` is not a multiple of the workgroup size.
    pub fn dispatch_linear(
        &self,
        frame: &mut FrameEncoder,
        bind_group: &wgpu::BindGroup,
        count: u32,
    ) {
        let wg = self.workgroup_size[0];
        let x = count.div_ceil(wg);
        self.dispatch(frame, bind_group, [x, 1, 1]);
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameCount;
    use crate::pipeline::binding::{BindEntry, BindKind, BindingLayout};
    use crate::shader::VALIDATION_CS_SPV;

    /// Builds a `ComputePipeline` for `validation.cs.hlsl` with the correct
    /// expected workgroup size and asserts the baked-in size is `[64, 1, 1]`.
    /// Also asserts the pipeline retains a clone of the `Arc<BindingLayout>`.
    ///
    /// Gated because it requires a DXC-compiled SPV (placeholder SPV is not
    /// a loadable module) and a Vulkan-capable GPU.
    #[test]
    #[ignore = "requires real GPU hardware (vulkan) and a DXC-built SPV; run with --ignored"]
    fn compute_pipeline_builds_for_validation_cs_with_matching_workgroup() {
        let ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine");

        let layout = Arc::new(
            BindingLayout::builder("validation")
                .add_entry(BindEntry {
                    binding: 1,
                    kind: BindKind::StorageBufferReadWrite { size: 4 },
                    visibility: wgpu::ShaderStages::COMPUTE,
                })
                .build(&ctx),
        );

        let pipeline = ComputePipeline::new(
            &ctx,
            ComputePipelineDescriptor {
                label: "validation",
                shader: ShaderSource::Spirv(VALIDATION_CS_SPV),
                entry_point: "main",
                layout: layout.clone(),
                expected_workgroup_size: Some([64, 1, 1]),
            },
        )
        .expect("ComputePipeline::new should succeed for the validation shader");

        assert_eq!(
            pipeline.workgroup_size(),
            [64, 1, 1],
            "baked-in workgroup size should match [numthreads(64, 1, 1)]",
        );

        // The pipeline stores one clone; the caller holds another.
        assert!(
            Arc::strong_count(pipeline.layout()) >= 2,
            "pipeline should retain a clone of the Arc<BindingLayout>",
        );
    }

    /// Building a `ComputePipeline` with the wrong `expected_workgroup_size`
    /// panics before any GPU call.
    ///
    /// Gated because the placeholder SPV has no entry points, so the test
    /// would fail at the reflect step rather than the assertion step — the
    /// panic message match would be wrong.
    #[test]
    #[should_panic(expected = "workgroup size mismatch")]
    #[ignore = "requires real GPU hardware (vulkan) and a DXC-built SPV; run with --ignored"]
    fn compute_pipeline_panics_on_wrong_expected_workgroup_size() {
        let ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine");

        let layout = Arc::new(
            BindingLayout::builder("validation")
                .add_entry(BindEntry {
                    binding: 1,
                    kind: BindKind::StorageBufferReadWrite { size: 4 },
                    visibility: wgpu::ShaderStages::COMPUTE,
                })
                .build(&ctx),
        );

        // [32, 1, 1] does not match the shader's [64, 1, 1] — should panic.
        let _ = ComputePipeline::new(
            &ctx,
            ComputePipelineDescriptor {
                label: "validation",
                shader: ShaderSource::Spirv(VALIDATION_CS_SPV),
                entry_point: "main",
                layout,
                expected_workgroup_size: Some([32, 1, 1]),
            },
        );
    }
}
