//! `ComputePipeline` with shader-reflected layout and workgroup-size assertion.
//!
//! [`ComputePipeline`] is the only path to dispatching a compute shader in
//! the renderer. It enforces three invariants at pipeline load time (inside
//! [`ComputePipeline::new`]), before any wgpu call, so a CPU-side mismatch
//! panics deterministically without touching the GPU:
//!
//! 1. **Workgroup-size assertion**: if
//!    `ComputePipelineDescriptor::expected_workgroup_size` is `Some(expected)`,
//!    the value is compared against the `LocalSize` execution mode reflected
//!    from the SPIR-V binary. A mismatch is a `panic!` with a message pointing
//!    at both the expected and actual values — the CPU dispatch code and the
//!    shader `[numthreads(...)]` are out of sync.
//!
//! 2. **GpuConsts size assertion**: the reflected SPIR-V must declare a uniform
//!    buffer at descriptor set 0 binding 0 (the slot [`GpuConstsData`] always
//!    occupies), and its byte size must equal `size_of::<GpuConstsData>()`.
//!    A missing or wrong-sized entry indicates a Rust/HLSL layout disagreement
//!    or a shader that does not include `gpu_consts.hlsl`.
//!
//! Both assertions fire at construction, not at dispatch — catching
//! CPU/shader mismatches at load means they surface immediately on startup
//! rather than on the first dispatch.
//!
//! ## wgpu::BindGroup leakage
//!
//! [`Commands::dispatch`](crate::commands::Commands::dispatch) and
//! [`Commands::dispatch_linear`](crate::commands::Commands::dispatch_linear)
//! accept a `&wgpu::BindGroup` directly. This is a deliberate exception to
//! the renderer's containment discipline (principle 3) — a thin wrapper around
//! `wgpu::BindGroup` would duplicate wgpu semantics without adding any safety
//! property. The same exception applies to `wgpu::ShaderStages` in
//! [`BindEntry`](crate::pipeline::binding::BindEntry), documented there.
//!
//! [`GpuConstsData`]: crate::gpu_consts::GpuConstsData

use crate::device::RendererContext;
use crate::gpu_consts::{GpuConsts, GpuConstsData};
use crate::pipeline::PipelineBindLayout;
use crate::pipeline::binding::{BindEntry, BindKind};
use crate::pipeline::bind_kind_to_wgpu_ty;
use crate::shader::ShaderModule;

// --- ComputePipelineDescriptor ---

/// Construction parameters for a [`ComputePipeline`].
///
/// See module-level documentation for the `wgpu::BindGroup` leakage rationale.
pub struct ComputePipelineDescriptor<'a> {
    /// Debug label forwarded to wgpu pipeline + pipeline layout.
    pub label: &'a str,
    /// Loaded shader module (produced by [`ShaderModule::load`]).
    pub shader: ShaderModule,
    /// If `Some`, the reflected workgroup size must equal this value. Triggers
    /// a `panic!` at construction if it does not, with a message pointing at
    /// both values. Pass `None` to skip the assertion (not recommended — it
    /// exists precisely to catch `[numthreads]` / dispatch-size drift).
    pub expected_workgroup_size: Option<[u32; 3]>,
    /// Immediate-data byte budget (previously called push constants). `0`
    /// means the pipeline declares no immediate data.
    pub immediate_size: u32,
}

// --- ComputePipeline ---

/// A compute pipeline with shader-reflected bind group layout.
///
/// The only path to dispatching compute shaders in the renderer. Workgroup-size
/// and GpuConsts-size assertions fire during [`Self::new`], before any wgpu
/// call, so mismatches are caught at pipeline load rather than at dispatch.
///
/// See module-level documentation.
pub struct ComputePipeline {
    pipeline     : wgpu::ComputePipeline,
    bg_layout    : wgpu::BindGroupLayout,
    bind_entries : Vec<BindEntry>,
    workgroup_size: [u32; 3],
    label        : String,
}

// --- ComputePipeline ---

impl ComputePipeline {
    /// Construct a new `ComputePipeline` from `desc`.
    ///
    /// 1. Workgroup-size assertion (if `expected_workgroup_size` is `Some`).
    /// 2. Slot-0 GpuConsts assertion — the shader must declare a
    ///    `UniformBuffer` at slot 0 sized to `GpuConstsData`.
    /// 3. Build `wgpu::BindGroupLayout` from the reflected entries.
    /// 4. Create the wgpu compute pipeline.
    ///
    /// # Panics
    ///
    /// Panics if `expected_workgroup_size` is `Some` and does not match the
    /// reflected workgroup size, or if a raster shader is passed, or if slot 0
    /// is absent, not a `UniformBuffer`, or sized incorrectly.
    pub fn new(ctx: &RendererContext, desc: ComputePipelineDescriptor<'_>) -> Self {
        use std::mem::size_of;

        // Step 1: workgroup-size assertion.
        if let Some(expected) = desc.expected_workgroup_size {
            match desc.shader.workgroup_size {
                Some(actual) if actual == expected => {}
                Some(actual) => panic!(
                    "workgroup size mismatch for pipeline `{}`:\n  \
                     expected {:?}, shader has {:?}\n  \
                     (CPU dispatch code and shader [numthreads(...)] are out of sync)",
                    desc.label, expected, actual,
                ),
                None => panic!(
                    "workgroup size mismatch for pipeline `{}`:\n  \
                     expected {:?}, but shader has no LocalSize \
                     (did you pass a raster shader to a compute pipeline?)",
                    desc.label, expected,
                ),
            }
        }

        // Step 2: slot-0 GpuConsts assertion.
        let slot0 = desc.shader.bind_entries.iter()
            .find(|(b, _)| *b == GpuConsts::SLOT);

        match slot0 {
            Some((_, BindKind::UniformBuffer { size }))
                if *size as usize == size_of::<GpuConstsData>() => {}
            Some((_, BindKind::UniformBuffer { size })) => panic!(
                "pipeline `{}`: slot {} is a UniformBuffer but its size is {} bytes; \
                 expected {} bytes (GpuConstsData) — \
                 check that the shader includes shaders/include/gpu_consts.hlsl",
                desc.label, GpuConsts::SLOT, size, size_of::<GpuConstsData>(),
            ),
            Some((_, kind)) => panic!(
                "pipeline `{}`: slot {} must be the GpuConsts uniform buffer but is \
                 {:?} — check that the shader includes shaders/include/gpu_consts.hlsl",
                desc.label, GpuConsts::SLOT, kind,
            ),
            None => panic!(
                "pipeline `{}`: slot {} (GpuConsts) is absent from the shader's \
                 descriptor set 0 — check that the shader includes \
                 shaders/include/gpu_consts.hlsl",
                desc.label, GpuConsts::SLOT,
            ),
        }

        // Step 3: build BindEntry list and wgpu::BindGroupLayout.
        let mut bind_entries: Vec<BindEntry> = desc.shader.bind_entries.iter()
            .map(|&(binding, kind)| BindEntry {
                binding,
                kind,
                visibility: desc.shader.stage,
            })
            .collect();
        bind_entries.sort_by_key(|e| e.binding);

        let wgpu_entries: Vec<wgpu::BindGroupLayoutEntry> = bind_entries.iter()
            .map(|e| wgpu::BindGroupLayoutEntry {
                binding   : e.binding,
                visibility: e.visibility,
                ty        : bind_kind_to_wgpu_ty(e.kind),
                count     : None,
            })
            .collect();

        let bg_layout = ctx.device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label  : Some(desc.label),
                entries: &wgpu_entries,
            });

        // Step 4: create the wgpu pipeline.
        let pipeline_layout =
            ctx.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label             : Some(desc.label),
                bind_group_layouts: &[Some(&bg_layout)],
                immediate_size    : desc.immediate_size,
            });

        let pipeline =
            ctx.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label              : Some(desc.label),
                layout             : Some(&pipeline_layout),
                module             : &desc.shader.inner,
                entry_point        : Some(desc.shader.entry_point.as_str()),
                compilation_options: Default::default(),
                cache              : None,
            });

        let workgroup_size = desc.shader.workgroup_size;

        Self {
            pipeline,
            bg_layout,
            bind_entries,
            workgroup_size: workgroup_size.unwrap_or([1, 1, 1]),
            label: desc.label.to_string(),
        }
    }

    /// The workgroup size reflected from the shader's `LocalSize` execution
    /// mode — `[x, y, z]`, matching `[numthreads(x, y, z)]` in HLSL.
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    /// The debug label this pipeline was constructed with.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The underlying wgpu compute pipeline.
    pub(crate) fn inner(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

impl PipelineBindLayout for ComputePipeline {
    fn bg_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bg_layout
    }

    fn bind_entries(&self) -> &[BindEntry] {
        &self.bind_entries
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameCount;
    use crate::shader::{ShaderModule, ShaderSource, VALIDATION_CS_SPV};

    /// Builds a `ComputePipeline` for `validation.cs.hlsl` with the correct
    /// expected workgroup size and asserts the baked-in size is `[64, 1, 1]`.
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

        let shader = ShaderModule::load(
            &ctx,
            "validation",
            ShaderSource::Spirv(VALIDATION_CS_SPV),
            "main",
        )
        .expect("ShaderModule::load should succeed for the validation shader");

        let pipeline = ComputePipeline::new(
            &ctx,
            ComputePipelineDescriptor {
                label:                  "validation",
                shader,
                expected_workgroup_size: Some([64, 1, 1]),
                immediate_size:         0,
            },
        );

        assert_eq!(
            pipeline.workgroup_size(),
            [64, 1, 1],
            "baked-in workgroup size should match [numthreads(64, 1, 1)]",
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

        let shader = ShaderModule::load(
            &ctx,
            "validation",
            ShaderSource::Spirv(VALIDATION_CS_SPV),
            "main",
        )
        .expect("ShaderModule::load should succeed for the validation shader");

        // [32, 1, 1] does not match the shader's [64, 1, 1] — should panic.
        ComputePipeline::new(
            &ctx,
            ComputePipelineDescriptor {
                label:                  "validation",
                shader,
                expected_workgroup_size: Some([32, 1, 1]),
                immediate_size:         0,
            },
        );
    }
}
