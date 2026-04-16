//! Shader module construction via wgpu's SPIR-V passthrough path.
//!
//! The renderer's shader toolchain is HLSL → DXC → SPIR-V → wgpu
//! passthrough, matching the old scaffold exactly. The `build.rs` invokes
//! DXC at build time and emits compiled SPV blobs under `$OUT_DIR/shaders/`;
//! Rust then picks them up with `include_bytes!` into `&'static [u8]`
//! constants and passes them to [`ShaderModule::load`], which performs SPIR-V
//! reflection, asserts `GpuConsts` layout, and uploads the module to the GPU.
//!
//! The toolchain is HLSL → DXC rather than WGSL because later subsystems
//! need `DrawIndex`, which naga can't round-trip — switching toolchains later
//! would cost two build-system transitions and force `types.hlsl` to be
//! written twice.
//!
//! `wgpu::ShaderModule` does not leak through this module's public API
//! (principle 3: wgpu is contained behind render abstractions). The
//! [`ShaderModule`] type exposes `inner` as `pub(crate)` so pipeline types
//! in this crate can use it without exposing it to callers.

use std::borrow::Cow;

use crate::device::RendererContext;
use crate::error::RendererError;
use crate::pipeline::binding::BindKind;

/// The source bytes of a shader module.
///
/// Only the SPIR-V passthrough variant exists in the first rewrite pass.
/// DXIL / HLSL / MSL / WGSL variants are not forbidden by design — wgpu's
/// passthrough descriptor supports them — but they are not needed yet and
/// adding them speculatively would widen the surface this module's tests
/// need to cover. Grow the enum when a real caller needs another variant.
pub enum ShaderSource {
    /// A binary SPIR-V module as a static byte slice. Typically sourced
    /// from `include_bytes!(concat!(env!("OUT_DIR"), "/shaders/<name>.spv"))`.
    /// Must begin with the SPIR-V magic number; [`ShaderModule::load`] panics
    /// via `wgpu::util::make_spirv_raw` otherwise.
    Spirv(&'static [u8]),
}

// --- ShaderModule ---

/// A loaded shader — SPIR-V reflected, GpuConsts-asserted, and uploaded to
/// the GPU as a `wgpu::ShaderModule`. Consumed by pipeline constructors.
///
/// `wgpu::ShaderModule` does not leak through this type's public interface
/// (principle 3); the `inner` field is `pub(crate)` so pipeline types in
/// this crate can use it without exposing it to callers.
pub struct ShaderModule {
    pub(crate) inner      : wgpu::ShaderModule,
    pub(crate) entry_point: String,
    /// Workgroup size reflected from the shader's `LocalSize` execution mode.
    /// `None` for raster (vertex/fragment) shaders, which have no `LocalSize`.
    pub workgroup_size      : Option<[u32; 3]>,
    /// Byte size of the `GpuConsts` uniform buffer at (set=0, binding=0),
    /// or `None` if the shader does not declare one.
    pub gpu_consts_byte_size: Option<u32>,
    /// All descriptor-set-0 bindings reflected from the shader, as
    /// `(binding_slot, kind)` pairs sorted by slot. Visibility is populated
    /// by the pipeline constructor that consumes this `ShaderModule`.
    pub bind_entries        : Vec<(u32, BindKind)>,
    /// The shader stage of this module's entry point.
    pub stage               : wgpu::ShaderStages,
}

impl ShaderModule {
    /// Load a shader from compiled SPIR-V.
    ///
    /// Performs SPIR-V reflection, asserts the `GpuConsts` uniform buffer size
    /// against `GpuConstsData` if the shader declares one, then uploads the
    /// module to the GPU via `wgpu`'s SPIR-V passthrough path.
    ///
    /// # Errors
    ///
    /// Returns [`RendererError::ShaderReflectionFailed`] if SPIR-V parsing
    /// fails, the entry point is not found, or a `LocalSizeId` execution mode
    /// is present (spec-constant workgroup sizes are unsupported).
    ///
    /// # Panics
    ///
    /// Panics if the reflected `GpuConsts` buffer size does not match
    /// `size_of::<GpuConstsData>()`, or if the SPIR-V bytes are malformed
    /// (not a multiple of 4, missing magic number).
    pub fn load(
        ctx: &RendererContext,
        label: &str,
        source: ShaderSource,
        entry_point: &str,
    ) -> Result<Self, RendererError> {
        use std::mem::size_of;
        use crate::gpu_consts::GpuConstsData;
        use crate::pipeline::reflect;

        let ShaderSource::Spirv(spv_bytes) = source;
        let reflected = reflect::reflect_spirv(spv_bytes, entry_point)?;
        let stage     = reflect::entry_point_stage(spv_bytes, entry_point)?;

        if let Some(size) = reflected.gpu_consts_byte_size {
            assert_eq!(
                size as usize,
                size_of::<GpuConstsData>(),
                "GpuConsts size mismatch for shader `{label}`: \
                 HLSL sees {size} bytes, Rust has {} bytes",
                size_of::<GpuConstsData>(),
            );
        }

        let inner       = create_wgpu_module(ctx, label, ShaderSource::Spirv(spv_bytes));
        let bind_entries = reflected.entries.into_iter()
            .map(|e| (e.binding, e.kind))
            .collect();

        Ok(Self {
            inner,
            entry_point     : entry_point.to_owned(),
            workgroup_size  : reflected.workgroup_size,
            gpu_consts_byte_size: reflected.gpu_consts_byte_size,
            bind_entries,
            stage,
        })
    }
}

// --- create_wgpu_module ---

/// Create a `wgpu::ShaderModule` from a [`ShaderSource`].
///
/// This is the single place the renderer calls
/// `create_shader_module_passthrough`. The call is `unsafe` per the wgpu
/// API because passthrough bypasses naga's validation — wgpu trusts the
/// bytes to be a well-formed SPIR-V module for the target backend. The
/// DXC-generated SPV that `build.rs` writes is that well-formed blob; the
/// placeholder SPV that `build.rs` falls back to when DXC is missing is
/// *not*, which is why the validation binary in Increment 10 refuses to
/// run against it and exits nonzero.
///
/// # Panics
///
/// Panics if the SPIR-V bytes are malformed — specifically, if the length
/// isn't a multiple of 4 or the magic number is absent. These are both
/// "build-time promise violated" bugs, not recoverable runtime conditions.
fn create_wgpu_module(
    ctx: &RendererContext,
    label: &str,
    source: ShaderSource,
) -> wgpu::ShaderModule {
    match source {
        ShaderSource::Spirv(bytes) => {
            let spirv: Cow<'static, [u32]> = wgpu::util::make_spirv_raw(bytes);
            // SAFETY: passthrough bypasses naga validation. The caller
            // promises the bytes are a well-formed SPIR-V module compiled
            // from a trusted source (DXC via `build.rs`). wgpu checks the
            // magic number and word alignment; beyond that it trusts us.
            unsafe {
                ctx.device().create_shader_module_passthrough(
                    wgpu::ShaderModuleDescriptorPassthrough {
                        label: Some(label),
                        spirv: Some(spirv),
                        ..Default::default()
                    },
                )
            }
        }
    }
}

/// The compiled SPIR-V for `shaders/validation.cs.hlsl`, produced by
/// `build.rs` and embedded at compile time.
///
/// Exposed as `pub(crate)` so in-crate tests and (later) the validation
/// binary can load it without duplicating the `include_bytes!` path.
/// External callers do not touch this — they'll receive a `ShaderModule`
/// handle through a higher-level API once `ComputePipeline` lands in
/// Increment 6.
#[allow(dead_code)] // First non-test caller: ComputePipeline tests in Increment 6.
pub(crate) const VALIDATION_CS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/validation.cs.spv"));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameCount;

    /// Build-time promise: `validation.cs.hlsl` compiled to a non-empty blob
    /// beginning with the SPIR-V magic number. Guard against DXC silently
    /// producing nothing — when DXC exits zero but writes no output, the
    /// placeholder blob is not a valid shader module.
    ///
    /// Accepts either byte order — SPIR-V is defined in host endianness,
    /// but the build-time placeholder writes native-endian bytes and real
    /// DXC output on every platform we build on is little-endian today.
    /// `make_spirv_raw` (used by the loader) handles the swap at runtime
    /// regardless, so the test mirrors that tolerance.
    #[test]
    fn validation_cs_spv_starts_with_spirv_magic_number() {
        const SPIRV_MAGIC: u32 = 0x0723_0203;

        assert!(
            VALIDATION_CS_SPV.len() >= 4,
            "validation.cs.spv is shorter than a SPIR-V header word; \
             did `build.rs` run? len={}",
            VALIDATION_CS_SPV.len(),
        );
        assert!(
            VALIDATION_CS_SPV.len().is_multiple_of(4),
            "validation.cs.spv length {} is not a multiple of 4",
            VALIDATION_CS_SPV.len(),
        );

        let first_word = u32::from_ne_bytes([
            VALIDATION_CS_SPV[0],
            VALIDATION_CS_SPV[1],
            VALIDATION_CS_SPV[2],
            VALIDATION_CS_SPV[3],
        ]);
        assert!(
            first_word == SPIRV_MAGIC || first_word == SPIRV_MAGIC.swap_bytes(),
            "validation.cs.spv does not begin with the SPIR-V magic number; \
             got {first_word:#010x}"
        );
    }

    /// GPU smoke test: `load_shader` passes the compiled validation shader
    /// through `create_shader_module_passthrough` without panicking, which
    /// exercises the full toolchain — DXC output, `make_spirv_raw`
    /// word-cast, the passthrough call, and the Vulkan driver's own
    /// SPIR-V acceptance.
    ///
    /// Gated with `#[ignore]` because it requires a working Vulkan stack
    /// *and* a real DXC build (the placeholder SPV is not a loadable
    /// module). Run locally with `cargo test -p renderer -- --ignored`.
    #[test]
    #[ignore = "requires real GPU hardware (vulkan) and a DXC-built SPV; run with --ignored"]
    fn validation_shader_loads_via_passthrough() {
        let ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine");

        let _module = ShaderModule::load(
            &ctx,
            "validation.cs",
            ShaderSource::Spirv(VALIDATION_CS_SPV),
            "main",
        )
        .expect("ShaderModule::load should succeed for the validation shader");
        // Reaching this point without panicking is the assertion — the
        // wgpu driver accepted the SPIR-V module and produced a handle.
    }
}
