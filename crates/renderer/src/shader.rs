//! Shader module construction via wgpu's SPIR-V passthrough path.
//!
//! The renderer's shader toolchain is HLSL → DXC → SPIR-V → wgpu
//! passthrough, matching the old scaffold exactly. The `build.rs` invokes
//! DXC at build time and emits compiled SPV blobs under `$OUT_DIR/shaders/`;
//! Rust then picks them up with `include_bytes!` into `&'static [u8]`
//! constants and hands them to [`load_shader`] which wraps
//! `wgpu::Device::create_shader_module_passthrough`.
//!
//! The decision to stay on DXC (rather than write WGSL for the first pass)
//! is argued in `.local/renderer_plan.md` §8.1: later subsystems will need
//! `DrawIndex`, which naga can't round-trip, so switching later would cost
//! two build-system transitions and force `types.hlsl` to be written twice.
//!
//! See `docs/renderer_rewrite_principles.md` principle 3 for the containment
//! rule — `wgpu::ShaderModule` does not leak through this module's public
//! API; higher layers consume [`load_shader`] and receive an opaque module
//! they pass into `ComputePipeline` (Increment 6).

use std::borrow::Cow;

use crate::device::RendererContext;

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
    /// Must begin with the SPIR-V magic number; [`load_shader`] panics via
    /// `wgpu::util::make_spirv_raw` otherwise.
    Spirv(&'static [u8]),
}

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
pub fn load_shader(
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
    /// beginning with the SPIR-V magic number. This is the "DXC silently
    /// produced nothing" guard from `.local/renderer_plan.md` §8.3.
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

        let _module = load_shader(
            &ctx,
            "validation.cs",
            ShaderSource::Spirv(VALIDATION_CS_SPV),
        );
        // Reaching this point without panicking is the assertion — the
        // wgpu driver accepted the SPIR-V module and produced a handle.
    }
}
