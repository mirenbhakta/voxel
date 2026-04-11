//! `BindingLayout` with forced [`GpuConstsData`] slot injection.
//!
//! Every `BindingLayout` built in the renderer has the shared constants
//! bound at slot 0 by construction: [`BindingLayoutBuilder::add_entry`] panics if
//! the caller targets the reserved slot, and [`BindingLayoutBuilder::build`]
//! unconditionally inserts the `GpuConstsData` entry at
//! [`BindingLayout::GPU_CONSTS_SLOT`] regardless of what the caller added.
//! This is the "forced injection" form of enforcement referenced in
//! `.local/renderer_plan.md` §4.1 and §5.2.
//!
//! A caller cannot forget to bind `GpuConstsData` at slot 0, and a shader
//! that `#include`s `shaders/include/types.hlsl` (which forces `g_consts` to
//! binding 0) will see the layout it expects in every pipeline it is used
//! with.

use std::num::NonZeroU64;

use crate::device::RendererContext;
use crate::gpu_consts::GpuConstsData;

/// Descriptor for a single binding in a [`BindingLayout`].
///
/// Produced by primitives like `UploadRing::bind_entry` / `ReadbackChannel::bind_entry`
/// (landing in later increments) and passed into [`BindingLayoutBuilder::add_entry`].
/// Callers never construct wgpu bind group layout entries directly — this is
/// the primitives-layer boundary that keeps binding descriptor shape in one
/// place.
#[derive(Clone, Debug)]
pub struct BindEntry {
    /// The bind group slot. Must not equal [`BindingLayout::GPU_CONSTS_SLOT`].
    pub binding: u32,
    /// The kind of resource (uniform / storage read-only / storage read-write).
    pub kind: BindKind,
    /// Shader stages that may access the binding. `wgpu::ShaderStages` is the
    /// one wgpu type that deliberately leaks through the primitives layer —
    /// the stage names are a closed set and a wrapper would add zero safety.
    /// Documented in `.local/renderer_plan.md` §4.1.
    pub visibility: wgpu::ShaderStages,
}

/// Kind of buffer resource exposed by a [`BindEntry`].
///
/// Textures and samplers are deferred until a primitive genuinely needs them
/// — see `.local/renderer_plan.md` §4.1.
#[derive(Clone, Copy, Debug)]
pub enum BindKind {
    /// A uniform buffer of the given size in bytes. Maps to wgpu's
    /// `BufferBindingType::Uniform`.
    UniformBuffer { size: u64 },
    /// A read-only storage buffer of the given size in bytes. Maps to wgpu's
    /// `BufferBindingType::Storage { read_only: true }`.
    StorageBufferReadOnly { size: u64 },
    /// A read-write storage buffer of the given size in bytes. Maps to wgpu's
    /// `BufferBindingType::Storage { read_only: false }`.
    StorageBufferReadWrite { size: u64 },
}

/// A stable slot assignment for a bind group, constructed once and shared
/// across all pipelines in a subsystem.
///
/// The `GpuConstsData` slot is reserved at construction; see the module-level
/// documentation for the forced-injection property.
///
/// The label passed to [`BindingLayout::builder`] is forwarded to
/// `wgpu::BindGroupLayoutDescriptor::label` during `build()` and is not
/// retained on the Rust side — wgpu owns it internally for debug output.
pub struct BindingLayout {
    entries: Vec<BindEntry>,
    wgpu_layout: wgpu::BindGroupLayout,
}

impl BindingLayout {
    /// The bind group slot reserved for [`GpuConstsData`]. Users cannot place
    /// a binding at this slot — [`BindingLayoutBuilder::add_entry`] panics on it
    /// and [`BindingLayoutBuilder::build`] injects the `GpuConstsData` entry
    /// there unconditionally.
    pub const GPU_CONSTS_SLOT: u32 = 0;

    /// Start building a new layout with the given label. The returned builder
    /// reserves slot 0 for `GpuConstsData`; user entries start at 1.
    ///
    /// Note: the plan's §4.1 sketch passes `ctx` to `builder()`, but the
    /// builder only accumulates CPU-side state. Taking `ctx` only at
    /// [`BindingLayoutBuilder::build`] — where it is actually needed for
    /// `device.create_bind_group_layout` — keeps the API free of unused
    /// parameters.
    pub fn builder(label: &str) -> BindingLayoutBuilder {
        BindingLayoutBuilder {
            label: label.to_string(),
            entries: Vec::new(),
        }
    }

    /// All binding entries in this layout, including the injected
    /// `GpuConstsData` entry at slot 0, in ascending binding-slot order.
    pub fn entries(&self) -> &[BindEntry] {
        &self.entries
    }

    /// The underlying wgpu layout handle, used by pipeline construction
    /// within this crate. Not exposed publicly — higher layers go through
    /// `ComputePipeline` / `RenderPipeline` which consume a `BindingLayout`.
    pub(crate) fn wgpu_layout(&self) -> &wgpu::BindGroupLayout {
        &self.wgpu_layout
    }
}

/// Builder for [`BindingLayout`]. See [`BindingLayout::builder`].
pub struct BindingLayoutBuilder {
    label: String,
    entries: Vec<BindEntry>,
}

impl BindingLayoutBuilder {
    /// Add a user binding to the layout.
    ///
    /// # Panics
    ///
    /// Panics if `entry.binding == BindingLayout::GPU_CONSTS_SLOT` — the
    /// GpuConsts slot is reserved and the builder injects it automatically at
    /// [`Self::build`].
    ///
    /// Panics on duplicate bindings at the same slot, which is a programmer
    /// error and would otherwise fail later in `device.create_bind_group_layout`
    /// with a much less readable diagnostic.
    ///
    /// Panics if a buffer `kind`'s `size` is zero. `NonZeroU64::new(0)` maps to
    /// `min_binding_size: None`, which silently disables wgpu's binding-size
    /// validation; catching it here keeps the error local to the caller's bug.
    ///
    /// All three checks belong at the "assert on programmer error" tier of the
    /// invariant hierarchy in `docs/renderer_rewrite_principles.md`.
    pub fn add_entry(mut self, entry: BindEntry) -> Self {
        assert!(
            entry.binding != BindingLayout::GPU_CONSTS_SLOT,
            "BindingLayoutBuilder::add_entry: slot {} is reserved for GpuConsts \
             (the builder injects it automatically at build())",
            BindingLayout::GPU_CONSTS_SLOT,
        );
        assert!(
            !self.entries.iter().any(|e| e.binding == entry.binding),
            "BindingLayoutBuilder::add_entry: duplicate binding at slot {}",
            entry.binding,
        );
        match entry.kind {
            BindKind::UniformBuffer { size }
            | BindKind::StorageBufferReadOnly { size }
            | BindKind::StorageBufferReadWrite { size } => {
                assert!(
                    size > 0,
                    "BindingLayoutBuilder::add_entry: buffer size at slot {} must be \
                     non-zero (zero maps to min_binding_size: None, skipping wgpu's \
                     binding size validation)",
                    entry.binding,
                );
            }
        }
        self.entries.push(entry);
        self
    }

    /// Finalize the layout. Unconditionally inserts the `GpuConstsData` entry
    /// at [`BindingLayout::GPU_CONSTS_SLOT`] regardless of what the caller
    /// added or omitted — this is the forced injection that makes principle 5
    /// enforceable.
    pub fn build(mut self, ctx: &RendererContext) -> BindingLayout {
        let gpu_consts_entry = BindEntry {
            binding: BindingLayout::GPU_CONSTS_SLOT,
            kind: BindKind::UniformBuffer {
                size: std::mem::size_of::<GpuConstsData>() as u64,
            },
            // GpuConsts is read by every shader stage — vertex, fragment, and
            // compute all need the ring slot / sentinel values it carries.
            visibility: wgpu::ShaderStages::VERTEX
                | wgpu::ShaderStages::FRAGMENT
                | wgpu::ShaderStages::COMPUTE,
        };
        self.entries.push(gpu_consts_entry);
        self.entries.sort_by_key(|e| e.binding);

        let wgpu_entries: Vec<wgpu::BindGroupLayoutEntry> = self
            .entries
            .iter()
            .map(|e| wgpu::BindGroupLayoutEntry {
                binding: e.binding,
                visibility: e.visibility,
                ty: bind_kind_to_wgpu_ty(e.kind),
                count: None,
            })
            .collect();

        let wgpu_layout =
            ctx.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&self.label),
                    entries: &wgpu_entries,
                });

        BindingLayout {
            entries: self.entries,
            wgpu_layout,
        }
    }
}

fn bind_kind_to_wgpu_ty(kind: BindKind) -> wgpu::BindingType {
    match kind {
        BindKind::UniformBuffer { size } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
        BindKind::StorageBufferReadOnly { size } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
        BindKind::StorageBufferReadWrite { size } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameCount;

    /// The reserved slot constant is zero — this is the single number that
    /// the shader-side `[[vk::binding(0, 0)]] ConstantBuffer<GpuConsts>` in
    /// `shaders/include/gpu_consts.hlsl` also commits to.
    #[test]
    fn gpu_consts_slot_is_zero() {
        assert_eq!(BindingLayout::GPU_CONSTS_SLOT, 0);
    }

    /// Attempting to add a user binding at the reserved slot is a programmer
    /// error and panics with a diagnostic pointing at `GpuConsts`.
    ///
    /// Pure CPU — the builder does not touch wgpu until `build()`.
    #[test]
    #[should_panic(expected = "reserved for GpuConsts")]
    fn add_entry_panics_on_reserved_slot() {
        let _ = BindingLayout::builder("test_layout").add_entry(BindEntry {
            binding: BindingLayout::GPU_CONSTS_SLOT,
            kind: BindKind::StorageBufferReadWrite { size: 64 },
            visibility: wgpu::ShaderStages::COMPUTE,
        });
    }

    /// Adding two entries at the same user slot is also a programmer error.
    #[test]
    #[should_panic(expected = "duplicate binding at slot 1")]
    fn add_entry_panics_on_duplicate_slot() {
        let _ = BindingLayout::builder("test_layout")
            .add_entry(BindEntry {
                binding: 1,
                kind: BindKind::StorageBufferReadOnly { size: 64 },
                visibility: wgpu::ShaderStages::COMPUTE,
            })
            .add_entry(BindEntry {
                binding: 1,
                kind: BindKind::StorageBufferReadWrite { size: 128 },
                visibility: wgpu::ShaderStages::COMPUTE,
            });
    }

    /// A buffer entry with `size: 0` silently disables wgpu's `min_binding_size`
    /// check (since `NonZeroU64::new(0)` produces `None`). Catch it early so the
    /// error is local to the caller.
    ///
    /// Pure CPU — the builder does not touch wgpu until `build()`.
    #[test]
    #[should_panic(expected = "buffer size at slot")]
    fn add_entry_panics_on_zero_size() {
        let _ = BindingLayout::builder("test_layout").add_entry(BindEntry {
            binding: 1,
            kind: BindKind::StorageBufferReadOnly { size: 0 },
            visibility: wgpu::ShaderStages::COMPUTE,
        });
    }

    /// GPU smoke test: `build()` injects the GpuConsts binding at slot 0 with
    /// a uniform-buffer type sized to `GpuConstsData`.
    ///
    /// Gated with `#[ignore]` — matches the Increment 3 pattern.
    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn build_injects_gpu_consts_at_slot_zero() {
        let ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine");

        let layout = BindingLayout::builder("empty_layout").build(&ctx);

        // Exactly one entry — the injected GpuConsts.
        assert_eq!(layout.entries().len(), 1);
        let entry = &layout.entries()[0];
        assert_eq!(entry.binding, BindingLayout::GPU_CONSTS_SLOT);
        match entry.kind {
            BindKind::UniformBuffer { size } => {
                assert_eq!(size as usize, std::mem::size_of::<GpuConstsData>());
            }
            other => panic!("expected UniformBuffer, got {other:?}"),
        }
        assert_eq!(
            entry.visibility,
            wgpu::ShaderStages::VERTEX
                | wgpu::ShaderStages::FRAGMENT
                | wgpu::ShaderStages::COMPUTE
        );
    }

    /// GPU smoke test: user bindings added in order end up after the
    /// injected GpuConsts entry, preserving their relative order.
    #[test]
    #[ignore = "requires real GPU hardware (vulkan); run with --ignored"]
    fn build_preserves_gpu_consts_with_user_bindings() {
        let ctx = pollster::block_on(RendererContext::new_headless(
            FrameCount::new(2).unwrap(),
        ))
        .expect("headless GPU context should construct on a vulkan-capable machine");

        let layout = BindingLayout::builder("two_user_bindings")
            .add_entry(BindEntry {
                binding: 1,
                kind: BindKind::StorageBufferReadOnly { size: 64 },
                visibility: wgpu::ShaderStages::COMPUTE,
            })
            .add_entry(BindEntry {
                binding: 2,
                kind: BindKind::StorageBufferReadWrite { size: 128 },
                visibility: wgpu::ShaderStages::COMPUTE,
            })
            .build(&ctx);

        assert_eq!(layout.entries().len(), 3);
        assert_eq!(layout.entries()[0].binding, BindingLayout::GPU_CONSTS_SLOT);
        assert!(matches!(
            layout.entries()[0].kind,
            BindKind::UniformBuffer { size } if size as usize == std::mem::size_of::<GpuConstsData>()
        ));
        assert_eq!(layout.entries()[1].binding, 1);
        assert!(matches!(
            layout.entries()[1].kind,
            BindKind::StorageBufferReadOnly { size: 64 }
        ));
        assert_eq!(layout.entries()[2].binding, 2);
        assert!(matches!(
            layout.entries()[2].kind,
            BindKind::StorageBufferReadWrite { size: 128 }
        ));
    }
}
