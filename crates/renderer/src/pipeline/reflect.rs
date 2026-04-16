//! SPIR-V workgroup-size and descriptor reflection via `rspirv`.
//!
//! Exposes [`reflect_spirv`] which parses a SPIR-V binary, locates a named
//! entry point, and returns [`Reflected`] — the workgroup size, the
//! uniform-buffer byte size at slot 0, and the full descriptor-set-0 entry
//! list.
//!
//! The implementation uses `rspirv`'s data-representation layer (`rspirv::dr`)
//! and scans the module's `entry_points`, `execution_modes`, `annotations`,
//! and `types_global_values` collections — no full instruction stream walk is
//! needed.
//!
//! [`GpuConstsData`]: crate::gpu_consts::GpuConstsData

use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{self, Decoration, Op, StorageClass, Word};

use crate::error::RendererError;
use crate::pipeline::binding::BindKind;

// --- ReflectedEntry ---

/// A single binding reflected from a SPIR-V descriptor set 0 variable.
#[derive(Debug, Clone)]
pub struct ReflectedEntry {
    /// The `[[vk::binding(N, 0)]]` slot number.
    pub binding : u32,
    /// Classification of the resource accessed at this slot.
    pub kind    : BindKind,
}

// --- Reflected ---

/// The properties extracted from a SPIR-V module by [`reflect_spirv`].
///
/// All fields are derived from the parsed module without GPU involvement.
#[derive(Debug)]
pub struct Reflected {
    /// The workgroup size declared by the entry point's `LocalSize` execution
    /// mode — `[x, y, z]`, matching `[numthreads(x, y, z)]` in HLSL.
    /// `None` for raster (vertex/fragment) shaders, which have no `LocalSize`.
    pub workgroup_size      : Option<[u32; 3]>,
    /// Size in bytes of the uniform buffer at descriptor set 0 binding 0, or
    /// `None` if the shader does not declare one.  The
    /// [`ComputePipeline`](crate::pipeline::compute::ComputePipeline)
    /// constructor asserts this value equals `size_of::<GpuConstsData>()`.
    pub gpu_consts_byte_size: Option<u32>,
    /// All bindings in descriptor set 0, sorted by binding number.
    pub entries             : Vec<ReflectedEntry>,
}

// --- Public API ---

/// Return the `wgpu::ShaderStages` flag for the named entry point.
///
/// Walks `module.entry_points` to find the `OpEntryPoint` whose name matches
/// `entry_point` and returns the corresponding stage flag.
///
/// Returns [`RendererError::ShaderReflectionFailed`] if the entry point is not
/// found.
pub fn entry_point_stage(spv: &[u8], entry_point: &str)
    -> Result<wgpu::ShaderStages, RendererError>
{
    let module = parse_module(spv)?;
    Ok(find_entry_point(&module, entry_point)?.stage)
}

/// Parse `spv` (raw SPIR-V bytes) and reflect the named `entry_point`.
///
/// Returns [`Reflected`] on success. Returns
/// [`RendererError::ShaderReflectionFailed`] for any structural problem in the
/// module: parse failure, missing entry point, unsupported `LocalSizeId`
/// execution mode, unsupported binding shape, or other malformed SPIR-V.
///
/// Only literal `LocalSize` execution modes are supported. If the shader uses
/// `LocalSizeId` (spec-constant workgroup sizes), this function returns an
/// error advising the caller to switch to literal `numthreads`. The DXC
/// toolchain emits `LocalSize` for literal `[numthreads(...)]`, so this
/// restriction is transparent for all shaders in the renderer.
pub fn reflect_spirv(spv: &[u8], entry_point: &str)
    -> Result<Reflected, RendererError>
{
    let module          = parse_module(spv)?;
    let ep              = find_entry_point(&module, entry_point)?;
    let workgroup_size  = find_workgroup_size(&module, ep.fn_id, entry_point)?;
    let mut entries     = reflect_set0_entries(&module)?;
    entries.sort_by_key(|e| e.binding);

    let gpu_consts_byte_size = entries.iter()
        .find(|e| e.binding == 0)
        .and_then(|e| match e.kind {
            BindKind::UniformBuffer { size } => Some(size as u32),
            _ => None,
        });

    Ok(Reflected { workgroup_size, gpu_consts_byte_size, entries })
}

// --- Error helper ---

/// Shorthand for `RendererError::ShaderReflectionFailed(format!(...))`.
macro_rules! refl_err {
    ($($arg:tt)*) => {
        RendererError::ShaderReflectionFailed(format!($($arg)*))
    };
}

// --- Module parsing ---

fn parse_module(spv: &[u8]) -> Result<Module, RendererError> {
    rspirv::dr::load_bytes(spv)
        .map_err(|e| refl_err!("failed to parse SPIR-V: {e}"))
}

// --- Operand helpers ---

fn as_id(op: Option<&Operand>) -> Option<Word> {
    match op {
        Some(Operand::IdRef(id)) => Some(*id),
        _ => None,
    }
}

fn as_lit(op: Option<&Operand>) -> Option<u32> {
    match op {
        Some(Operand::LiteralBit32(v)) => Some(*v),
        _ => None,
    }
}

fn as_deco(op: Option<&Operand>) -> Option<Decoration> {
    match op {
        Some(Operand::Decoration(d)) => Some(*d),
        _ => None,
    }
}

// --- Type-table lookup ---

/// Find the instruction in `module.types_global_values` matching both
/// `opcode` and `result_id`.
fn find_type(module: &Module, opcode: Op, id: Word) -> Option<&Instruction> {
    module.types_global_values.iter()
        .find(|i| i.class.opcode == opcode && i.result_id == Some(id))
}

/// Find the instruction in `module.types_global_values` whose `result_id`
/// matches, regardless of opcode.
fn find_type_any(module: &Module, id: Word) -> Option<&Instruction> {
    module.types_global_values.iter()
        .find(|i| i.result_id == Some(id))
}

/// Iterate the `IdRef` operands of an instruction.
fn id_refs(inst: &Instruction) -> impl Iterator<Item = Word> + '_ {
    inst.operands.iter().filter_map(|op| {
        if let Operand::IdRef(id) = op { Some(*id) } else { None }
    })
}

// --- Decoration iteration ---

/// Yield `(target_id, decoration, extras)` for every `OpDecorate` in
/// `module.annotations`, where `extras` is the slice of operands after the
/// decoration tag (so `extras.first()` is the decoration's first literal).
fn decorates(module: &Module)
    -> impl Iterator<Item = (Word, Decoration, &[Operand])>
{
    module.annotations.iter().filter_map(|inst| {
        if inst.class.opcode != Op::Decorate {
            return None;
        }

        let target = as_id(inst.operands.first())?;
        let deco   = as_deco(inst.operands.get(1))?;
        Some((target, deco, inst.operands.get(2..).unwrap_or(&[])))
    })
}

/// Yield `(struct_id, member_index, decoration, extras)` for every
/// `OpMemberDecorate` in `module.annotations`.
fn member_decorates(module: &Module)
    -> impl Iterator<Item = (Word, u32, Decoration, &[Operand])>
{
    module.annotations.iter().filter_map(|inst| {
        if inst.class.opcode != Op::MemberDecorate {
            return None;
        }

        let sid  = as_id(inst.operands.first())?;
        let idx  = as_lit(inst.operands.get(1))?;
        let deco = as_deco(inst.operands.get(2))?;
        Some((sid, idx, deco, inst.operands.get(3..).unwrap_or(&[])))
    })
}

// --- Entry point lookup ---

/// Parsed `OpEntryPoint`: the function id it targets and the stage flag for
/// its execution model.
struct EntryPoint {
    fn_id: Word,
    stage: wgpu::ShaderStages,
}

/// Walk `module.entry_points` for the `OpEntryPoint` with a matching name
/// operand and return its `(fn_id, stage)`.
///
/// `OpEntryPoint` operand layout: `[ExecutionModel, IdRef(fn_id),
/// LiteralString(name), IdRef*(interface)]`.
fn find_entry_point(module: &Module, name: &str)
    -> Result<EntryPoint, RendererError>
{
    for inst in &module.entry_points {
        let (
            Some(Operand::ExecutionModel(model)),
            Some(Operand::IdRef(fn_id)),
            Some(Operand::LiteralString(ep_name)),
        ) = (
            inst.operands.first(),
            inst.operands.get(1),
            inst.operands.get(2),
        )
        else {
            continue;
        };

        if ep_name != name {
            continue;
        }

        let stage = match model {
            spirv::ExecutionModel::Vertex    => wgpu::ShaderStages::VERTEX,
            spirv::ExecutionModel::Fragment  => wgpu::ShaderStages::FRAGMENT,
            spirv::ExecutionModel::GLCompute => wgpu::ShaderStages::COMPUTE,
            other => return Err(refl_err!(
                "entry point '{name}' has unsupported execution model {other:?}",
            )),
        };

        return Ok(EntryPoint { fn_id: *fn_id, stage });
    }

    Err(refl_err!(
        "no entry point named '{name}' found in SPIR-V module"
    ))
}

/// Walk `module.execution_modes` to find the `OpExecutionMode` for `fn_id`
/// with mode `LocalSize`. Returns `Some([x, y, z])` for compute shaders,
/// `None` for raster shaders that have no `LocalSize` execution mode.
fn find_workgroup_size(
    module     : &Module,
    fn_id      : Word,
    entry_point: &str,
)
    -> Result<Option<[u32; 3]>, RendererError>
{
    for inst in &module.execution_modes {
        if as_id(inst.operands.first()) != Some(fn_id) {
            continue;
        }

        let Some(Operand::ExecutionMode(mode)) = inst.operands.get(1) else {
            continue;
        };

        match mode {
            spirv::ExecutionMode::LocalSize => {
                let x = extract_literal_bit32(&inst.operands, 2, "LocalSize x")?;
                let y = extract_literal_bit32(&inst.operands, 3, "LocalSize y")?;
                let z = extract_literal_bit32(&inst.operands, 4, "LocalSize z")?;

                if x == 0 || y == 0 || z == 0 {
                    return Err(refl_err!(
                        "entry point '{entry_point}' has invalid LocalSize \
                         [{x}, {y}, {z}]; all dimensions must be ≥ 1 \
                         (Vulkan spec: WorkgroupSize must be at least 1 \
                         in each dimension)",
                    ));
                }

                return Ok(Some([x, y, z]));
            }

            spirv::ExecutionMode::LocalSizeId => {
                return Err(refl_err!(
                    "entry point '{entry_point}' uses LocalSizeId (spec-constant \
                     workgroup size); use literal numthreads instead — \
                     LocalSizeId is not supported in the first rewrite pass",
                ));
            }

            _ => {}
        }
    }

    // No LocalSize found — raster shader (vertex/fragment).
    Ok(None)
}

/// Extract a `LiteralBit32` from `operands[index]`, or return an error.
fn extract_literal_bit32(
    operands: &[Operand],
    index   : usize,
    label   : &str,
)
    -> Result<u32, RendererError>
{
    as_lit(operands.get(index)).ok_or_else(|| refl_err!(
        "expected LiteralBit32 for {label}, got {:?}", operands.get(index),
    ))
}

// --- Descriptor set 0 reflection ---

/// Walk all `module.annotations` to find every variable decorated with
/// `DescriptorSet = 0`, then classify each one into a [`ReflectedEntry`].
fn reflect_set0_entries(module: &Module)
    -> Result<Vec<ReflectedEntry>, RendererError>
{
    let set0_bindings = collect_set0_bindings(module);
    let mut entries   = Vec::with_capacity(set0_bindings.len());

    for (var_id, binding) in set0_bindings {
        let kind = classify_var(module, var_id, binding)?;
        entries.push(ReflectedEntry { binding, kind });
    }

    Ok(entries)
}

/// Collect `(var_id, binding)` for every variable that has both
/// `OpDecorate DescriptorSet 0` and `OpDecorate Binding N`.
fn collect_set0_bindings(module: &Module) -> Vec<(Word, u32)> {
    let mut set0_vars  : Vec<Word>         = Vec::new();
    let mut binding_map: Vec<(Word, u32)>  = Vec::new();

    for (target, deco, extras) in decorates(module) {
        match deco {
            Decoration::DescriptorSet if as_lit(extras.first()) == Some(0) => {
                set0_vars.push(target);
            }
            Decoration::Binding => {
                if let Some(n) = as_lit(extras.first()) {
                    binding_map.push((target, n));
                }
            }
            _ => {}
        }
    }

    binding_map.into_iter()
        .filter(|(id, _)| set0_vars.contains(id))
        .collect()
}

/// Classify a single descriptor-set-0 variable into a `BindKind`.
///
/// - `Uniform` + `Block` → uniform buffer (size from member offsets).
/// - `Uniform` + `BufferBlock` (legacy SPIR-V) OR `StorageBuffer` + `Block`
///   → storage buffer (size = element stride of the trailing runtime array).
///   Read-only when the variable carries `NonWritable`, read-write otherwise.
/// - Anything else → `ShaderReflectionFailed`.
fn classify_var(module: &Module, var_id: Word, binding: u32)
    -> Result<BindKind, RendererError>
{
    let var_inst = find_type(module, Op::Variable, var_id)
        .ok_or_else(|| refl_err!(
            "binding {binding}: no OpVariable found for id {var_id}",
        ))?;

    let Some(Operand::StorageClass(sc)) = var_inst.operands.first() else {
        return Err(refl_err!(
            "binding {binding}: OpVariable has no StorageClass operand",
        ));
    };

    let ptr_id = var_inst.result_type.ok_or_else(|| refl_err!(
        "binding {binding}: OpVariable has no result type",
    ))?;

    let struct_id = find_type(module, Op::TypePointer, ptr_id)
        .and_then(|i| as_id(i.operands.get(1)))
        .ok_or_else(|| refl_err!(
            "binding {binding}: cannot follow pointer type {ptr_id} to a struct",
        ))?;

    let block = struct_block_decoration(module, struct_id);

    match (sc, block) {
        (StorageClass::Uniform, Some(Decoration::Block)) => {
            let size = compute_struct_byte_size(module, struct_id)
                .ok_or_else(|| refl_err!(
                    "binding {binding}: cannot compute byte size of uniform struct {struct_id}",
                ))?;

            Ok(BindKind::UniformBuffer { size: size as u64 })
        }

        (StorageClass::Uniform,       Some(Decoration::BufferBlock))
        | (StorageClass::StorageBuffer, Some(Decoration::Block)) => {
            let size = storage_buffer_size_from_struct(module, struct_id)
                .ok_or_else(|| refl_err!(
                    "binding {binding}: storage buffer struct {struct_id} has no \
                     runtime-array last member with ArrayStride decoration",
                ))?;

            let kind = if var_is_non_writable(module, var_id) {
                BindKind::StorageBufferReadOnly { size }
            }
            else {
                BindKind::StorageBufferReadWrite { size }
            };

            Ok(kind)
        }

        (sc, deco) => Err(refl_err!(
            "binding {binding}: unsupported variable shape \
             (StorageClass={sc:?}, decoration={deco:?}); \
             only Uniform+Block, Uniform+BufferBlock, and StorageBuffer+Block are supported",
        )),
    }
}

// --- Decoration queries ---

/// Whether `var_id` carries an `OpDecorate NonWritable`.
fn var_is_non_writable(module: &Module, var_id: Word) -> bool {
    decorates(module).any(|(t, d, _)| t == var_id && d == Decoration::NonWritable)
}

/// Returns `Block` or `BufferBlock` if `struct_id` is decorated with either,
/// `None` otherwise.
fn struct_block_decoration(module: &Module, struct_id: Word) -> Option<Decoration> {
    decorates(module)
        .find(|(t, d, _)| {
            *t == struct_id
                && matches!(d, Decoration::Block | Decoration::BufferBlock)
        })
        .map(|(_, d, _)| d)
}

/// Find the `ArrayStride` decoration on an `OpTypeRuntimeArray` type id.
fn runtime_array_stride(module: &Module, rta_id: Word) -> Option<u32> {
    decorates(module)
        .find(|(t, d, _)| *t == rta_id && *d == Decoration::ArrayStride)
        .and_then(|(_, _, extras)| as_lit(extras.first()))
}

// --- Struct size computation ---

/// Compute the total byte size of a struct type.
///
/// Strategy: if every member has an `OpMemberDecorate Offset`, return
/// `offset_of_last + size_of_last`. Otherwise sum each member's scalar
/// bit-width in bytes. For `GpuConstsData` (8 × u32, offsets 0, 4, … 28) both
/// paths produce 32.
fn compute_struct_byte_size(module: &Module, struct_id: Word) -> Option<u32> {
    let struct_inst = find_type(module, Op::TypeStruct, struct_id)?;
    let members: Vec<Word> = id_refs(struct_inst).collect();

    if members.is_empty() {
        return Some(0);
    }

    let mut offsets: Vec<Option<u32>> = vec![None; members.len()];

    for (sid, idx, deco, extras) in member_decorates(module) {
        if sid != struct_id || deco != Decoration::Offset {
            continue;
        }

        let idx = idx as usize;

        if idx < offsets.len()
            && let Some(off) = as_lit(extras.first())
        {
            offsets[idx] = Some(off);
        }
    }

    if offsets.iter().all(Option::is_some) {
        let last      = members.len() - 1;
        let last_off  = offsets[last]?;
        let last_size = member_byte_size(module, members[last])?;
        Some(last_off + last_size)
    }
    else {
        members.iter()
            .try_fold(0u32, |acc, &tid| Some(acc + member_byte_size(module, tid)?))
    }
}

/// For a struct whose last member is `OpTypeRuntimeArray`, return the array's
/// element stride from the `ArrayStride` decoration. Used as the reported
/// "size" of a storage buffer binding.
fn storage_buffer_size_from_struct(module: &Module, struct_id: Word) -> Option<u64> {
    let struct_inst = find_type(module, Op::TypeStruct, struct_id)?;
    let last_member = id_refs(struct_inst).last()?;

    find_type(module, Op::TypeRuntimeArray, last_member)?;
    runtime_array_stride(module, last_member).map(|s| s as u64)
}

/// Return the byte size of a SPIR-V scalar type (`OpTypeInt` or
/// `OpTypeFloat`). Returns `None` for any other type — callers treat that as
/// "size couldn't be computed".
fn member_byte_size(module: &Module, type_id: Word) -> Option<u32> {
    let inst = find_type_any(module, type_id)?;

    match inst.class.opcode {
        Op::TypeInt | Op::TypeFloat => as_lit(inst.operands.first()).map(|w| w / 8),
        _ => None,
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RendererError;

    /// Build a minimal little-endian SPIR-V compute shader binary with the
    /// given `[numthreads(x, y, z)]` and no descriptor bindings.
    ///
    /// Equivalent HLSL (x=64, y=1, z=1):
    /// ```hlsl
    /// [numthreads(64, 1, 1)]
    /// void main() {}
    /// ```
    ///
    /// IDs used: %void=1, %voidfn=2, %main=3, %label=4 (bound=5).
    fn minimal_compute_spirv(x: u32, y: u32, z: u32) -> Vec<u8> {
        #[rustfmt::skip]
        let words: &[u32] = &[
            // Header
            0x07230203, // magic
            0x00010100, // version 1.1
            0x00000000, // generator
            0x00000005, // bound (IDs 1–4 used)
            0x00000000, // schema

            // OpCapability Shader  (op=17, 2 words)
            0x00020011, 0x00000001,

            // OpMemoryModel Logical GLSL450  (op=14, 3 words)
            0x0003000E, 0x00000000, 0x00000001,

            // OpEntryPoint GLCompute %main "main"  (op=15, 5 words)
            // "main\0" occupies 2 words (5 bytes padded to 8).
            0x0005000F, 0x00000005, 0x00000003, 0x6E69616D, 0x00000000,

            // OpExecutionMode %main LocalSize x y z  (op=16, 6 words)
            0x00060010, 0x00000003, 0x00000011, x, y, z,

            // %void = OpTypeVoid  (op=19, 2 words)
            0x00020013, 0x00000001,

            // %voidfn = OpTypeFunction %void  (op=33, 3 words)
            0x00030021, 0x00000002, 0x00000001,

            // %main = OpFunction %void None %voidfn  (op=54, 5 words)
            0x00050036, 0x00000001, 0x00000003, 0x00000000, 0x00000002,

            // %label = OpLabel  (op=248, 2 words)
            0x000200F8, 0x00000004,

            // OpReturn  (op=253, 1 word)
            0x000100FD,

            // OpFunctionEnd  (op=56, 1 word)
            0x00010038,
        ];

        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Build a SPIR-V compute shader with `[numthreads(x, y, z)]` and a
    /// uniform buffer at `(set=0, binding=0)` containing `num_u32_members`
    /// `u32` fields with `std140` member offsets at `0, 4, 8, …`.
    ///
    /// IDs: %1=void, %2=voidfn, %3=main, %4=label,
    ///      %5=uint, %6=struct, %7=pointer, %8=variable. Bound=9.
    fn compute_spirv_with_uniform(
        x: u32, y: u32, z: u32,
        num_u32_members: u32,
    ) -> Vec<u8> {
        assert!(num_u32_members > 0);

        let mut w: Vec<u32> = Vec::new();

        // --- Header ---
        w.extend_from_slice(&[
            0x07230203, // magic
            0x00010100, // version 1.1
            0x00000000, // generator
            0x00000009, // bound (IDs 1–8)
            0x00000000, // schema
        ]);

        // OpCapability Shader
        w.extend_from_slice(&[0x00020011, 0x00000001]);

        // OpMemoryModel Logical GLSL450
        w.extend_from_slice(&[0x0003000E, 0x00000000, 0x00000001]);

        // OpEntryPoint GLCompute %3 "main"
        w.extend_from_slice(&[
            0x0005000F, 0x00000005, 0x00000003, 0x6E69616D, 0x00000000,
        ]);

        // OpExecutionMode %3 LocalSize x y z
        w.extend_from_slice(&[0x00060010, 0x00000003, 0x00000011, x, y, z]);

        // --- Annotations ---

        // OpDecorate %8 DescriptorSet 0
        w.extend_from_slice(&[0x00040047, 0x00000008, 0x00000022, 0x00000000]);

        // OpDecorate %8 Binding 0
        w.extend_from_slice(&[0x00040047, 0x00000008, 0x00000021, 0x00000000]);

        // OpDecorate %6 Block
        w.extend_from_slice(&[0x00030047, 0x00000006, 0x00000002]);

        // OpMemberDecorate %6 i Offset (i*4) for each member
        for i in 0..num_u32_members {
            w.extend_from_slice(&[0x00050048, 0x00000006, i, 0x00000023, i * 4]);
        }

        // --- Types ---

        // %1 = OpTypeVoid
        w.extend_from_slice(&[0x00020013, 0x00000001]);

        // %2 = OpTypeFunction %1
        w.extend_from_slice(&[0x00030021, 0x00000002, 0x00000001]);

        // %5 = OpTypeInt 32 0
        w.extend_from_slice(&[0x00040015, 0x00000005, 0x00000020, 0x00000000]);

        // %6 = OpTypeStruct %5 %5 ... (num_u32_members times)
        let struct_wc = 2 + num_u32_members;
        w.push((struct_wc << 16) | 0x001E);
        w.push(0x00000006);
        w.extend(std::iter::repeat_n(0x00000005u32, num_u32_members as usize));

        // %7 = OpTypePointer Uniform %6
        w.extend_from_slice(&[0x00040020, 0x00000007, 0x00000002, 0x00000006]);

        // %8 = OpVariable %7 Uniform
        w.extend_from_slice(&[0x0004003B, 0x00000007, 0x00000008, 0x00000002]);

        // --- Function ---

        // %3 = OpFunction %1 None %2
        w.extend_from_slice(&[
            0x00050036, 0x00000001, 0x00000003, 0x00000000, 0x00000002,
        ]);

        // %4 = OpLabel
        w.extend_from_slice(&[0x000200F8, 0x00000004]);

        // OpReturn
        w.push(0x000100FD);

        // OpFunctionEnd
        w.push(0x00010038);

        w.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    /// Hand-crafted minimal SPIR-V (no DXC required). Verifies the success
    /// path: correct workgroup size returned, and `None` for `gpu_consts_byte_size`
    /// since the shader declares no descriptor bindings.
    #[test]
    fn reflect_spirv_succeeds_on_minimal_compute_shader() {
        let spv = minimal_compute_spirv(64, 1, 1);
        let reflected = reflect_spirv(&spv, "main")
            .expect("reflect_spirv should succeed on a valid minimal SPIR-V");

        assert_eq!(
            reflected.workgroup_size,
            Some([64, 1, 1]),
            "workgroup size should match the LocalSize 64 1 1 in the binary",
        );
        assert_eq!(
            reflected.gpu_consts_byte_size,
            None,
            "gpu_consts_byte_size should be None: shader has no descriptor bindings",
        );
    }

    /// Hand-crafted SPV with a uniform buffer at (set=0, binding=0)
    /// containing 8 × u32 (32 bytes, matching `GpuConstsData`). Exercises
    /// the full `reflect_spirv` success path — workgroup size *and*
    /// `gpu_consts_byte_size` — without requiring DXC.
    #[test]
    fn reflect_spirv_reports_gpu_consts_byte_size_from_hand_crafted_spv() {
        let spv = compute_spirv_with_uniform(64, 1, 1, 8);
        let reflected = reflect_spirv(&spv, "main")
            .expect("reflect_spirv should succeed on hand-crafted SPV with uniform buffer");

        assert_eq!(
            reflected.workgroup_size,
            Some([64, 1, 1]),
            "workgroup size should match the LocalSize in the binary",
        );
        assert_eq!(
            reflected.gpu_consts_byte_size,
            Some(32),
            "gpu_consts_byte_size should be Some(32) for 8 × u32 members",
        );
    }

    /// Verify the byte-size calculation isn't hard-coded to 32: a 4-member
    /// struct should reflect as 16 bytes.
    #[test]
    fn reflect_spirv_reports_correct_byte_size_for_different_member_count() {
        let spv = compute_spirv_with_uniform(32, 1, 1, 4);
        let reflected = reflect_spirv(&spv, "main")
            .expect("reflect_spirv should succeed on 4-member uniform buffer");

        assert_eq!(reflected.workgroup_size, Some([32, 1, 1]));
        assert_eq!(
            reflected.gpu_consts_byte_size,
            Some(16),
            "gpu_consts_byte_size should be Some(16) for 4 × u32 members",
        );
    }

    /// A LocalSize with any zero dimension is invalid per the Vulkan spec.
    /// `reflect_spirv` must reject it rather than letting `dispatch_linear`
    /// divide by zero later.
    #[test]
    fn reflect_spirv_errors_on_zero_workgroup_dimension() {
        for (x, y, z) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            let spv = minimal_compute_spirv(x, y, z);
            let result = reflect_spirv(&spv, "main");

            assert!(
                matches!(result, Err(RendererError::ShaderReflectionFailed(_))),
                "expected ShaderReflectionFailed for LocalSize [{x}, {y}, {z}], \
                 got {result:?}",
            );
        }
    }

    /// Fewer than 5 words (20 bytes) can never be a valid SPIR-V header.
    /// `rspirv`'s parser rejects this as `HeaderIncomplete`.
    #[test]
    fn reflect_spirv_errors_on_truncated_bytes() {
        let result = reflect_spirv(&[0u8; 3], "main");

        assert!(
            matches!(result, Err(RendererError::ShaderReflectionFailed(_))),
            "expected ShaderReflectionFailed, got {result:?}",
        );
    }

    /// Five words (20 bytes) with a wrong magic number. `rspirv` rejects this
    /// as `HeaderIncorrect`.
    #[test]
    fn reflect_spirv_errors_on_bad_magic() {
        // A 5-word (20-byte) blob with a wrong first word.
        let bad_magic: [u8; 20] = [
            0xDE, 0xAD, 0xBE, 0xEF, // wrong magic
            0x00, 0x01, 0x00, 0x00, // version 1.0
            0x00, 0x00, 0x00, 0x00, // generator
            0x00, 0x00, 0x00, 0x00, // bound
            0x00, 0x00, 0x00, 0x00, // schema / reserved
        ];
        let result = reflect_spirv(&bad_magic, "main");

        assert!(
            matches!(result, Err(RendererError::ShaderReflectionFailed(_))),
            "expected ShaderReflectionFailed, got {result:?}",
        );
    }

    /// Requesting a non-existent entry point name on the embedded validation
    /// shader (which the placeholder SPV has no entry points for, and which
    /// the real DXC-compiled SPV has only "main"). Either way "nonexistent"
    /// won't match.
    #[test]
    fn reflect_spirv_errors_on_missing_entry_point() {
        let result = reflect_spirv(crate::shader::VALIDATION_CS_SPV, "nonexistent");

        assert!(
            matches!(result, Err(RendererError::ShaderReflectionFailed(_))),
            "expected ShaderReflectionFailed, got {result:?}",
        );
    }

    /// Build a minimal Fragment-shader SPIR-V with no `LocalSize` execution
    /// mode — raster shaders (vertex/fragment) have no workgroup concept.
    ///
    /// IDs: %void=1, %voidfn=2, %main=3, %label=4 (bound=5).
    fn minimal_fragment_spirv() -> Vec<u8> {
        // Fragment shader: OpEntryPoint Fragment, no LocalSize.
        // IDs: %void=1, %voidfn=2, %main=3, %label=4 (bound=5).
        let words: &[u32] = &[
            0x07230203, // magic
            0x00010100, // version 1.1
            0x00000000, // generator
            0x00000005, // bound
            0x00000000, // schema
            // OpCapability Shader
            0x00020011, 0x00000001,
            // OpMemoryModel Logical GLSL450
            0x0003000E, 0x00000000, 0x00000001,
            // OpEntryPoint Fragment %main "main"  (execution model 4 = Fragment)
            0x0005000F, 0x00000004, 0x00000003, 0x6E69616D, 0x00000000,
            // No OpExecutionMode — raster shaders have no LocalSize
            // %void = OpTypeVoid
            0x00020013, 0x00000001,
            // %voidfn = OpTypeFunction %void
            0x00030021, 0x00000002, 0x00000001,
            // %main = OpFunction %void None %voidfn
            0x00050036, 0x00000001, 0x00000003, 0x00000000, 0x00000002,
            // %label = OpLabel
            0x000200F8, 0x00000004,
            // OpReturn
            0x000100FD,
            // OpFunctionEnd
            0x00010038,
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// A fragment shader has no `LocalSize` execution mode — `workgroup_size`
    /// must be `None`. `gpu_consts_byte_size` is also `None` since this shader
    /// declares no descriptor bindings.
    #[test]
    fn reflect_spirv_returns_none_workgroup_size_for_raster_shader() {
        let spv = minimal_fragment_spirv();
        let reflected = reflect_spirv(&spv, "main")
            .expect("reflect_spirv should succeed on a valid fragment shader");

        assert_eq!(
            reflected.workgroup_size,
            None,
            "workgroup_size should be None for a raster (fragment) shader",
        );
        assert_eq!(
            reflected.gpu_consts_byte_size,
            None,
            "gpu_consts_byte_size should be None: shader has no descriptor bindings",
        );
    }

    /// Reflects `VALIDATION_CS_SPV` and asserts the workgroup size matches
    /// the `[numthreads(64, 1, 1)]` declared in `validation.cs.hlsl`.
    ///
    /// Gated because the placeholder SPV (produced when DXC is absent) is a
    /// bare 5-word header with no entry points. The real check requires a
    /// DXC-compiled SPV.
    #[test]
    #[ignore = "requires DXC-built SPV; run with --ignored on a machine with DXC installed"]
    fn reflect_spirv_reports_workgroup_size_for_validation_cs() {
        let reflected = reflect_spirv(crate::shader::VALIDATION_CS_SPV, "main")
            .expect("reflect_spirv should succeed on a DXC-compiled validation shader");

        assert_eq!(
            reflected.workgroup_size,
            Some([64, 1, 1]),
            "expected [64, 1, 1] matching [numthreads(64, 1, 1)] in validation.cs.hlsl",
        );
    }

    /// Reflects `VALIDATION_CS_SPV` and asserts the GpuConsts uniform buffer
    /// at (set=0, binding=0) has the same byte size as `GpuConstsData` (32).
    ///
    /// Gated for the same reason as `reflect_spirv_reports_workgroup_size`.
    #[test]
    #[ignore = "requires DXC-built SPV; run with --ignored on a machine with DXC installed"]
    fn reflect_spirv_reports_32_byte_gpu_consts_for_validation_cs() {
        let reflected = reflect_spirv(crate::shader::VALIDATION_CS_SPV, "main")
            .expect("reflect_spirv should succeed on a DXC-compiled validation shader");

        assert_eq!(
            reflected.gpu_consts_byte_size,
            Some(32),
            "expected Some(32) matching GpuConstsData (8 × u32)",
        );
    }

    // --- Storage buffer reflection helpers and tests ---

    /// Build a SPIR-V compute shader with:
    /// - A uniform buffer at (set=0, binding=0): `num_uniform_members` × u32.
    /// - A read-write storage buffer at (set=0, binding=1): struct wrapping a
    ///   runtime array of u32 with `ArrayStride = element_stride`, using the
    ///   modern `StorageBuffer` storage class + `Block` decoration.
    ///
    /// IDs: %1=void, %2=voidfn, %3=main, %4=label,
    ///      %5=uint, %6=uniform_struct, %7=uniform_ptr, %8=uniform_var,
    ///      %9=rta, %10=storage_struct, %11=storage_ptr, %12=storage_var.
    ///      Bound = 13.
    fn compute_spirv_with_uniform_and_storage_rw(
        x: u32, y: u32, z: u32,
        num_uniform_members: u32,
        element_stride: u32,
    ) -> Vec<u8> {
        assert!(num_uniform_members > 0);

        let mut w: Vec<u32> = Vec::new();

        // Header (bound = 13)
        w.extend_from_slice(&[0x07230203, 0x00010100, 0x00000000, 0x0000000D, 0x00000000]);

        // OpCapability Shader
        w.extend_from_slice(&[0x00020011, 0x00000001]);

        // OpMemoryModel Logical GLSL450
        w.extend_from_slice(&[0x0003000E, 0x00000000, 0x00000001]);

        // OpEntryPoint GLCompute %3 "main"
        w.extend_from_slice(&[0x0005000F, 0x00000005, 0x00000003, 0x6E69616D, 0x00000000]);

        // OpExecutionMode %3 LocalSize x y z
        w.extend_from_slice(&[0x00060010, 0x00000003, 0x00000011, x, y, z]);

        // --- Annotations ---

        // Uniform buffer (binding 0):
        // OpDecorate %8 DescriptorSet 0
        w.extend_from_slice(&[0x00040047, 0x00000008, 0x00000022, 0x00000000]);
        // OpDecorate %8 Binding 0
        w.extend_from_slice(&[0x00040047, 0x00000008, 0x00000021, 0x00000000]);
        // OpDecorate %6 Block
        w.extend_from_slice(&[0x00030047, 0x00000006, 0x00000002]);
        // OpMemberDecorate %6 i Offset (i*4)
        for i in 0..num_uniform_members {
            w.extend_from_slice(&[0x00050048, 0x00000006, i, 0x00000023, i * 4]);
        }

        // Storage buffer (binding 1):
        // OpDecorate %12 DescriptorSet 0
        w.extend_from_slice(&[0x00040047, 0x0000000C, 0x00000022, 0x00000000]);
        // OpDecorate %12 Binding 1
        w.extend_from_slice(&[0x00040047, 0x0000000C, 0x00000021, 0x00000001]);
        // OpDecorate %10 Block  (modern storage buffer)
        w.extend_from_slice(&[0x00030047, 0x0000000A, 0x00000002]);
        // OpDecorate %9 ArrayStride element_stride
        w.extend_from_slice(&[0x00040047, 0x00000009, 0x00000006, element_stride]);

        // --- Types ---

        // %1 = OpTypeVoid  (op=19=0x13, wc=2)
        w.extend_from_slice(&[0x00020013, 0x00000001]);
        // %2 = OpTypeFunction %1  (op=33=0x21, wc=3)
        w.extend_from_slice(&[0x00030021, 0x00000002, 0x00000001]);
        // %5 = OpTypeInt 32 0  (op=21=0x15, wc=4)
        w.extend_from_slice(&[0x00040015, 0x00000005, 0x00000020, 0x00000000]);

        // %6 = OpTypeStruct %5 × num_uniform_members  (op=30=0x1E)
        let ubuf_struct_wc = 2 + num_uniform_members;
        w.push((ubuf_struct_wc << 16) | 0x001E);
        w.push(0x00000006);
        w.extend(std::iter::repeat_n(0x00000005u32, num_uniform_members as usize));

        // %7 = OpTypePointer Uniform %6  (Uniform = 2, op=32=0x20, wc=4)
        w.extend_from_slice(&[0x00040020, 0x00000007, 0x00000002, 0x00000006]);
        // %8 = OpVariable %7 Uniform  (op=59=0x3B, wc=4)
        w.extend_from_slice(&[0x0004003B, 0x00000007, 0x00000008, 0x00000002]);

        // %9 = OpTypeRuntimeArray %5  (op=29=0x1D, wc=3)
        w.extend_from_slice(&[0x0003001D, 0x00000009, 0x00000005]);
        // %10 = OpTypeStruct %9  (op=30=0x1E, wc=3)
        w.extend_from_slice(&[0x0003001E, 0x0000000A, 0x00000009]);
        // %11 = OpTypePointer StorageBuffer %10  (StorageBuffer=12=0xC, op=32=0x20, wc=4)
        w.extend_from_slice(&[0x00040020, 0x0000000B, 0x0000000C, 0x0000000A]);
        // %12 = OpVariable %11 StorageBuffer  (op=59=0x3B, wc=4)
        w.extend_from_slice(&[0x0004003B, 0x0000000B, 0x0000000C, 0x0000000C]);

        // --- Function ---

        // %3 = OpFunction %1 None %2  (op=54=0x36, wc=5)
        w.extend_from_slice(&[0x00050036, 0x00000001, 0x00000003, 0x00000000, 0x00000002]);
        // %4 = OpLabel  (op=248=0xF8, wc=2)
        w.extend_from_slice(&[0x000200F8, 0x00000004]);
        // OpReturn  (op=253=0xFD, wc=1)
        w.push(0x000100FD);
        // OpFunctionEnd  (op=56=0x38, wc=1)
        w.push(0x00010038);

        w.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    /// A 2-binding shader (uniform at 0, storage-rw at 1) reflects the correct
    /// (binding, kind) pairs in ascending binding order.
    #[test]
    fn reflect_spirv_entries_uniform_and_storage_rw() {
        let spv = compute_spirv_with_uniform_and_storage_rw(64, 1, 1, 8, 4);
        let reflected = reflect_spirv(&spv, "main")
            .expect("reflect_spirv should succeed on 2-binding shader");

        assert_eq!(reflected.entries.len(), 2, "expected 2 entries");

        assert_eq!(reflected.entries[0].binding, 0);
        assert!(
            matches!(reflected.entries[0].kind, BindKind::UniformBuffer { size: 32 }),
            "slot 0 should be UniformBuffer {{ size: 32 }}, got {:?}",
            reflected.entries[0].kind,
        );

        assert_eq!(reflected.entries[1].binding, 1);
        assert!(
            matches!(reflected.entries[1].kind, BindKind::StorageBufferReadWrite { size: 4 }),
            "slot 1 should be StorageBufferReadWrite {{ size: 4 }}, got {:?}",
            reflected.entries[1].kind,
        );
    }

    /// Uniform buffer entries show up in `entries` and `gpu_consts_byte_size`
    /// is derived from the slot-0 entry.
    #[test]
    fn reflect_spirv_entries_match_gpu_consts_byte_size() {
        let spv = compute_spirv_with_uniform(64, 1, 1, 8);
        let reflected = reflect_spirv(&spv, "main")
            .expect("reflect_spirv should succeed");

        assert_eq!(reflected.entries.len(), 1);
        assert_eq!(reflected.entries[0].binding, 0);
        assert!(
            matches!(reflected.entries[0].kind, BindKind::UniformBuffer { size: 32 }),
        );
        assert_eq!(reflected.gpu_consts_byte_size, Some(32));
    }
}
