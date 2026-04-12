//! SPIR-V workgroup-size and descriptor reflection via `rspirv`.
//!
//! Exposes [`reflect_spirv`] which parses a SPIR-V binary, locates a named
//! entry point, and returns [`Reflected`] — the workgroup size extracted from
//! the `LocalSize` execution mode, and the uniform-buffer byte size at
//! descriptor set 0 binding 0 (the slot [`GpuConstsData`] occupies in every
//! pipeline).
//!
//! The implementation uses `rspirv`'s data-representation layer (`rspirv::dr`)
//! and scans the module's `entry_points`, `execution_modes`, `annotations`,
//! and `types_global_values` collections — no full instruction stream walk is
//! needed.
//!
//! [`GpuConstsData`]: crate::gpu_consts::GpuConstsData

use rspirv::dr::Operand;
use rspirv::spirv;

use crate::error::RendererError;

// --- Reflected ---

/// The properties extracted from a SPIR-V module by [`reflect_spirv`].
///
/// All fields are derived from the parsed module without GPU involvement.
#[derive(Debug)]
pub struct Reflected {
    /// The workgroup size declared by the entry point's `LocalSize` execution
    /// mode — `[x, y, z]`, matching `[numthreads(x, y, z)]` in HLSL.
    pub workgroup_size: [u32; 3],
    /// Size in bytes of the uniform buffer at descriptor set 0 binding 0, or
    /// `None` if the shader does not declare one. This slot is always
    /// `GpuConstsData` in the renderer's binding model (forced injection by
    /// [`BindingLayout`](crate::pipeline::binding::BindingLayout)); the
    /// [`ComputePipeline`](crate::pipeline::compute::ComputePipeline)
    /// constructor asserts this value equals `size_of::<GpuConstsData>()`.
    pub gpu_consts_byte_size: Option<u32>,
}

// --- reflect_spirv ---

/// Parse `spv` (raw SPIR-V bytes) and reflect the named `entry_point`.
///
/// Returns [`Reflected`] on success. Returns
/// [`RendererError::ShaderReflectionFailed`] for any structural problem in the
/// module: parse failure, missing entry point, unsupported `LocalSizeId`
/// execution mode, or other malformed SPIR-V.
///
/// Only literal `LocalSize` execution modes are supported. If the shader uses
/// `LocalSizeId` (spec-constant workgroup sizes), this function returns an
/// error advising the caller to switch to literal `numthreads`. The DXC
/// toolchain emits `LocalSize` for literal `[numthreads(...)]`, so this
/// restriction is transparent for all shaders in the renderer.
pub fn reflect_spirv(spv: &[u8], entry_point: &str)
    -> Result<Reflected, RendererError>
{
    let module = rspirv::dr::load_bytes(spv)
        .map_err(|e| RendererError::ShaderReflectionFailed(
            format!("failed to parse SPIR-V: {e}"),
        ))?;

    let fn_id = find_entry_point_fn_id(&module, entry_point)?;
    let workgroup_size = find_workgroup_size(&module, fn_id, entry_point)?;
    let gpu_consts_byte_size = find_gpu_consts_byte_size(&module);

    Ok(Reflected { workgroup_size, gpu_consts_byte_size })
}

// --- private helpers ---

/// Walk `module.entry_points` to find the `OpEntryPoint` whose name operand
/// matches `name`. Returns the function-id (`IdRef`) that the `OpEntryPoint`
/// instruction references.
///
/// `OpEntryPoint` operand layout: `[ExecutionModel, IdRef(fn_id),
/// LiteralString(name), IdRef*(interface)]`.
fn find_entry_point_fn_id(
    module: &rspirv::dr::Module,
    name: &str,
)
    -> Result<spirv::Word, RendererError>
{
    for inst in &module.entry_points {
        // Operands[1] = IdRef(fn_id), Operands[2] = LiteralString(name).
        if let (
            Some(Operand::IdRef(fn_id)),
            Some(Operand::LiteralString(ep_name)),
        ) = (inst.operands.get(1), inst.operands.get(2))
            && ep_name == name
        {
            return Ok(*fn_id);
        }
    }

    Err(RendererError::ShaderReflectionFailed(format!(
        "no entry point named '{name}' found in SPIR-V module"
    )))
}

/// Walk `module.execution_modes` to find the `OpExecutionMode` for `fn_id`
/// with mode `LocalSize`. Returns `[x, y, z]`.
///
/// `OpExecutionMode` operand layout: `[IdRef(fn_id), ExecutionMode(mode),
/// LiteralBit32(x), LiteralBit32(y), LiteralBit32(z)]` for `LocalSize`.
fn find_workgroup_size(
    module: &rspirv::dr::Module,
    fn_id: spirv::Word,
    entry_point: &str,
)
    -> Result<[u32; 3], RendererError>
{
    for inst in &module.execution_modes {
        let Some(Operand::IdRef(id)) = inst.operands.first() else {
            continue;
        };

        if *id != fn_id {
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
                    return Err(RendererError::ShaderReflectionFailed(format!(
                        "entry point '{entry_point}' has invalid LocalSize \
                         [{x}, {y}, {z}]; all dimensions must be ≥ 1 \
                         (Vulkan spec: WorkgroupSize must be at least 1 \
                         in each dimension)"
                    )));
                }

                return Ok([x, y, z]);
            }

            spirv::ExecutionMode::LocalSizeId => {
                return Err(RendererError::ShaderReflectionFailed(format!(
                    "entry point '{entry_point}' uses LocalSizeId (spec-constant \
                     workgroup size); use literal numthreads instead — \
                     LocalSizeId is not supported in the first rewrite pass"
                )));
            }

            _ => {}
        }
    }

    Err(RendererError::ShaderReflectionFailed(format!(
        "no LocalSize execution mode found for entry point '{entry_point}'"
    )))
}

/// Extract a `LiteralBit32` from `operands[index]`, or return an error.
fn extract_literal_bit32(
    operands: &[Operand],
    index: usize,
    label: &str,
)
    -> Result<u32, RendererError>
{
    match operands.get(index) {
        Some(Operand::LiteralBit32(v)) => Ok(*v),
        other => Err(RendererError::ShaderReflectionFailed(format!(
            "expected LiteralBit32 for {label}, got {other:?}"
        ))),
    }
}

/// Scan annotations + types to find the uniform buffer at descriptor set 0
/// binding 0, then compute its byte size from the pointed-to struct's members.
///
/// Returns `None` if no variable is decorated with `(set=0, binding=0)`.
/// Returns `Some(size)` with the sum of member widths in bytes otherwise.
///
/// Implementation strategy:
/// 1. Collect all variable ids decorated with `DescriptorSet = 0`.
/// 2. From those, find the one also decorated with `Binding = 0`.
/// 3. Follow: `OpVariable.result_type` → `OpTypePointer` → struct id.
/// 4. Walk the struct's member `OpTypeInt` (or other scalar) types and sum
///    widths. Member `Offset` decorations are used if present, falling back to
///    a straight sum for contiguous layouts.
fn find_gpu_consts_byte_size(module: &rspirv::dr::Module) -> Option<u32> {
    // Step 1–2: find variable id at (set=0, binding=0).
    let var_id = find_var_at_set0_binding0(module)?;

    // Step 3: find the OpVariable, follow its result_type (a pointer type id)
    // to the pointed-to struct type id.
    let struct_id = find_pointed_struct_id(module, var_id)?;

    // Step 4: compute byte size from struct member types + optional offsets.
    compute_struct_byte_size(module, struct_id)
}

/// Find the SPIR-V variable id decorated with both `DescriptorSet = 0` and
/// `Binding = 0`, scanning `module.annotations`.
fn find_var_at_set0_binding0(module: &rspirv::dr::Module) -> Option<spirv::Word> {
    let mut set0_vars: Vec<spirv::Word> = Vec::new();
    let mut binding0_vars: Vec<spirv::Word> = Vec::new();

    for inst in &module.annotations {
        let opcode = inst.class.opcode;

        if opcode != spirv::Op::Decorate {
            continue;
        }

        // OpDecorate operands: [IdRef(target), Decoration, ...extra]
        let (Some(Operand::IdRef(target)), Some(Operand::Decoration(deco))) =
            (inst.operands.first(), inst.operands.get(1))
        else {
            continue;
        };

        match deco {
            spirv::Decoration::DescriptorSet => {
                if let Some(Operand::LiteralBit32(0)) = inst.operands.get(2) {
                    set0_vars.push(*target);
                }
            }
            spirv::Decoration::Binding => {
                if let Some(Operand::LiteralBit32(0)) = inst.operands.get(2) {
                    binding0_vars.push(*target);
                }
            }
            _ => {}
        }
    }

    // The variable must appear in both lists.
    set0_vars
        .iter()
        .find(|id| binding0_vars.contains(id))
        .copied()
}

/// Given a variable id, walk `module.types_global_values` to find the
/// `OpVariable` with that result id, then follow its `result_type` (a pointer
/// type id) through `OpTypePointer` to get the pointee struct type id.
fn find_pointed_struct_id(
    module: &rspirv::dr::Module,
    var_id: spirv::Word,
)
    -> Option<spirv::Word>
{
    // Find the OpVariable to get its result_type (the pointer type id).
    // Only accept Uniform storage class — storage buffers also live at
    // descriptor set 0 / binding 0 in some shaders and must not be mistaken
    // for the GpuConsts uniform buffer.
    // OpVariable operand layout: [StorageClass, optional initializer].
    let pointer_type_id = module.types_global_values.iter()
        .find(|inst| {
            inst.class.opcode == spirv::Op::Variable
                && inst.result_id == Some(var_id)
                && matches!(
                    inst.operands.first(),
                    Some(Operand::StorageClass(spirv::StorageClass::Uniform))
                )
        })
        .and_then(|inst| inst.result_type)?;

    // Find the OpTypePointer with that result id.
    // OpTypePointer operand layout: [StorageClass, IdRef(pointee)].
    // Require Uniform storage class as a belt-and-suspenders check: a
    // storage-buffer pointer (StorageBuffer class) pointing to a struct must
    // not be treated as the GpuConsts slot.
    let struct_id = module.types_global_values.iter()
        .find(|inst| {
            inst.class.opcode == spirv::Op::TypePointer
                && inst.result_id == Some(pointer_type_id)
                && matches!(
                    inst.operands.first(),
                    Some(Operand::StorageClass(spirv::StorageClass::Uniform))
                )
        })
        .and_then(|inst| {
            if let Some(Operand::IdRef(pointee)) = inst.operands.get(1) {
                Some(*pointee)
            }
            else {
                None
            }
        })?;

    // Verify the pointee is actually a struct, then return its id.
    let is_struct = module.types_global_values.iter()
        .any(|inst| {
            inst.class.opcode == spirv::Op::TypeStruct
                && inst.result_id == Some(struct_id)
        });

    if is_struct { Some(struct_id) } else { None }
}

/// Compute the total byte size of a struct type.
///
/// Strategy: collect all member `Offset` decorations from
/// `module.annotations` (`OpMemberDecorate`). If offsets are present,
/// size = `offset_of_last_member + size_of_last_member`. If absent, fall back
/// to summing each member's scalar bit-width in bytes.
///
/// For `GpuConstsData` (8 × u32, no vector members, offsets at 0, 4, 8, …
/// 28) both paths produce 32.
fn compute_struct_byte_size(
    module: &rspirv::dr::Module,
    struct_id: spirv::Word,
)
    -> Option<u32>
{
    // Collect the struct's member type ids (IdRef operands of OpTypeStruct).
    let struct_inst = module.types_global_values.iter()
        .find(|inst| {
            inst.class.opcode == spirv::Op::TypeStruct
                && inst.result_id == Some(struct_id)
        })?;

    let member_type_ids: Vec<spirv::Word> = struct_inst.operands.iter()
        .filter_map(|op| {
            if let Operand::IdRef(id) = op { Some(*id) } else { None }
        })
        .collect();

    if member_type_ids.is_empty() {
        return Some(0);
    }

    // Collect OpMemberDecorate Offset values for this struct.
    // offsets[member_index] = offset in bytes, if declared.
    let mut offsets: Vec<Option<u32>> = vec![None; member_type_ids.len()];

    for inst in &module.annotations {
        if inst.class.opcode != spirv::Op::MemberDecorate {
            continue;
        }

        // OpMemberDecorate: [IdRef(struct_id), LiteralBit32(member), Decoration, ...extra]
        let (
            Some(Operand::IdRef(sid)),
            Some(Operand::LiteralBit32(member_idx)),
            Some(Operand::Decoration(spirv::Decoration::Offset)),
            Some(Operand::LiteralBit32(offset)),
        ) = (
            inst.operands.first(),
            inst.operands.get(1),
            inst.operands.get(2),
            inst.operands.get(3),
        ) else {
            continue;
        };

        if *sid == struct_id {
            let idx = *member_idx as usize;

            if idx < offsets.len() {
                offsets[idx] = Some(*offset);
            }
        }
    }

    let all_offsets_present = offsets.iter().all(|o| o.is_some());

    if all_offsets_present {
        // Use offset of last member + size of last member.
        let last_idx = member_type_ids.len() - 1;
        let last_offset = offsets[last_idx]?;
        let last_type_id = member_type_ids[last_idx];
        let last_size = member_byte_size(module, last_type_id)?;

        Some(last_offset + last_size)
    }
    else {
        // Fall back: sum all member sizes in order.
        let mut total: u32 = 0;

        for type_id in &member_type_ids {
            total += member_byte_size(module, *type_id)?;
        }

        Some(total)
    }
}

/// Return the byte size of a SPIR-V scalar/vector type by looking it up in
/// `module.types_global_values`.
///
/// Handles `OpTypeInt` (width/8) and `OpTypeFloat` (width/8). Returns `None`
/// for unrecognised types — the caller treats `None` as "couldn't compute
/// size."
fn member_byte_size(module: &rspirv::dr::Module, type_id: spirv::Word) -> Option<u32> {
    let inst = module.types_global_values.iter()
        .find(|i| i.result_id == Some(type_id))?;

    match inst.class.opcode {
        spirv::Op::TypeInt | spirv::Op::TypeFloat => {
            // operands[0] = LiteralBit32(width in bits)
            if let Some(Operand::LiteralBit32(width)) = inst.operands.first() {
                Some(width / 8)
            }
            else {
                None
            }
        }
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
            [64, 1, 1],
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
            [64, 1, 1],
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

        assert_eq!(reflected.workgroup_size, [32, 1, 1]);
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
            [64, 1, 1],
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
}
