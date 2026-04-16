//! Build script for the renderer crate.
//!
//! Compiles every HLSL shader under `shaders/` to SPIR-V via DXC and places
//! the output in `$OUT_DIR/shaders/<basename>.spv`, matching the toolchain
//! the old scaffold crate used. Entry point suffixes are how the stage is
//! inferred:
//!
//! - `*.cs.hlsl` → `cs_6_0`
//! - `*.vs.hlsl` → `vs_6_0`
//! - `*.ps.hlsl` → `ps_6_0`
//!
//! Files under `shaders/include/` are treated as headers: they are tracked
//! for rerun purposes but not compiled directly. The entry-point shaders
//! `#include "include/<name>.hlsl"` them via `-I <shaders_dir>`.
//!
//! If DXC cannot be located (via `VULKAN_SDK/bin/dxc` or `PATH`), a minimal
//! placeholder SPIR-V header is written instead — just the 5-word
//! magic/version/generator/bound/schema prelude. This is enough to satisfy
//! the CPU-side magic-number test and leaves Rust builds green on machines
//! without the Vulkan SDK. It is *not* a loadable shader module; anyone who
//! tries to pass the placeholder through `create_shader_module_passthrough`
//! at runtime will fail loudly, which is the intended behaviour for CI.
//! The game crate's `--validate` mode adds a higher-level check that
//! pattern-matches this sentinel and prints an actionable message.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"),
    );
    let shaders_dir = manifest_dir.join("shaders");
    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR must be set by cargo"));
    let spv_out_dir = out_dir.join("shaders");
    std::fs::create_dir_all(&spv_out_dir)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", spv_out_dir.display()));

    // Rerun on any shader source change, env change, or build.rs edit.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    if shaders_dir.is_dir() {
        track_dir_recursively(&shaders_dir);
    }

    let dxc = find_dxc();

    // Compile every top-level *.cs.hlsl / *.vs.hlsl / *.ps.hlsl. Files under
    // `include/` are headers and skipped here — they are still tracked above.
    for (src, stage) in collect_entry_points(&shaders_dir) {
        let out = spv_out_dir.join(spv_output_name(&src));
        match &dxc {
            Some(dxc_path) => compile_with_dxc(dxc_path, &src, &out, stage, &shaders_dir),
            None => write_placeholder_spv(&out),
        }
    }
}

/// Entry-point shader stage suffixes → DXC `-T` profile.
const STAGE_SUFFIXES: &[(&str, &str)] = &[
    (".cs.hlsl", "cs_6_0"),
    (".vs.hlsl", "vs_6_0"),
    (".ps.hlsl", "ps_6_0"),
];

/// Walk `dir` recursively and emit `cargo:rerun-if-changed=` for every file.
/// The `shaders/include/` tree is picked up this way without being special-cased.
fn track_dir_recursively(dir: &Path) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            track_dir_recursively(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Collect the entry-point shader files at the top of `shaders_dir` (not recursive).
/// Returns `(absolute source path, dxc target profile)` pairs.
fn collect_entry_points(shaders_dir: &Path) -> Vec<(PathBuf, &'static str)> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(shaders_dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        if let Some(stage) = STAGE_SUFFIXES
            .iter()
            .find_map(|(suffix, stage)| name.ends_with(suffix).then_some(*stage))
        {
            out.push((path, stage));
        }
    }
    out
}

/// Map `validation.cs.hlsl` → `validation.cs.spv`.
fn spv_output_name(src: &Path) -> String {
    let name = src
        .file_name()
        .and_then(OsStr::to_str)
        .expect("entry-point shader must have a unicode file name");
    let stem = name
        .strip_suffix(".hlsl")
        .unwrap_or_else(|| panic!("entry-point shader `{name}` must end in .hlsl"));
    format!("{stem}.spv")
}

/// Try `$VULKAN_SDK/bin/dxc` first, then `dxc` on `PATH`.
fn find_dxc() -> Option<PathBuf> {
    if let Some(vulkan_sdk) = std::env::var_os("VULKAN_SDK") {
        let candidate = PathBuf::from(vulkan_sdk).join("bin").join("dxc");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    // `Command::new("dxc").arg("--version").output()` returns `Err` if the
    // binary cannot be spawned (typical "not on PATH" failure).
    if Command::new("dxc").arg("--version").output().is_ok() {
        return Some(PathBuf::from("dxc"));
    }
    None
}

/// Invoke DXC. Any non-zero exit aborts the build with the shader path.
fn compile_with_dxc(dxc: &Path, src: &Path, out: &Path, stage: &str, include_dir: &Path) {
    let status = Command::new(dxc)
        .arg("-spirv")
        .arg("-fspv-target-env=vulkan1.1")
        // Use D3D constant-buffer packing rules instead of std140. Without
        // this flag DXC pads float3 to 16-byte alignment (std140), so
        // float3+float occupies 20 bytes rather than 16 — silently
        // misaligning every field after the first float3 against repr(C)
        // Rust structs where [f32;3]+f32 packs tightly to 16 bytes.
        .arg("-fvk-use-dx-layout")
        .arg("-O3")
        .arg("-T")
        .arg(stage)
        .arg("-E")
        .arg("main")
        .arg("-I")
        .arg(include_dir)
        .arg(src)
        .arg("-Fo")
        .arg(out)
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn dxc for {}: {e}", src.display()));
    if !status.success() {
        panic!(
            "dxc failed for {}: exit status {}",
            src.display(),
            status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "<signal>".to_string()),
        );
    }
}

/// Write a minimal SPIR-V header — just enough bytes for the CPU test to see
/// the magic number. Not a loadable module.
fn write_placeholder_spv(out: &Path) {
    // 5 u32 words: magic, version 1.0, generator, id bound, schema. The
    // bytes are written in native endian so the wgpu host-side tests see the
    // magic number at offset 0 on every platform we build on (all LE today).
    const PLACEHOLDER_WORDS: [u32; 5] = [
        0x0723_0203, // SPIR-V magic number
        0x0001_0000, // version 1.0 (major.minor in the high two bytes)
        0x0000_0000, // generator: unknown
        0x0000_0000, // id bound
        0x0000_0000, // schema / reserved
    ];
    let mut bytes = Vec::with_capacity(PLACEHOLDER_WORDS.len() * 4);
    for word in PLACEHOLDER_WORDS {
        bytes.extend_from_slice(&word.to_ne_bytes());
    }
    std::fs::write(out, bytes)
        .unwrap_or_else(|e| panic!("failed to write placeholder SPV at {}: {e}", out.display()));
}
