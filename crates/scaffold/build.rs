//! Build script for compiling HLSL shaders to SPIR-V via DXC.
//!
//! Searches for DXC in the Vulkan SDK and common system paths. Each
//! `.hlsl` file in `src/shaders/` is compiled with the appropriate
//! target profile derived from its filename suffix:
//!
//!   `.vs.hlsl` -> `vs_6_0`  (vertex shader)
//!   `.ps.hlsl` -> `ps_6_0`  (pixel shader)
//!   `.cs.hlsl` -> `cs_6_0`  (compute shader)
//!
//! Files in `src/shaders/include/` are header-only and not compiled
//! directly. The include path is set so `#include "include/foo.hlsl"`
//! resolves correctly.
//!
//! Output `.spv` files are written to `OUT_DIR` and loaded at compile
//! time via `include_bytes!`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let shader_dir = Path::new("src/shaders");
    let out_dir    = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Re-run if any shader source changes.
    println!("cargo:rerun-if-changed=src/shaders");

    let dxc = match find_dxc() {
        Some(path) => path,
        None => {
            // DXC not available. Write empty placeholder .spv files so
            // include_bytes! in the Rust source compiles. The actual
            // shaders will be compiled when DXC is in PATH.
            println!(
                "cargo:warning=DXC not found. Writing placeholder .spv \
                 files. Set VULKAN_SDK or add DXC to PATH to compile \
                 shaders.",
            );

            write_placeholder_spvs(shader_dir, &out_dir);
            return;
        }
    };

    // Compile each top-level .hlsl file (skip include/).
    for entry in fs::read_dir(shader_dir).unwrap() {
        let entry = entry.unwrap();
        let path  = entry.path();

        if !path.is_file() {
            continue;
        }

        let name = path.file_name().unwrap().to_str().unwrap();

        if !name.ends_with(".hlsl") {
            continue;
        }

        let profile = match () {
            _ if name.ends_with(".vs.hlsl") => "vs_6_0",
            _ if name.ends_with(".ps.hlsl") => "ps_6_0",
            _ if name.ends_with(".cs.hlsl") => "cs_6_0",
            _ => {
                panic!(
                    "Unknown shader type for {name}. Expected \
                     .vs.hlsl, .ps.hlsl, or .cs.hlsl suffix.",
                );
            }
        };

        let entry_point = "main";

        // Output filename: replace .hlsl with .spv.
        let spv_name = name.replace(".hlsl", ".spv");
        let spv_path = out_dir.join(&spv_name);

        let status = Command::new(&dxc)
            .arg("-spirv")
            .arg("-T").arg(profile)
            .arg("-E").arg(entry_point)
            .arg("-I").arg(shader_dir)
            .arg("-Fo").arg(&spv_path)
            .arg("-fspv-target-env=vulkan1.1")
            .arg("-O3")
            .arg(&path)
            .status()
            .unwrap_or_else(|e| {
                panic!("Failed to run DXC for {name}: {e}");
            });

        if !status.success() {
            panic!("DXC compilation failed for {name}");
        }
    }
}

/// Write empty placeholder .spv files for all shaders when DXC is unavailable.
///
/// This allows the Rust code to compile (the `include_bytes!` calls need
/// the files to exist), but the shaders will not actually work until
/// recompiled with DXC.
fn write_placeholder_spvs(shader_dir: &Path, out_dir: &Path) {
    // Minimal valid SPIR-V: just the magic number and version header.
    // This won't create a working shader but lets include_bytes! succeed.
    let placeholder: &[u8] = &[
        0x03, 0x02, 0x23, 0x07, // SPIR-V magic
        0x00, 0x00, 0x01, 0x00, // version 1.0
        0x00, 0x00, 0x00, 0x00, // generator
        0x01, 0x00, 0x00, 0x00, // bound
        0x00, 0x00, 0x00, 0x00, // reserved
    ];

    for entry in fs::read_dir(shader_dir).unwrap() {
        let entry = entry.unwrap();
        let path  = entry.path();

        if !path.is_file() {
            continue;
        }

        let name = path.file_name().unwrap().to_str().unwrap();

        if !name.ends_with(".hlsl") {
            continue;
        }

        let spv_name = name.replace(".hlsl", ".spv");
        let spv_path = out_dir.join(&spv_name);

        fs::write(&spv_path, placeholder).unwrap_or_else(|e| {
            panic!("Failed to write placeholder {}: {e}", spv_path.display());
        });
    }
}

/// Search for the DXC executable in common locations.
fn find_dxc() -> Option<PathBuf> {
    // Check explicit environment variable first.
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let candidate = PathBuf::from(&sdk).join("bin/dxc");

        if candidate.exists() {
            return Some(candidate);
        }
    }

    // Check PATH.
    if let Ok(output) = Command::new("which").arg("dxc").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout)
                .trim()
                .to_string();

            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    // Common Linux paths.
    let candidates = [
        "/usr/bin/dxc",
        "/usr/local/bin/dxc",
    ];

    for path in candidates {
        if Path::new(path).exists() {
            return Some(PathBuf::from(path));
        }
    }

    None
}
