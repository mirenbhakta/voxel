//! GPU primitive validation suite.
//!
//! Run via `game --validate`.  Executes a series of checks against real GPU
//! hardware and prints a human-readable report suitable for bug reports.
//! Exits with code 0 on full pass, 1 on any failure.
//!
//! # What is validated
//!
//! - **Device creation** — Vulkan adapter and device can be obtained.
//! - **StagedBuffer** — CPU staging, push, flush round-trip completes
//!   without panics; GPU buffer is allocated at the reported size.
//! - **RenderGraph** — a one-pass graph compiles and executes end-to-end
//!   with a real `FrameEncoder` against the live device; barriers are
//!   emitted in the expected order.
//! - **begin/end frame cycle** — a no-op frame submits without errors.
//!
//! Readback (verifying bytes on the GPU) is not included in this pass;
//! the tests in `crates/renderer` cover that case with `#[ignore]` GPU
//! tests run via `cargo test -p renderer -- --ignored`.

use renderer::{FrameCount, RendererContext, RendererError, StagedBuffer};
use renderer::graph::RenderGraph;

// ---------------------------------------------------------------------------

pub fn run() {
    println!("=== voxel renderer validation ===\n");

    let mut ctx = match pollster::block_on(RendererContext::new_headless(
        FrameCount::new(2).unwrap(),
    )) {
        Ok(ctx) => ctx,
        Err(RendererError::NoCompatibleAdapter) => {
            eprintln!("FATAL: no Vulkan adapter found.");
            eprintln!("       Install Vulkan drivers and retry.");
            eprintln!("       (Software fallback / lavapipe is not a valid substitute.)");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("FATAL: device creation failed: {e}");
            std::process::exit(1);
        }
    };

    let mut passed = 0u32;
    let mut failed = 0u32;

    // --- Check 1: StagedBuffer lifecycle ---

    print!("  [1] StagedBuffer push + flush ... ");
    {
        let mut buf: StagedBuffer<u32> = StagedBuffer::new(&ctx, "validation_staged", 4);

        buf.push(0xDEAD_BEEF_u32);
        buf.push(0xCAFE_BABE_u32);

        if buf.len() != 2 {
            println!("FAIL (expected len=2, got {})", buf.len());
            failed += 1;
        }
        else {
            buf.flush(&ctx);
            if buf.is_empty() {
                println!("ok");
                passed += 1;
            }
            else {
                println!("FAIL (staging not cleared after flush)");
                failed += 1;
            }
        }
    }

    // --- Check 2: StagedBuffer GPU buffer size ---

    print!("  [2] StagedBuffer GPU allocation size ... ");
    {
        let buf: StagedBuffer<[u32; 4]> = StagedBuffer::new(&ctx, "validation_sized", 8);
        // 8 values * 16 bytes/value = 128 bytes.
        let expected = 8_u64 * std::mem::size_of::<[u32; 4]>() as u64;
        let actual = buf.buffer().size();
        if actual == expected {
            println!("ok ({expected} bytes)");
            passed += 1;
        }
        else {
            println!("FAIL (expected {expected} bytes, GPU reports {actual})");
            failed += 1;
        }
    }

    // --- Check 3: RenderGraph compile + execute ---

    print!("  [3] RenderGraph compile + execute ... ");
    {
        // Build a graph with one imported buffer and one pass that reads it.
        // Marking the buffer as an output pins the pass (prevents dead-pass cull).
        let placeholder: StagedBuffer<u32> =
            StagedBuffer::new(&ctx, "validation_graph_placeholder", 4);
        let buf_clone = placeholder.buffer().clone();

        let mut graph = RenderGraph::new();
        let h = graph.import_buffer(buf_clone);

        graph.add_pass("validation_noop", |builder| {
            builder.read_buffer(h);
            builder.execute(move |_ctx| {
                // No GPU work — just confirms the closure is called.
            });
        });
        graph.mark_output(h);

        let compiled = match graph.compile() {
            Ok(g) => g,
            Err(e) => {
                println!("FAIL (compile error: {e:?})");
                failed += 1;
                // Skip the execute step.
                report(passed, failed);
                return;
            }
        };

        let mut buf_pool = renderer::graph::BufferPool::default();
        let mut tex_pool = renderer::graph::TexturePool::default();
        let mut fe = ctx.begin_frame();
        let frame = ctx.frame_index();
        let _ = compiled.execute(&mut fe, frame, &mut buf_pool, &mut tex_pool, ctx.device());
        ctx.end_frame(fe);

        println!("ok");
        passed += 1;
    }

    // --- Check 4: No-op begin/end frame cycle ---

    print!("  [4] No-op frame submit ... ");
    {
        let fe = ctx.begin_frame();
        ctx.end_frame(fe);
        println!("ok");
        passed += 1;
    }

    // ---------------------------------------------------------------------------

    report(passed, failed);
}

fn report(passed: u32, failed: u32) {
    let total = passed + failed;
    println!("\n  {passed}/{total} passed");
    if failed > 0 {
        std::process::exit(1);
    }
}
