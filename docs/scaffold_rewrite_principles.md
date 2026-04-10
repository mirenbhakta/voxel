# Scaffold Rewrite Principles

Design principles for the next iteration of the `scaffold` crate. Recorded
after a debugging session that surfaced accumulated architectural debt which
cannot be resolved by patches in place.


## Why a rewrite

The `scaffold` crate began as a feasibility spike for the rendering design
proposal and has grown into a working renderer. Over time the CPU↔GPU
boundary has accumulated a specific class of bug that keeps recurring in new
forms. A non-exhaustive list from the git history and Agentic Memory:

1. `GpuAllocator::upload` overwrote unconsumed GPU free list entries each
   frame, causing permanent range leaks during mass unloads (fixed by moving
   to GPU-autonomous allocation).
2. Chunks removed while a build was in flight silently leaked their GPU
   ranges because CPU `process_build_feedback` couldn't find the chunk
   (fixed by the GPU Phase 1 scan).
3. Alloc overflow left `quad_count` visible to the cull shader, which
   rendered chunks with missing materials (fixed by zeroing on overflow).
4. Quad allocator leaked ranges via bump retraction + `set_bump_offset`
   interaction during rebuilds (fixed by removing the CPU mirror entirely).
5. GPU free list `push_to_free_list` has no bounds check. Under rapid
   view-distance changes the unbounded push rate exceeds the bounded
   drain rate; once the count exceeds `FREE_LIST_MAX` every subsequent
   allocation corrupts the list via swap-with-last reading OOB zeros.
   Observed in a capture: 93% of the quad buffer leaked (973,931 of
   1,048,576 quads), 1,495 phantom entries past buffer end, 115 live
   entries already zero-wiped by swap-with-last.
6. `MAX_CHUNKS` defined independently in `world.rs` (Rust) and
   `bindings.hlsl` (HLSL) with no link between them. A debug-time
   reduction in one file was not caught by the build, silently producing
   out-of-bounds loads in the alloc shader.

Every one of these traces back to the same root cause: the CPU and GPU
share mutable state, and the CPU either simulates the GPU to predict that
state or expects the GPU to keep up with commands synchronously. Neither
assumption is robust. Patches have converted each instance of the bug
into a new instance, but the failure mode is structural.


## Principles

These are the invariants that the next iteration must establish by
construction. Each one exists because the current crate violates it and
a bug has been paid for that violation.


### 1. CPU↔GPU messaging is per-frame-in-flight and bounded

Every buffer carrying a message between CPU and GPU is a ring replicated
across `N` copies where `N` is the number of frames in flight. The CPU
writes the copy for frame `k+N` while the GPU reads copy `k`. No
single-copy buffer is ever mutated on one side while the other reads it.

This applies to:

- Upload channels (occupancy edits, build requests, slot zeroing, buffer
  shape descriptors).
- Readback channels (quad counts, overflow flags, command acks, state
  snapshots).
- Indirect dispatch parameter buffers the CPU writes for the GPU to read.

What it rules out: the current pattern where the CPU writes `chunk_meta`
zeros directly into the buffer that the alloc shader will read on the
*next* dispatch. That pattern works when the GPU happens to drain fast
enough and corrupts state when it does not.

**Rings are bounded channels.** A ring has finite capacity. When the CPU
tries to push faster than the GPU drains, the ring fills. At that point
the caller picks exactly one of three overflow policies:

- **Block** until the GPU catches up. Appropriate when the work will
  eventually complete and a brief stall is acceptable.
- **Drop with failure** so the caller can back off or coalesce.
  Appropriate when the caller has a better response than waiting.
- **Timeout-crash** if the GPU's watermark has not advanced in `M`
  frames. This is the signal that the GPU is not processing commands at
  all — crash with the full pending command log and last-known state.

Backpressure is a first-class feature, not an afterthought. It is the
mechanism that turns "the user asked for too much" into latency rather
than corruption, and the mechanism that distinguishes "slow" from
"broken." Without it, bursty CPU writes race a bounded GPU drain and
produce exactly the class of bug that triggered the rewrite.

Concretely, this pushes toward two reusable primitives:

- `UploadRing<T>` — a CPU-writable, GPU-readable ring with a runtime
  frame-count (passed at construction from the surface configuration),
  a bounded per-frame capacity, and a configurable overflow policy.
  Rotation is automatic per frame.
- `ReadbackChannel<T>` — a GPU-writable, CPU-readable ring with the same
  runtime frame-count, fence-based retirement, and a frame/command
  watermark so callers can line up retired snapshots with their
  pending-command history.

Every other abstraction in the rewrite builds on these two. There is no
third "shared buffer" primitive. Any case that seems to need one is
either a message (use a ring) or GPU-owned state (see principle 2).


### 2. No CPU-side mirrors of GPU state

The CPU submits a contract (*"allocate this"*, *"free that"*, *"build
this chunk"*) and trusts the GPU to fulfill it. The CPU does not maintain
parallel data structures that simulate GPU execution in order to predict
state.

What it rules out: the historical `ContiguousAllocator` on the CPU that
mirrored the GPU bump pointer, the CPU free list seeding pattern, and any
future attempt to "keep a copy of the page table on the CPU so we can
reason about it locally". All of these create divergence points.

The stronger form of this rule: **divergence between CPU and GPU state
is impossible by construction, not just unlikely.** There is no
CPU-side state that represents allocator internals, so there is nothing
to diverge *from*. The class of bug "CPU and GPU disagree about slot X"
cannot exist because the CPU has no opinion about slot X. What remains
possible is GPU-internal inconsistency — the allocator has a bug that
breaks its own invariants — and that class is handled by the shadow
ledger pattern in §6.

One clean operating rule: **if the GPU owns a data structure, the CPU
never writes to its contents after initialization.** The free list is
owned by the GPU. The bump pointer is owned by the GPU. The chunk
metadata table is owned by the GPU. The CPU submits commands through
upload rings and reads retired snapshots through readback channels —
nothing else.

"Initialization" is a distinct, auditable re-init path. A re-init tears
a GPU-owned structure down and rebuilds it wholesale — for example, on
world reset. It is not a mutation of live state. The CPU cannot reach
into a GPU-owned buffer to fix up a piece of it mid-frame.

**Messages about buffer shape travel on the normal channel.** When a
GPU-owned buffer grows (e.g. a multi-buffer capacity expansion), the
CPU describes the new shape in a message on an upload ring like any
other command. There is no separate "metadata buffer" concept that the
CPU writes directly. A capacity descriptor is a command that happens to
describe geometry rather than world state.

This has one coordination rule: the bind group swap that introduces the
new buffer and the capacity-grew message that tells shaders about it
must come from a single atomic operation. A `GpuBufferArray::push()`
method rebuilds the bind group and enqueues the message together;
callers cannot do one without the other. This prevents the "first frame
after growth reads OOB" class of bug by construction.


### 3. wgpu is contained behind render abstractions

Today `wgpu` types leak into nearly every file in the crate. `Buffer`,
`BindGroup`, `Queue`, `Device`, `CommandEncoder`, `ComputePassDescriptor`
are threaded through `world.rs`, `chunk_manager.rs`, `build.rs`,
`cull.rs`, and `gpu_alloc.rs`. There is no layer that says *"a GPU
resource with these semantics"* — every subsystem reaches directly for
`queue.write_buffer` or `encoder.copy_buffer_to_buffer` wherever it needs
them.

The rewrite introduces a core render layer that owns all contact with
wgpu. Higher layers work in terms of `UploadRing`, `ReadbackChannel`,
`ComputePass`, `Pipeline`, `BindingLayout` — abstractions that encode the
rules of principles 1 and 2 in their signatures. A caller that wants to
send data to the GPU cannot construct a single-copy buffer by accident,
because the only path to the GPU is through the ring primitives.

This is not an anti-wgpu position. wgpu is the right foundation. But a
renderer built on wgpu should not have wgpu leaking through every module
boundary — that is what makes the frames-in-flight discipline
unenforceable.


### 4. HLSL types are first-class

The current alloc shader is roughly 300 lines of `buf.Store(offset + N * 4,
value)` with field offsets hand-coded inline. Every call site must agree
on the layout. This has produced bugs (the quad range prefix sums, the
material sub-block mask encoding) and makes the shader hostile to
modification.

The rewrite requires that every GPU-visible type has a single definition
with accessor functions for field reads/writes. Call sites use
`ChunkMeta::load(buf, slot)` and `ChunkMeta::store(buf, slot, value)`,
not raw byte offsets. HLSL does not give us `#[repr(C)]` structs inside
`RWByteAddressBuffer`, but accessor helpers get most of the benefit for
a fraction of the layout-bug surface.

Type layout divergence is a different problem from runtime constant divergence
(see §5). A mismatched runtime constant fails silently — the shader uses
the wrong bounds and produces corruption or OOB reads with no obvious
indication. A mismatched type layout fails loudly — the shader reads the
wrong field and produces visibly wrong output or a GPU fault, both of which
are caught quickly during development. Behavioral verification is therefore
sufficient: if the shader produces correct output across its test workloads,
the layouts agree.

If a trivial codegen path exists — a build-script tool that emits the HLSL
accessor header from Rust type definitions, or vice versa — use it. The
benefit is real and the cost is low. If no trivial path exists, the
behavioral verification property makes manual maintenance acceptable; it is
not the same risk as hand-maintaining runtime constants.


### 5. Constants flow from one source

Shared constants desynced across files is the load-bearing evidence from
the rewrite-triggering session (`MAX_CHUNKS = 4096` in `bindings.hlsl`
next to `MAX_CHUNKS = 1024` in `world.rs`). It must not be possible by
construction in the next iteration. Two complementary rules handle the
two kinds of constant.

**Shared constants live in a GPU-uploaded struct.** Any value used by
both CPU and shader code — buffer capacities, slot counts, layout
bounds, thresholds — lives in a single `GpuConsts` uniform buffer bound
at a fixed location across every pipeline. There is exactly one `u32`
per value, written by the CPU at startup (and on explicit updates like
multi-buffer growth), read by shaders via the uniform. Nothing is
compile-time on the shader side because nothing needs to be: bounds
checks, capacity reads, and threshold comparisons all work against a
loaded value with no meaningful perf difference.

This enforces the rule harder than codegen would. Codegen relies on the
convention that nobody writes a stray `#define`; runtime statics make
the stray `#define` impossible because there is nothing to define — the
shader must load from the uniform.

**Shader-local compile-time constants live in the shader.** A small
subset of values genuinely has to be literal in the shader source:
workgroup sizes (`[numthreads(...)]`), `groupshared` array dimensions,
and anywhere the compiler needs a literal for correctness. These are
defined once in the shader file that uses them. They are shader-local,
so the cross-file drift problem does not apply.

Where CPU dispatch code needs to know a workgroup size (to compute
`ceil(n / wg_size)`), it does not carry its own copy. Instead, the
pipeline-creation path reflects the shader's workgroup size out of the
SPIR-V and asserts it matches the expected value. Any drift between the
shader literal and the CPU expectation fires at pipeline load, not at
dispatch time.

What this rules out: any constant shared across files being defined in
more than one place, whether by `#define`, hand-maintained Rust `const`,
or generated header. The `GpuConsts` uniform is the only shared-constant
channel; the pipeline-creation reflection check is the only compile-time
verification channel.


### 6. Validation via shadow ledgers

Principles 1 and 2 eliminate CPU↔GPU divergence bugs by construction.
They do not, on their own, detect GPU-internal logic bugs — the case
where the allocator has a bug that breaks its own invariants and the
CPU has no way to know. Without a detection mechanism the only tool
left for diagnosing a broken GPU is renderdoc, which is the heavy
hammer the current codebase depends on today.

The shadow ledger pattern gives subsystems a detection mechanism without
reintroducing CPU-side state that could diverge. The CPU keeps a
**ledger** of commands it has sent. The GPU publishes **aggregate state
snapshots** via a readback channel. On retirement, the subsystem asks
*"given the commands I know were applied by frame k, does the snapshot
match the invariants I expect?"* On mismatch, crash with both the
expected and observed state printed.

The load-bearing distinction from the old CPU mirror: **the ledger is a
witness, not a source of truth.** Nothing upstream ever consults the
ledger to make a decision. It exists solely to validate the GPU's
observable behavior. The moment the ledger starts influencing what the
CPU asks for next, it has become a mirror and principle 2 is violated.

Ledgers are per-subsystem because the invariants they enforce are
subsystem-specific. `AllocLedger` knows what an allocation request does
to the bump pointer and free count. `ChunkLedger` knows what a load
command does to the metadata table. There is no generic primitive for
this — it is a pattern built on the two real primitives (`UploadRing`
and `ReadbackChannel`).

**Three tiers of validation, with a hard ceiling on GPU cost.**

- **Tier 1 — command watermark (always on).** Every subsystem that
  pushes commands through an `UploadRing` maintains a single `u32` on
  the GPU side: "highest command index processed." The shader bumps
  it after processing each command. It is written to the readback ring
  once per frame. The CPU uses the watermark to know which pending
  commands have retired, and therefore which can be dropped from the
  ledger. Cost: one atomic add plus one store per subsystem per frame.

- **Tier 2 — aggregate invariant snapshot (opt-in per subsystem).** A
  subsystem with nontrivial state dumps a small struct of scalar
  aggregates — bump pointer, free count, running totals, overflow
  flags — to the readback ring once per frame. These are values the
  shader already holds in registers; the snapshot is a memcpy, not a
  computation. The CPU checks aggregate invariants on retirement (e.g.
  `allocs - frees == live_count`, `free_count <= FREE_LIST_MAX`).
  Cost: one tiny "publish snapshot" kernel at end of frame, writing a
  handful of `u32`s.

- **Tier 3 — full state dump (debug-only, manual trigger).** A
  copy-to-readback of the full metadata table, triggered by a flag or
  debug key. Not in the steady-state loop. Never runs in release
  builds without an explicit opt-in. Used offline to diff against the
  ledger when tier-2 invariants fire and per-entity detail is needed.

**Snapshots are scalar aggregates, not state copies.** This is the rule
that keeps GPU cost bounded forever. A subsystem's steady-state snapshot
contains running totals, bounds, and counters — values the shader
already maintains. It does not contain copies of per-entity state. The
moment per-entity state enters the steady-state snapshot, the shader
cost stops being bounded and the CPU starts reasoning about GPU
internals, which is exactly the failure mode principle 2 prevents. If a
validation requires per-entity comparison, it goes in tier 3, not tier
1 or 2.

The total steady-state GPU cost of the ledger pattern across the whole
crate is on the order of 20-30 `u32` stores per frame. It is below the
noise floor of any measurement. In exchange, every bug class on the
rewrite-triggering list becomes detectable:

- Free list OOB → `free_count > FREE_LIST_MAX` on snapshot.
- Mass unload leak → `allocs - frees` diverges from `bump - overflow`.
- Alloc overflow silent → `overflow_count > 0` with chunk context from
  ledger replay.
- Bump retraction leak → aggregate `live_count` drifts from expected.
- Stale build feedback for unloaded chunks → command watermark shows
  the build completed after the chunk's unload command, CPU drops the
  feedback silently instead of leaking its range.


## Scope for the first rewrite pass

The rewrite is large. Attempting it in one step is not realistic. The
minimum viable first pass is the core primitives layer — everything
that the existing rendering logic will eventually be ported onto.

In scope for the first pass:

- `UploadRing<T, N>` and `ReadbackChannel<T, N>` primitives with bounded
  capacity, configurable overflow policy (block / drop / timeout-crash),
  and frame/command watermarks on retirement.
- `Pipeline` and `BindingLayout` abstractions that own wgpu descriptor
  construction.
- `GpuConsts` uniform buffer machinery — shared constants uploaded from
  Rust, bound at a fixed location across every pipeline.
- Pipeline-creation workgroup-size reflection and assertion for
  shader-local compile-time constants.
- HLSL accessor helpers (`include/types.hlsl`) for the shared struct
  set, with a single authoring location per type.
- A standalone validation binary that exercises the primitives under
  realistic conditions: ring backpressure (submit faster than drain,
  verify the overflow policy fires), a two-subsystem shadow ledger
  pattern end-to-end (command watermark + aggregate snapshot + an
  intentional invariant violation that crashes cleanly), and a
  multi-frame roundtrip that validates frames-in-flight correctness
  (distinct sentinel per frame, verified in order on retirement). This
  binary runs against real GPU hardware; software emulation is not a
  reliable substitute.

Out of scope for the first pass (but must remain *achievable* on top of
the primitives):

- Porting the full render pipeline onto the new primitives.
- Redesigning the allocator — see *What this does not attempt to fix*.
- The cull / MDI path.
- Material packing.

The goal of the first pass is not to replace any functionality. It is
to establish a layer where the principles above are enforced by
construction, so that every subsequent port lands in a place that
cannot repeat the class of bug that triggered the rewrite.


## What this does not attempt to fix

The principles above eliminate the CPU↔GPU divergence class of bug and
make GPU-internal logic bugs detectable. They do *not* fix throughput
limits of the current allocator design. Two throughput issues and a
fragmentation issue are explicitly out of scope for this rewrite and
belong to follow-up passes whose shapes are informed by, but not
scoped into, this work.


### Allocator throughput under bursty mass frees

The current bump + free list allocator has a structural throughput
ceiling: every free pushes to a single bounded data structure, and
bursty frees (e.g. a view-distance change from 16 to 1, freeing ~17,000
chunks at once) exceed the steady-state drain rate regardless of how
large the free list is made. The rewrite's shadow ledger catches this
as a clean `free_count > FREE_LIST_MAX` crash with full diagnostic
context instead of silent 93% buffer corruption — a large improvement
over current behavior — but "cleaner crash" is not "works."

**The primitives pass does not pick an allocator.** The allocator redesign
is a separate pass that happens after the primitives are stable, so it
can be built on top of them rather than around them. The design should
be chosen based on the actual access patterns and burst characteristics
observed once the ledger gives clean diagnostic data.


### Phase 1 scan throughput

The unbounded per-dispatch scan work in the chunk cleanup path is a
separate throughput issue. The frames-in-flight discipline and ring
backpressure make it less acute — work that cannot keep up with the
frame rate naturally stalls the ring rather than silently corrupting
state — but the scan algorithm itself is a separate concern. Follow-up
pass, likely bundled with the allocator redesign since both touch the
same region of the pipeline.


### Allocator fragmentation

The tiny-entry fragmentation observed in the final capture (size 1-5
entries from first-fit shrink-on-alloc) is orthogonal to everything
above. It belongs to the same allocator redesign pass. Candidate
strategies (coalescing, size classes, in-place reuse) are all
compatible with the tombstone direction and with the rewrite
primitives, but none is picked here.
