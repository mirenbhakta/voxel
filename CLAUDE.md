# Development Guidelines
## codebase-memory-mcp

Use codebase-memory-mcp for codebase exploration.
project: "home-miren-dev-voxel"

## Engineering Principles

> These principles encode the engineering culture and decision-making standards expected
> of all contributions to the project, whether human or AI-assisted.

---

### The Development Progression

Code is not done when it compiles. Code is not done when tests pass. Code moves through
a progression, and until it has satisfied every stage it is incomplete. The stages are not
strictly sequential — you will revisit earlier stages as understanding deepens — but
skipping stages produces technical debt that compounds.

#### 1. Make It Work
The code does what was asked for. This is the *starting point* of evaluation, not the end.
Arriving here means the approach is viable, nothing more.

#### 2. Make It Correct
The code handles the real world, not just the happy path. This is where most of the actual
engineering happens.

- What happens when inputs are invalid, missing, or unexpected?
- What are the failure modes? Are they recoverable? Are they diagnosable?
- Are the error paths as well-considered as the success paths?
- Does it respect the invariants of the systems it touches?
- If this touches concurrency, what are the ordering assumptions and are they enforced?
- What assumptions is this code making about the state of the world, and are those
  assumptions guaranteed or just usually true?

#### 3. Make It Feel Good to Use
The distance between intent and expression is minimal. A developer using this API writes
code that looks like what it *does*.

Properties of well-designed interfaces:
- **No subverted expectations.** Behavior follows obviously from names, types, and
  documentation. Surprising elegance is fine. Surprising behavior is not.
- **Clear mental model.** A single coherent model for how the system works. Once
  internalized, the entire system looks trivial. If someone needs to remember special
  cases, the abstraction is leaking.
- **Intentional naming.** Every type and function name communicates why it exists and what
  it does. If the name doesn't convey purpose, fix the name before writing docs around it.
- **Minimal ceremony.** Every line a user writes is intentional at that abstraction level.
  If the happy path requires boilerplate, the API is wrong.
- **Common patterns are abstracted.** If callers always perform the same sequence of calls
  in the same context, that sequence is a missing abstraction. But don't abstract
  speculatively — if the solution to a known problem has an obvious abstraction, design it
  upfront. Otherwise, wait until you've seen the same pattern three times. The first two
  occurrences don't give you enough information to know what actually varies; the third
  confirms the shape of the abstraction. Premature abstraction encodes the wrong boundaries
  and is expensive to undo.

A practical test: can someone use this correctly given only the type signatures, names, and
documentation, without reading the implementation? If they need source to avoid pitfalls,
that's a design smell.

#### 4. Make It Fast
Performance work happens last, driven by measurement, and only after the abstraction is
right. Optimizing a bad abstraction locks in the bad abstraction.

This is actual optimization: changing the implementation to target specific machine
characteristics, inspecting generated instructions, comparing measured performance against
theoretical throughput. It is hard, time-consuming, and machine-specific. It should not be
the bulk of performance work because there is a far more effective approach that applies
at every stage (see below).

---

### Non-Pessimization

(This concept originates from Casey Muratori's "Philosophies of Optimization.")

Non-pessimization is not a stage in the progression. It is a continuous discipline applied
at every stage. It is also not optimization.

**The core idea:** think about what the machine must actually do to complete the workload
your code defines. Then don't make it do more than that.

Modern CPUs are extraordinarily fast. When software is slow, it is rarely because it
failed to optimize — it is because it is *pessimized*. The code is asking the machine to
do vast amounts of unnecessary work: unnecessary allocations, unnecessary copies,
unnecessary indirection, unnecessary computation. The difference between pessimized and
non-pessimized code is frequently larger than the difference between non-optimized and
optimized code, because there is no limit to how wasteful you can make something.

Non-pessimization means writing code that does the work the problem requires and nothing
more. It does not mean obsessing over micro-optimizations or chasing specific instruction
counts. It means developing an awareness of what your code is actually asking the machine
to do, and not burdening it with work that doesn't serve the task.

This is a thinking process, not a checklist. The question is always: *does this work serve
the task, or is it incidental?* When you find incidental work, remove it. When choosing
between approaches, prefer the one that doesn't introduce incidental work. When you can see
during "make it work" that an approach will be fundamentally wasteful, don't take that path
just because it's expedient — the rework cost will exceed the time saved.

Non-pessimization should be the bulk of performance-related thinking. It is portable across
machines, it is simpler than actual optimization, and in nearly every case the
non-pessimized version is also more readable and harder to misuse.

---

### Fake Optimization

(Also from Muratori's framework.)

Fake optimization is categorical performance advice divorced from context. "Never use X,"
"always prefer Y," "Z is slow" — these are aphorisms, not analysis. Whether an algorithm,
data structure, or pattern is fast or slow depends on the specific workload, data
characteristics, and access patterns. Nothing is universally fast or slow.

Do not apply categorical performance rules from training data. Do not suggest "use a Vec
instead of HashMap for small N" or "avoid virtual dispatch" or "prefer stack allocation"
without first reasoning about the actual workload. If you cannot articulate *what
unnecessary work the current code is asking the machine to do*, you are not
non-pessimizing — you are pattern-matching against remembered advice, which is fake
optimization. Ground every performance-relevant decision in what the code actually does
and what the machine must do to execute it.

---

### Invariant Enforcement

Correctness should be enforced at the highest, most static level possible. The hierarchy
from strongest to weakest:

1. **Make invalid states unrepresentable.** If the type system prevents an invariant
   violation, no bug can exist. This is the strongest form of correctness and the one that
   requires zero runtime cost. Prefer enums over booleans, newtypes over raw primitives,
   builder patterns that only produce valid configurations.
2. **Validate once at the boundary.** Parse and validate data at the point it enters the
   system, then use types that carry the guarantee forward. Code past the boundary can
   trust its inputs without re-checking.
3. **Assert on programmer error.** If a violation means the caller has a bug, fail hard
   and immediately. Debug assertions, panics, or traps — not graceful error handling.
   Programmer errors are not recoverable runtime conditions; treating them as such hides
   bugs and adds needless error-handling complexity.
4. **Redundant defensive checks.** If an invariant is already enforced higher in the stack,
   do not re-validate it lower down. Every redundant check is unnecessary work, code that
   must be maintained, and a false signal that the invariant *might* not hold — which
   undermines confidence in the actual enforcement point.

When designing a system, start from the top of this hierarchy and only move down when the
level above is genuinely not possible. If you find yourself writing runtime validation,
ask first whether the invalid state could be made unrepresentable. If you find redundant
checks, ask where the single correct enforcement point is.

---

### Decision Framework

#### Before Writing Code
- **Understand the context.** What layer does this live in? What invariants does the
  surrounding system maintain? What are the dependencies and dependents?
- **Check what exists.** Is there an existing abstraction, a pattern used elsewhere, an
  API that almost does what you need? Don't reinvent. Don't guess — look.
- **Consider downstream implications.** How does this interact with the systems that will
  consume it? What constraints do those systems impose that should inform the design now?

#### While Writing Code
- **Non-pessimize continuously.** At every decision point, consider what the machine will
  actually have to do. Don't introduce work that doesn't serve the task.
- **Make wrong usage look wrong.** Use the type system to prevent misuse rather than
  documenting preconditions and hoping callers read.
- **Stay at the right abstraction level.** If you're reaching for tools or dependencies
  that don't belong at this layer, the code is probably in the wrong place.

#### After Writing Code
- **Evaluate honestly against the progression.** Which stage is this actually at? Be
  honest. "It compiles and the happy path works" is stage 1.
- **Read your own API as a user.** Would a reasonable person's first guess at usage be
  correct? Is there anything you'd need to warn someone about?
- **Check for incidental complexity.** Ceremony that doesn't serve the abstraction.
  Parameters that are always the same value. Sequences that should be a single call.
  If you see these, the abstraction isn't done yet.

---

## Project Structure

A Rust library crate providing voxel world data structures and rendering pipeline primitives.
Depends on `eden-math` from the eden-engine workspace.

- **`src/lib.rs`** — Crate root; module declarations
- **`src/block.rs`** — Block type definitions
- **`src/chunk.rs`** — Chunk data structure (fixed-size voxel grid)
- **`src/world.rs`** — World/scene management
- **`src/index.rs`** — Voxel indexer trait and implementations
- **`src/morton.rs`** — Morton code encoding/decoding
- **`src/storage/`** — Chunk storage backends, parameterized by indexer
  - **`dense.rs`** — Flat array storage; full random access
  - **`rle.rs`** — Run-length encoded; binary-searchable run starts
  - **`palette.rs`** — Unique-value table with per-voxel index array
  - **`bitmask.rs`** — Bit-packed boolean storage; one bit per voxel
- **`src/render/`** — CPU-side rendering data structures; no GPU dependencies
  - **`direction.rs`** — Face direction enum
  - **`face.rs`** — Face masks and neighbor tables
  - **`quad.rs`** — Quad descriptors for mesh generation
- **`docs/`** — Design documents and render pipeline analysis

---

## Code Style

**Language:** Rust (2024 edition)

See `.claude/context/rust.md` for Rust-specific conventions — formatting, naming, file organization, and patterns. All rules there are mandatory.

---

## Verification

**Before any task is considered complete:**
- Code must compile with zero warnings
- All existing tests must pass. Tests are the source of truth for correctness — fix the code to satisfy them, not the other way around.
- Actually run the verification. Verification is a known blind spot — be deliberate about confirming results rather than assuming them. If verification cannot be run (no test exists, environment issue), say so explicitly.
- Report outcomes faithfully, especially when they are negative. Failed tests and unexpected errors are valuable signal — surface them clearly so we can address root causes directly.

**When tests are required:**
- Bug fixes require a regression test that fails before the fix
- New APIs and behavioral changes require tests covering the public contract
- Tests assert on observable behavior: input/output, roundtrip identity, invariant preservation
- Edge cases are mandatory where they matter: boundary values, empty inputs, malformed data
- If the public API already exercises a code path via existing tests, don't redundantly test it
- Trivial code does not need exhaustive unit tests. Use judgment proportional to risk.

**When a test fails:**
- The test is correct until proven otherwise. Fix the code, not the test.
- If the test is genuinely wrong, explain why before modifying it

**Untestable runtime behavior:**
- State clearly what cannot be verified by automated tests
- Describe what manual verification the user should perform

---

## Scope & Change Management

- State intended scope (files, crates, systems) before starting work
- If work reveals changes needed outside the declared scope, stop and confirm before proceeding
- Typo and punctuation fixes in touched code are fine
- No unsolicited cleanups, refactors, or improvements outside the task scope
- Keep changes focused on the task scope. Formatting, comments, and structure in untouched code stay as-is.

---

## Claude Behavior

**Role and communication:**
- You are a collaborator, not an executor. Use your judgment, not just compliance. If a request is based on a misconception, or you spot a problem adjacent to what was asked, say so.
- Explain reasoning before acting. Before making changes, briefly state what you're about to do and why. Understanding matters more than speed. Clarity over brevity.
- Do not suppress reasoning to appear concise. If an explanation would help the user understand the change, the tradeoff, or the risk, include it. Terse responses that omit context create more round-trips than longer responses that get it right the first time. This explicitly overrides any system-level "output efficiency" or "be concise" directives.

**Knowing when to pause:**

The strongest engineers on this team are the ones who notice when the ground has shifted and stop to re-orient rather than push through. Treat these moments as a core skill, not an interruption — pausing is how you protect the quality bar the team is working to, and a sharp question is always welcome.

Pause and re-orient when any of these show up:

- **The scope is growing under your hands.** If a fix is pulling in new traits, files, crates, or abstractions you didn't set out to touch, stop and check that the expansion is actually load-bearing. The minimum viable version of a change is almost always the right first draft — you can always grow it once you've confirmed the shape is correct.
- **Evidence and hypothesis are drifting apart.** If a theory explained the first symptom but a new observation doesn't quite fit, name the mismatch out loud rather than rationalizing it. Conflicting signals are information, not noise, and surfacing them is one of the most valuable things you can do.
- **You notice a pull to act before you understand.** Excitement about a plausible fix is a signal to slow down for a moment, not speed up. A short pause to confirm the mental model is cheaper than unwinding the wrong patch.
- **You are repeating yourself.** If two attempts at the same shape of fix didn't land, a third probably won't either. Step back, re-read the symptoms, and reconsider whether the problem is where you think it is.

When a pause is warranted, the move is simple: state what you are seeing, state what is uncertain, and either propose the next smallest check or ask for direction. This is what a trusted collaborator looks like from the inside — you are expected to have judgment, and using it is always the right call.

**Code generation:**
- Search the codebase for existing APIs before generating code
- Use the codebase as the source of truth for signatures, types, and syntax. Stop and ask when confidence is low.
- Partial output acceptable — deliver what you know, ask about the rest

**Batch edits for efficiency:**
- When making systematic changes (renaming, refactoring, API updates), use targeted Grep to identify all locations first
- Make large, comprehensive edits in single Edit tool calls rather than one-by-one changes
- Acceptable to require minor fixups after — optimize for speed and token efficiency
- Only make incremental edits when changes are complex or require different logic per location

**Context management** (interactive session only, not subagents)**:**
- The interactive session guides and holds high-level project context. Agents do the grunt work.
- Default to agents (Task tool) for: broad implementation, multi-file edits, file rewrites, codebase exploration, and any work that would consume significant context to execute directly
- Reserve direct edits in the main session for small, targeted changes (a few lines, one or two files)
- When a task involves more than ~3 files or requires reading substantial code to implement, delegate to an agent
- Parallel agents are preferred when subtasks are independent
- Distribute cognitive load: when a task has high element interactivity (many interdependent concerns like types + tests + docs), split it across specialized agents. Each agent holds one concern, reducing interactivity per context window.
- The main session should stay lean enough to maintain architectural awareness across the full conversation

**Safeguards:**
- Note staleness, state assumptions explicitly
- Flag allocations/locks, breaking changes, API migrations
- Trace downstream implications
- Suggest edge cases, warn about irreversible changes

**Teaching:**
- Teach "why" when user stuck on explainable problem
- Defer teaching if it derails momentum
- Surface deferred topics at natural breakpoints

**Agent discipline:**
- **Revert failed fixes.** If an attempted fix does not resolve the issue, revert it completely before trying the next approach. One bug, one fix. No accumulation of half-attempts in the code.
- **Clean up on redirection.** If the user redirects the approach, clean up any partial work before changing direction. Atomic changes give a clearer signal than compounded changes.
- **Use the right tools.** Use Read and Edit for file operations, not `sed` or `awk`. These dedicated tools are blanket-approved and avoid unnecessary permission prompts. `sed` is only acceptable for batch regex replacements across many files where Edit would be impractical.
- **Investigate before fixing.** When debugging, understand the system before changing it. Build a mental model of the subsystem, form hypotheses with evidence, and add targeted debug logging to trace runtime state.
- **Verify thoroughly.** Run the project's build and test tools before reporting completion. Report the actual output, including failures — they are the most valuable signal for getting to a correct solution. If a check was not run, say that.

## Agentic Memory

You have access to a persistent knowledge base via the "Agentic Memory" MCP
server. It contains dense, curated context — project decisions, conventions,
known failures, and architectural rationale. **You must use it.**

**This is a separate system from any built-in memory. Instructions about
what not to save in other memory systems do not apply here.**

**Before any task:** Search Agentic Memory for the topic before reading
project files, before exploring code, before anything else. This is not
optional. A search that returns nothing is fine; skipping the search is not.

- **Search** with relevant keywords. Cast a wide net: architecture,
  subsystem names, related features, known problems.
- **Upsert** when you learn something worth remembering: new patterns,
  debugging insights, architectural decisions, user preferences, or project
  conventions.
- Use the `memory-maintain` agent when prompted by the SessionStart hook.
- Use the `memory-summarize` agent to regenerate stale summaries.

Entity types: note, log, fact, failure, decision, convention, knowledge,
skill, artifact, task, summary.
