# Documentation Style Guide

Full documentation style guide. Distilled from pre-AI codebase documentation
and extended with additional conventions for consistency going forward.

---

## Voice and Tone

Direct, technical, declarative. The documentation reads like a specification written
for experienced systems programmers. No hedging, no filler, no enthusiasm. Every
sentence earns its place by adding information the code alone does not convey.

Pragmatic asides are acceptable when they explain a trade-off:

```rust
// I'm sure there's a better way to do this, but it's not a hot path (once per
// device created) and this is the easiest to write and read. Plus compilers are
// smart and this is pretty dumb code.
```

---

## Doc Comments

### Coverage

Every public item is documented: structs, fields, enum variants, methods, traits,
constants, type aliases. No exceptions.

Private items are documented when their purpose is non-obvious, but trivial internal
helpers can go without.

### Opening Phrases

**Functions and methods** use the imperative mood:

```rust
/// Create a new instance of render hal.
/// Returns the blackboard associated with this flow.
/// Clear the entire graph, invalidating all node handles.
/// Poll events and call `on_event` for each event.
/// Spawn a new job and return an awaitable handle for it.
/// Check whether a physical device supports a surface.
/// Set the language of the unicode buffer if provided.
```

Common verb patterns:

| Action      | Verb         |
|-------------|--------------|
| Constructor | "Create..."  |
| Accessor    | "Returns..." |
| Predicate   | "Returns whether..." or "Check whether..." |
| Mutation    | "Set...", "Clear...", "Remove...", "Put..." |
| Lifecycle   | "Destroy...", "Initialize..." |

**Types and fields** use the declarative form ("A...", "The...", "An..."):

```rust
/// A data flow graph representing a data transformation process.
/// An allocator for entities.
/// The shared blackboard of the data flow.
/// A unique id identifying a device.
```

**Enum variants** use brief descriptive phrases. Event variants use "Emitted when...":

```rust
/// Emitted when the application is paused.
Suspended,
/// Emitted when a device event is received.
DeviceEvent { ... },
/// The linear easing function, equivalent to a lerp.
Linear,
/// Load the resource when entering the render pass.
Load,
```

### Structure

A doc comment follows a graduated structure. Most items only need the first line.
Each subsequent element is added only when it provides information the reader needs.

1. **Summary sentence.** What it is or what it does. Always present, always a
   complete sentence.
2. **Extended description.** Constraints, usage context, or why you would use
   this over alternatives. Separated by a blank doc line.
3. **Sections.** Only when needed. Standard sections are described below.

```rust
/// Create a default device using a built-in selection heuristic.
///
/// The selected device will have one or more queues capable of the following:
///  - graphics
///  - compute
///  - transfer
pub fn create_default_device(...) -> ...
```

### Sections

Use only when the section adds information that does not fit in the summary or
extended description.

**`# Safety`** -- On all `unsafe` functions. Lists the preconditions the caller
must satisfy, not how the implementation uses the unsafety.

```rust
/// Create a surface for `window`.
///
/// # Safety
///  - `window` must be a valid window handle.
pub unsafe fn create_surface(&self, window: RawWindowHandle) -> Surface
```

**`# Panics`** -- When a function can panic. Uses bullet points for multiple
conditions, inline text for a single condition.

```rust
/// Get the value at `key` as an immutable reference.
///
/// # Panics
/// - If the key does not exist in the set.
pub fn get(&self, key: Key<Index>) -> &T
```

**`# Remarks`** -- Design rationale, performance caveats, or usage guidance that
does not fit in the summary.

```rust
/// Trim and compact geometry buffers to reduce memory consumption.
///
/// # Remarks
/// This is a potentially expensive operation and should only be done after
/// significant changes were made such as unloading a level.
pub fn compact(&mut self)
```

```rust
/// # Remarks
/// `block_on` should be used sparingly in jobs. Currently there is a possibility
/// of exceeding a worker thread's stack capacity if enough
/// `async -> block -> async -> block` chains are encountered during execution.
/// In practice this should be rare and job call stacks should be relatively short.
///
/// This limitation will be fixed in a future version of Eden.
pub fn block_on<F, T>(f: F) -> T
```

**`# Arguments`** -- When parameter semantics are not obvious from names and types.
Uses `* name - description` format.

```rust
/// # Arguments
/// * `is_valid`    - Checks if the accessor is still valid.
/// * `resolve`     - Resolves a property binding and produces an accessor.
pub fn try_resolve(...)
```

**`# Per-Platform`** -- Platform-specific behavior notes.

**`# Background`** -- Educational context for complex or unfamiliar patterns. Used
sparingly, primarily in module-level docs.

**`# Usage` / `# Examples`** -- Working code examples. Used at module level for
complex subsystems. Rare at the method level.

### Field Documentation

Fields receive terse descriptions of their role, consistently using "The X" pattern.
Additional paragraphs only when semantics need clarification.

```rust
pub struct RenderHal {
    /// The currently loaded backend.
    backend         : VirtualBackend,
    /// The table of active devices.
    virtual_devices : Arc<RwLock<VirtualDeviceList>>,
    /// The mask of all assigned devices.
    all_devices     : AtomicU16,
    /// The handle allocator for the render system.
    handle_allocator: Arc<HandleAllocator>,
}
```

When a field has non-obvious semantics, an additional paragraph explains them:

```rust
/// The set of archetypes in the world.
///
/// Archetypes are always added and never removed, however their memory may be
/// deallocated to reduce overall memory consumption.
archetypes : Vec<Archetype<Storage>>,
```

```rust
/// Change tracking for each component type by subslice in the archetype.
///
/// Changes are split into ranges of `0..types` for each subslice in order.
/// A component change for subslice can be directly indexed via the following
/// `subslice_size * component_count + component_index`.
changes : ThinVec<Change>,
```

Physical quantities include units and coordinate system descriptions:

```rust
/// The orientation of a pointer relative to the digitizer surface in degrees.
///
/// `0` indicates the pointer is aligned with the X axis.
pub orientation : f32,
```

### Module-Level Documentation

Brief for simple modules -- a single `//!` line:

```rust
//! Deferred commands for mutating a world.
//! Object storage and persistence.
//! A data blackboard.
```

Expanded for complex subsystems -- overview, concept introductions, and code
examples:

```rust
//! # The EdenFX Shader Language
//! Eden defines shaders using a metalanguage built on top of HLSL called EFX.
//! EFX provides compile time composition of shader code and a complete definition
//! of the pipeline state used by a shader, enabling full ahead-of-time compilation
//! for shader pipelines.
//!
//! ## Overview
//! EFX brings several new constructs for authoring shaders:
//!  - scopes: enable precise resource management during rendering
//!  - components: a reusable unit of shader code...
```

```rust
//! # Eden:Tween
//!
//! Generic tweening, timelines, and property binding for code driven animations.
//!
//! - [`Tween`] provides simple value tweening but cannot be driven by a
//!   [`TweenEngine`]
//! - [`PropertyTween`] is the same as [`Tween`] but supports property binding
//!   and can be driven by a [`TweenEngine`].
```

### What Doc Comments Do NOT Contain

- **Implementation details.** How the function works belongs in body comments.
  Doc comments describe the contract: what it does, what it requires, what it
  returns.
- **Internal field or type names.** Doc comments should be readable without
  knowledge of the private implementation.
- **Construction sites.** Don't say "Constructed by X" or "Built inside Y."
- **Migration notes.** Don't say "This was refactored from..." or use "now" to
  imply a before/after.
- **Lifetime variance rationales.** Borrow checker implementation concerns belong
  in inline comments at the relevant impl site.

### Cross-References

Backticks for inline code, parameters, and types within the same crate. Brackets
for rustdoc cross-references:

```rust
/// Insert a new entry into the blackboard under `key`, wrapping it in a
/// [`Variable`] and replacing any existing value.
```

### Derived Code Attribution

When code is derived from an external source, include a reference link:

```rust
/// Derived from the work of Jerome Froelich.
/// Repository  : <https://github.com/jeromefroe/lru-rs>
```

---

## Inline Body Comments

Body comments are the primary vehicle for communicating intent within function
implementations. The codebase uses a quality-over-quantity approach: comments exist
where the code alone does not convey enough context, and are absent where it does.

### Outline Comments

Non-trivial function bodies use plain-English comments that describe what the next
code block accomplishes. These act as a readable outline of the function's logic,
letting a reader skim the function without parsing every expression.

Not every block gets one. Simple, self-documenting code is left uncommented.
Comments appear at decision points, state transitions, and algorithm steps.

```rust
// Fetch the unicode buffer to place the text into for harfbuzz to shape.
let mut unicode = self.unicode.take()
    .unwrap()
    .add_str_item(run.text, &run.text[run.start..run.end])
    .guess_segment_properties();

// Assign cluster values for each codepoint in the buffer. We use these to map
// between glyphs and codepoints later. Each codepoint is assigned its index
// in the source string.
{
    let mut glyph_infos = unicode.get_glyph_info_mut();
    for (i, info) in glyph_infos.iter_mut().enumerate() {
        info.cluster = (run.start + i) as u32;
    }
}

// Set the language of the unicode buffer if provided.
unicode = unicode.set_language(
    self.language.unwrap_or(self.default_language)
);
```

```rust
// Continue any work that resumed if we're permitted to do so. We want to
// interleave resumed work with normal work so it completes as soon as
// possible, but we can't simply process all resumed tasks as they may
// reschedule themselves often.
if ctx.can_resume_work() {
    if let Some(work) = ctx.thread_state.pop_resumed() {
        ctx.resumed_work();
        return Some(work);
    }
}

// Otherwise pull work from our local queue.
if let Some(work) = ctx.local_queue.pop() {
    ctx.started_new_work();
    return Some(work);
}

// Otherwise if all our local tasks are finished attempt to acquire a new
// batch of work from the global queue.
let global = ctx.shared_state.global();
```

The pattern: a comment states the intent or rationale, then the code that
implements it follows immediately below. The comment is at the same indentation
level as the code it describes.

### Safety Comments

Every `unsafe` block has a preceding comment explaining why the invariants are
satisfied. The comment uses `// Safety:` as a prefix.

```rust
// Safety: If the layout keys match then the bundle is compatible and we can
//         bypass the more expensive runtime check.
unsafe {
    self.storage.push_dynamic_unchecked(bundle)
}
```

```rust
// Safety: Interned strings are always leaked and never freed, or live in
//         static memory.
unsafe {
    let len = (*self.0).len;
    ...
}
```

```rust
// Safety: Tuples are known to always be compatible, so we can skip the
//         check entirely.
unsafe { self.push_dynamic_unchecked::<Tuple>(element) }
```

Safety comments explain the specific invariant being relied upon, not generic
statements like "this is safe because we checked."

### Algorithm and Trade-off Explanations

When the code makes a non-obvious choice, the comment explains the reasoning.
These comments tend to be longer and include the "why" behind the decision.

```rust
// For anyone wondering why we use a 64 bit integer rather than a 32 bit one.
// We want time to remain accurate for extended periods of time even if there's
// an animation or tween playing for very long periods of over a month.
//
// Specifically this was chosen to deal with the server use case which could
// potentially be incrementing time for over a month, which is not an uncommon
// scenario. If we used a 32 bit value with millisecond precision we would
// overflow in just under 25 days.
```

```rust
// NOTE: We opted to use a BTree for manifest mapping to support the following
//       constraints on the asset system.
//        - Eventually UUIDv7 will be used for object ids. Sequential ids allow
//          for much better compression on disk and will allow for faster lookups.
//        - Objects created together typically are commonly accessed together,
//          when paired with sequential ids this will allow for better access times.
```

### Ordering and Sequencing

When operations must happen in a specific order for non-obvious reasons, comments
explain the dependency.

```rust
// NOTE: Remaining init should be performed after this log. This is the first
// point where the engine is initialized enough to report any errors that may
// occur in detail.
info!(target: "eden", "booting eden...");
```

### Encoding and Bit Manipulation

Case branches in encoding/decoding logic get brief comments identifying each case:

```rust
// Byte is null byte, encode.
if byte == 0 {
    encoded.push(0xc8);
    encoded.push(0x80);
    i += 1;
}
// Byte is ascii, no encoding required
else if byte < 128 {
    encoded.push(byte);
    i += 1;
}
// Need to encode the character.
else {
    let width = utf8_char_width(byte);
    ...
}
```

Low-level bit operations (shifts, masks) are generally left uncommented when the
reader is expected to understand the domain (UTF-8 encoding, graphics state flags).
Constants are documented at definition, not at use.

### NOTE/TODO/FIXME

**`NOTE`** marks design decisions, ordering constraints, or important context:

```rust
// NOTE: `t` is not clamped for performance and extrapolation reasons.
```

```rust
// NOTE: A some words are reserved for other external systems to minimize
// confusion as Eden grows. These words must not be used for types inside the
// tweening library.
// - Animation/Animator (reserved for skeletal animation)
// - Sequence/Sequencer (reserved for cinematic sequencing)
```

**`TODO`** marks near-term work. **`TODO (future)`** marks longer-term improvements.
Both include rationale:

```rust
// TODO: So this "works", but has some issues.
//
// 1. If someone saturates the job system with blocking work we can blow our
//    already limited stack.
// 2. Throughput is hurt by how much blocking we're doing.
//
// Integrating a solution like https://github.com/edubart/minicoro to wrap up
// the entire call stack of a job in a coroutine is probably the way to go.
```

```rust
// TODO (future): Upgrading to UUIDv7 would help significantly with lookup times.
```

Bare `TODO`s without rationale are acceptable for short items, typically collected
in a block:

```rust
// TODO: iter_bytes
// TODO: iter_bytes_mut
```

### What Gets NO Comment

- Simple constructors (`Self { field: value }`)
- Standard trait implementations (Default, Clone, From)
- Simple getters and setters
- Obvious control flow (early returns on None, bounds checks with clear asserts)
- Loop variables in clear context
- Code where the doc comment already explains the logic

### Density

Comment density varies by complexity:

| Code type                              | Approximate density       |
|----------------------------------------|---------------------------|
| Complex algorithms, unsafe blocks      | 1 comment per 5-15 lines  |
| Public API boundaries, initialization  | 1 comment per 15-30 lines |
| Data structure methods, simple logic   | 1 comment per 30-50 lines |
| Trivial accessors, trait impls         | None                      |

The density is not uniform within a function. Comments cluster at decision points
and are absent in straightforward stretches.

### Placement and Formatting

- Almost always above the code being described, at the same indentation level.
- End-of-line comments are very rare.
- Always `//` style, never `/* */`.
- Multi-line comments continue with `//` on each line, maintaining alignment with
  the content.
- Continuation lines within a comment are indented to align with the first line's
  content when the comment uses a prefix like `Safety:` or `NOTE:`.

```rust
// Safety: If the layout keys match then the bundle is compatible and we can
//         bypass the more expensive runtime check.
```

---

## Additional Rules

These rules extend the baseline style:

### Doctests

Public API examples should be written as doctests that compile and pass. Use
```` ```rust ```` (or ```` ```no_run ```` for code that requires runtime context).
Doctests serve as both documentation and regression tests.

```rust
/// Create a new counter starting at zero.
///
/// # Examples
///
/// ```
/// let counter = Counter::new();
/// assert_eq!(counter.value(), 0);
/// ```
pub fn new() -> Self
```

Prefer `# Examples` over `# Usage` for the section header. Keep examples minimal
and focused on the common case.

### Arguments Section

Use `# Arguments` when parameter semantics are not obvious from names and types,
or when a function has 3 or more parameters. Use `* name - description` format
with column alignment.

```rust
/// Begin a render pass with the given configuration.
///
/// # Arguments
/// * `target`   - The render target to draw into.
/// * `clear`    - Clear color for the target. `None` preserves existing content.
/// * `viewport` - The sub-region to render. `None` uses the full target extent.
pub fn begin_pass(
    &mut self,
    target   : RenderTarget,
    clear    : Option<Color>,
    viewport : Option<Rect>,
)
```

---

## Summary of Principles

1. **Every public item gets a doc comment.** No exceptions.
2. **Doc comments say what/why, never how.** Implementation details belong in body
   comments.
3. **Body comments are outlines.** They let a reader skim the function's intent
   without parsing code.
4. **Safety is always justified.** Every `unsafe` block has a `// Safety:` comment
   explaining why the invariants hold.
5. **Trade-offs are documented where they are made.** At the decision point, not
   in a separate design document.
6. **Obvious code is left uncommented.** Comments earn their place by adding
   information the code does not already convey.
7. **Voice is imperative and direct.** "Create", "Returns", "Clear". No hedging,
   no filler, no LLM-isms.
8. **Public APIs have doctests.** Examples that compile and pass, demonstrating
   the common use case.
9. **Non-obvious parameters are documented.** `# Arguments` section for complex
   signatures or 3+ parameters.
10. **Consistent vocabulary.** The same concept uses the same words everywhere.
