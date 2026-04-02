# Rust Conventions

Formatting rules and patterns for Rust code. Referenced by
`CLAUDE.md`. All rules here are mandatory.

## Formatting Examples

**Function signatures** progress through three levels as they grow:

1. Inline: `fn foo(a: i32, b: i32) -> i32`
2. Return on new line: `fn foo(a: i32, b: i32)\n    -> i32`
3. All args on separate lines, indented once. `)` unindented. `-> ReturnType` indented once. `{` unindented.

```rust
// No where clause.
pub fn method(
    &mut self,
    name       : Type,
    other_name : OtherType,
)
    -> Result<ReturnType>
{

// Explicit where clause.
pub fn method<F>(
    &mut self,
    name       : Type,
    other_name : F,
)
    -> Result<ReturnType> where F: Copy
{
```

**Blank line after closing `}`.** Always leave a blank line after a closing
brace before the next statement or expression. No code immediately after a block.

**let-else:** `else {` the body goes on the next line after the expression.

```rust
let Some(x) = expr else {
    return;
};
```

**If/else assignments:** single line or wrap in block (never bare multiline).

```rust
// Good: single line
let x = if cond { a } else { b };

// Good: wrapped in block
let x = {
    if cond {
        a
    }
    else {
        b
    }
};

// Bad: bare multiline
let x = if cond {
    a
}
else {
    b
};
```

**Flow control** statements are never single-line.

```rust
// Good
if cond {
    return;
}

// Bad
if cond { return; }
```

**Flow control spacing** relative to other code:

```rust
// Good: space between flow control and next expression.
if cond {
    a
}

if cond {
    b
}

let x = y;

// Good: related single line expressions are close together.
let x = expr;
if x {
    do_thing();
}
```

**Argument columns:** always align by comma, not by value.

```rust
set("value"         , 1.02    );
set("value-b"       , 1.024532);
set("value-charlie" , 1.2     );
```

## Imports and Modules

- Import symbols into file scope with `use` statements. Avoid qualified paths like `foo::bar::Baz` in code bodies. Exception: when two types share a name, qualify the less-used one.
- Use new-style module paths (`foo.rs` + `foo/bar.rs`), never `mod.rs`.

## Generic Functions

Prefer type erasure over large generic bodies. If a function body is non-trivial,
use `&dyn Trait` dispatch over monomorphization. Short generic wrappers delegating
to non-generic implementations is the standard pattern. Reduces code bloat,
improves compile times and binary size.

```rust
// Short generic wrapper
pub fn process<T: Trait>(value: T) -> Result<(), Error> {
    __process(&value as &dyn Trait)
}

// Non-generic implementation
#[inline(never)]
fn __process(value: &dyn Trait) -> Result<(), Error> {
    // Bulk of logic here
}
```

## Macros

- When implementing a trait across many types, use `macro_rules!` rather than copy-pasting implementations.
- Extend existing macros when adding new types rather than writing standalone impls.
- Prefer declarative macros over proc macros for simple repetitive patterns.

## File Organization

- Section headers use `// --- <Name> ---` format exclusively.
- Error type definitions go at the bottom of the file.
- Impl block ordering by visibility: `pub` -> `pub(<path>)` -> private.
- Constructors go at the top of their impl block (just below struct/enum definition).

```rust
pub struct Foo {
    value: u32,
}

// --- Foo ---

impl Foo {
    /// Creates a new foo with the given value.
    pub fn new(value: u32) -> Self {
        Self { value }
    }

    /// Returns the current value.
    pub fn value(&self) -> u32 {
        self.value
    }
}

impl Foo {
    /// Internal helper for validation.
    fn validate(&self) -> bool {
        self.value > 0
    }
}

// --- Error ---

#[derive(Debug)]
pub enum FooError {
    InvalidValue,
}
```

## Builder Pattern

- Use `create() -> Builder` as the entry point (not `new()` on the main type).
- Builders can have their own `new()` constructor if needed.
- Example: `Foo::create().with_bar(x).build()` not `Foo::new()`.
