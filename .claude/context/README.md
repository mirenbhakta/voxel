# Context Files

Context files are project-specific reference material that agents read before starting
work. They contain the conventions, formatting rules, and patterns that are too
language-specific or project-specific for `CLAUDE.md`.

## How It Works

Every code-writing agent has a Setup step that says "read all context files in
`.claude/context/`." This means any `.md` file you place here will be loaded as
authoritative project rules.

## Creating a Context File

1. Create a `.md` file in this directory.
2. Name it descriptively: `language.md`, `documentation.md`, `architecture.md`,
   `testing.md`, etc.
3. Write concrete, actionable rules. Agents need to be able to follow them
   mechanically. Examples and counter-examples are more useful than abstract
   principles.

## What Makes a Good Context File

- **Concrete examples.** Show the right way and the wrong way. Agents pattern-match
  against examples better than they follow abstract descriptions.
- **Mechanical rules.** "Column-align struct fields at the next tab stop" is
  enforceable. "Make code readable" is not.
- **Scope.** Each file covers one topic. A language conventions file covers formatting,
  naming, and patterns. A documentation file covers voice, structure, and coverage.
  Don't combine unrelated concerns.
- **Grounded in existing code.** If possible, distill conventions from the codebase
  as it already exists, rather than inventing new rules. The goal is consistency with
  what's already there.

## Recommended Files

| File | Contents |
|------|----------|
| `language.md` | Language-specific naming, formatting, patterns, file organization |
| `documentation.md` | Doc comment voice, structure, inline comment conventions |
| `architecture.md` | System layering, module boundaries, dependency rules |
| `testing.md` | Test framework patterns, what to test, test file organization |

You don't need all of these. Start with what matters most to your project and add
more as you encounter convention drift.

## Language-Specific Agents

If your project uses multiple languages, you can also create language-specific agents
(e.g., `agents/lang.python.md`, `agents/lang.cpp.md`) that read the relevant context
files and enforce language-specific rules. See the agent files in `.claude/agents/` for
the pattern.

## Example Stubs

The `.example.md` files in this directory are templates showing what sections to include.
Rename them (removing `.example`) and fill in your project's conventions.
