---
description: >
  Create a git commit following project conventions. Use when the user
  asks to commit, or after completing a task that should be committed.
---

# Commit

## Process

1. Run `git status` (never use `-uall`), `git diff --cached`, and `git log --oneline -10` in parallel.
2. Review all staged and unstaged changes. Stage relevant files by name (never `git add -A` or `git add .`).
3. Do not commit files that likely contain secrets (.env, credentials, etc.).
4. Draft the commit message following the conventions below.
5. Create the commit using a HEREDOC for the message.
6. Run `git status` after to verify success.

## Message Format

```
<title line>

<body>

<trailers>
```

### Title Line

**Format:** `<type>(<scope>): <description>`

Use [Conventional Commits](https://www.conventionalcommits.org/) format.
Max ~72 characters. Examples:

```
feat(auth): add JWT token refresh (#42)
fix(parser): handle empty input without panic (#15)
docs: update API reference for v2 endpoints
refactor(db): extract connection pooling into shared module
```

For changes spanning many areas, the scope can be omitted:

```
feat: add user notification system (#88)
fix: resolve cross-platform path handling (#73)
```

**Type table:**

| Type | Use when |
|------|----------|
| feat | New feature or capability |
| fix | Bug fix |
| docs | Documentation only |
| refactor | Code restructuring, no behavior change |
| test | Adding or updating tests |
| build | Build system or dependency changes |
| ci | CI/CD configuration |
| perf | Performance improvement |
| chore | Maintenance, tooling, config |

If the commit resolves a PR, include `(#N)` at the end of the title.

**Common action verbs** (for the description portion):

| Action | Use when |
|--------|----------|
| Add | Wholly new feature or capability |
| Fix | Bug fix |
| Replace | Swapping one implementation for another |
| Remove | Deleting a feature or module |
| Rename | Renaming without behavior change |
| Extract | Pulling code into a new module |
| Migrate | Moving to a new version or API |
| Update | Enhancing an existing feature |

> **Customization:** If your project uses a different commit format (e.g.,
> `[scope] - Action`), replace the Title Line section above.

### Body

Structured paragraphs describing what changed and why. For non-trivial commits:

- Open with a 1-2 sentence summary of the change and its motivation.
- Use section headers for distinct areas of change (module names, subsystem names).
- Bullet points for individual changes within a section.
- Include `Resolves #N` or `Closes #N` on its own line to auto-close GitHub issues.
  Multiple issues get separate lines.

For trivial commits (typo fixes, single-line changes), the body can be omitted.

**Example body:**
```
Custom measurement harness and orchestrator CLI replacing the previous
benchmarking framework.

Harness (lib/bench):
- Interleaved chunk execution with warmup detection
- Adaptive warmup with time-budgeted iterations
- Closure-based API eliminating macro boilerplate

Orchestrator (tools/bench-cli):
- Target discovery, compilation, multi-invocation execution
- Statistical comparison with dual-gate significance testing

Resolves #355
```

## Commit HEREDOC

Always use this format for the commit command:

```bash
git commit -m "$(cat <<'EOF'
Title line here

Body here.

Resolves #123
EOF
)"
```

## Anti-Patterns

- Never amend a previous commit unless explicitly asked.
- Never skip hooks (`--no-verify`).
- Never use `-A` or `.` to stage files.
- Never create empty commits.
- If a pre-commit hook fails, fix the issue and create a NEW commit.
