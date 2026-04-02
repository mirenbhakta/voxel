---
description: >
  Create a pull request with proper formatting and issue linking. Use when
  the user asks to create a PR, or after completing work on a branch.
---

# Pull Request Creation

## Process

1. Run in parallel:
   - `git status` (never `-uall`)
   - `git diff` for any uncommitted changes
   - `git log --oneline main..HEAD` to see all commits on this branch
   - `git diff main...HEAD --stat` for a file-level summary
   - Check if current branch tracks a remote and is up to date

   > **Note:** Replace `main` with your default branch name if different.

2. Analyze ALL commits on the branch (not just the latest). Draft the PR.

3. Identify any GitHub issues this PR resolves by:
   - Checking commit messages for `Resolves #N`, `Closes #N`, `Fixes #N`
   - Checking if the branch name references an issue number
   - Asking the user if unclear

4. Push and create the PR.

## PR Format

```bash
gh pr create --title "<title>" --body "$(cat <<'EOF'
## Summary

<1-5 bullet points describing what changed and why>

## Changes

<Grouped by module or subsystem. Brief description of each change area.>

## Test plan

- [ ] Build check -- no warnings
- [ ] Tests -- all pass
- [ ] <any manual verification needed>

Closes #N
Closes #M

EOF
)"
```

## Issue Linking

**This is critical.** Every PR that resolves an issue MUST include closing
keywords in the PR body. GitHub recognizes these keywords followed by an
issue reference:

- `Closes #N`
- `Resolves #N`
- `Fixes #N`

Place them at the end of the body. Each issue gets its own line.

**How to find related issues:**

```bash
# Search open issues by keyword
gh issue list --state open --search "<keyword>"

# Check if a specific issue exists
gh issue view <N>
```

If the work clearly relates to an open issue but doesn't fully resolve it,
use `Related to #N` instead (this creates a reference without closing).

## Title Convention

Keep under 70 characters. Use imperative verb style:

- `Add user authentication flow`
- `Fix pagination for large result sets`
- `Update API rate limiting configuration`

Do NOT include `(#N)` in the PR title (GitHub adds it automatically).

## Labels

Add labels to the PR matching the issue label taxonomy (type, area, impact).
Use `--label` flags on `gh pr create`, same as issues:

```bash
gh pr create \
  --title "Fix login timeout on slow connections" \
  --label "type: bug" \
  --label "area: backend" \
  --label "impact: high" \
  --body "..."
```

## Pre-Flight Checks

Before creating the PR:

- All changes committed (no dirty working tree)
- Branch pushed to remote with `-u`
- Build check passes with zero warnings (not just "no new" -- zero total)
- Tests pass

If any of these fail, fix them before creating the PR.
