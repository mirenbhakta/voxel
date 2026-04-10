---
description: >
  Create a pull request with proper formatting. Use when the user asks
  to create a PR, or after completing work on a branch.
---

# Pull Request Creation

## Process

1. Run in parallel:
   - `git status` (never `-uall`)
   - `git diff` for any uncommitted changes
   - `git log --oneline main..HEAD` to see all commits on this branch
   - `git diff main...HEAD --stat` for a file-level summary
   - Check if current branch tracks a remote and is up to date

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

<Grouped by area. Brief description of each change.>

## Test plan

- [ ] <verification steps>

Closes #N

EOF
)"
```

## Issue Linking

When a PR resolves or closes an issue, include closing keywords in the PR body:

- `Closes #N`
- `Resolves #N`
- `Fixes #N`

Place them at the end of the body. Each issue gets its own line.

## Title Convention

Keep under 70 characters. Imperative verb style:

- `Add user authentication middleware`
- `Fix request handler null pointer`
- `Update CI pipeline caching`

Do NOT include `(#N)` in the PR title (GitHub adds it automatically).

## Pre-Flight Checks

Before creating the PR:

- All changes committed (no dirty working tree)
- Branch pushed to remote with `-u`
- Tests pass

If any of these fail, fix them before creating the PR.
