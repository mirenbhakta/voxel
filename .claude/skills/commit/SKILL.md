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
```

### Title Line

**Format:** `<Action> <what changed>`

Imperative sentence, max ~72 characters. Examples:

```
Add user authentication middleware
Fix null pointer in request handler
Update CI pipeline to cache dependencies
```

Common action verbs:

| Action | Use when |
|--------|----------|
| Add | New feature, module, or capability |
| Fix | Bug fix |
| Update | Enhancing existing functionality |
| Remove | Deleting a feature or file |
| Rename | Renaming without behavior change |
| Refactor | Restructuring without behavior change |
| Migrate | Moving to a new version or API |

### Body

For non-trivial commits, structured paragraphs describing what changed and why:

- Open with a 1-2 sentence summary of the change and its motivation.
- Bullet points for individual changes.
- Include `Resolves #N` or `Closes #N` on its own line to auto-close GitHub issues.

For trivial commits (typo fixes, single-line changes), the body can be omitted.

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

## Commit Hygiene

- Amend a previous commit only when explicitly asked. Prefer new commits.
- Always let hooks run (`--no-verify` bypasses the safety net).
- Stage files individually rather than using `-A` or `.` to avoid accidental inclusions.
- If a pre-commit hook fails, fix the issue and create a new commit.
