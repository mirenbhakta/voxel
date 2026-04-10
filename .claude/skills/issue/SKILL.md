---
description: >
  Create a GitHub issue. Use when the user asks to file an issue, or
  when work reveals a problem that should be tracked.
---

# Issue Creation

## Process

1. Determine the issue type and affected area.
2. Create the issue with `gh issue create`.

## Command Format

```bash
gh issue create \
  --title "Issue title" \
  --body "$(cat <<'EOF'
## Description

What is wrong or what is needed.

## Context

Why this matters, what triggered it.

## Reproduction (for bugs)

Steps to reproduce, or a minimal test case.
EOF
)"
```

## Linking Issues

When creating issues that relate to existing ones:

```bash
# Reference in body
gh issue create --title "..." --body "Related to #123"
```
