---
description: >
  Create a GitHub issue with proper labels. Use when the user asks to file
  an issue, or when work reveals a problem that should be tracked.
---

# Issue Creation

## Process

1. Determine the issue type, affected area(s), impact, and target platform(s).
2. Select labels from the taxonomy below.
3. Create the issue with `gh issue create` using the label flags.

## Label Taxonomy

Every issue gets at least one `type:` label and one `area:` label.

### Type (required, pick one)

| Label | Use when |
|-------|----------|
| `type: bug` | Something is broken |
| `type: enhancement` | Improvement to existing feature |
| `type: feature` | New capability |
| `type: refactor` | Structural change spanning multiple issues |
| `type: documentation` | Docs additions or changes |
| `type: research` | Investigation or exploration |
| `type: task` | Doesn't fit other categories |
| `type: tracking` | Meta-issue tracking a larger initiative |

### Area (required, pick all that apply)

> **Replace these with your project's subsystems.** The labels below are
> examples to show the pattern.

| Label | Scope |
|-------|-------|
| `area: frontend` | Client-side UI and rendering |
| `area: backend` | Server-side logic and APIs |
| `area: infrastructure` | Deployment, CI/CD, hosting |
| `area: documentation` | Docs, guides, references |
| `area: tooling` | Developer tools and scripts |

### Impact (recommended)

| Label | Criteria |
|-------|----------|
| `impact: high` | Affects most users or is critical |
| `impact: med` | Affects many users, not critical |
| `impact: low` | Affects few users |

### Target Platform (when platform-specific)

`target: windows`, `target: linux`, `target: macos`, `target: android`, `target: ios`

### Tags (as needed)

| Label | Use when |
|-------|----------|
| `tag: blocked` | Blocked by another issue |
| `tag: backlog` | Can wait indefinitely |
| `tag: upstream` | Relates to external dependency |
| `tag: needs-information` | Awaiting more details |
| `tag: needs-repro` | Needs reproduction case |
| `tag: needs-triage` | New, uncategorized |

## Command Format

```bash
gh issue create \
  --title "Issue title" \
  --label "type: bug" \
  --label "area: backend" \
  --label "impact: high" \
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

# After creation, if blocked
gh issue edit <N> --add-label "tag: blocked"
```
