#!/usr/bin/env bash
#
# PreToolUse + PostToolUse hook for automatic agent worktree isolation.
#
# PreToolUse (Agent): creates a worktree for edit agents and injects
# the path into the agent's prompt.
#
# PostToolUse (Agent): collects changes from the worktree back into
# the spawning worktree and cleans up.

set -euo pipefail

INPUT=$(cat)
EVENT=$(echo "$INPUT" | jq -r '.hook_event_name')
AGENT_TYPE=$(echo "$INPUT" | jq -r '.tool_input.subagent_type // .tool_response.agentType // "unknown"')
TOOL_USE_ID=$(echo "$INPUT" | jq -r '.tool_use_id')
CWD=$(echo "$INPUT" | jq -r '.cwd')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_SCRIPT="$SCRIPT_DIR/agent-worktree.sh"

# Agent types that modify files and need worktree isolation.
needs_worktree() {
    case "$1" in
        Edit|"Edit - Minor"|Debug)
            return 0 ;;
        *)
            return 1 ;;
    esac
}

# Shorten tool_use_id for worktree name (last 12 chars).
worktree_name() {
    echo "agent-${1: -12}"
}

# -----------------------------------------------------------------------

if [ "$EVENT" = "PreToolUse" ]; then
    if ! needs_worktree "$AGENT_TYPE"; then
        exit 0
    fi

    NAME=$(worktree_name "$TOOL_USE_ID")
    WORKTREE_PATH=$(cd "$CWD" && bash "$WORKTREE_SCRIPT" create "$NAME" 2>/dev/null)

    if [ -z "$WORKTREE_PATH" ]; then
        # Creation failed — fall through without isolation.
        exit 0
    fi

    ORIGINAL_PROMPT=$(echo "$INPUT" | jq -r '.tool_input.prompt')

    # Inject worktree path into the agent's prompt.
    PREAMBLE="You are running in an isolated worktree at: $WORKTREE_PATH

Use absolute paths under this directory for all file operations. For Bash
commands, cd here first.
Example: Read \"$WORKTREE_PATH/lib/foo/src/bar.rs\"

---

"

    UPDATED_PROMPT="${PREAMBLE}${ORIGINAL_PROMPT}"

    jq -n \
        --arg prompt "$UPDATED_PROMPT" \
        '{
            hookSpecificOutput: {
                hookEventName: "PreToolUse",
                permissionDecision: "allow",
                permissionDecisionReason: "agent worktree created",
                updatedInput: { prompt: $prompt },
                additionalContext: "Agent is running in an isolated worktree. Changes will be merged back on completion."
            }
        }'
    exit 0
fi

if [ "$EVENT" = "PostToolUse" ]; then
    NAME=$(worktree_name "$TOOL_USE_ID")
    WORKTREE_DIR="$CWD/.claude/worktrees/$NAME"

    # Check if we created a worktree for this agent. More reliable than
    # matching on agent type, since updatedInput can alter the type
    # visible in PostToolUse.
    if [ ! -d "$WORKTREE_DIR" ]; then
        exit 0
    fi

    # Collect changes back into the spawning worktree.
    COLLECT_OUTPUT=$(cd "$CWD" && bash "$WORKTREE_SCRIPT" collect "$NAME" 2>&1) || true

    # Clean up the worktree.
    (cd "$CWD" && bash "$WORKTREE_SCRIPT" cleanup "$NAME" 2>/dev/null) || true

    # Log what was collected (visible in hook output if needed).
    if [ -n "$COLLECT_OUTPUT" ]; then
        echo "$COLLECT_OUTPUT" >&2
    fi

    exit 0
fi

exit 0
