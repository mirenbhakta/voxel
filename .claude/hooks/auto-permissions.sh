#!/usr/bin/env bash
#
# PreToolUse hook: auto-approve safe commands and enforce worktree isolation.
#
# Two responsibilities:
#
# 1. WORKTREE FENCE — When the agent's CWD is inside a worktree
#    (.claude/worktrees/ or .local/git/), deny any tool call whose target
#    path resolves into the parent repository. Other external paths
#    (system headers, toolchains, ~/.cargo, etc.) are allowed. This
#    applies to Read, Edit, Write, Glob, Grep, and Bash tools.
#
# 2. BASH AUTO-APPROVE — Patterns loaded from .permissions and
#    .permissions.local control which Bash commands are auto-approved.
#    Each line is:
#      MODE REGEX
#    where MODE is:
#      read      - Always approve (safe read-only command)
#      write     - Approve only when permission_mode allows writes
#      worktree  - Approve only when running in an isolated agent worktree
#      nolog     - Do not approve, but suppress fallthrough logging
#      dir       - Adds a directory to the allowed path set
#
#    Patterns may use ${DIRS} as a placeholder for allowed directory prefixes.
#    This expands to a regex alternation of all allowed paths (CWD, /tmp, and
#    any dir entries). Commands containing ../ are resolved and re-validated.
#
# 3. STRUCTURED MATCHER — After regex patterns fall through, the hook
#    invokes match-command.py which parses the command into a token-level
#    AST (pipeline of commands, with classified arguments) and matches it
#    against rules expressed as Python pattern matches. This is where
#    cargo, bare git, gh, rustup, and exec wrapper rules live, including
#    nolog rules for always-ask operations.
#
# Outputs JSON with permissionDecision for matching operations.
# Exits silently (no output) to fall through to normal permission flow.

set -euo pipefail

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name')
CWD=$(echo "$INPUT" | jq -r '.cwd')

# --- Permission checks ---

in_worktree() {
    [[ "$CWD" == */.claude/worktrees/* || "$CWD" == */.local/git/* ]]
}

approve() {
    jq -n --arg reason "$1" '{
        hookSpecificOutput: {
            hookEventName: "PreToolUse",
            permissionDecision: "allow",
            permissionDecisionReason: $reason
        }
    }'
    exit 0
}

deny() {
    jq -n --arg reason "$1" '{
        hookSpecificOutput: {
            hookEventName: "PreToolUse",
            permissionDecision: "deny",
            permissionDecisionReason: $reason
        }
    }'
    exit 0
}

# Derive the parent repository root and worktree root from the current
# working directory. Sets PARENT_REPO and WORKTREE_ROOT. Returns 1 if
# not in a recognized worktree layout.
worktree_fence_dirs() {
    PARENT_REPO=""
    WORKTREE_ROOT=""

    case "$CWD" in
        */.claude/worktrees/*)
            PARENT_REPO="${CWD%%/.claude/worktrees/*}"
            local remainder="${CWD#*/.claude/worktrees/}"
            local wt_name="${remainder%%/*}"
            WORKTREE_ROOT="$PARENT_REPO/.claude/worktrees/$wt_name"
            ;;
        */.local/git/*)
            PARENT_REPO="${CWD%%/.local/git/*}"
            local remainder="${CWD#*/.local/git/}"
            local wt_name="${remainder%%/*}"
            WORKTREE_ROOT="$PARENT_REPO/.local/git/$wt_name"
            ;;
        *)
            return 1
            ;;
    esac
}

# Check if a resolved absolute path violates the worktree fence.
# Returns 0 (true = violation) if the path is inside the parent repo
# but outside the worktree.
path_violates_fence() {
    local resolved="$1"
    [[ "$resolved" == "$PARENT_REPO" || "$resolved" == "$PARENT_REPO"/* ]] \
        && [[ "$resolved" != "$WORKTREE_ROOT"/* ]]
}

# --- Worktree fence (file-access tools) ---
#
# When running in a worktree, deny file-access tools that target paths
# within the parent repository. Other external paths (system headers,
# toolchains, etc.) are allowed.

if [[ "$TOOL" != "Bash" ]]; then
    # Only enforce fence in worktrees — outside worktrees, the main
    # session's permission mode governs these tools.
    if ! in_worktree; then
        exit 0
    fi

    worktree_fence_dirs || exit 0

    # Extract target path from tool input.
    case "$TOOL" in
        Read|Edit|Write)
            TARGET=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
            ;;
        Glob|Grep)
            TARGET=$(echo "$INPUT" | jq -r '.tool_input.path // empty')
            ;;
        *)
            exit 0
            ;;
    esac

    # No explicit path — tool defaults to CWD (within the worktree).
    if [[ -z "$TARGET" ]]; then
        exit 0
    fi

    # Resolve relative paths from CWD.
    if [[ "$TARGET" != /* ]]; then
        TARGET="$CWD/$TARGET"
    fi

    RESOLVED=$(realpath -m "$TARGET" 2>/dev/null || echo "$TARGET")

    if path_violates_fence "$RESOLVED"; then
        deny "WORKTREE ISOLATION: Access to the parent repository is denied. All file operations must target paths within your worktree at: $WORKTREE_ROOT"
    fi

    exit 0
fi

COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')
PERM_MODE=$(echo "$INPUT" | jq -r '.permission_mode')

# Handle "cd <path> && <cmd>" prefixes. Strip the cd and resolve the target;
# validation against ALLOWED_DIRS happens after that array is built.
CD_TARGET=""
if [[ "$COMMAND" =~ ^cd[[:space:]]+([^[:space:]&]+)[[:space:]]*\&\&[[:space:]]*(.*) ]]; then
    CD_TARGET="${BASH_REMATCH[1]}"
    COMMAND="${BASH_REMATCH[2]}"

    # Resolve the cd target relative to CWD if not absolute.
    if [[ "$CD_TARGET" == /* ]]; then
        CD_TARGET=$(realpath -m "$CD_TARGET" 2>/dev/null) || exit 0
    else
        CD_TARGET=$(realpath -m "$CWD/$CD_TARGET" 2>/dev/null) || exit 0
    fi
fi

# Check cd target against worktree fence before ALLOWED_DIRS can exit.
if [[ -n "$CD_TARGET" ]] && in_worktree && worktree_fence_dirs; then
    if path_violates_fence "$CD_TARGET"; then
        deny "WORKTREE ISOLATION: Bash command changes directory into the parent repository. All operations must target paths within your worktree at: $WORKTREE_ROOT"
    fi
fi

CLAUDE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERMISSIONS_FILE="$CLAUDE_DIR/.permissions"
PERMISSIONS_LOCAL="$CLAUDE_DIR/.permissions.local"

# Build combined patterns list (local overrides shared, first match wins).
COMBINED=$(mktemp)
trap 'rm -f "$COMBINED"' EXIT
[ -f "$PERMISSIONS_LOCAL" ] && cat "$PERMISSIONS_LOCAL" >> "$COMBINED"
[ -f "$PERMISSIONS_FILE" ] && cat "$PERMISSIONS_FILE" >> "$COMBINED"

if [[ ! -s "$COMBINED" ]]; then
    exit 0
fi

# --- Collect allowed directories ---

ALLOWED_DIRS=("$CWD" "/tmp")

while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    mode="${line%% *}"
    if [[ "$mode" == "dir" ]]; then
        dir="${line#* }"
        dir="${dir#"${dir%%[![:space:]]*}"}"
        dir="${dir/#\~/$HOME}"
        ALLOWED_DIRS+=("$dir")
    fi
done < "$COMBINED"

# If the command had a cd prefix, validate the target is within the allowed
# dirs we just collected. If it passes, treat it as the effective CWD so that
# DIRS_RE and validate_paths resolve paths relative to the cd target.
if [[ -n "$CD_TARGET" ]]; then
    cd_ok=false
    for d in "${ALLOWED_DIRS[@]}"; do
        if [[ "$CD_TARGET" == "$d" || "$CD_TARGET" == "$d/"* ]]; then
            cd_ok=true
            break
        fi
    done
    if ! $cd_ok; then
        exit 0
    fi
    CWD="$CD_TARGET"
fi

# --- Build ${DIRS} regex ---

escape_regex() {
    printf '%s' "$1" | sed 's/[.[\*^$()+?{|\\]/\\&/g'
}

dir_parts=()
for d in "${ALLOWED_DIRS[@]}"; do
    dir_parts+=("$(escape_regex "$d")")
done
# Also allow relative paths (non-/ start): they can only resolve within CWD,
# and ../ escapes are caught by validate_paths below.
dir_parts+=("[^/-]")
DIRS_RE="($(IFS='|'; printf '%s' "${dir_parts[*]}"))"

# --- Path validation ---
#
# After a regex match, verify that any ../ in the command doesn't escape
# allowed directories. If no .. is present, the prefix match is sufficient.

validate_paths() {
    local cmd="$1"

    # Fast path: no .. means prefix match is sufficient.
    if [[ "$cmd" != *..* ]]; then
        return 0
    fi

    # Extract tokens, resolve any containing .., check against allowed dirs.
    local -a tokens
    read -ra tokens <<< "$cmd"

    for ((i = 1; i < ${#tokens[@]}; i++)); do
        local tok="${tokens[$i]}"

        # Skip flags.
        [[ "$tok" == -* ]] && continue

        # Only care about tokens with ..
        [[ "$tok" != *..* ]] && continue

        # Resolve the path.
        local resolved
        if [[ "$tok" == /* ]]; then
            resolved=$(realpath -m "$tok" 2>/dev/null) || return 1
        else
            resolved=$(realpath -m "$CWD/$tok" 2>/dev/null) || return 1
        fi

        # Check against allowed dirs.
        local ok=false
        for d in "${ALLOWED_DIRS[@]}"; do
            if [[ "$resolved" == "$d" || "$resolved" == "$d/"* ]]; then
                ok=true
                break
            fi
        done

        if ! $ok; then
            return 1
        fi
    done

    return 0
}

# --- Permission checks ---

writes_allowed() {
    case "$PERM_MODE" in
        acceptEdits|auto|bypassPermissions) return 0 ;;
        *) return 1 ;;
    esac
}

# --- Worktree fence (Bash commands) ---
#
# When in a worktree, scan path-like tokens for references to the parent
# repository. This runs before pattern matching so the fence cannot be
# bypassed by an approve rule.

if in_worktree && worktree_fence_dirs; then
    read -ra _fence_tokens <<< "$COMMAND"
    for _tok in "${_fence_tokens[@]}"; do
        # Skip flags and non-path tokens.
        [[ "$_tok" == -* ]] && continue

        # Resolve the token as a path.
        if [[ "$_tok" == /* ]]; then
            _resolved=$(realpath -m "$_tok" 2>/dev/null) || continue
        elif [[ "$_tok" == */* ]]; then
            # Only resolve tokens that look like paths (contain a slash).
            _resolved=$(realpath -m "$CWD/$_tok" 2>/dev/null) || continue
        else
            continue
        fi

        if path_violates_fence "$_resolved"; then
            deny "WORKTREE ISOLATION: Bash command references the parent repository. All operations must target paths within your worktree at: $WORKTREE_ROOT"
        fi
    done
fi

# --- Match patterns ---

SKIP_LOG=0

while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    [[ -z "$line" || "$line" == \#* ]] && continue

    # Skip dir entries (already processed).
    [[ "$line" == dir\ * || "$line" == dir$'\t'* ]] && continue

    mode="${line%% *}"
    pattern="${line#* }"
    pattern="${pattern#"${pattern%%[![:space:]]*}"}"

    # Expand ${DIRS} placeholder.
    expanded="${pattern//\$\{DIRS\}/$DIRS_RE}"

    if echo "$COMMAND" | grep -qE "$expanded"; then
        # If pattern uses ${DIRS}, validate ../ paths.
        if [[ "$pattern" == *'${DIRS}'* ]] && ! validate_paths "$COMMAND"; then
            continue
        fi

        case "$mode" in
            read)
                approve "read-only command"
                ;;
            write)
                if writes_allowed; then
                    approve "write command (writes enabled)"
                fi
                ;;
            worktree)
                if in_worktree; then
                    approve "isolated worktree operation"
                fi
                ;;
            nolog)
                # Match suppresses fallthrough logging but does NOT approve.
                # Used for "always ask" commands so the log only collects
                # patterns we might want to extend.
                SKIP_LOG=1
                ;;
        esac
    fi
done < "$COMBINED"

# --- Structured matcher (parsed-form rules) ---
#
# After regex patterns fall through, hand the command to the python matcher
# which parses it into a typed pipeline AST and matches rules expressed as
# Python pattern matches. The matcher prints one of:
#
#   ALLOW <reason>   -- auto-approve
#   NOLOG <reason>   -- skip logging but still ask the user
#   (empty)          -- fall through to normal permission flow

MATCHER="$CLAUDE_DIR/hooks/match_command.py"
if [[ -x "$MATCHER" || -f "$MATCHER" ]]; then
    MATCHER_OUT=$(printf '%s' "$COMMAND" | python3 "$MATCHER" 2>/dev/null || true)
    if [[ "$MATCHER_OUT" == ALLOW* ]]; then
        approve "${MATCHER_OUT#ALLOW }"
    elif [[ "$MATCHER_OUT" == NOLOG* ]]; then
        SKIP_LOG=1
    fi
fi

# --- Fall-through logging ---
#
# If we reach this point, the command was not auto-approved by any rule and
# will fall through to the normal permission flow (user prompt). When this
# happens inside a worktree, log the command to the parent repository so we
# can mine the corpus to extend .permissions over time. The log lives in the
# parent repo (not the worktree) so it survives worktree cleanup and so a
# single repo accumulates entries from all its worktrees.

if [[ "$SKIP_LOG" -eq 0 ]] && in_worktree && worktree_fence_dirs; then
    LOG_DIR="$PARENT_REPO/.claude/log/permissions"
    LOG_FILE="$LOG_DIR/fallthrough.log"
    # Escape embedded newlines so heredoc commands log as a single entry.
    COMMAND_ESCAPED="${COMMAND//$'\n'/\\n}"
    mkdir -p "$LOG_DIR" 2>/dev/null && \
        printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$COMMAND_ESCAPED" >> "$LOG_FILE"
fi

exit 0
