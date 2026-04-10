#!/usr/bin/env bash
#
# Agent worktree lifecycle management.
#
# Usage:
#   agent-worktree.sh create <name>    Create an isolated worktree for an agent.
#   agent-worktree.sh collect <name>   Merge agent changes back to the spawning worktree.
#   agent-worktree.sh cleanup <name>   Remove the worktree and baseline snapshot.
#
# The spawning worktree's uncommitted state is copied into the agent
# worktree. A baseline snapshot of the modified files is saved so that
# after the agent runs, we can diff against it to extract exactly the
# agent's changes — no commits needed.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKTREE_DIR="$REPO_ROOT/.claude/worktrees"

usage() {
    echo "Usage: $0 {create|collect|cleanup} <name>" >&2
    exit 1
}

[ $# -ge 2 ] || usage

ACTION="$1"
NAME="$2"
AGENT_DIR="$WORKTREE_DIR/$NAME"
BASELINE_DIR="$WORKTREE_DIR/.baseline-$NAME"

# -----------------------------------------------------------------------
# create: set up an isolated worktree with the spawning tree's state
# -----------------------------------------------------------------------
do_create() {
    if [ -d "$AGENT_DIR" ]; then
        echo "error: worktree '$NAME' already exists at $AGENT_DIR" >&2
        exit 1
    fi

    mkdir -p "$WORKTREE_DIR"

    # Create a detached worktree at the current HEAD.
    git worktree add --detach "$AGENT_DIR" HEAD 2>/dev/null

    # The post-checkout hook handles submodule symlinks and
    # settings.local.json copying. If it didn't run (older git),
    # do it manually.
    if [ ! -f "$AGENT_DIR/.claude/settings.local.json" ] && [ -f "$REPO_ROOT/.claude/settings.local.json" ]; then
        mkdir -p "$AGENT_DIR/.claude"
        cp "$REPO_ROOT/.claude/settings.local.json" "$AGENT_DIR/.claude/settings.local.json"
    fi

    # --- Copy uncommitted state from spawning worktree ---

    # Staged + unstaged changes.
    diff_output="$(git diff HEAD)" || true
    if [ -n "$diff_output" ]; then
        echo "$diff_output" | (cd "$AGENT_DIR" && git apply --allow-empty 2>/dev/null) || {
            echo "warning: some hunks failed to apply, continuing" >&2
        }
    fi

    # Untracked files (excluding the worktrees directory itself).
    git ls-files --others --exclude-standard -z | \
        grep -zv '^\.claude/worktrees/' | \
        while IFS= read -r -d '' file; do
            dir="$(dirname "$file")"
            mkdir -p "$AGENT_DIR/$dir"
            cp "$file" "$AGENT_DIR/$file"
        done

    # --- Snapshot baseline for diffing later ---
    #
    # Copy every file that differs from HEAD into the baseline directory.
    # After the agent runs, `git diff --no-index baseline/ worktree/`
    # gives us exactly the agent's changes.

    mkdir -p "$BASELINE_DIR"

    # Modified/added tracked files.
    (cd "$AGENT_DIR" && git diff --name-only HEAD) | while read -r file; do
        if [ -f "$AGENT_DIR/$file" ]; then
            dir="$(dirname "$file")"
            mkdir -p "$BASELINE_DIR/$dir"
            cp "$AGENT_DIR/$file" "$BASELINE_DIR/$file"
        fi
    done

    # Untracked files.
    (cd "$AGENT_DIR" && git ls-files --others --exclude-standard) | while read -r file; do
        if [ -f "$AGENT_DIR/$file" ]; then
            dir="$(dirname "$file")"
            mkdir -p "$BASELINE_DIR/$dir"
            cp "$AGENT_DIR/$file" "$BASELINE_DIR/$file"
        fi
    done

    # Also save the list of all files at baseline for deletion detection.
    (cd "$AGENT_DIR" && git ls-files && git ls-files --others --exclude-standard) | \
        sort -u > "$BASELINE_DIR/.filelist"

    echo "$AGENT_DIR"
}

# -----------------------------------------------------------------------
# collect: extract agent changes and merge them into the spawning worktree
# -----------------------------------------------------------------------
do_collect() {
    if [ ! -d "$AGENT_DIR" ]; then
        echo "error: worktree '$NAME' does not exist" >&2
        exit 1
    fi

    local had_conflicts=0

    # --- Find files the agent modified ---
    #
    # For files that exist in the baseline, use git diff --no-index
    # to get a proper patch. For new files (not in baseline), copy
    # directly. For deleted files, remove from spawning worktree.

    # Collect the set of files currently in the agent worktree.
    (cd "$AGENT_DIR" && git ls-files && git ls-files --others --exclude-standard) | \
        sort -u > "$BASELINE_DIR/.filelist-final"

    # Files deleted by the agent (in baseline but not in final).
    { grep -Fxv -f "$BASELINE_DIR/.filelist-final" "$BASELINE_DIR/.filelist" || true; } | while read -r file; do
        if [ -f "$REPO_ROOT/$file" ]; then
            rm "$REPO_ROOT/$file"
            echo "D  $file"
        fi
    done

    # Files created by the agent (in final but not in baseline).
    { grep -Fxv -f "$BASELINE_DIR/.filelist" "$BASELINE_DIR/.filelist-final" || true; } | while read -r file; do
        if [ -f "$AGENT_DIR/$file" ]; then
            dir="$(dirname "$file")"
            mkdir -p "$REPO_ROOT/$dir"
            cp "$AGENT_DIR/$file" "$REPO_ROOT/$file"
            echo "A  $file"
        fi
    done

    # Files that existed in both — check for modifications.
    { grep -Fx -f "$BASELINE_DIR/.filelist" "$BASELINE_DIR/.filelist-final" || true; } | while read -r file; do
        [ -f "$AGENT_DIR/$file" ] || continue

        # Determine the baseline version of this file. Files that were
        # modified from HEAD at baseline time have a copy in the baseline
        # dir. Files that were clean at baseline time don't — use the
        # spawning worktree's copy as the reference (it had the same
        # content at creation time).
        if [ -f "$BASELINE_DIR/$file" ]; then
            baseline_file="$BASELINE_DIR/$file"
        elif [ -f "$REPO_ROOT/$file" ]; then
            baseline_file="$REPO_ROOT/$file"
        else
            continue
        fi

        # Skip if unchanged from baseline.
        if cmp -s "$baseline_file" "$AGENT_DIR/$file"; then
            continue
        fi

        # Check if the spawning worktree also changed this file since
        # we created the baseline (another agent merged changes).
        if [ -f "$REPO_ROOT/$file" ] && ! cmp -s "$baseline_file" "$REPO_ROOT/$file"; then
            # Both sides changed. Attempt 3-way merge.
            # git merge-file modifies the first file in place.
            # Args: ours base theirs — writes result into ours.
            cp "$REPO_ROOT/$file" "$REPO_ROOT/$file.ours"
            cp "$baseline_file" "$REPO_ROOT/$file.base"
            if git merge-file \
                "$REPO_ROOT/$file.ours" \
                "$REPO_ROOT/$file.base" \
                "$AGENT_DIR/$file" 2>/dev/null
            then
                mv "$REPO_ROOT/$file.ours" "$REPO_ROOT/$file"
                rm -f "$REPO_ROOT/$file.base"
                echo "M  $file (merged)"
            else
                mv "$REPO_ROOT/$file.ours" "$REPO_ROOT/$file"
                rm -f "$REPO_ROOT/$file.base"
                echo "C  $file (conflicts — resolve manually)"
                had_conflicts=1
            fi
        else
            # Only the agent changed it. Direct copy.
            cp "$AGENT_DIR/$file" "$REPO_ROOT/$file"
            echo "M  $file"
        fi
    done

    if [ "$had_conflicts" -ne 0 ]; then
        echo ""
        echo "Some files have merge conflicts. Resolve them before continuing."
        return 1
    fi
}

# -----------------------------------------------------------------------
# cleanup: remove worktree and baseline
# -----------------------------------------------------------------------
do_cleanup() {
    if [ -d "$AGENT_DIR" ]; then
        git worktree remove --force "$AGENT_DIR" 2>/dev/null || {
            # If git can't remove it cleanly, force delete.
            rm -rf "$AGENT_DIR"
            git worktree prune 2>/dev/null
        }
    fi

    rm -rf "$BASELINE_DIR"
    echo "cleaned up worktree '$NAME'"
}

# -----------------------------------------------------------------------
case "$ACTION" in
    create)  do_create  ;;
    collect) do_collect ;;
    cleanup) do_cleanup ;;
    *)       usage      ;;
esac
