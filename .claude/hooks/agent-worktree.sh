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
# agent's changes — no commits needed. For files that were clean at
# create time, the baseline is recovered at collect time from the base
# commit via `git show $BASE_SHA:$file`, so the fallback does not depend
# on the spawning worktree's current state (which other agents or the
# main session may have changed concurrently).

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

    mkdir -p "$BASELINE_DIR"

    # Record the base commit SHA. Collect uses this to recover the
    # baseline for files that were clean at create time.
    git -C "$REPO_ROOT" rev-parse HEAD > "$BASELINE_DIR/.base-sha"

    # --- Copy uncommitted state from spawning worktree ---

    # Staged + unstaged changes. Prefer strict apply; fall back to
    # --3way (which can merge around small offsets by consulting the
    # blob objects). Loudly report any failure so a silent partial
    # state cannot mask missing work.
    diff_output="$(git diff HEAD)" || true
    if [ -n "$diff_output" ]; then
        if ! echo "$diff_output" | (cd "$AGENT_DIR" && git apply --allow-empty 2>/dev/null); then
            if ! echo "$diff_output" | (cd "$AGENT_DIR" && git apply --allow-empty --3way >&2); then
                echo "error: could not apply spawning tree's uncommitted diff; agent runs on HEAD only" >&2
            else
                echo "warning: spawning diff applied via --3way; worktree may contain conflict markers" >&2
            fi
        fi
    fi

    # Untracked files (excluding the worktrees directory itself).
    while IFS= read -r -d '' file; do
        dir="$(dirname "$file")"
        mkdir -p "$AGENT_DIR/$dir"
        cp "$file" "$AGENT_DIR/$file"
    done < <(git ls-files --others --exclude-standard -z | grep -zv '^\.claude/worktrees/' || true)

    # --- Snapshot baseline for diffing later ---
    #
    # Copy every file that differs from HEAD into the baseline directory.
    # After the agent runs, `git diff --no-index baseline/ worktree/`
    # gives us exactly the agent's changes.

    # Modified/added tracked files.
    while read -r file; do
        if [ -f "$AGENT_DIR/$file" ]; then
            dir="$(dirname "$file")"
            mkdir -p "$BASELINE_DIR/$dir"
            cp "$AGENT_DIR/$file" "$BASELINE_DIR/$file"
        fi
    done < <(cd "$AGENT_DIR" && git diff --name-only HEAD)

    # Untracked files.
    while read -r file; do
        if [ -f "$AGENT_DIR/$file" ]; then
            dir="$(dirname "$file")"
            mkdir -p "$BASELINE_DIR/$dir"
            cp "$AGENT_DIR/$file" "$BASELINE_DIR/$file"
        fi
    done < <(cd "$AGENT_DIR" && git ls-files --others --exclude-standard)

    # Save the list of all files at baseline for deletion detection.
    # Filter by filesystem existence — `git ls-files` alone reports files
    # still in the index even if deleted from the working tree, which would
    # suppress deletion detection at collect time.
    while IFS= read -r file; do
        [ -n "$file" ] && [ -e "$AGENT_DIR/$file" ] && printf '%s\n' "$file"
    done < <(cd "$AGENT_DIR" && git ls-files && git ls-files --others --exclude-standard) | \
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
    local had_errors=0
    local file dir

    # Load the base commit SHA for clean-at-baseline file lookups.
    local base_sha=""
    if [ -f "$BASELINE_DIR/.base-sha" ]; then
        base_sha="$(cat "$BASELINE_DIR/.base-sha")"
    fi

    # Produce the baseline content for $file into $out_path. Returns 0
    # on success, 1 if no baseline is available (e.g., agent-created file).
    get_baseline() {
        local _file="$1"
        local _out="$2"
        if [ -f "$BASELINE_DIR/$_file" ]; then
            cp "$BASELINE_DIR/$_file" "$_out"
            return 0
        fi
        if [ -n "$base_sha" ]; then
            if git -C "$REPO_ROOT" show "$base_sha:$_file" > "$_out" 2>/dev/null; then
                return 0
            fi
        fi
        return 1
    }

    local tmp_base tmp_ours
    tmp_base="$(mktemp)"
    tmp_ours="$(mktemp)"

    # Collect the set of files currently present in the agent worktree.
    # Filter by filesystem existence so that files deleted from the
    # working tree (but still listed by git ls-files via the index)
    # correctly fall out of the final set and are detected as deletions.
    while IFS= read -r file; do
        [ -n "$file" ] && [ -e "$AGENT_DIR/$file" ] && printf '%s\n' "$file"
    done < <(cd "$AGENT_DIR" && git ls-files && git ls-files --others --exclude-standard) | \
        sort -u > "$BASELINE_DIR/.filelist-final"

    # --- Deletions (in baseline but not in final) ---
    while IFS= read -r file; do
        [ -z "$file" ] && continue
        [ -f "$REPO_ROOT/$file" ] || continue

        # If the spawner has modified the file since baseline while the
        # agent deleted it, don't silently drop the spawner's work.
        if get_baseline "$file" "$tmp_base" && ! cmp -s "$tmp_base" "$REPO_ROOT/$file"; then
            echo "C  $file (agent deleted, spawner modified — keeping spawner version)" >&2
            had_conflicts=1
            continue
        fi

        if rm "$REPO_ROOT/$file" 2>/dev/null; then
            echo "D  $file"
        else
            echo "error: failed to delete $file" >&2
            had_errors=1
        fi
    done < <(grep -Fxv -f "$BASELINE_DIR/.filelist-final" "$BASELINE_DIR/.filelist" 2>/dev/null || true)

    # --- Creations (in final but not in baseline) ---
    while IFS= read -r file; do
        [ -z "$file" ] && continue
        [ -f "$AGENT_DIR/$file" ] || continue

        # If the spawner also created this file (e.g., a concurrent agent),
        # don't overwrite — flag as a conflict unless contents match.
        if [ -f "$REPO_ROOT/$file" ]; then
            if cmp -s "$AGENT_DIR/$file" "$REPO_ROOT/$file"; then
                continue
            fi
            echo "C  $file (created in both agent and spawner)" >&2
            had_conflicts=1
            continue
        fi

        dir="$(dirname "$file")"
        if mkdir -p "$REPO_ROOT/$dir" && cp "$AGENT_DIR/$file" "$REPO_ROOT/$file"; then
            echo "A  $file"
        else
            echo "error: failed to create $file" >&2
            had_errors=1
        fi
    done < <(grep -Fxv -f "$BASELINE_DIR/.filelist" "$BASELINE_DIR/.filelist-final" 2>/dev/null || true)

    # --- Intersection — files present in both baseline and final ---
    while IFS= read -r file; do
        [ -z "$file" ] && continue
        [ -f "$AGENT_DIR/$file" ] || continue

        if ! get_baseline "$file" "$tmp_base"; then
            # No baseline available. Only propagate if the agent's version
            # genuinely differs from what's currently in the repo.
            if [ -f "$REPO_ROOT/$file" ] && ! cmp -s "$AGENT_DIR/$file" "$REPO_ROOT/$file"; then
                if cp "$AGENT_DIR/$file" "$REPO_ROOT/$file"; then
                    echo "M  $file (no baseline)"
                else
                    echo "error: failed to copy $file" >&2
                    had_errors=1
                fi
            fi
            continue
        fi

        # Agent didn't touch this file — nothing to do.
        if cmp -s "$tmp_base" "$AGENT_DIR/$file"; then
            continue
        fi

        if [ -f "$REPO_ROOT/$file" ] && ! cmp -s "$tmp_base" "$REPO_ROOT/$file"; then
            # Both agent and repo diverged from baseline — 3-way merge.
            if ! cp "$REPO_ROOT/$file" "$tmp_ours"; then
                echo "error: failed to stage $file for merge" >&2
                had_errors=1
                continue
            fi
            if git merge-file "$tmp_ours" "$tmp_base" "$AGENT_DIR/$file" 2>/dev/null; then
                if cp "$tmp_ours" "$REPO_ROOT/$file"; then
                    echo "M  $file (merged)"
                else
                    echo "error: failed to write merged $file" >&2
                    had_errors=1
                fi
            else
                # git merge-file returns >0 when conflicts remain; $tmp_ours
                # now contains the file with conflict markers. Write it so
                # the user can resolve in place.
                if cp "$tmp_ours" "$REPO_ROOT/$file"; then
                    echo "C  $file (conflicts — resolve manually)"
                    had_conflicts=1
                else
                    echo "error: failed to write conflicted $file" >&2
                    had_errors=1
                fi
            fi
        else
            # Only the agent changed it relative to baseline. Direct copy.
            if cp "$AGENT_DIR/$file" "$REPO_ROOT/$file"; then
                echo "M  $file"
            else
                echo "error: failed to copy $file" >&2
                had_errors=1
            fi
        fi
    done < <(grep -Fx -f "$BASELINE_DIR/.filelist" "$BASELINE_DIR/.filelist-final" 2>/dev/null || true)

    rm -f "$tmp_base" "$tmp_ours"

    if [ "$had_conflicts" -ne 0 ]; then
        echo "" >&2
        echo "Some files have merge conflicts. Resolve them before continuing." >&2
    fi
    if [ "$had_errors" -ne 0 ]; then
        echo "Some operations failed during collect; worktree preserved for inspection." >&2
    fi
    if [ "$had_conflicts" -ne 0 ] || [ "$had_errors" -ne 0 ]; then
        return 1
    fi
    return 0
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
