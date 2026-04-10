#!/usr/bin/env bash
#
# Version check hook for claude-workflow managed projects.
#
# Compares the locally stamped framework version against the upstream
# repository on GitHub. Warns if the local copy is behind.
#
# Designed to run as a PreToolUse hook. Caches the result daily in /tmp
# so only the first invocation per day per project does a network call.

set -euo pipefail

CLAUDE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="$CLAUDE_DIR/.workflow-version"

# Nothing to check if no version stamp exists.
if [[ ! -f "$VERSION_FILE" ]]; then
    exit 0
fi

# Read stamped version.
STAMPED_REPO=""
STAMPED_SHA=""
while IFS='=' read -r key value; do
    case "$key" in
        repo) STAMPED_REPO="$value" ;;
        sha)  STAMPED_SHA="$value" ;;
    esac
done < "$VERSION_FILE"

if [[ -z "$STAMPED_REPO" || -z "$STAMPED_SHA" ]]; then
    exit 0
fi

# Cache key: project directory hash + date. One check per project per day.
PROJECT_HASH=$(echo "$CLAUDE_DIR" | md5sum | cut -d' ' -f1)
CACHE_FILE="/tmp/.claude-workflow-version-${PROJECT_HASH}-$(date +%Y%m%d)"

if [[ -f "$CACHE_FILE" ]]; then
    # Replay cached warning if any.
    CACHED=$(cat "$CACHE_FILE")
    if [[ -n "$CACHED" ]]; then
        echo "$CACHED" >&2
    fi
    exit 0
fi

# Extract owner/repo from the remote URL.
OWNER_REPO=""
if [[ "$STAMPED_REPO" =~ github\.com[:/]([^/]+/[^/.]+) ]]; then
    OWNER_REPO="${BASH_REMATCH[1]}"
fi

if [[ -z "$OWNER_REPO" ]]; then
    touch "$CACHE_FILE"
    exit 0
fi

# Fetch latest SHA from GitHub API (timeout 5s, fail silently).
UPSTREAM_SHA=$(curl -sf --max-time 5 \
    "https://api.github.com/repos/${OWNER_REPO}/commits/main" \
    -H "Accept: application/vnd.github.sha" 2>/dev/null) || true

if [[ -z "$UPSTREAM_SHA" ]]; then
    # Network failure — skip silently, don't cache so we retry tomorrow.
    touch "$CACHE_FILE"
    exit 0
fi

if [[ "$UPSTREAM_SHA" != "$STAMPED_SHA" ]]; then
    WARNING="[claude-workflow] Framework has updates available (local: ${STAMPED_SHA:0:8}, upstream: ${UPSTREAM_SHA:0:8}). Run setup.sh assemble from the framework repo to update."
    echo "$WARNING" > "$CACHE_FILE"
    echo "$WARNING" >&2
else
    # Up to date — cache empty result.
    touch "$CACHE_FILE"
fi

exit 0
