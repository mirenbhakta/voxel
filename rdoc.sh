#!/usr/bin/env bash
# Launch eden-renderer under RenderDoc's command-line capture tool.
# Usage: ./rdoc.sh [path-to-binary]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${1:-target/debug/game}"

# Resolve to absolute path if relative.
if [[ "$BIN" != /* ]]; then
    BIN="$SCRIPT_DIR/$BIN"
fi

if [[ ! -f "$BIN" ]]; then
    echo "error: binary not found: $BIN" >&2
    exit 1
fi

echo "binary:    $BIN"
echo "workdir:   $SCRIPT_DIR"
echo "rpath:     $(readelf -d "$BIN" 2>/dev/null | grep -i 'rpath\|runpath' || echo '(none)')"
echo "---"

# Launch under renderdoccmd so we get full terminal output.
exec renderdoccmd capture \
    --working-dir "$SCRIPT_DIR" \
    --opt-api-validation \
    "$BIN"
