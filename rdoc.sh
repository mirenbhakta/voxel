#!/usr/bin/env bash
# Launch eden-renderer under RenderDoc's command-line capture tool.
#
# Usage:
#   ./rdoc.sh [path-to-binary] [-- game-args...]
#
# The first positional is treated as the binary path only if it does NOT
# start with `-`; otherwise the default binary is used and all args are
# passed through. Everything after a `--` sentinel is always passed
# through, even if the first token looks like a path.
#
# Examples:
#   ./rdoc.sh
#   ./rdoc.sh target/release/game
#   ./rdoc.sh --pos -119.27,9.36,-53.17 --yaw 81.6 --pitch -46.9
#   ./rdoc.sh target/release/game -- --pos 0,0,0 --yaw 45
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="target/debug/game"

# Optional binary override: first positional iff it doesn't start with
# `-`. Anything that does start with `-` (or a leading `--` sentinel)
# falls through to the game.
if [[ $# -gt 0 && "$1" != -* ]]; then
    BIN="$1"
    shift
fi
# Strip a leading `--` sentinel so users can force "skip the binary slot"
# with `./rdoc.sh -- --pos ...`.
if [[ $# -gt 0 && "$1" == "--" ]]; then
    shift
fi

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
echo "game args: ${*:-(none)}"
echo "---"

# Launch under renderdoccmd so we get full terminal output.
#
# renderdoccmd's CLI parser does NOT recognise `--` as an end-of-options
# sentinel. `renderdoccmd capture [options] <binary> [binary-args...]`
# expects renderdoccmd's own options first, then the binary path, then
# any program arguments. Everything positional after `<binary>` is handed
# to the launched process verbatim. So we emit `"$BIN" "$@"` with no
# separator and rely on renderdoccmd treating the positional tail as the
# program's argv.
exec renderdoccmd capture \
    --working-dir "$SCRIPT_DIR" \
    --opt-api-validation \
    "$BIN" "$@"
