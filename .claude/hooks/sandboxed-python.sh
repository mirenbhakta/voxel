#!/usr/bin/env bash
#
# Sandboxed python3: read-only filesystem access with writable /tmp.
#
# Uses bubblewrap (bwrap) to restrict python to read-only access on the
# entire filesystem. Only /tmp is writable (for python's own tempfiles).
#
# Falls back to agentic-python if available, then raw python3 if bwrap
# is not installed (which will require normal permission approval).
#
# Usage:
#   sandboxed-python.sh -c "print('hello')"
#   sandboxed-python.sh script.py

set -euo pipefail

# Prefer agentic-python if available (already sandboxed).
if command -v agentic-python &>/dev/null; then
    exec agentic-python "$@"
fi

# Use bwrap if available.
if command -v bwrap &>/dev/null; then
    exec bwrap \
        --ro-bind / / \
        --bind /tmp /tmp \
        --dev /dev \
        --proc /proc \
        --unshare-net \
        --die-with-parent \
        python3 "$@"
fi

# No sandbox available. Refuse to run unsandboxed — the auto-approve
# hook trusts this script to provide isolation. Use python3 directly
# (which will go through normal permission approval).
echo "error: no sandbox available (install bubblewrap or agentic-python)" >&2
exit 1
