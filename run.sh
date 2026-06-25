#!/usr/bin/env bash
# Run the smoke test with the project's venv python.
# We call .venv/bin/python directly rather than `source activate` so a stale
# PATH from the parent shell can't shadow the venv interpreter.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_bootstrap.sh"
exec "$HERE/.venv/bin/python" "$HERE/smoke_test.py" "$@"
