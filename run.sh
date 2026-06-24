#!/usr/bin/env bash
# Run the smoke test with the project's venv python.
# We call .venv/bin/python directly rather than `source activate` so a stale
# PATH from the parent shell can't shadow the venv interpreter.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$HERE/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export VIRTUAL_ENV="$HERE/.venv"
exec "$HERE/.venv/bin/python" "$HERE/smoke_test.py" "$@"
