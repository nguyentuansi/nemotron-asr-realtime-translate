#!/usr/bin/env bash
# Launch the interactive mic test with the project's venv python.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$HERE/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export VIRTUAL_ENV="$HERE/.venv"
exec "$HERE/.venv/bin/python" "$HERE/mic_test.py" "$@"
