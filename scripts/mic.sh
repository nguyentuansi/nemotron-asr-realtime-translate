#!/usr/bin/env bash
# Launch the interactive mic test with the project's venv python.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$ROOT/_bootstrap.sh"
exec "$ROOT/.venv/bin/python" "$HERE/mic_test.py" "$@"
