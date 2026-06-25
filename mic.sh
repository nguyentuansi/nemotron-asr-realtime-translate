#!/usr/bin/env bash
# Launch the interactive mic test with the project's venv python.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_bootstrap.sh"
exec "$HERE/.venv/bin/python" "$HERE/mic_test.py" "$@"
