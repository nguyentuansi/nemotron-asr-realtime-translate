#!/usr/bin/env bash
# Launch the Nemo ơi voice assistant with the project's venv python.
# Same pattern as stream_translate.sh: source _bootstrap.sh (which auto-creates
# .venv and installs requirements.txt on first run), then exec assistant.py
# with any user-passed flags.
#
# Special flag: --setup routes to the config wizard (assistant_setup.py) instead
# of the main assistant loop. Everything else goes to assistant.py directly.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_bootstrap.sh"

if [[ "$1" == "--setup" ]]; then
  exec "$HERE/.venv/bin/python" "$HERE/assistant_setup.py"
fi

exec "$HERE/.venv/bin/python" "$HERE/assistant.py" "$@"
