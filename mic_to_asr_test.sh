#!/usr/bin/env bash
# Record mic via the shim, transcribe with the production ASR. Diagnoses
# whether mic capture or the streaming pipeline is at fault.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_bootstrap.sh"
exec "$HERE/.venv/bin/python" "$HERE/mic_to_asr_test.py" "$@"
