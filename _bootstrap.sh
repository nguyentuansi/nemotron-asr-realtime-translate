# Sourced by run.sh / mic.sh / stream.sh / stream_translate.sh / stream_web.sh.
# Ensures .venv exists (creates it from requirements.txt on first run) and
# exports HF_HOME / VIRTUAL_ENV. Wrappers set HERE; if invoked directly we
# fall back to the directory containing this script (BASH_SOURCE works whether
# sourced or executed).
set -e
: "${HERE:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export HF_HOME="$HERE/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export VIRTUAL_ENV="$HERE/.venv"

if [[ ! -x "$HERE/.venv/bin/python" ]]; then
  # Need Python >= 3.11.4 (NeMo's tar_utils uses the tarfile data-filter
  # kwarg added by PEP 706, which earlier 3.11.x lacks). Try 3.13 first,
  # then 3.12, then 3.11. Override with PYTHON=/path/to/python.
  PY="${PYTHON:-}"
  if [[ -z "$PY" ]]; then
    for candidate in python3.13 python3.12 python3.11 python3; do
      if command -v "$candidate" >/dev/null 2>&1; then
        PY="$candidate"; break
      fi
    done
  fi
  if [[ -z "$PY" ]] || ! command -v "$PY" >/dev/null 2>&1; then
    echo "[setup] No suitable python interpreter found. Install python3.13 (brew install python@3.13) or set PYTHON=/path/to/python." >&2
    exit 1
  fi
  if ! "$PY" -c "import sys; sys.exit(0 if sys.version_info >= (3,11,4) else 1)"; then
    echo "[setup] $PY is $($PY --version) — need >= 3.11.4 for NeMo's tarfile data-filter." >&2
    echo "[setup] Install a newer interpreter (brew install python@3.13) or set PYTHON=..." >&2
    exit 1
  fi
  echo "[setup] Creating .venv with $($PY --version) (one-time, ~6 GB)..."
  "$PY" -m venv "$HERE/.venv"
  "$HERE/.venv/bin/pip" install --upgrade pip
  "$HERE/.venv/bin/pip" install -r "$HERE/requirements.txt"
  echo "[setup] .venv ready."
fi
