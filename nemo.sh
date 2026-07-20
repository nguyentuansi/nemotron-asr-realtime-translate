#!/usr/bin/env bash
# nemo.sh — single entry point for every mode of the project.
#
# Usage:
#   ./nemo.sh                          # interactive menu
#   ./nemo.sh <command> [args...]      # direct dispatch
#   ./nemo.sh --help                   # list commands
#
# All subcommands ultimately shell out to the individual wrappers
# (stream_translate.sh, stream_web.sh, assistant.sh, ...) so anything you
# could pass to those, you can pass here after the subcommand.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_help() {
  cat <<'EOF'
Nemo — Vietnamese-first streaming ASR + real-time translation + voice assistant

Usage: ./nemo.sh <command> [args...]

Streaming (mic → ASR → translation):
  stream           Real-time ASR + translation, terminal UI
  web              Same pipeline, browser UI at http://127.0.0.1:8765
  smoke            One-shot smoke test on bundled audio (no mic)

Voice assistant ("Nemo ơi" wake word):
  assistant        Start the always-on assistant (needs a trained wake model)
  ptt              Push-to-talk assistant (no wake model needed — press ENTER)
  setup            First-run wizard (writes ~/.config/nemo-assistant.yaml)
  wake-train       Prepare / train the "Nemo ơi" wake-word model
    wake-train prepare   Generate ~500 Piper-synthesized positives
    wake-train train     Invoke openWakeWord training on your data

Diagnostics / bench:
  mic-test [SECS]  Record N seconds of mic, transcribe, print (default 8s)
  bench rtf        Measure ASR real-time factor on audio/
  bench wake       Measure wake-word FAR/FRR on labeled audio
  regress          Run demo/simulate_assistant.py against a JSON script

Meta:
  menu             Show the interactive picker (default when no args)
  help, --help     This message

Every subcommand forwards any extra flags to the underlying script, e.g.:
  ./nemo.sh stream --translator nllb --lang vi-VN
  ./nemo.sh assistant --no-tts --wake-only
EOF
}

_menu() {
  cat <<'EOF'

Nemo — pick a mode:

  Streaming
    1) stream         Terminal ASR + translation
    2) web            Web UI (browser)
    3) smoke          Smoke test on bundled audio

  Assistant
    4) assistant      Always-on "Nemo ơi" (needs trained wake model)
    5) ptt            Push-to-talk (no wake model needed — press ENTER)
    6) setup          First-run config wizard
    7) wake-train     Prepare / train wake-word model

  Diagnostics
    8) mic-test       Record + transcribe test
    9) bench rtf      ASR speed benchmark
   10) bench wake     Wake word FAR/FRR benchmark
   11) regress        End-to-end regression from wavs

    q) quit

EOF
  # `read -p` doesn't work everywhere; prompt-then-read is portable.
  printf "Choice: "
  read -r choice
  case "$choice" in
    1)  exec "$HERE/nemo.sh" stream ;;
    2)  exec "$HERE/nemo.sh" web ;;
    3)  exec "$HERE/nemo.sh" smoke ;;
    4)  exec "$HERE/nemo.sh" assistant ;;
    5)  exec "$HERE/nemo.sh" ptt ;;
    6)  exec "$HERE/nemo.sh" setup ;;
    7)  exec "$HERE/nemo.sh" wake-train prepare ;;
    8)  exec "$HERE/nemo.sh" mic-test ;;
    9)  exec "$HERE/nemo.sh" bench rtf ;;
    10) exec "$HERE/nemo.sh" bench wake ;;
    11) exec "$HERE/nemo.sh" regress ;;
    q|Q|quit|exit) exit 0 ;;
    *) echo "Unknown choice: $choice"; exit 1 ;;
  esac
}

# --------------------------------------------------------------------
# Bootstrap on demand — only for subcommands that actually need Python.
# `--help` and the interactive menu don't need it, so we skip. The
# individual sub-wrappers (stream_translate.sh, assistant.sh, ...) source
# _bootstrap.sh themselves, so we don't need to when exec'ing into them.
# We only source here for the "run this .py directly" cases below.
# --------------------------------------------------------------------
_bootstrap() { source "$HERE/_bootstrap.sh"; }

cmd="${1:-menu}"
[[ $# -gt 0 ]] && shift || true

case "$cmd" in
  # ---------- streaming ----------
  stream|stream-translate)
    exec "$HERE/stream_translate.sh" "$@" ;;

  web|stream-web)
    exec "$HERE/stream_web.sh" "$@" ;;

  smoke|smoke-test)
    # smoke_test.py loads the model and transcribes audio/sample*.flac
    _bootstrap
    exec "$HERE/.venv/bin/python" "$HERE/smoke_test.py" "$@" ;;

  # ---------- assistant ----------
  assistant)
    exec "$HERE/assistant.sh" "$@" ;;

  ptt|push-to-talk)
    # Push-to-talk mode: no wake model needed, just press ENTER to start.
    exec "$HERE/assistant.sh" --no-wake "$@" ;;

  setup|assistant-setup)
    exec "$HERE/assistant.sh" --setup ;;

  wake-train)
    # Sub-subcommand: prepare | train
    subcmd="${1:-prepare}"
    [[ $# -gt 0 ]] && shift || true
    case "$subcmd" in
      prepare|train)
        _bootstrap
        exec "$HERE/.venv/bin/python" "$HERE/scripts/train_wake_model.py" "$subcmd" "$@" ;;
      *)
        echo "wake-train: unknown sub-command '$subcmd' (expected: prepare | train)"
        exit 1 ;;
    esac ;;

  # ---------- diagnostics / bench ----------
  mic-test)
    # Default 8 seconds if not specified.
    secs="${1:-8}"
    [[ $# -gt 0 ]] && shift || true
    exec "$HERE/mic_to_asr_test.sh" --secs "$secs" "$@" ;;

  bench)
    subcmd="${1:-rtf}"
    [[ $# -gt 0 ]] && shift || true
    case "$subcmd" in
      rtf)
        _bootstrap
        exec "$HERE/.venv/bin/python" "$HERE/bench/measure_rtf.py" "$@" ;;
      wake|wake-far-frr)
        _bootstrap
        exec "$HERE/.venv/bin/python" "$HERE/bench/wake_far_frr.py" "$@" ;;
      *)
        echo "bench: unknown sub-command '$subcmd' (expected: rtf | wake)"
        exit 1 ;;
    esac ;;

  regress|regression|simulate)
    _bootstrap
    exec "$HERE/.venv/bin/python" "$HERE/demo/simulate_assistant.py" "$@" ;;

  # ---------- meta ----------
  menu)
    _menu ;;

  help|--help|-h)
    _help ;;

  *)
    echo "Unknown command: $cmd"
    echo "Run './nemo.sh --help' for a list of commands."
    exit 1 ;;
esac
