"""assistant.py — main event loop for the Nemo ơi voice assistant.

See docs/assistant/00-build-story.md Chapter 6 for design notes.

State machine:

    IDLE ──wake fires──▶ CAPTURING ──silence VAD commits──▶ THINKING
                                                                 │
                                                            skill returns
                                                                 ▼
    ┌──────────────────────────────────────── SPEAKING ◀─────────┘
    │                                             │
    │                                        TTS done
    │                                             ▼
    └────────────────────────────────────────  IDLE

Single-threaded for the core: mic ingress, wake gate, ASR loop, intent
routing all live on the main thread. TTS runs on a background thread so
we can honor barge-in during long responses.

CLI:
    ./assistant.sh                 # start assistant (terminal)
    ./assistant.sh --lang en-US    # command language (default vi-VN)
    ./assistant.sh --no-tts        # print responses instead of speaking
    ./assistant.sh --wake-only     # test WakeGate; skip ASR + skills
    ./assistant.sh --setup         # first-run config wizard (deferred to setup.py)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# NeMo + OneLogger noise suppression, mirroring stream_translate.py's pattern.
if os.environ.get("NEMO_VERBOSE") != "1":
    logging.getLogger("nemo_logger").setLevel(logging.WARNING)
    logging.getLogger("nv_one_logger").setLevel(logging.ERROR)

# Suppress transitive-dep noise that clutters the debug log without helping
# debugging (matplotlib/PIL come in via acoustics; urllib3/asyncio at DEBUG
# are just internal HTTP + event-loop chatter).
for _noisy in ("matplotlib", "PIL", "urllib3", "asyncio", "fsspec", "filelock",
               "h5py", "numba"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import numpy as np

# --- Reuse from the streaming pipeline ---
# We import a handful of primitives from stream_translate.py rather than
# duplicating: MicProducer (mic capture), load_asr (with ONNX/MPS handling),
# SR/CH/FMT constants, LANG_TAG_RE, SENTENCE_END, ATT_CONTEXT.
from stream_translate import (
    MicProducer,
    load_asr,
    strip_lang_tags,
    LANG_TAG_RE,
    SR,
    ATT_CONTEXT,
)

# --- New assistant modules ---
from wake_gate import WakeGate, WakeEvent
from intent_router import IntentRouter, strip_wake_phrase
from tts_speaker import TTSSpeaker

# --- Skills ---
from skills import time_skill, translate_skill, alarm_skill, home_assistant_skill, help_skill


LOG = logging.getLogger("assistant")


# --- Config: model paths and thresholds ---

WAKE_MODEL = HERE / "models" / "wake" / "nemo_oi.onnx"
PIPER_VOICE = HERE / "models" / "piper" / "vi_VN-vais1000-medium.onnx"
WAKE_THRESHOLD = 0.55
# Silence VAD reads `producer.peak()` (smoothed peak amplitude). Threshold is
# LOWER than stream_translate's default (0.15) because assistant commands
# have different needs than dictation:
#   - dictation tolerates cutting one long utterance into commit chunks;
#     0.15 catches natural pauses between sentences and moves on
#   - commands need to survive brief pauses mid-phrase ("thời tiết ... hôm nay")
#     and also work when the user speaks softly (peak 0.05-0.10 range)
# 0.05 catches quiet speech as speech, only true silence (background noise
# only) triggers silence-run.
SILENCE_THRESHOLD = 0.05
# STREAMING chunks are 560 ms each. 3 chunks = 1.68 s of silence required
# before commit. Longer than dictation's 2 because we're more forgiving of
# mid-command pauses ("Nemo ơi... đặt báo thức... sáu giờ sáng").
SILENCE_CHUNKS_TO_COMMIT = 3
MAX_COMMAND_SECS = 8.0                  # hard cap on command length


# --- Skill registration ---

def register_skills(router: IntentRouter) -> None:
    """Wire the v0 skills into the router.

    Order matters (see intent_router.py): specific patterns first, catch-alls
    last. We interleave TIME and DATE variants together since they share a
    single skill module but different `kind` slots.
    """
    # Help / intro — registered FIRST so it wins over the more general
    # patterns below. "Bạn có thể làm gì" is the most natural first question
    # a new user asks; answering it well matters.
    #
    # Match any short sentence containing "làm gì" / "giúp gì" / "được gì",
    # plus the standalone commands "trợ giúp" / "help" / "danh sách lệnh".
    # We use search() semantics (no ^ anchor) so both "Bạn có thể làm gì?" and
    # "Mình muốn biết bạn làm được gì" route here.
    router.register_skill(
        "help",
        re.compile(
            r"\b(làm|giúp|làm được|giúp được|làm việc)\s+(gì|được gì)\b"
            r"|^\s*trợ giúp\s*[?.!]*\s*$"
            r"|^\s*help\s*[?.!]*\s*$"
            r"|^\s*danh sách lệnh\s*[?.!]*\s*$",
            re.IGNORECASE,
        ),
        help_skill.handle,
    )

    # Time patterns
    router.register_skill(
        "time",
        re.compile(r"^\s*(bây giờ (là )?)?mấy giờ", re.IGNORECASE),
        _bind(time_skill.handle, kind="time"),
    )
    router.register_skill(
        "time",
        re.compile(r"^\s*hôm nay là thứ mấy", re.IGNORECASE),
        _bind(time_skill.handle, kind="weekday"),
    )
    router.register_skill(
        "time",
        re.compile(r"^\s*hôm nay là ngày (mấy|bao nhiêu)", re.IGNORECASE),
        _bind(time_skill.handle, kind="day"),
    )
    router.register_skill(
        "time",
        re.compile(r"^\s*tháng (mấy|bao nhiêu)", re.IGNORECASE),
        _bind(time_skill.handle, kind="month"),
    )

    # Alarms & timers. The named-group `time_spec` captures everything after
    # the verb; the skill's parser sorts out clock vs relative form.
    router.register_skill(
        "alarm_set_clock",
        re.compile(r"^\s*(đặt |cài )?báo thức (?P<time_spec>.+)", re.IGNORECASE),
        _bind(alarm_skill.handle, op="set", kind="clock"),
    )
    router.register_skill(
        "alarm_set_timer",
        re.compile(r"^\s*hẹn giờ (?P<time_spec>.+)", re.IGNORECASE),
        _bind(alarm_skill.handle, op="set", kind="timer"),
    )
    router.register_skill(
        "alarm_cancel",
        re.compile(r"^\s*(hủy|huỷ|xóa|xoá) (báo thức|hẹn giờ)", re.IGNORECASE),
        _bind(alarm_skill.handle, op="cancel"),
    )
    router.register_skill(
        "alarm_count",
        re.compile(r"^\s*(còn|có) .*báo thức", re.IGNORECASE),
        _bind(alarm_skill.handle, op="count"),
    )

    # Translate. Two shapes: "dịch sang tiếng X: body" and "dịch body sang tiếng X".
    # Non-greedy capture of body so we don't eat trailing markers.
    router.register_skill(
        "translate_target_first",
        re.compile(
            r"^\s*dịch\s+(sang\s+)?(?P<target_name>tiếng [^:]+?)\s*[:,]\s*(?P<body>.+)",
            re.IGNORECASE,
        ),
        translate_skill.handle,
    )
    router.register_skill(
        "translate_body_first",
        re.compile(
            r"^\s*dịch\s+(?P<body>.+?)\s+sang\s+(?P<target_name>tiếng .+)",
            re.IGNORECASE,
        ),
        translate_skill.handle,
    )

    # Home Assistant.
    router.register_skill(
        "ha_turn_on",
        re.compile(r"^\s*bật (?P<alias>.+)", re.IGNORECASE),
        _bind(home_assistant_skill.handle, op="turn_on"),
    )
    router.register_skill(
        "ha_turn_off",
        re.compile(r"^\s*tắt (?P<alias>.+)", re.IGNORECASE),
        _bind(home_assistant_skill.handle, op="turn_off"),
    )
    router.register_skill(
        "ha_activate",
        re.compile(r"^\s*kích hoạt (cảnh )?(?P<alias>.+)", re.IGNORECASE),
        _bind(home_assistant_skill.handle, op="activate"),
    )
    router.register_skill(
        "ha_state",
        re.compile(r"^\s*(trạng thái|tình trạng) (?P<alias>.+)", re.IGNORECASE),
        _bind(home_assistant_skill.handle, op="state"),
    )


def _bind(handler, **defaults):
    """Return a wrapped handler that merges `defaults` into slots before calling.

    Lets us register the same skill.handle() function under multiple patterns,
    each carrying a different fixed slot (e.g. `kind="time"` vs `kind="weekday"`).
    Regex-captured slots take precedence over defaults on key collision.
    """
    def wrapped(slots: dict) -> str:
        merged = {**defaults, **slots}
        return handler(merged)
    return wrapped


# --- Streaming ASR helpers ---

def _rms(chunk: np.ndarray) -> float:
    """Root-mean-square amplitude of a chunk. Not used in the main VAD path
    (we read producer.peak() there, which is a smoothed peak with decay), but
    kept as a helper in case a future test wants a per-chunk energy read.
    """
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def _capture_command(
    model,
    producer: MicProducer,
    pre_roll: np.ndarray,
    chunk_samples: int,
    att_ctx: list[int],
) -> str:
    """Run streaming ASR from wake fire until silence-VAD commit.

    Reuses the exact same code paths stream_translate.py's inner loop uses:
    the model's `conformer_stream_step` interface and NeMo's
    `CacheAwareStreamingAudioBuffer`. The differences vs the streaming
    pipeline are:
      - We stop after ONE commit (a command is one utterance).
      - We include pre-roll audio from before the wake event.
      - We ignore the translator worker — that's not part of the assistant flow.

    Returns the committed Vietnamese text with wake-phrase and lang-tags stripped.
    """
    import torch
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    # Fresh cache state per command — no cross-command context leaking.
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    cache_lc, cache_lt, cache_lcl = model.encoder.get_initial_cache_state(
        batch_size=1, device=dev,
    )
    buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=True)
    cfg = model.encoder.streaming_cfg

    prev_hyp = None
    pred_out = None
    committed = ""
    silence_run = 0
    step = 0
    start_ts = time.time()

    def _push_and_step(chunk_audio_raw: np.ndarray) -> tuple[bool, str]:
        """Push audio, run any full-chunks through the model. Return (committed?, text)."""
        nonlocal cache_lc, cache_lt, cache_lcl, prev_hyp, pred_out, step
        sid = -1 if buf.buffer is None else 0
        buf.append_audio(chunk_audio_raw, stream_id=sid)
        latest_text = ""
        while True:
            if buf.buffer is None:
                break
            req = cfg.chunk_size[0] if step == 0 else cfg.chunk_size[1]
            if buf.buffer.size(-1) - buf.buffer_idx < req:
                break
            try:
                ca, cl = next(iter(buf))
            except StopIteration:
                break
            drop = cfg.drop_extra_pre_encoded if step != 0 else 0
            if ca.dtype != dtype:
                ca = ca.to(dtype)
            with torch.inference_mode():
                out = model.conformer_stream_step(
                    processed_signal=ca,
                    processed_signal_length=cl,
                    cache_last_channel=cache_lc,
                    cache_last_time=cache_lt,
                    cache_last_channel_len=cache_lcl,
                    keep_all_outputs=False,
                    previous_hypotheses=prev_hyp,
                    previous_pred_out=pred_out,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
            pred_out, transcribed, cache_lc, cache_lt, cache_lcl, prev_hyp = out
            step += 1
            hyp = transcribed[0] if transcribed else None
            if hyp is not None:
                latest_text = getattr(hyp, "text", "") or ""
        return False, latest_text

    # Feed pre-roll first (a single append; the streaming buffer will re-chunk).
    if pre_roll.size:
        _, _ = _push_and_step(pre_roll.astype(np.float32))

    # Then keep pulling live mic chunks until commit or timeout.
    latest_raw = ""
    while True:
        if time.time() - start_ts > MAX_COMMAND_SECS:
            LOG.info("command timed out at %.1fs, committing what we have",
                     MAX_COMMAND_SECS)
            break

        chunk = producer.take(chunk_samples)
        if chunk is None:
            time.sleep(0.02)
            continue

        _, txt = _push_and_step(chunk)
        if txt:
            latest_raw = txt

        # Silence VAD — read the MicProducer's smoothed peak (same signal
        # stream_translate.py uses). Reading a per-chunk RMS instead is a
        # 3-5× different scale and would flag all speech as silence.
        peak = producer.peak()
        if peak < SILENCE_THRESHOLD:
            silence_run += 1
        else:
            silence_run = 0

        LOG.debug("cap step=%d peak=%.3f silc=%d raw=%r",
                  step, peak, silence_run, latest_raw[:60])

        # Commit conditions:
        #   1. Silence VAD: enough consecutive quiet chunks AND we already have
        #      SOME text (don't commit on pre-speech silence).
        #   2. End-of-utterance lang tag emitted by the model.
        if silence_run >= SILENCE_CHUNKS_TO_COMMIT and latest_raw:
            LOG.debug("silence commit after %d quiet chunks", silence_run)
            committed = latest_raw
            break
        # Lang tag at end signals model thinks utterance is done
        tags = LANG_TAG_RE.findall(latest_raw.rstrip())
        if tags and latest_raw.rstrip().endswith(tags[-1]):
            # Only honor lang tag at word boundary (see stream_translate.py:838-853)
            before_tag = latest_raw.rstrip()[:-len(tags[-1])]
            if not before_tag or before_tag[-1] in " \t\n.,;:!?…":
                LOG.debug("lang tag commit")
                committed = latest_raw
                break

    if not committed:
        committed = latest_raw

    # Clean the committed text: strip lang tags, then strip wake phrase.
    cleaned = strip_lang_tags(committed).strip()
    return strip_wake_phrase(cleaned)


# --- Main event loop ---

def main() -> None:
    ap = argparse.ArgumentParser(description="Nemo ơi voice assistant")
    ap.add_argument("--lang", default="vi-VN", help="ASR prompt language")
    ap.add_argument("--device", default="default", help="mic device")
    ap.add_argument("--chunk", choices=list(ATT_CONTEXT), default="560ms")
    ap.add_argument("--wake-model", default=str(WAKE_MODEL),
                    help="path to openWakeWord ONNX for 'Nemo ơi'")
    ap.add_argument("--wake-word-key", default="nemo_oi",
                    help="output label of the openWakeWord model (change when "
                         "using a community model e.g. hey_jarvis)")
    ap.add_argument("--wake-threshold", type=float, default=WAKE_THRESHOLD)
    ap.add_argument("--piper-voice", default=str(PIPER_VOICE))
    ap.add_argument("--no-tts", action="store_true",
                    help="print responses instead of speaking them")
    ap.add_argument("--wake-only", action="store_true",
                    help="only test wake detection; skip ASR/skills")
    ap.add_argument("--no-wake", action="store_true",
                    help="push-to-talk mode: skip wake-word detection entirely "
                         "and prompt for ENTER to start each command. Useful "
                         "for testing the pipeline before you've trained a "
                         "'Nemo ơi' wake-word model.")
    ap.add_argument("--log-file", default="auto",
                    help="path to debug log file. 'auto' → logs/assistant-<ts>.log; "
                         "'-' to disable file logging (stderr only).")
    args = ap.parse_args()

    # --- Logging: dual sink (terminal + persistent file) ---
    # Terminal at INFO — matches what users see today.
    # File at DEBUG — captures every wake score, every ASR partial, every
    # skill dispatch. Same pattern stream_translate.py uses so anyone
    # familiar with that log format knows what to expect here.
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.log_file == "-":
        log_path = None
    elif args.log_file == "auto":
        log_path = HERE / "logs" / f"assistant-{ts}.log"
    else:
        log_path = Path(args.log_file)

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(name)-16s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if log_path else logging.INFO)
    # Terminal handler at INFO
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    # File handler at DEBUG (full detail)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        print(f"[log]    debug log -> {log_path}", flush=True)
        LOG.info("=== session start: %s", " ".join(sys.argv))

    # --- Load ASR (may take ~30s first time) ---
    LOG.info("loading ASR...")
    model = load_asr()
    model.set_inference_prompt(args.lang)
    att_ctx = ATT_CONTEXT[args.chunk]
    model.encoder.set_default_att_context_size(att_ctx)
    chunk_secs = (1 + att_ctx[1]) * 0.08
    chunk_samples = int(chunk_secs * SR)

    # --- Assistant pieces ---
    # WakeGate is loaded lazily when --no-wake is off. In push-to-talk mode
    # we skip it entirely so the assistant works BEFORE the wake model exists.
    wake: WakeGate | None = None
    if not args.no_wake:
        LOG.info("loading wake gate...")
        wake = WakeGate(
            model_path=args.wake_model,
            wake_word_key=args.wake_word_key,
            threshold=args.wake_threshold,
        )
    else:
        LOG.info("push-to-talk mode — wake-word detection skipped")

    router = IntentRouter()
    register_skills(router)
    LOG.info("registered %d skill patterns", len(router._skills))

    tts: TTSSpeaker | None = None
    if not args.no_tts:
        LOG.info("loading TTS voice...")
        tts = TTSSpeaker(voice_model=args.piper_voice)

    # --- Mic capture ---
    producer = MicProducer(args.device, max_buffer_samples=int(1.5 * SR))
    producer.start()

    def _respond(text: str) -> None:
        """Emit a response to the user — TTS if enabled, stdout otherwise."""
        print(f"    Nemo: {text}", flush=True)
        if tts is not None:
            tts.speak(text, blocking=True)

    # --- Push-to-talk branch: no wake word, prompt for ENTER ---
    if args.no_wake:
        print("\nPush-to-talk mode. Nhấn ENTER để bắt đầu nói, Ctrl-C để thoát.\n",
              flush=True)
        try:
            while True:
                try:
                    input("→ ENTER để nói: ")
                except EOFError:
                    break
                print("    (đang nghe...)", flush=True)
                # No wake pre-roll in this mode — start from silence. The ASR
                # will pick up as soon as the user speaks.
                t0 = time.time()
                command = _capture_command(
                    model=model,
                    producer=producer,
                    pre_roll=np.zeros(0, dtype=np.float32),
                    chunk_samples=chunk_samples,
                    att_ctx=att_ctx,
                )
                capture_ms = (time.time() - t0) * 1000
                print(f"    You: {command}  ({capture_ms:.0f}ms)", flush=True)
                if not command.strip():
                    _respond("Nemo không nghe rõ, bạn nói lại nhé.")
                    continue
                result = router.route(command)
                if result.skill_name is None:
                    _respond("Xin lỗi, Nemo chưa hiểu câu đó.")
                else:
                    try:
                        response = result.handler(result.slots)
                    except Exception as e:
                        LOG.exception("skill %s failed", result.skill_name)
                        response = f"Nemo gặp lỗi: {type(e).__name__}"
                    _respond(response)
        except KeyboardInterrupt:
            print("\n[thoát]", flush=True)
        finally:
            producer.stop()
        return

    # --- Wake-word branch ---
    print(f"\nSẵn sàng — nói 'Nemo ơi' để bắt đầu (Ctrl-C để thoát)\n", flush=True)

    try:
        while True:
            chunk = producer.take(1024)
            if chunk is None:
                time.sleep(0.02)
                continue

            ev = wake.process(chunk)
            if ev is None:
                continue

            print(f"[wake fired: score={ev.score:.2f}]", flush=True)

            if args.wake_only:
                wake.reset()
                continue

            # --- CAPTURING ---
            t0 = time.time()
            command = _capture_command(
                model=model,
                producer=producer,
                pre_roll=ev.pre_roll,
                chunk_samples=chunk_samples,
                att_ctx=att_ctx,
            )
            capture_ms = (time.time() - t0) * 1000
            print(f"    You: {command}  ({capture_ms:.0f}ms)", flush=True)

            if not command.strip():
                _respond("Nemo không nghe rõ, bạn nói lại nhé.")
                wake.reset()
                continue

            # --- THINKING ---
            result = router.route(command)
            if result.skill_name is None:
                LOG.info("no rule matched: %r", command)
                _respond("Xin lỗi, Nemo chưa hiểu câu đó.")
            else:
                LOG.info("intent=%s slots=%s", result.skill_name, result.slots)
                try:
                    response = result.handler(result.slots)
                except Exception as e:
                    LOG.exception("skill %s failed", result.skill_name)
                    response = f"Nemo gặp lỗi khi xử lý: {type(e).__name__}"
                _respond(response)

            wake.reset()

    except KeyboardInterrupt:
        print("\n[thoát]", flush=True)
    finally:
        LOG.info("=== session end")
        producer.stop()


if __name__ == "__main__":
    main()
