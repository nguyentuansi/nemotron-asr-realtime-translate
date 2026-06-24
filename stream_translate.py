"""Real-time streaming ASR + lightweight NLLB-200 translation.

Same streaming ASR loop as stream_demo.py — cache-aware via conformer_stream_step
with persistent encoder caches. When the model commits an utterance (end-of-
utterance <lang> tag or watchdog), the finalized text is pushed onto a background
translation queue. A CTranslate2 NLLB-200 worker (CPU, int8) translates and
prints the translation underneath the original.

Translation runs only on committed utterances, never on the live partial. The
ASR loop never blocks on translation. The GPU stays fully owned by ASR.

Display:
    A lô a lô nói gì đi tại sao cậu không nghe được vậy?
      ↳ Hello, hello, say something, why can't you hear?
    Bây giờ mình sẽ nói một câu chuyện về chú thỏ và con rua.
      ↳ Now I'm going to tell a story about the rabbit and the turtle.
    [ 25.7s #  45 ▕███····▏] Ngày xửa ngày xưa            <-- live partial
"""
import argparse
import datetime as _dt
import logging
import math
import os
import queue
import re
import shutil
import sys
import threading
import time
from pathlib import Path

LOG = logging.getLogger("stream_translate")


def setup_logging(log_path: Path | None, level=logging.DEBUG):
    """Configure file logging without polluting the live terminal display.

    No console handler — stdout is reserved for the live transcript/translation.
    The log captures every chunk's hypothesis, each commit + its trigger,
    translation latencies and any errors, so we can post-mortem 'why did it
    stop transcribing' without re-running the session.
    """
    LOG.handlers.clear()
    LOG.setLevel(level)
    if log_path is None:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    ))
    LOG.addHandler(fh)
    LOG.propagate = False
    return log_path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import alsaaudio
import numpy as np
import torch

SR = 16000
CH = 1
FMT = alsaaudio.PCM_FORMAT_S16_LE

ATT_CONTEXT = {
    "80ms":   [56, 0],
    "160ms":  [56, 1],
    "320ms":  [56, 3],
    "560ms":  [56, 6],
    "1120ms": [56, 13],
}

LANG_TAG_RE = re.compile(r"<[a-zA-Z]{2,3}(?:-[A-Z]{2})?>")
NLLB_MODEL_DIR = HERE / "nllb-200-distilled-600M-ct2-int8"


def load_asr():
    import nemo.collections.asr as nemo_asr
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)
    t0 = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
        map_location="cpu",
    )
    print(f"[asr]  loaded in {time.time()-t0:.1f}s", flush=True)
    if torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1":
        torch.cuda.empty_cache()
        model = model.to("cuda").eval()
        print(f"[asr]  fp32 weights on GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB",
              flush=True)
    else:
        model = model.eval()
        print("[asr]  running on CPU", flush=True)
    return model


class MicProducer(threading.Thread):
    def __init__(self, device):
        super().__init__(daemon=True)
        self.device = device
        self._lock = threading.Lock()
        self._buf = np.zeros(0, dtype=np.float32)
        self._stop = threading.Event()
        self._error = None
        self._recent_peak = 0.0

    def run(self):
        try:
            pcm = alsaaudio.PCM(
                type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL,
                device=self.device, channels=CH, rate=SR, format=FMT,
                periodsize=1024,
            )
        except alsaaudio.ALSAAudioError as e:
            self._error = e
            return
        try:
            while not self._stop.is_set():
                length, data = pcm.read()
                if length <= 0:
                    continue
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                with self._lock:
                    self._buf = np.concatenate([self._buf, arr])
                    p = float(np.abs(arr).max()) if arr.size else 0.0
                    self._recent_peak = max(p, self._recent_peak * 0.85)
        finally:
            pcm.close()

    def take(self, n):
        with self._lock:
            if self._buf.size < n:
                return None
            out = self._buf[:n].copy()
            self._buf = self._buf[n:]
            return out

    def peak(self):
        with self._lock:
            return self._recent_peak

    def stop(self):
        self._stop.set()
        if self._error is not None:
            raise self._error


def level_bar(peak_f32, width=10):
    if peak_f32 <= 1e-6:
        n = 0
    else:
        db = 20.0 * np.log10(peak_f32)
        n = int(np.clip((db + 60.0) / 60.0 * width, 0, width))
    return "▕" + "█" * n + "·" * (width - n) + "▏"


def strip_lang_tags(text):
    return LANG_TAG_RE.sub("", text).rstrip()


class Display:
    """Thread-safe console writer. Owns the 'live partial' slot at the bottom of
    the visible region. Prints from any thread route through here so the live
    line is always redrawn after a scrolling write.

    Critical: the live partial MUST be truncated to terminal width. If it wraps,
    '\\r' only returns to col 0 of the current visual line — wrap residue stays
    and the next write paints below it, producing the stair-step bug.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._partial = ""

    def _term_width(self):
        try:
            return max(20, shutil.get_terminal_size((100, 24)).columns)
        except OSError:
            return 100

    def _truncate(self, text):
        w = self._term_width()
        # Trim from the LEFT so the most recent words stay visible.
        if len(text) > w:
            text = "…" + text[-(w - 1):]
        return text

    def update_partial(self, text):
        with self._lock:
            self._partial = text
            sys.stdout.write(f"\r\x1b[2K{self._truncate(text)}")
            sys.stdout.flush()

    def commit(self, orig, translated=None):
        """Append a finalized utterance above the live partial line."""
        with self._lock:
            sys.stdout.write(f"\r\x1b[2K{orig}\n")
            if translated is not None:
                sys.stdout.write(f"  ↳ {translated}\n")
            sys.stdout.write(self._truncate(self._partial))
            sys.stdout.flush()

    def append_translation(self, translated):
        """Print just a translation line (the original was already shown)."""
        with self._lock:
            sys.stdout.write(f"\r\x1b[2K  ↳ {translated}\n")
            sys.stdout.write(self._truncate(self._partial))
            sys.stdout.flush()


class Translator:
    """Background NLLB-200 worker. Pulls (utt_id, text) tuples off a queue,
    translates, and emits the translation through Display.append_translation().
    """
    def __init__(self, model_dir, src_lang, tgt_lang, display, device="cpu",
                 compute_type="int8", beam_size=2):
        from translator import NLLBTranslator
        self.model = NLLBTranslator(
            model_dir, device=device, compute_type=compute_type, beam_size=beam_size,
        )
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.display = display
        self.queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def submit(self, text):
        self.queue.put(text)

    def shutdown(self, drain_timeout=10.0):
        """Signal the worker to drain its queue then exit. Called from main on
        Ctrl-C — without this, ctranslate2's C++ threads get killed mid-call and
        std::terminate() fires SIGABRT after Python exits."""
        self.queue.put(None)
        self._thread.join(timeout=drain_timeout)

    def _run(self):
        while True:
            text = self.queue.get()
            if text is None:
                LOG.debug("translator worker received sentinel, exiting")
                return
            try:
                t0 = time.time()
                out = self.model.translate(text, self.src_lang, self.tgt_lang)
                dt_ms = (time.time() - t0) * 1000
                LOG.info(
                    "trans %dms in_len=%d out_len=%d  in=%r -> out=%r",
                    int(dt_ms), len(text), len(out), text, out,
                )
                self.display.append_translation(f"{out}  [{dt_ms:.0f}ms]")
            except Exception as e:
                LOG.exception("translation failed for %r", text)
                self.display.append_translation(f"[trans error: {type(e).__name__}: {e}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="default",
                    help="ALSA capture PCM (default: 'default' -> PipeWire)")
    ap.add_argument("--lang", default="vi-VN",
                    help="source language spoken into the mic (default: vi-VN)")
    ap.add_argument("--target-lang", default="en-US",
                    help="translation target language (default: en-US)")
    ap.add_argument("--chunk", choices=list(ATT_CONTEXT), default="560ms",
                    help="streaming chunk size")
    ap.add_argument("--no-translate", action="store_true",
                    help="run ASR only; do not load the translation model")
    ap.add_argument("--translator-device", choices=["cpu", "cuda"], default="cpu",
                    help="device for the NLLB translator (default: cpu; keeps GPU for ASR)")
    ap.add_argument("--watchdog", type=int, default=25,
                    help="commit + reset RNNT after N chunks with no new tokens (0=off)")
    ap.add_argument("--max-utterance-secs", type=float, default=6.0,
                    help="force a commit when the current utterance has been running for "
                         "more than N seconds (prevents continuous speech from never committing)")
    ap.add_argument("--max-utterance-chars", type=int, default=120,
                    help="hard cap on running-partial length before a forced commit")
    ap.add_argument("--silence-secs", type=float, default=1.0,
                    help="commit when mic audio stays below --silence-threshold for this long "
                         "(VAD-style endpointing — the main 'feels realtime' lever; 0 to disable)")
    ap.add_argument("--silence-threshold", type=float, default=0.025,
                    help="audio peak below this counts as silence (linear, 0..1)")
    ap.add_argument("--render-hz", type=float, default=10.0,
                    help="live partial redraw rate (lower = less terminal flicker)")
    ap.add_argument("--beam-size", type=int, default=2,
                    help="NLLB decoding beam size (1=greedy, 4=quality, 2=balanced)")
    ap.add_argument("--log-file", default="auto",
                    help="path to debug log file, 'auto' for logs/stream-<ts>.log, '-' to disable")
    args = ap.parse_args()

    if args.log_file == "-":
        log_path = None
    elif args.log_file == "auto":
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = HERE / "logs" / f"stream-{ts}.log"
    else:
        log_path = Path(args.log_file)
    setup_logging(log_path)
    if log_path is not None:
        print(f"[log]    debug log -> {log_path}", flush=True)
        LOG.info("=== session start: %s", " ".join(sys.argv))

    att_ctx = ATT_CONTEXT[args.chunk]
    model = load_asr()
    model.encoder.set_default_att_context_size(att_ctx)
    try:
        model.set_inference_prompt(args.lang)
    except AttributeError:
        print("[warn] model has no set_inference_prompt — not prompt-conditioned?",
              flush=True)

    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    dev_torch = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    cache_last_channel, cache_last_time, cache_last_channel_len = (
        model.encoder.get_initial_cache_state(batch_size=1, device=dev_torch)
    )
    streaming_buffer = CacheAwareStreamingAudioBuffer(model=model, online_normalization=True)

    chunk_secs = (1 + att_ctx[1]) * 0.08
    chunk_samples = int(chunk_secs * SR)
    cfg = model.encoder.streaming_cfg

    def required_remaining(step_num):
        cs = cfg.chunk_size
        if step_num == 0 and isinstance(cs, list):
            return cs[0]
        return cs[1] if isinstance(cs, list) else cs

    display = Display()
    translator = None
    if not args.no_translate:
        t0 = time.time()
        translator = Translator(
            NLLB_MODEL_DIR, args.lang, args.target_lang, display,
            device=args.translator_device, beam_size=args.beam_size,
        )
        translator.start()
        print(f"[nmt]  NLLB-200 int8 loaded in {time.time()-t0:.1f}s "
              f"on {args.translator_device} beam_size={args.beam_size}", flush=True)
        LOG.info("translator loaded device=%s beam_size=%d",
                 args.translator_device, args.beam_size)

    print(f"\n[stream] src={args.lang} -> tgt={args.target_lang} chunk={args.chunk} "
          f"device={args.device}", flush=True)
    print(f"[cfg]    chunk_samples={chunk_samples} streaming_cfg.chunk_size={cfg.chunk_size}",
          flush=True)
    if args.watchdog > 0:
        print(f"[cfg]    watchdog={args.watchdog} chunks ({args.watchdog * chunk_secs:.1f}s)",
              flush=True)
    print("[*] speak — Ctrl-C to stop\n", flush=True)

    producer = MicProducer(args.device)
    producer.start()
    time.sleep(0.05)

    previous_hypotheses = None
    pred_out_stream = None
    step = 0
    last_partial = ""
    chunks_since_change = 0
    t_start = time.time()
    last_render_t = 0.0
    render_interval = 1.0 / max(1.0, args.render_hz)
    utterance_started_at = time.time()
    # The model's hypothesis text is the WHOLE running transcript. To split it
    # into utterances for display + translation, we track how much we've already
    # committed and only show/submit the slice past that point. Crucially, we
    # DO NOT reset previous_hypotheses here — resetting mid-stream broke the
    # encoder↔RNNT coupling and caused the model to emit blanks for the rest
    # of the session.
    committed_raw_len = 0
    latest_raw_text = ""  # most recent hypothesis text, used by Ctrl-C handler.
    chunks_below_silence = 0
    # Round UP so silence_secs=1.0 with a 0.56s chunk gives 2 chunks (1.12s),
    # not 1 chunk (0.56s) — a 0.56s gap is just a between-word pause.
    silence_chunks_needed = (max(2, math.ceil(args.silence_secs / chunk_secs))
                             if args.silence_secs > 0 else 0)
    if silence_chunks_needed > 0:
        print(f"[cfg]    silence VAD: peak<{args.silence_threshold} for "
              f"{silence_chunks_needed} chunks ({silence_chunks_needed * chunk_secs:.1f}s) "
              f"-> commit", flush=True)
    # Sentence terminators: covers Latin punctuation + Vietnamese spoken-style endings.
    SENTENCE_END = ".!?。！？"

    def render_partial(force=False):
        nonlocal last_render_t
        now = time.time()
        if not force and now - last_render_t < render_interval:
            return
        last_render_t = now
        elapsed = now - t_start
        bar = level_bar(producer.peak())
        body = strip_lang_tags(last_partial) if last_partial else ""
        display.update_partial(f"[{elapsed:6.1f}s #{step:4d} {bar}] {body}")

    def commit_utterance(raw_text_now, reason):
        """Take everything in raw_text_now after the last commit point, finalize
        it as one line, submit for translation. We do NOT reset RNNT state —
        the model keeps decoding into the same growing hypothesis; we just move
        our committed-up-to pointer forward.
        """
        nonlocal committed_raw_len, last_partial, chunks_since_change
        nonlocal utterance_started_at
        new_raw = raw_text_now[committed_raw_len:]
        finalized = strip_lang_tags(new_raw).strip()
        if not finalized:
            LOG.debug("commit_skipped reason=%s empty new_raw=%r", reason, new_raw)
            return
        LOG.info(
            "COMMIT reason=%s text=%r  (committed_so_far=%d -> %d)",
            reason, finalized, committed_raw_len, len(raw_text_now),
        )
        display.commit(finalized)
        if translator is not None:
            translator.submit(finalized)
        committed_raw_len = len(raw_text_now)
        last_partial = ""
        chunks_since_change = 0
        utterance_started_at = time.time()

    try:
        while True:
            chunk_audio_raw = producer.take(chunk_samples)
            if chunk_audio_raw is None:
                render_partial()
                time.sleep(0.05)
                continue
            sid = -1 if streaming_buffer.buffer is None else 0
            streaming_buffer.append_audio(chunk_audio_raw, stream_id=sid)
            while (streaming_buffer.buffer.size(-1) - streaming_buffer.buffer_idx
                   >= required_remaining(step)):
                gen = iter(streaming_buffer)
                try:
                    chunk_audio, chunk_lengths = next(gen)
                except StopIteration:
                    break
                drop_extra_pre_encoded = cfg.drop_extra_pre_encoded if step != 0 else 0
                if chunk_audio.dtype != model_dtype:
                    chunk_audio = chunk_audio.to(model_dtype)
                with torch.inference_mode():
                    result = model.conformer_stream_step(
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=False,
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=drop_extra_pre_encoded,
                        return_transcription=True,
                    )
                (pred_out_stream, transcribed_texts, cache_last_channel,
                 cache_last_time, cache_last_channel_len, previous_hypotheses) = result
                step += 1
                hyp = transcribed_texts[0] if transcribed_texts else None
                raw_text = getattr(hyp, "text", "") if hyp is not None else ""
                latest_raw_text = raw_text

                # Everything in the running hypothesis past the last commit
                # point is the current utterance-in-progress.
                new_raw = raw_text[committed_raw_len:]
                new_clean = strip_lang_tags(new_raw).rstrip()

                if new_raw != last_partial:
                    chunks_since_change = 0
                else:
                    chunks_since_change += 1

                LOG.debug(
                    "step=%d t=%.2fs peak=%.3f silc=%d raw_len=%d committed=%d new=%r",
                    step, time.time() - t_start, producer.peak(),
                    chunks_below_silence, len(raw_text), committed_raw_len, new_clean[:80],
                )

                # Track silence chunks (audio-level VAD).
                cur_peak = producer.peak()
                if cur_peak < args.silence_threshold:
                    chunks_below_silence += 1
                else:
                    chunks_below_silence = 0

                # End-of-utterance lang tag in the un-committed portion.
                stripped = new_raw.rstrip()
                tags = LANG_TAG_RE.findall(stripped)
                if tags and stripped.endswith(tags[-1]):
                    commit_utterance(raw_text, reason="lang_tag")
                    chunks_below_silence = 0
                    continue
                # Sentence-final punctuation.
                if new_clean and new_clean[-1] in SENTENCE_END:
                    commit_utterance(raw_text, reason="punctuation")
                    chunks_below_silence = 0
                    continue
                # Silence-based: mic dropped below threshold for long enough.
                # This is the main 'feels realtime' lever — commits ~1s after
                # speech ends, not 6+ seconds via the time fallback.
                if (silence_chunks_needed > 0
                        and chunks_below_silence >= silence_chunks_needed
                        and new_clean):
                    commit_utterance(
                        raw_text,
                        reason=f"silence_{chunks_below_silence}chk_peak={cur_peak:.3f}",
                    )
                    chunks_below_silence = 0
                    continue
                # Time-based: long-running utterance, fallback.
                utt_age = time.time() - utterance_started_at
                if (args.max_utterance_secs > 0 and new_clean
                        and utt_age >= args.max_utterance_secs):
                    commit_utterance(raw_text, reason=f"max_secs={args.max_utterance_secs}")
                    continue
                # Hard char cap.
                if args.max_utterance_chars > 0 and len(new_clean) >= args.max_utterance_chars:
                    commit_utterance(raw_text, reason=f"max_chars={args.max_utterance_chars}")
                    continue

                last_partial = new_raw

                # Watchdog: text hasn't changed for N chunks.
                if args.watchdog > 0 and chunks_since_change >= args.watchdog and new_clean:
                    commit_utterance(raw_text, reason=f"watchdog={args.watchdog}")
                    continue

            render_partial()
    except KeyboardInterrupt:
        if latest_raw_text and len(latest_raw_text) > committed_raw_len:
            commit_utterance(latest_raw_text, reason="ctrl_c")
        if translator is not None:
            sys.stdout.write("\n[wait] draining translator queue...\n")
            sys.stdout.flush()
            translator.shutdown(drain_timeout=15.0)
        sys.stdout.write("[done]\n")
        sys.stdout.flush()
        LOG.info("=== session end")
    finally:
        producer.stop()


if __name__ == "__main__":
    main()
