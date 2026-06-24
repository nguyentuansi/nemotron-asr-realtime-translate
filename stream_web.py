"""Web UI for the real-time streaming ASR + translation combo.

Same ASR + translator core as stream_translate.py, but the display layer is
replaced with a FastAPI + WebSocket broadcaster. Open http://127.0.0.1:8765
in a browser — you'll see two stacked panes:

    SOURCE       <- finalized Vietnamese flows here, with the current live
                    partial appended in italic light-grey at the end
    TRANSLATION  <- finalized English flows here, with the current draft
                    translation appended in italic light-grey at the end

Drafts re-translate the running partial every ~1s so a translation is
visible BEFORE you pause. Commit fires on real sentence end / 2s silence
/ time-cap; the italic-grey live text locks in to normal style.

Run:
    ./stream_web.sh --lang vi-VN
    # then open http://127.0.0.1:8765 in a browser
"""
import argparse
import asyncio
import datetime as _dt
import json
import logging
import math
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import alsaaudio
import numpy as np
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
import uvicorn

LOG = logging.getLogger("stream_web")

SR = 16000
CH = 1
FMT = alsaaudio.PCM_FORMAT_S16_LE
NLLB_MODEL_DIR = HERE / "nllb-200-distilled-600M-ct2-int8"
WEB_INDEX = HERE / "web" / "index.html"

ATT_CONTEXT = {
    "80ms":   [56, 0],
    "160ms":  [56, 1],
    "320ms":  [56, 3],
    "560ms":  [56, 6],
    "1120ms": [56, 13],
}

LANG_TAG_RE = re.compile(r"<[a-zA-Z]{2,3}(?:-[A-Z]{2})?>")


# ---------- logging ----------

def setup_logging(log_path: Path | None, level=logging.DEBUG):
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


def strip_lang_tags(text):
    return LANG_TAG_RE.sub("", text).rstrip()


# ---------- ASR model load + mic capture ----------

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
    """Pulls raw int16 frames from ALSA, converts to float32, appends to a thread-safe buffer."""
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


# ---------- WebSocket broadcaster ----------

class Broadcaster:
    """Bridges sync ASR/translator threads to async WebSocket clients.

    State is kept in this process so a browser that joins mid-session gets a
    full snapshot (all committed lines, current partial + draft, stats, config)
    before live events resume.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        # Persistent state mirrored by the frontend.
        self.source_lines: list[str] = []
        self.translation_lines: list[str] = []
        self.current_partial: str = ""
        self.current_draft: str = ""
        self.stats: dict = {"step": 0, "peak": 0.0, "commits": 0, "elapsed_s": 0.0}
        self.config: dict = {}

    def set_loop(self, loop):
        self._loop = loop

    def set_config(self, **kwargs):
        with self._lock:
            self.config.update(kwargs)
        self.emit({"type": "config", **kwargs})

    def snapshot(self):
        with self._lock:
            return {
                "type": "state",
                "config": dict(self.config),
                "source_lines": list(self.source_lines),
                "translation_lines": list(self.translation_lines),
                "current_partial": self.current_partial,
                "current_draft": self.current_draft,
                "stats": dict(self.stats),
            }

    async def add_client(self, ws: WebSocket):
        with self._lock:
            self._clients.add(ws)
        try:
            await ws.send_text(json.dumps(self.snapshot()))
        except Exception:
            LOG.exception("failed to send snapshot to new client")

    async def remove_client(self, ws: WebSocket):
        with self._lock:
            self._clients.discard(ws)

    def emit(self, event: dict):
        # Update mirrored state first so a new connection during this call
        # gets the latest snapshot.
        with self._lock:
            t = event.get("type")
            if t == "partial":
                self.current_partial = event.get("text", "")
                for k in ("step", "peak", "elapsed_s"):
                    if k in event:
                        self.stats[k] = event[k]
            elif t == "draft":
                self.current_draft = event.get("text", "")
            elif t == "commit_source":
                self.source_lines.append(event.get("text", ""))
                self.current_partial = ""
                self.stats["commits"] = len(self.source_lines)
            elif t == "commit_translation":
                self.translation_lines.append(event.get("text", ""))
                self.current_draft = ""
            elif t == "reset":
                # Preserve history by default (a full ASR reset isn't a "clear screen").
                if not event.get("preserve", True):
                    self.source_lines = []
                    self.translation_lines = []
                self.current_partial = ""
                self.current_draft = ""
        # Then fire to all connected clients.
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(event), self._loop)

    async def _broadcast(self, event):
        msg = json.dumps(event, ensure_ascii=False)
        dead = []
        with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    self._clients.discard(ws)


# ---------- Translator with draft + final queues ----------

class Translator:
    def __init__(self, model_dir, src_lang, tgt_lang, broadcaster: Broadcaster,
                 device="cpu", compute_type="int8", beam_size=2):
        from translator import NLLBTranslator
        self.model = NLLBTranslator(
            model_dir, device=device, compute_type=compute_type, beam_size=beam_size,
        )
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.broadcaster = broadcaster
        self._final_queue: queue.Queue = queue.Queue()
        self._draft_text: str | None = None
        self._draft_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def submit_final(self, text: str):
        self._final_queue.put(text)
        self._wake.set()

    def submit_draft(self, text: str):
        with self._draft_lock:
            self._draft_text = text
        self._wake.set()

    def clear_draft(self):
        with self._draft_lock:
            self._draft_text = None

    def shutdown(self, drain_timeout=10.0):
        self._stop.set()
        self._final_queue.put(None)
        self._wake.set()
        self._thread.join(timeout=drain_timeout)

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            # FINAL first.
            while True:
                try:
                    text = self._final_queue.get_nowait()
                except queue.Empty:
                    break
                if text is None:
                    LOG.debug("translator worker received sentinel, exiting")
                    return
                self._translate(text, is_final=True)
            with self._draft_lock:
                draft = self._draft_text
                self._draft_text = None
            if draft:
                self._translate(draft, is_final=False)

    def _translate(self, text: str, is_final: bool):
        try:
            t0 = time.time()
            out = self.model.translate(text, self.src_lang, self.tgt_lang)
            dt_ms = int((time.time() - t0) * 1000)
            kind = "FINAL" if is_final else "draft"
            LOG.info(
                "trans %s %dms in_len=%d out_len=%d  in=%r -> out=%r",
                kind, dt_ms, len(text), len(out), text, out,
            )
            if is_final:
                self.broadcaster.emit({"type": "commit_translation", "text": out, "ms": dt_ms})
            else:
                self.broadcaster.emit({"type": "draft", "text": out, "ms": dt_ms})
        except Exception as e:
            LOG.exception("translation failed for %r", text)
            if is_final:
                self.broadcaster.emit({
                    "type": "commit_translation",
                    "text": f"[trans error: {type(e).__name__}: {e}]",
                })


# ---------- ASR streaming loop ----------

def run_asr_loop(args, broadcaster: Broadcaster, translator: Translator | None):
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

    broadcaster.set_config(
        src_lang=args.lang,
        tgt_lang=args.target_lang,
        chunk_label=args.chunk,
        chunk_samples=chunk_samples,
    )

    def required_remaining(step_num):
        cs = cfg.chunk_size
        if step_num == 0 and isinstance(cs, list):
            return cs[0]
        return cs[1] if isinstance(cs, list) else cs

    producer = MicProducer(args.device)
    producer.start()
    time.sleep(0.05)

    previous_hypotheses = None
    pred_out_stream = None
    step = 0
    last_partial = ""
    chunks_since_change = 0
    t_start = time.time()
    last_emit_t = 0.0
    emit_interval = 1.0 / max(1.0, args.emit_hz)
    utterance_started_at = time.time()
    committed_raw_len = 0
    latest_raw_text = ""
    chunks_below_silence = 0
    silence_chunks_needed = (max(2, math.ceil(args.silence_secs / chunk_secs))
                             if args.silence_secs > 0 else 0)
    draft_enabled = args.draft_secs > 0 and translator is not None
    last_draft_at = 0.0
    last_draft_text = ""
    commits_since_reset = 0
    last_full_reset_at = time.time()
    SENTENCE_END = ".!?。！？"

    def emit_partial(force=False):
        nonlocal last_emit_t
        now = time.time()
        if not force and now - last_emit_t < emit_interval:
            return
        last_emit_t = now
        broadcaster.emit({
            "type": "partial",
            "text": strip_lang_tags(last_partial) if last_partial else "",
            "step": step,
            "peak": producer.peak(),
            "elapsed_s": now - t_start,
        })

    def commit_utterance(raw_text_now, reason):
        nonlocal committed_raw_len, last_partial, chunks_since_change
        nonlocal utterance_started_at, last_draft_text, last_draft_at, commits_since_reset
        new_raw = raw_text_now[committed_raw_len:]
        finalized = strip_lang_tags(new_raw).strip()
        if not finalized:
            LOG.debug("commit_skipped reason=%s empty new_raw=%r", reason, new_raw)
            return
        LOG.info("COMMIT reason=%s text=%r  (committed_so_far=%d -> %d)",
                 reason, finalized, committed_raw_len, len(raw_text_now))
        broadcaster.emit({"type": "commit_source", "text": finalized})
        if translator is not None:
            translator.clear_draft()
            translator.submit_final(finalized)
        committed_raw_len = len(raw_text_now)
        last_partial = ""
        chunks_since_change = 0
        utterance_started_at = time.time()
        last_draft_text = ""
        last_draft_at = time.time()
        commits_since_reset += 1

    def maybe_full_reset():
        nonlocal cache_last_channel, cache_last_time, cache_last_channel_len
        nonlocal previous_hypotheses, pred_out_stream, committed_raw_len
        nonlocal streaming_buffer, step, commits_since_reset, last_full_reset_at
        nonlocal last_partial, latest_raw_text, chunks_since_change
        nonlocal chunks_below_silence, last_draft_text, last_draft_at, utterance_started_at

        need, why = False, ""
        if args.full_reset_after > 0 and commits_since_reset >= args.full_reset_after:
            need, why = True, f"commits={commits_since_reset}"
        elif (args.full_reset_secs > 0
              and time.time() - last_full_reset_at >= args.full_reset_secs):
            need, why = True, f"age={time.time()-last_full_reset_at:.0f}s"
        if not need:
            return False
        LOG.info("FULL_RESET %s", why)
        cache_last_channel, cache_last_time, cache_last_channel_len = (
            model.encoder.get_initial_cache_state(batch_size=1, device=dev_torch)
        )
        previous_hypotheses = None
        pred_out_stream = None
        committed_raw_len = 0
        streaming_buffer = CacheAwareStreamingAudioBuffer(model=model, online_normalization=True)
        step = 0
        commits_since_reset = 0
        last_full_reset_at = time.time()
        last_partial = ""
        latest_raw_text = ""
        chunks_since_change = 0
        chunks_below_silence = 0
        last_draft_text = ""
        last_draft_at = time.time()
        utterance_started_at = time.time()
        # Tell the UI we reset but keep its scroll-back history.
        broadcaster.emit({"type": "reset", "preserve": True})
        return True

    try:
        while True:
            chunk_audio_raw = producer.take(chunk_samples)
            if chunk_audio_raw is None:
                emit_partial()
                time.sleep(0.05)
                continue
            sid = -1 if streaming_buffer.buffer is None else 0
            streaming_buffer.append_audio(chunk_audio_raw, stream_id=sid)
            while True:
                if streaming_buffer.buffer is None:
                    break
                if (streaming_buffer.buffer.size(-1) - streaming_buffer.buffer_idx
                        < required_remaining(step)):
                    break
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

                cur_peak = producer.peak()
                if cur_peak < args.silence_threshold:
                    chunks_below_silence += 1
                else:
                    chunks_below_silence = 0

                stripped = new_raw.rstrip()
                tags = LANG_TAG_RE.findall(stripped)
                if tags and stripped.endswith(tags[-1]):
                    commit_utterance(raw_text, reason="lang_tag")
                    chunks_below_silence = 0
                    if maybe_full_reset():
                        break
                    continue
                if new_clean and new_clean[-1] in SENTENCE_END:
                    commit_utterance(raw_text, reason="punctuation")
                    chunks_below_silence = 0
                    if maybe_full_reset():
                        break
                    continue
                if (silence_chunks_needed > 0
                        and chunks_below_silence >= silence_chunks_needed
                        and new_clean):
                    commit_utterance(raw_text,
                                     reason=f"silence_{chunks_below_silence}chk_peak={cur_peak:.3f}")
                    chunks_below_silence = 0
                    if maybe_full_reset():
                        break
                    continue
                utt_age = time.time() - utterance_started_at
                if (args.max_utterance_secs > 0 and new_clean
                        and utt_age >= args.max_utterance_secs):
                    commit_utterance(raw_text, reason=f"max_secs={args.max_utterance_secs}")
                    if maybe_full_reset():
                        break
                    continue
                if args.max_utterance_chars > 0 and len(new_clean) >= args.max_utterance_chars:
                    commit_utterance(raw_text, reason=f"max_chars={args.max_utterance_chars}")
                    if maybe_full_reset():
                        break
                    continue

                last_partial = new_raw

                if args.watchdog > 0 and chunks_since_change >= args.watchdog and new_clean:
                    commit_utterance(raw_text, reason=f"watchdog={args.watchdog}")
                    if maybe_full_reset():
                        break
                    continue

                # Streaming draft submission.
                if draft_enabled and new_clean:
                    now = time.time()
                    grew_enough = abs(len(new_clean) - len(last_draft_text)) >= 3
                    if (now - last_draft_at >= args.draft_secs
                            and grew_enough
                            and new_clean != last_draft_text):
                        translator.submit_draft(new_clean)
                        last_draft_text = new_clean
                        last_draft_at = now
                        LOG.debug("submit_draft len=%d text=%r", len(new_clean), new_clean[:80])

            emit_partial()
    finally:
        producer.stop()


# ---------- FastAPI app ----------

def build_app(broadcaster: Broadcaster) -> FastAPI:
    app = FastAPI(title="nemotron-asr live translate", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    async def _on_startup():
        broadcaster.set_loop(asyncio.get_event_loop())

    @app.get("/", response_class=HTMLResponse)
    async def index():
        if not WEB_INDEX.exists():
            return PlainTextResponse(f"missing {WEB_INDEX}", status_code=500)
        return HTMLResponse(WEB_INDEX.read_text(encoding="utf-8"))

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        await broadcaster.add_client(ws)
        try:
            while True:
                # The browser doesn't send anything; receive_text just keeps the
                # connection open and detects disconnect.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            LOG.exception("ws_endpoint error")
        finally:
            await broadcaster.remove_client(ws)

    return app


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="default")
    ap.add_argument("--lang", default="vi-VN")
    ap.add_argument("--target-lang", default="en-US")
    ap.add_argument("--chunk", choices=list(ATT_CONTEXT), default="560ms")
    ap.add_argument("--no-translate", action="store_true")
    ap.add_argument("--translator-device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--watchdog", type=int, default=25)
    ap.add_argument("--max-utterance-secs", type=float, default=12.0)
    ap.add_argument("--max-utterance-chars", type=int, default=240)
    ap.add_argument("--silence-secs", type=float, default=2.0)
    ap.add_argument("--silence-threshold", type=float, default=0.025)
    ap.add_argument("--draft-secs", type=float, default=1.0)
    ap.add_argument("--full-reset-after", type=int, default=4)
    ap.add_argument("--full-reset-secs", type=float, default=45.0)
    ap.add_argument("--beam-size", type=int, default=2)
    ap.add_argument("--emit-hz", type=float, default=10.0,
                    help="rate at which we push 'partial' events to the browser")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log-file", default="auto",
                    help="path to debug log file, 'auto' for logs/web-<ts>.log, '-' to disable")
    args = ap.parse_args()

    if args.log_file == "-":
        log_path = None
    elif args.log_file == "auto":
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = HERE / "logs" / f"web-{ts}.log"
    else:
        log_path = Path(args.log_file)
    setup_logging(log_path)
    if log_path is not None:
        print(f"[log]    debug log -> {log_path}", flush=True)
        LOG.info("=== session start: %s", " ".join(sys.argv))

    broadcaster = Broadcaster()
    translator = None
    if not args.no_translate:
        t0 = time.time()
        translator = Translator(
            NLLB_MODEL_DIR, args.lang, args.target_lang, broadcaster,
            device=args.translator_device, beam_size=args.beam_size,
        )
        translator.start()
        print(f"[nmt]  NLLB-200 int8 loaded in {time.time()-t0:.1f}s "
              f"on {args.translator_device} beam_size={args.beam_size}", flush=True)

    asr_thread = threading.Thread(
        target=run_asr_loop, args=(args, broadcaster, translator), daemon=True,
    )
    asr_thread.start()

    app = build_app(broadcaster)
    url = f"http://{args.host}:{args.port}"
    print(f"\n[web]  open {url} in a browser", flush=True)
    print(f"[stream] src={args.lang} -> tgt={args.target_lang} chunk={args.chunk}", flush=True)

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning",
                            access_log=False)
    server = uvicorn.Server(config)
    try:
        server.run()
    finally:
        if translator is not None:
            translator.shutdown(drain_timeout=5.0)
        LOG.info("=== session end")


if __name__ == "__main__":
    main()
