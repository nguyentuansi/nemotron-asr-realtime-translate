"""Real-time cache-aware streaming ASR from microphone.

Uses model.conformer_stream_step() with persistent encoder caches — the actual
streaming inference path the model was designed for. Each chunk of audio updates
a running partial transcript; when the model emits an end-of-utterance language
tag (e.g. <vi-VN>) the partial is committed as a fixed line and the RNNT
decoder is reset for the next utterance. Encoder caches keep flowing.

Latency = chunk size. Choose with --chunk:
    80ms | 160ms | 320ms | 560ms (default) | 1120ms

Lower chunk = lower latency, slightly worse WER (see model card §Performance).
"""
import argparse
import os
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

try:
    import alsaaudio
except ImportError:
    import alsa_shim as alsaaudio
import numpy as np
import torch

SR = 16000
CH = 1
FMT = alsaaudio.PCM_FORMAT_S16_LE

# att_context_size = [left_context_frames, right_context_frames]; frame = 80ms.
# Chunk size = (1 + right_context) frames.
ATT_CONTEXT = {
    "80ms":   [56, 0],
    "160ms":  [56, 1],
    "320ms":  [56, 3],
    "560ms":  [56, 6],
    "1120ms": [56, 13],
}

LANG_TAG_RE = re.compile(r"<[a-zA-Z]{2,3}(?:-[A-Z]{2})?>")


def load_model():
    import nemo.collections.asr as nemo_asr
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)
    t0 = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
        map_location="cpu",
    )
    print(f"[load] {time.time()-t0:.1f}s", flush=True)
    if torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1":
        torch.cuda.empty_cache()
        # fp32: conformer_stream_step() doesn't auto-cast like model.transcribe()
        # does. fp32 weights are ~2.4 GB on the 6 GB GPU.
        model = model.to("cuda").eval()
        print(f"[gpu]  fp32 weights resident: {torch.cuda.memory_allocated()/1e9:.2f} GB",
              flush=True)
    else:
        model = model.eval()
        print("[cpu]  running on CPU", flush=True)
    return model


class MicProducer(threading.Thread):
    """Pulls raw int16 frames from ALSA, converts to float32, appends to a thread-safe buffer.

    Also exports a rolling peak so the UI can show a level meter.
    """
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
                type=alsaaudio.PCM_CAPTURE,
                mode=alsaaudio.PCM_NORMAL,
                device=self.device,
                channels=CH,
                rate=SR,
                format=FMT,
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
                    # Exponential decay so meter doesn't latch high after a single transient.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="default",
                    help="ALSA capture PCM (default: 'default' -> PipeWire)")
    ap.add_argument("--lang", default="en-US",
                    help="target language (e.g. en-US, vi-VN, es-ES, auto)")
    ap.add_argument("--chunk", choices=list(ATT_CONTEXT), default="560ms",
                    help="streaming chunk size; smaller = lower latency, slightly worse WER")
    ap.add_argument("--keep-tag", action="store_true",
                    help="keep the trailing language tag (e.g. <en-US>) in committed lines")
    ap.add_argument("--watchdog", type=int, default=25,
                    help="force utterance commit + RNNT reset after N chunks without new tokens (0 = off)")
    args = ap.parse_args()

    att_ctx = ATT_CONTEXT[args.chunk]
    model = load_model()
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

    print(f"\n[stream] lang={args.lang} chunk={args.chunk} att_ctx={att_ctx} "
          f"device={args.device}", flush=True)
    print(f"[cfg]    chunk_samples={chunk_samples} streaming_cfg.chunk_size={cfg.chunk_size} "
          f"shift_size={cfg.shift_size} pre_encode_cache_size={cfg.pre_encode_cache_size}",
          flush=True)
    if args.watchdog > 0:
        wd_secs = args.watchdog * chunk_secs
        print(f"[cfg]    watchdog={args.watchdog} chunks ({wd_secs:.1f}s) -> commit+reset",
              flush=True)
    print("[*] speak — Ctrl-C to stop\n", flush=True)

    producer = MicProducer(args.device)
    producer.start()
    time.sleep(0.05)

    # Streaming state
    previous_hypotheses = None
    pred_out_stream = None
    step = 0
    # Display state
    last_partial = ""
    chunks_since_change = 0
    t_start = time.time()

    def render_partial(partial):
        elapsed = time.time() - t_start
        bar = level_bar(producer.peak())
        # \r + ANSI clear-to-end-of-line, then the live partial.
        sys.stdout.write(
            f"\r\x1b[2K[{elapsed:6.1f}s #{step:4d} {bar}] {partial}"
        )
        sys.stdout.flush()

    def commit(partial):
        """Move the current partial to a finalized line above, leaving the
        live line empty for the next utterance."""
        nonlocal previous_hypotheses, pred_out_stream, last_partial, chunks_since_change
        finalized = partial if args.keep_tag else strip_lang_tags(partial)
        finalized = finalized.strip()
        # Erase the live line, print finalized text + newline, leave cursor on fresh line.
        sys.stdout.write(f"\r\x1b[2K{finalized}\n")
        sys.stdout.flush()
        # Reset RNNT decoder state. Encoder caches stay — acoustic context continues.
        previous_hypotheses = None
        pred_out_stream = None
        last_partial = ""
        chunks_since_change = 0

    try:
        while True:
            chunk_audio_raw = producer.take(chunk_samples)
            if chunk_audio_raw is None:
                # Keep heartbeat going even when waiting for samples.
                render_partial(last_partial)
                time.sleep(0.02)
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
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = result
                step += 1
                hyp = transcribed_texts[0] if transcribed_texts else None
                raw_text = getattr(hyp, "text", "") if hyp is not None else ""

                if raw_text != last_partial:
                    chunks_since_change = 0
                else:
                    chunks_since_change += 1

                # If the running text now ends with an end-of-utterance lang tag,
                # the model considers the utterance complete. Commit + reset.
                stripped = raw_text.rstrip()
                if stripped.endswith(">") and LANG_TAG_RE.search(stripped):
                    if stripped.endswith(LANG_TAG_RE.findall(stripped)[-1]):
                        commit(raw_text)
                        continue

                last_partial = raw_text

                # Watchdog: if we've been emitting the same partial for too long,
                # force a commit + reset so the RNNT state stays bounded.
                if args.watchdog > 0 and chunks_since_change >= args.watchdog and raw_text:
                    commit(raw_text)
                    continue

            render_partial(last_partial)
    except KeyboardInterrupt:
        # Flush whatever was pending.
        if last_partial:
            sys.stdout.write(
                f"\r\x1b[2K{(last_partial if args.keep_tag else strip_lang_tags(last_partial)).strip()}\n"
            )
        sys.stdout.write("[done]\n")
        sys.stdout.flush()
    finally:
        producer.stop()


if __name__ == "__main__":
    main()
