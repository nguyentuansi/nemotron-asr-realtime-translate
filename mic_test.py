"""Interactive mic test for nvidia/nemotron-3.5-asr-streaming-0.6b.

Push-to-talk: hit ENTER to start recording, ENTER again to stop and transcribe.
Commands at the prompt:
    q              quit
    d              list capture devices
    dev <name>     switch ALSA capture device (default: 'default' -> pipewire)
    lang <code>    switch target language (default: en-US)
    s <sec>        timed record for N seconds (no second ENTER needed)
    h              help
"""
import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import soundfile as sf
try:
    import alsaaudio
except ImportError:
    import alsa_shim as alsaaudio
import torch

SR = 16000
CH = 1
FMT = alsaaudio.PCM_FORMAT_S16_LE
PERIOD = 1024  # frames per ALSA read

# ---------- capture ----------

class MicRecorder:
    """Background thread that drains a single ALSA capture PCM into a list of int16 chunks."""
    def __init__(self, device="default", sr=SR):
        self.device = device
        self.sr = sr
        self._stop = threading.Event()
        self._chunks = []
        self._peak = 0
        self._thread = None
        self._error = None

    def _run(self):
        try:
            pcm = alsaaudio.PCM(
                type=alsaaudio.PCM_CAPTURE,
                mode=alsaaudio.PCM_NORMAL,
                device=self.device,
                channels=CH,
                rate=self.sr,
                format=FMT,
                periodsize=PERIOD,
            )
        except alsaaudio.ALSAAudioError as e:
            self._error = e
            return
        try:
            while not self._stop.is_set():
                length, data = pcm.read()
                if length <= 0:
                    continue
                arr = np.frombuffer(data, dtype=np.int16)
                if arr.size:
                    self._chunks.append(arr.copy())
                    p = int(np.abs(arr).max())
                    if p > self._peak:
                        self._peak = p
        finally:
            pcm.close()

    def start(self):
        self._stop.clear()
        self._chunks.clear()
        self._peak = 0
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def peak(self):
        p = self._peak
        self._peak = 0
        return p

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._error is not None:
            raise self._error
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._chunks).astype(np.float32) / 32768.0
        return audio


def level_bar(peak_i16, width=30):
    # peak in dBFS roughly: 20*log10(peak/32767). Map [-60, 0] dB -> [0, width].
    if peak_i16 <= 0:
        n = 0
    else:
        db = 20.0 * np.log10(peak_i16 / 32767.0)
        n = int(np.clip((db + 60.0) / 60.0 * width, 0, width))
    return "[" + "#" * n + "-" * (width - n) + f"] {peak_i16:5d}"


# ---------- model ----------

def load_model():
    import nemo.collections.asr as nemo_asr
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"[env] HF_HOME={os.environ['HF_HOME']}")
    t0 = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
        map_location="cpu",
    )
    print(f"[load] {time.time()-t0:.1f}s on cpu, type={type(model).__name__}")
    use_gpu = torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1"
    if use_gpu:
        torch.cuda.empty_cache()
        model = model.half().to("cuda").eval()
        print(f"[gpu]  fp16 weights resident: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    else:
        model = model.eval()
        print("[cpu]  running on CPU")
    return model


def _trim_and_normalize(audio_f32, sr):
    """Trim leading/trailing silence (RMS-window VAD), normalize peak to 0.9.

    The mic-captured audio often has a second of pre/post silence and sits
    around -10 dBFS. The model returns empty text on this input — verified
    by side-by-side diagnostic — so we preprocess to match the level/structure
    of the LibriSpeech samples the model handles confidently.
    """
    if audio_f32.size == 0:
        return audio_f32
    win = int(0.025 * sr)
    hop = int(0.010 * sr)
    if len(audio_f32) <= win:
        norm = audio_f32 * (0.9 / max(float(np.abs(audio_f32).max()), 1e-6))
        return norm
    rms = np.sqrt(np.array([
        np.mean(audio_f32[i:i + win] ** 2)
        for i in range(0, len(audio_f32) - win, hop)
    ]))
    thr = max(0.01, float(np.percentile(rms, 25)) * 3)
    active = rms > thr
    if not active.any():
        out = audio_f32
    else:
        first = int(np.argmax(active)) * hop
        last = (len(active) - 1 - int(np.argmax(active[::-1]))) * hop + win
        pad = hop * 5
        out = audio_f32[max(0, first - pad): min(len(audio_f32), last + pad)]
    out = out * (0.9 / max(float(np.abs(out).max()), 1e-6))
    return out


def transcribe(model, audio_f32, lang):
    """Write a one-cut manifest with the right 'lang' field, then call model.transcribe()."""
    import json
    audio_proc = _trim_and_normalize(audio_f32, SR)
    tmp_wav = HERE / "audio" / "_mic_buffer.wav"
    tmp_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp_wav), audio_proc, SR, subtype="PCM_16")
    manifest = HERE / "audio" / "_mic_manifest.json"
    with manifest.open("w") as fp:
        # Match smoke_test.py: a giant duration sentinel keeps Lhotse from
        # filtering the cut by min/max-duration buckets. Using the real
        # duration silently returned empty transcriptions in testing.
        fp.write(json.dumps({
            "audio_filepath": str(tmp_wav),
            "duration": 100000,
            "text": "",
            "lang": lang,
        }) + "\n")
    t0 = time.time()
    out = model.transcribe(audio=[str(manifest)], batch_size=1, target_lang=lang)
    dur = time.time() - t0
    text = out[0]
    text = getattr(text, "text", text)
    return text, dur


# ---------- ui ----------

HELP = """\
commands:
  ENTER          start recording; ENTER again to stop and transcribe
  s <sec>        timed record for N seconds, then transcribe
  d              list capture devices
  dev <name>     switch ALSA capture device
  lang <code>    switch target language (e.g. en-US, vi-VN)
  h              this help
  q              quit
"""


def list_devices():
    print("ALSA capture PCMs (use 'dev <name>'):")
    for p in alsaaudio.pcms(alsaaudio.PCM_CAPTURE):
        print(f"  {p}")


def record_with_meter(rec, stop_event):
    """Print a one-line peak meter while the capture thread runs."""
    started = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - started
        bar = level_bar(rec.peak())
        sys.stdout.write(f"\r  rec {elapsed:5.1f}s  {bar}")
        sys.stdout.flush()
        time.sleep(0.05)
    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="default",
                    help="ALSA capture PCM name (default: 'default' -> pipewire)")
    ap.add_argument("--lang", default="en-US",
                    help="target language code (default: en-US)")
    args = ap.parse_args()

    device = args.device
    lang = args.lang

    model = load_model()
    print()
    print(HELP)
    print(f"[mic ] device={device}  lang={lang}")
    print("ready.")

    while True:
        try:
            line = input(f"[{lang}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if line == "":
            # Push-to-talk: ENTER to start, ENTER to stop.
            rec = MicRecorder(device=device)
            rec.start()
            time.sleep(0.05)
            print("  recording... press ENTER to stop")
            stop = threading.Event()

            def wait_enter():
                try:
                    input()
                finally:
                    stop.set()

            t = threading.Thread(target=wait_enter, daemon=True)
            t.start()
            try:
                record_with_meter(rec, stop)
            except KeyboardInterrupt:
                stop.set()
            audio = rec.stop()
        elif line.startswith("s "):
            try:
                sec = float(line.split()[1])
            except (IndexError, ValueError):
                print("usage: s <seconds>")
                continue
            rec = MicRecorder(device=device)
            rec.start()
            print(f"  recording {sec:.1f}s...")
            stop = threading.Event()
            t0 = time.time()
            try:
                while time.time() - t0 < sec:
                    bar = level_bar(rec.peak())
                    sys.stdout.write(f"\r  rec {time.time()-t0:5.1f}s  {bar}")
                    sys.stdout.flush()
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass
            sys.stdout.write("\r" + " " * 70 + "\r")
            audio = rec.stop()
        elif line in ("q", "quit", "exit"):
            break
        elif line in ("h", "help", "?"):
            print(HELP); continue
        elif line == "d":
            list_devices(); continue
        elif line.startswith("dev "):
            device = line[4:].strip()
            print(f"[mic ] device={device}")
            continue
        elif line.startswith("lang "):
            lang = line[5:].strip()
            print(f"[mic ] lang={lang}")
            continue
        else:
            print("unknown. 'h' for help.")
            continue

        if audio.size == 0:
            print("  (no audio captured)")
            continue
        dur_s = audio.size / SR
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        print(f"  captured {dur_s:.2f}s  rms={rms:.4f}  peak={float(np.abs(audio).max()):.3f}")
        if rms < 1e-4:
            print("  WARNING: audio looks silent. Check mic / 'dev <name>'.")
        try:
            text, infer_s = transcribe(model, audio, lang)
            rtf = infer_s / max(dur_s, 1e-6)
            print(f"  [{lang}] ({infer_s:.2f}s, rtf={rtf:.2f}) {text}")
        except Exception as e:
            print(f"  transcribe failed: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
