"""Second pass: try silence-trimmed + amplified, and try Vietnamese."""
import os, sys, json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import soundfile as sf
import numpy as np
import torch

BUF = HERE / "audio" / "_mic_buffer.wav"
audio, sr = sf.read(str(BUF))
audio = audio.astype(np.float32)
print(f"[buf] {len(audio)/sr:.2f}s peak={float(np.abs(audio).max()):.3f} rms={float(np.sqrt(np.mean(audio**2))):.4f}")

# Trim leading/trailing silence using a simple RMS-window VAD
win = int(0.025 * sr)
hop = int(0.010 * sr)
rms = np.sqrt(np.array([np.mean(audio[i:i+win]**2) for i in range(0, max(1, len(audio)-win), hop)]))
thr = max(0.01, float(np.percentile(rms, 25)) * 3)
active = rms > thr
if active.any():
    first = int(np.argmax(active)) * hop
    last = (len(active) - 1 - int(np.argmax(active[::-1]))) * hop + win
    trimmed = audio[max(0, first - hop*5): min(len(audio), last + hop*5)]
else:
    trimmed = audio
print(f"[trim] {len(trimmed)/sr:.2f}s active_thr={thr:.4f}")

# Amplify trimmed to peak ~0.95
amp = trimmed * (0.95 / max(float(np.abs(trimmed).max()), 1e-6))
amp_path = HERE / "audio" / "_mic_trim_amp.wav"
sf.write(str(amp_path), amp, sr, subtype="PCM_16")

import nemo.collections.asr as nemo_asr
t0 = time.time()
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
    map_location="cpu",
)
print(f"[load] {time.time()-t0:.1f}s")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    model = model.half().to("cuda").eval()
else:
    model = model.eval()

def run(label, file_path, lang):
    man = HERE / "audio" / f"_diag_{label.replace(' ','_')}.json"
    man.write_text(json.dumps({"audio_filepath": str(file_path), "duration": 100000, "text": "", "lang": lang}) + "\n")
    try:
        t = time.time()
        out = model.transcribe(audio=[str(man)], batch_size=1, target_lang=lang)
        dur = time.time() - t
        text = getattr(out[0], "text", out[0])
        print(f"  {label:<32s} [{lang}] -> ({dur:.2f}s) {text!r}")
    except Exception as e:
        print(f"  {label:<32s} [{lang}] -> ERROR {type(e).__name__}: {e}")

for lang in ("en-US", "vi-VN"):
    run("orig buffer", BUF, lang)
    run("trimmed+amplified", amp_path, lang)
