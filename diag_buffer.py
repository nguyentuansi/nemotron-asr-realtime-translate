"""Diagnose why mic_test returned empty text.

Replays the existing audio/_mic_buffer.wav through three transcribe configs:
  A) manifest with real duration         (what mic_test.py just used)
  B) manifest with duration=100000       (what smoke_test.py uses)
  C) raw audio path, no manifest, target_lang only
"""
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
if not BUF.exists():
    print(f"missing {BUF} — record something with mic_test.py first.")
    sys.exit(1)

audio, sr = sf.read(str(BUF))
print(f"[buf] {BUF.name}: sr={sr} dur={len(audio)/sr:.2f}s "
      f"rms={float(np.sqrt(np.mean(audio**2))):.4f} peak={float(np.abs(audio).max()):.3f}")

import nemo.collections.asr as nemo_asr
t0 = time.time()
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
    map_location="cpu",
)
print(f"[load] {time.time()-t0:.1f}s on cpu")
if torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1":
    torch.cuda.empty_cache()
    model = model.half().to("cuda").eval()
    print(f"[gpu]  fp16 weights resident: {torch.cuda.memory_allocated()/1e9:.2f} GB")
else:
    model = model.eval()
    print("[cpu]  running on CPU")

LANG = "en-US"

def run(label, audio_arg, **kw):
    try:
        t = time.time()
        out = model.transcribe(audio=audio_arg, batch_size=1, **kw)
        dur = time.time() - t
        text = out[0]
        text = getattr(text, "text", text)
        print(f"  {label:<40s} -> ({dur:.2f}s) {text!r}")
    except Exception as e:
        print(f"  {label:<40s} -> ERROR {type(e).__name__}: {e}")

# A) real duration
manA = HERE / "audio" / "_diag_A.json"
manA.write_text(json.dumps({"audio_filepath": str(BUF), "duration": float(len(audio)/sr), "text": "", "lang": LANG}) + "\n")
run("A real-duration manifest", [str(manA)], target_lang=LANG)

# B) duration=100000 (smoke_test pattern)
manB = HERE / "audio" / "_diag_B.json"
manB.write_text(json.dumps({"audio_filepath": str(BUF), "duration": 100000, "text": "", "lang": LANG}) + "\n")
run("B duration=100000 manifest", [str(manB)], target_lang=LANG)

# C) raw path, no manifest
run("C raw path + target_lang only", [str(BUF)], target_lang=LANG)

# D) control — sample1.flac through B-style manifest. If this transcribes,
# the model is fine and the mic buffer is the input quality problem.
SAMPLE = HERE / "audio" / "sample1.flac"
if SAMPLE.exists():
    manD = HERE / "audio" / "_diag_D.json"
    manD.write_text(json.dumps({"audio_filepath": str(SAMPLE), "duration": 100000, "text": "", "lang": LANG}) + "\n")
    run("D sample1.flac control", [str(manD)], target_lang=LANG)

# E) re-encode mic buffer to FLAC at 16kHz to rule out WAV/PCM-vs-FLAC quirks.
flac_path = HERE / "audio" / "_mic_buffer.flac"
sf.write(str(flac_path), audio, sr, format="FLAC")
manE = HERE / "audio" / "_diag_E.json"
manE.write_text(json.dumps({"audio_filepath": str(flac_path), "duration": 100000, "text": "", "lang": LANG}) + "\n")
run("E mic buffer re-encoded as FLAC", [str(manE)], target_lang=LANG)

# F) normalize mic buffer to peak ~0.9 to rule out level. mic peak was 0.34.
norm = audio.astype(np.float32) * (0.9 / max(float(np.abs(audio).max()), 1e-6))
norm_path = HERE / "audio" / "_mic_buffer_norm.wav"
sf.write(str(norm_path), norm, sr, subtype="PCM_16")
manF = HERE / "audio" / "_diag_F.json"
manF.write_text(json.dumps({"audio_filepath": str(norm_path), "duration": 100000, "text": "", "lang": LANG}) + "\n")
run("F mic buffer normalized to 0.9", [str(manF)], target_lang=LANG)
