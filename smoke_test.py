"""Smoke test: load nvidia/nemotron-3.5-asr-streaming-0.6b and transcribe two clips."""
import os, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Help with fragmentation on the 6GB RTX 2060.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
print(f"[env] HF_HOME={os.environ['HF_HOME']}")

import nemo.collections.asr as nemo_asr
print("[env] nemo imported")

LANG = "en-US"
# Prompt-conditioned models need 'lang' on each cut. The transcribe() mixin only
# accepts str paths or a single .json manifest. So we write a NeMo-format manifest
# with the 'lang' field set (Lhotse reads it as supervision.language).
import json
MANIFEST = HERE / "audio" / "smoke_manifest.json"
with MANIFEST.open("w") as fp:
    for name in ("sample1.flac", "sample2.flac"):
        fp.write(json.dumps({
            "audio_filepath": str(HERE / "audio" / name),
            "duration": 100000,
            "text": "",
            "lang": LANG,
        }) + "\n")
AUDIO = [str(MANIFEST)]

# Load on CPU first — NeMo's from_pretrained eagerly puts modules on cuda if available,
# which OOMs a 6 GB GPU because it double-allocates during state_dict loading.
t0 = time.time()
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
    map_location="cpu",
)
print(f"[load] {time.time()-t0:.1f}s on cpu, type={type(model).__name__}")

USE_GPU = torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1"
if USE_GPU:
    torch.cuda.empty_cache()
    # fp16 halves weight memory (~1.2 GB instead of ~2.4 GB).
    model = model.half().to("cuda").eval()
    print(f"[gpu]  fp16 weights resident: {torch.cuda.memory_allocated()/1e9:.2f} GB")
else:
    model = model.eval()
    print("[cpu]  running on CPU")

def try_transcribe(**kw):
    t = time.time()
    return model.transcribe(audio=AUDIO, batch_size=2, **kw), time.time() - t

try:
    out, dur = try_transcribe(target_lang=LANG)
    print(f"[infer] target_lang={LANG} -> {dur:.2f}s")
except Exception as e:
    print(f"[infer] failed: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

for i, item in enumerate(out):
    text = getattr(item, "text", item)
    print(f"[result] sample{i+1}: {text}")
