"""Step 02 §2 — verify ONNX encoder output matches PyTorch within tolerance."""
import os, sys, logging
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
logging.getLogger("nemo_logger").setLevel(logging.WARNING)
logging.getLogger("nv_one_logger").setLevel(logging.ERROR)

import numpy as np, torch, onnxruntime as ort
import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nl; nl.setLevel(logging.ERROR)

m = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu"
).eval()
m.encoder.set_default_att_context_size([56, 6])  # matches stream_translate ATT_CONTEXT["560ms"]

sess = ort.InferenceSession(str(HERE / "models" / "encoder.onnx"),
                            providers=["CPUExecutionProvider"])

# Use a *real* input from a known wav to be representative
import soundfile as sf
audio_raw, sr = sf.read(str(HERE / "audio" / "sample1.flac"))
assert sr == 16000
# Take ~0.56s of audio for one chunk
chunk = torch.from_numpy(audio_raw[:8960].astype(np.float32)).unsqueeze(0)

# Convert to mel via the model's preprocessor (the encoder's input is mel features)
with torch.inference_mode():
    mel, mel_len = m.preprocessor(input_signal=chunk,
                                  length=torch.tensor([8960], dtype=torch.int64))

clc, clt, clcl = m.encoder.get_initial_cache_state(batch_size=1, device="cpu")

with torch.inference_mode():
    pt_out = m.encoder(audio_signal=mel, length=mel_len,
                       cache_last_channel=clc, cache_last_time=clt,
                       cache_last_channel_len=clcl)

ort_out = sess.run(None, {
    "audio_signal": mel.numpy(), "length": mel_len.numpy().astype(np.int64),
    "cache_last_channel": clc.numpy(), "cache_last_time": clt.numpy(),
    "cache_last_channel_len": clcl.numpy().astype(np.int64),
})

names = ["outputs", "encoded_lengths", "cache_last_channel_next",
         "cache_last_time_next", "cache_last_channel_len_next"]
max_drift = 0.0
for name, pt, o in zip(names, pt_out, ort_out):
    pt = pt.numpy() if torch.is_tensor(pt) else pt
    if pt.dtype.kind == "f":
        diff = float(np.abs(pt - o).max())
        max_drift = max(max_drift, diff)
        print(f"  {name:30s} pt={pt.shape} ort={o.shape}  max_abs_diff={diff:.2e}")
    else:
        eq = np.array_equal(pt, o)
        print(f"  {name:30s} pt={pt.shape} ort={o.shape}  exact_match={eq}")

print(f"\n[verify] max float drift across all outputs: {max_drift:.2e}")
if max_drift < 1e-2:
    print("[verify] PASS — drift within tolerance, safe to use ONNX in production")
else:
    print("[verify] FAIL — drift > 1e-2, an op was approximated badly")
