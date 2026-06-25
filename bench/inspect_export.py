"""Step 02 §1 — try the default NeMo encoder export and inspect what came out.

Reports: input/output names, shapes, opset, file size. Tells you whether the
cache state is in the graph (success) or only forward(audio, length) is
(needs §5 patching).
"""
import os, sys, logging
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
logging.getLogger("nemo_logger").setLevel(logging.WARNING)
logging.getLogger("nv_one_logger").setLevel(logging.ERROR)

import torch
import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nl; nl.setLevel(logging.ERROR)

print("[export] loading model...")
m = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu"
).eval()
m.encoder.set_default_att_context_size([56, 6])  # matches stream_translate ATT_CONTEXT["560ms"]

OUT = HERE / "models"
OUT.mkdir(exist_ok=True)
onnx_path = OUT / "encoder_raw.onnx"

print(f"[export] attempting m.encoder.export({onnx_path})")
try:
    m.encoder.export(str(onnx_path))
    print(f"[export] SUCCESS — wrote {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")
except Exception as e:
    print(f"[export] FAILED: {type(e).__name__}: {e}")
    raise

print("\n[inspect] reading ONNX graph...")
import onnx
g = onnx.load(str(onnx_path)).graph
print("INPUTS:")
for i in g.input:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]
    print(f"  {i.name:35s} shape={dims}")
print("OUTPUTS:")
for o in g.output:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]
    print(f"  {o.name:35s} shape={dims}")

inputs = {i.name for i in g.input}
cache_inputs = {"cache_last_channel", "cache_last_time", "cache_last_channel_len"}
has_cache = cache_inputs.issubset(inputs)
print(f"\n[verdict] cache I/O present: {has_cache}")
if has_cache:
    print("  -> proceed to verify_onnx.py (Step 02 §2)")
else:
    print("  -> default export skipped cache state; need export_with_cache.py (Step 02 §5)")
