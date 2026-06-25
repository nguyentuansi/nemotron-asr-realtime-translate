"""Step 02 §5 — export encoder with cache I/O via torch.onnx.export.

NeMo's default model.encoder.export() emits a 2-arg forward that drops the
streaming cache. We need the 5-arg forward (audio_signal, length, cache_lc,
cache_lt, cache_lcl) for the streaming pipeline. Produces models/encoder.onnx.
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
import torch.nn as nn
import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nl; nl.setLevel(logging.ERROR)

print("[export] loading model...")
m = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu"
).eval()
m.encoder.set_default_att_context_size([56, 6])  # matches stream_translate.py ATT_CONTEXT['560ms']


class StreamingEncoderForExport(nn.Module):
    """Pure 5-arg forward so torch.onnx.export sees the streaming signature."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal, length,
                cache_last_channel, cache_last_time, cache_last_channel_len):
        return self.encoder(
            audio_signal=audio_signal, length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )


wrapper = StreamingEncoderForExport(m.encoder).eval()

# Build representative example tensors. Frames vary across chunks; use the
# later-chunk size (56) since that's what runs >99% of the time.
clc, clt, clcl = m.encoder.get_initial_cache_state(batch_size=1, device="cpu")
audio = torch.randn(1, 128, 56)
length = torch.tensor([56], dtype=torch.int64)

print(f"[export] input shapes:")
print(f"  audio_signal           {tuple(audio.shape)}")
print(f"  length                 {tuple(length.shape)}")
print(f"  cache_last_channel     {tuple(clc.shape)}")
print(f"  cache_last_time        {tuple(clt.shape)}")
print(f"  cache_last_channel_len {tuple(clcl.shape)}")

# Verify forward works with this input combo
with torch.inference_mode():
    ref = wrapper(audio, length, clc, clt, clcl)
print(f"[export] reference forward OK — output shapes:")
for i, o in enumerate(ref):
    print(f"  [{i}] {tuple(o.shape)} {o.dtype}")

OUT = HERE / "models" / "encoder.onnx"
OUT.parent.mkdir(exist_ok=True)
print(f"\n[export] exporting to {OUT} (this takes a few minutes for a 600M model)...")
torch.onnx.export(
    wrapper,
    (audio, length, clc, clt, clcl),
    str(OUT),
    input_names=["audio_signal", "length",
                 "cache_last_channel", "cache_last_time", "cache_last_channel_len"],
    output_names=["outputs", "encoded_lengths",
                  "cache_last_channel_next", "cache_last_time_next",
                  "cache_last_channel_len_next"],
    dynamic_axes={
        "audio_signal": {0: "batch", 2: "frames"},
        "length": {0: "batch"},
        # cache shapes: lc=(24,B,70,1024) lt=(24,B,1024,8) lcl=(B,)
        # Only batch dim is dynamic for a streaming pipeline.
        "cache_last_channel": {1: "batch"},
        "cache_last_time": {1: "batch"},
        "cache_last_channel_len": {0: "batch"},
        "outputs": {0: "batch", 2: "out_frames"},
        "encoded_lengths": {0: "batch"},
        "cache_last_channel_next": {1: "batch"},
        "cache_last_time_next": {1: "batch"},
        "cache_last_channel_len_next": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)
size_mb = OUT.stat().st_size / 1e6
print(f"[export] wrote {OUT.name} ({size_mb:.1f} MB main + external weight files)")

# Re-inspect to confirm cache I/O made it
import onnx
g = onnx.load(str(OUT), load_external_data=False).graph
print(f"\n[verify] INPUTS:  {[i.name for i in g.input]}")
print(f"[verify] OUTPUTS: {[o.name for o in g.output]}")
cache_in_ok = {"cache_last_channel", "cache_last_time", "cache_last_channel_len"}.issubset(
    {i.name for i in g.input})
cache_out_ok = {"cache_last_channel_next", "cache_last_time_next", "cache_last_channel_len_next"}.issubset(
    {o.name for o in g.output})
print(f"\n[verify] cache inputs present:  {cache_in_ok}")
print(f"[verify] cache outputs present: {cache_out_ok}")
assert cache_in_ok and cache_out_ok, "cache I/O missing — export is broken"
print("[verify] DONE — proceed to verify_onnx.py")
