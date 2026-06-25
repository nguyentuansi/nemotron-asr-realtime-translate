# Step 03 — wrap as a drop-in encoder, integrate, benchmark

**Why**: the optimization is worthless until `stream_translate.sh` actually
uses it. This step wraps the ONNX session so it's interface-compatible with
`model.encoder` and the rest of the streaming loop stays untouched.

**Time**: 2-3 hours.

**Output**:
- `onnx_encoder.py` — drop-in replacement for `model.encoder`
- `load_asr()` in `stream_translate.py` + `stream_web.py` patched to use it
- Final RTF + WER numbers vs Chapter 01's baseline

## 1. What `model.encoder` actually exposes

Before writing a wrapper, list exactly which methods/attributes the streaming
loop touches:

```bash
grep -nE "model\.encoder\." stream_translate.py
```

The streaming loop in `stream_translate.py` calls (verified by grep):

| Call site | What it needs |
|---|---|
| `model.encoder.set_default_att_context_size([70, 6])` | a method that updates internal config |
| `model.encoder.get_initial_cache_state(batch_size=1, device=dev_torch)` | returns 3 tensors |
| `model.encoder.streaming_cfg.chunk_size` + `drop_extra_pre_encoded` | a config object with attributes |
| `model.conformer_stream_step(processed_signal=..., cache_last_channel=..., ...)` | this calls `self.encoder(...)` internally |

The wrapper must duck-type all four. We'll keep the original PyTorch encoder
alive in the background to delegate `set_default_att_context_size`,
`get_initial_cache_state`, and `streaming_cfg` to it (they're cheap and
stateful).

## 2. The wrapper

```python
# onnx_encoder.py
"""Drop-in replacement for nemo_asr Conformer encoder, backed by ONNX Runtime.

Only the heavy forward call is routed through ONNX. The auxiliary methods
(cache state init, config) stay on the original PyTorch encoder — they're
called rarely and aren't the bottleneck.

Usage:
    from onnx_encoder import wrap_encoder_with_onnx
    model = nemo_asr.models.ASRModel.from_pretrained(..., map_location="cpu").eval()
    wrap_encoder_with_onnx(model, "models/encoder.fp16.onnx", providers=...)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch


class ONNXEncoderWrapper:
    """Duck-typed encoder. Delegates everything but forward() to the original."""

    def __init__(self, original_encoder, onnx_path: str | Path, providers):
        self._orig = original_encoder
        self._sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_names = [i.name for i in self._sess.get_inputs()]
        self._output_names = [o.name for o in self._sess.get_outputs()]

    # --- delegated to original PyTorch encoder ---
    def set_default_att_context_size(self, *a, **kw):
        return self._orig.set_default_att_context_size(*a, **kw)

    def get_initial_cache_state(self, *a, **kw):
        return self._orig.get_initial_cache_state(*a, **kw)

    @property
    def streaming_cfg(self):
        return self._orig.streaming_cfg

    # --- the hot path: route through ONNX ---
    def __call__(self, *, audio_signal, length,
                 cache_last_channel, cache_last_time, cache_last_channel_len):
        feeds = {
            "audio_signal": audio_signal.detach().cpu().numpy(),
            "length": length.detach().cpu().numpy(),
            "cache_last_channel": cache_last_channel.detach().cpu().numpy(),
            "cache_last_time": cache_last_time.detach().cpu().numpy(),
            "cache_last_channel_len": cache_last_channel_len.detach().cpu().numpy(),
        }
        outs = self._sess.run(self._output_names, feeds)
        # Convert back to torch on CPU. The downstream RNNT decoder is CPU-only
        # in this configuration, so no device juggling needed.
        return tuple(torch.from_numpy(o) for o in outs)


def wrap_encoder_with_onnx(model, onnx_path, providers=None):
    """Mutate `model` so its encoder forward routes through ONNX. Returns model."""
    if providers is None:
        providers = ["CPUExecutionProvider"]
    model.encoder = ONNXEncoderWrapper(model.encoder, onnx_path, providers)
    return model
```

### Key design notes

- We **don't** subclass `nn.Module`. The original encoder is gone from the
  model's parameter list — but we never trained, so that's fine. If you ever
  want to mix-and-match (PyTorch encoder for warmup, ONNX for steady state),
  subclass `nn.Module` and wire `forward()`.
- `__call__` instead of `forward()` because `conformer_stream_step` does
  `self.encoder(...)` not `self.encoder.forward(...)`. Both work, but `__call__`
  is what gets invoked.
- Cache tensors are large (~1-2 MB each at FP32 / ~0.5-1 MB at FP16). The
  per-chunk numpy↔torch overhead is real — measure it with cProfile if
  RTF gain is less than expected.

## 3. Patch `load_asr()` in both stream scripts

```python
# stream_translate.py, in load_asr() after `model = model.to(...).eval()` etc.

ONNX_PATHS = [
    HERE / "models" / "encoder.fp16.onnx",
    HERE / "models" / "encoder.int8.onnx",
]
onnx_path = next((p for p in ONNX_PATHS if p.exists()), None)
if onnx_path and os.environ.get("NO_ONNX") != "1":
    from onnx_encoder import wrap_encoder_with_onnx
    providers = _pick_onnx_providers()
    wrap_encoder_with_onnx(model, onnx_path, providers=providers)
    print(f"[asr]  encoder routed through ONNX ({onnx_path.name}, providers={[p[0] if isinstance(p, tuple) else p for p in providers]})",
          flush=True)


def _pick_onnx_providers():
    import onnxruntime as ort
    avail = ort.get_available_providers()
    chain = []
    if "CoreMLExecutionProvider" in avail:
        chain.append(("CoreMLExecutionProvider", {
            "MLComputeUnits": "CPUAndNeuralEngine",  # tries ANE first, falls back
            "ModelFormat": "MLProgram",
        }))
    chain.append("CPUExecutionProvider")  # always last as fallback
    return chain
```

Add `NO_ONNX=1` escape hatch — if ONNX ever produces bad output, the user
can disable it without deleting the file.

## 4. Install onnxruntime properly

```bash
# macOS Apple Silicon (gets CoreML EP via the onnxruntime wheel)
./.venv/bin/pip install onnxruntime

# Linux x86_64 / Windows
./.venv/bin/pip install onnxruntime

# If you have CUDA and want GPU EP too:
./.venv/bin/pip install onnxruntime-gpu
```

Then add to `requirements.txt`:

```
# ONNX runtime — used for the optimized encoder. CoreML EP ships in the
# onnxruntime macOS wheel; no separate install needed on Apple Silicon.
onnxruntime
onnxconverter-common; sys_platform == "darwin"
```

## 5. Re-measure RTF + WER

Re-run the two scripts from Chapter 01:

```bash
./.venv/bin/python bench/measure_rtf.py    # writes rtf_baseline.json again — rename first!
./.venv/bin/python bench/measure_wer.py
```

Compare:

```python
# bench/diff_bench.py
import json
b = json.loads(open("bench/rtf_baseline.json").read())
n = json.loads(open("bench/rtf_onnx.json").read())     # rename after re-run
bw = json.loads(open("bench/wer_baseline.json").read())
nw = json.loads(open("bench/wer_onnx.json").read())

print(f"RTF: {b['avg_rtf']:.2f} -> {n['avg_rtf']:.2f}  ({(1-n['avg_rtf']/b['avg_rtf'])*100:+.0f}%)")
print(f"chunk_ms: {b['avg_chunk_ms']:.0f} -> {n['avg_chunk_ms']:.0f}")
print(f"WER: {bw['wer']*100:.2f}% -> {nw['wer']*100:.2f}%  (Δ={nw['wer']-bw['wer']:+.4f})")
print(f"CER: {bw['cer']*100:.2f}% -> {nw['cer']*100:.2f}%  (Δ={nw['cer']-bw['cer']:+.4f})")
```

## 6. Accept / reject criteria

Ship the ONNX path only if **all four hold**:

| Criterion | Threshold |
|---|---|
| Avg RTF | < 0.8 (ideally < 0.5) |
| Chunk_ms | < 400 ms |
| WER drift | < 5% relative (i.e. baseline 20% → new ≤ 21%) |
| CER drift | < 5% relative (tone marks for Vietnamese) |

If RTF improved but WER/CER drifted too far: you quantized too aggressively.
- Re-do Chapter 02 in FP16 instead of INT8 (Mac)
- Or use static INT8 with Vietnamese calibration (Linux/Win)

If WER held but RTF didn't drop enough:
- Profile with cProfile to find where time goes — likely numpy↔torch copies
- Confirm CoreML EP actually ran (set `ORT_LOG_LEVEL=VERBOSE` and look for `CoreML: Adding ...` lines)
- If CoreML fell back to CPU for every node, your ONNX uses ops CoreML doesn't support — Chapter 04 (Plan B)

## 7. Smoke-test in the actual app

```bash
./stream_translate.sh --no-translate --lang en-US
# Watch for: "[asr]  encoder routed through ONNX (encoder.fp16.onnx, providers=[...])"
# Speak something. Confirm transcript appears with low latency.
```

Then with translation:

```bash
./stream_translate.sh --translator envit5
# Speak Vietnamese. Latency from end-of-speech to displayed translation
# should be ~1s, vs ~5-10s on CPU baseline.
```

## 8. Roll back if needed

```bash
NO_ONNX=1 ./stream_translate.sh
```

Sticks back to PyTorch encoder. Use this immediately if you see bad transcripts
in production while you debug.

## Stop and assess before Chapter 04

You should now have:
- An ONNX-routed encoder that meets all four accept criteria
- A clean rollback path via `NO_ONNX=1`
- Final RTF/WER numbers committed to `bench/`

If all green, you're done. Chapter 04 is only for cases where the ONNX
approach didn't work or didn't speed things up enough.
