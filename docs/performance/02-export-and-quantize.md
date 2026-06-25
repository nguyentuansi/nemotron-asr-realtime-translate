# Step 02 — export encoder to ONNX, choose precision

**Why**: the encoder is 85% of streaming wall time. Everything else in this
plan depends on having a working ONNX file that round-trips outputs against
PyTorch within tolerance.

**Time**: 1-2 hours if cache I/O export works first try; up to a day if you
have to subclass `forward_for_export`.

**Output**: `models/encoder.fp16.onnx` (Mac path) **or** `models/encoder.int8.onnx`
(Linux/Win path), plus a verification script that proves output matches PyTorch.

## 1. The big risk to confirm first

NeMo's `model.encoder.export(path)` exports the **forward** function. The
streaming path uses a custom forward signature with **5 extra cache I/O
tensors**:

```python
out, encoded_len, new_cache_lc, new_cache_lt, new_cache_lcl = encoder(
    audio_signal, length,
    cache_last_channel, cache_last_time, cache_last_channel_len,
)
```

If those cache inputs/outputs don't appear in the exported graph, the export
is useless for streaming — you'll get full-file inference only, which we
already have.

**Spend the first 30 minutes verifying export shape before doing anything else.**

```python
# scripts/inspect_export.py
import os, logging
os.environ["HF_HOME"] = os.path.abspath(".hf-cache")
logging.getLogger("nemo_logger").setLevel(logging.WARNING)
logging.getLogger("nv_one_logger").setLevel(logging.ERROR)

import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nl; nl.setLevel(logging.ERROR)

m = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu"
).eval()
m.encoder.set_default_att_context_size([70, 6])

# Try the default export first
m.encoder.export("models/encoder_raw.onnx")

import onnx
g = onnx.load("models/encoder_raw.onnx").graph
print("INPUTS :", [i.name for i in g.input])
print("OUTPUTS:", [o.name for o in g.output])
```

**What you want to see**:
```
INPUTS : ['audio_signal', 'length', 'cache_last_channel', 'cache_last_time', 'cache_last_channel_len']
OUTPUTS: ['outputs', 'encoded_lengths', 'cache_last_channel_next', 'cache_last_time_next', 'cache_last_channel_len_next']
```

**What you might see instead** (the default `forward_for_export`):
```
INPUTS : ['audio_signal', 'length']
OUTPUTS: ['outputs', 'encoded_lengths']
```

If you got the second case, jump to **§5 — patching `forward_for_export`**
below. If you got the first, continue to §2.

## 2. Verify ONNX output matches PyTorch within tolerance

Numerical drift between PyTorch and ONNX is normal (1e-5 to 1e-3 depending on
ops used). Drift > 1e-2 means an op was approximated badly and downstream
decoding will misbehave.

```python
# scripts/verify_onnx.py
import numpy as np, onnxruntime as ort, torch
import nemo.collections.asr as nemo_asr

m = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu"
).eval()
m.encoder.set_default_att_context_size([70, 6])

sess = ort.InferenceSession("models/encoder_raw.onnx", providers=["CPUExecutionProvider"])

# Random-but-shape-correct inputs matching what the streaming loop sends
audio = torch.randn(1, 80, 49)  # (batch, mel_features, frames) - check your model's expected shape!
length = torch.tensor([49])
clc, clt, clcl = m.encoder.get_initial_cache_state(batch_size=1, device="cpu")

with torch.inference_mode():
    pt_out = m.encoder(audio_signal=audio, length=length,
                       cache_last_channel=clc, cache_last_time=clt,
                       cache_last_channel_len=clcl)

ort_out = sess.run(None, {
    "audio_signal": audio.numpy(), "length": length.numpy(),
    "cache_last_channel": clc.numpy(), "cache_last_time": clt.numpy(),
    "cache_last_channel_len": clcl.numpy(),
})

for i, (a, b) in enumerate(zip(pt_out, ort_out)):
    a = a.numpy() if torch.is_tensor(a) else a
    diff = np.abs(a - b).max()
    print(f"output[{i}] max abs diff = {diff:.2e}  shape pt={a.shape} ort={b.shape}")
    assert diff < 1e-2, f"output {i} drift too large"

print("VERIFY OK")
```

**Tolerance interpretation**:
- < 1e-5 → perfect, FP32 round-trip exact
- 1e-5 to 1e-3 → expected for FP32 ONNX
- 1e-3 to 1e-2 → acceptable, watch for downstream WER changes
- &gt; 1e-2 → an op got approximated wrong; do NOT proceed

If you fail this, the ONNX is broken — don't bother quantizing.

## 3. Decide FP16 vs INT8

**On macOS Apple Silicon** → FP16, always. The Neural Engine only runs FP16
and gives you 3-5× speedup. INT8 forces CoreML to fall back to CPU/GPU and you
lose the ANE win.

**On Linux/Windows CPU** → INT8, always. No ANE to target; INT8 on
x86_64 (AVX-VNNI) or ARM (NEON dotprod) gives 1.5-2× over FP32.

**On Linux/CUDA** → don't bother; PyTorch CUDA already runs this model at
RTF 0.1-0.2.

### 3a. FP16 conversion (Mac path)

```python
# scripts/convert_fp16.py
from onnxconverter_common import float16
import onnx
model = onnx.load("models/encoder_raw.onnx")
# keep_io_types=True so the wrapper still hands in FP32 (CoreML EP casts internally)
fp16_model = float16.convert_float_to_float16(model, keep_io_types=True)
onnx.save(fp16_model, "models/encoder.fp16.onnx")
```

Install dep if missing:
```bash
./.venv/bin/pip install onnxconverter-common
```

Re-run `verify_onnx.py` against `encoder.fp16.onnx`. Tolerance budget rises
from 1e-3 to ~5e-3 — that's the FP16 rounding error. Anything above 1e-1
means an op is poorly represented in FP16; fall back to FP32+CoreML.

### 3b. INT8 dynamic quantization (Linux/Win path)

```python
# scripts/quantize_int8.py
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    "models/encoder_raw.onnx",
    "models/encoder.int8.onnx",
    weight_type=QuantType.QInt8,
    # Conformer has LayerNorm + GELU; some quantizers crash on those — exclude:
    nodes_to_exclude=[],
    op_types_to_quantize=["MatMul", "Gemm", "Conv"],
)
```

INT8 dynamic only quantizes weights (activations stay FP32). For better
accuracy on tone-language Vietnamese, use **static quantization with
Vietnamese calibration audio** — see §4.

### 3c. Static INT8 with Vietnamese calibration (better accuracy)

```python
# scripts/quantize_int8_static.py
import numpy as np, soundfile as sf
from pathlib import Path
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

class ViCalibReader(CalibrationDataReader):
    def __init__(self, wav_dir):
        self.wavs = sorted(Path(wav_dir).glob("*.wav"))
        self.idx = 0
    def get_next(self):
        if self.idx >= len(self.wavs): return None
        # Produce one valid input dict matching the ONNX graph's input names + shapes.
        # Use the same preprocessing the model does internally — easiest is to
        # call model.preprocessor(audio) once and stash that as numpy.
        ...
        self.idx += 1
        return {"audio_signal": ..., "length": ..., "cache_last_channel": ...,
                "cache_last_time": ..., "cache_last_channel_len": ...}

quantize_static(
    "models/encoder_raw.onnx", "models/encoder.int8.onnx",
    ViCalibReader("audio/bench"),
    quant_format=QuantType.QInt8,
)
```

100-500 Vietnamese clips is enough to calibrate. Use your `audio/bench/`
clips — they're already representative of your real use.

## 4. Sanity-check with Vietnamese audio + WER

This is the gate. Compare WER on `audio/bench/` between PyTorch and the
quantized ONNX (you'll need the full integration from Chapter 03 to do this
properly, but a quick spot-check on 3-5 clips is enough to catch a disaster).

| Precision | Expected WER drift vs FP32 PyTorch | If you see worse |
|---|---|---|
| FP32 ONNX | identical (< 0.5% absolute) | a graph op was wrong — revisit §2 |
| FP16 ONNX | 0-2% absolute | acceptable; ship |
| INT8 dynamic | 2-8% absolute | borderline; try static |
| INT8 static (with Vi calibration) | 1-3% absolute | acceptable; ship |
| INT8 dynamic with > 10% WER drift | INT8 is breaking tone discrimination | fall back to FP16, or skip quantization entirely |

If Vietnamese drift is too high but English/FLEURS is fine, you're hitting the
tone-language INT8 problem. Static calibration with Vietnamese audio is the
proper fix. Don't trust generic English-calibrated INT8 for Vietnamese.

## 5. Patching `forward_for_export` (only if §1 showed no cache I/O)

Subclass to expose the streaming forward as the exporter's entry point:

```python
# scripts/export_with_cache.py
import torch
import nemo.collections.asr as nemo_asr

m = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu"
).eval()

class StreamingEncoderForExport(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
    def forward(self, audio_signal, length,
                cache_last_channel, cache_last_time, cache_last_channel_len):
        out = self.enc(
            audio_signal=audio_signal, length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        return out  # tuple of 5

m.encoder.set_default_att_context_size([70, 6])
wrapper = StreamingEncoderForExport(m.encoder)

# Build example inputs with correct shapes
clc, clt, clcl = m.encoder.get_initial_cache_state(batch_size=1, device="cpu")
audio = torch.randn(1, 80, 49)  # check your feature dim!
length = torch.tensor([49])

torch.onnx.export(
    wrapper, (audio, length, clc, clt, clcl),
    "models/encoder.onnx",
    input_names=["audio_signal", "length",
                 "cache_last_channel", "cache_last_time", "cache_last_channel_len"],
    output_names=["outputs", "encoded_lengths",
                  "cache_last_channel_next", "cache_last_time_next",
                  "cache_last_channel_len_next"],
    dynamic_axes={
        "audio_signal": {0: "batch", 2: "frames"},
        "length": {0: "batch"},
        "cache_last_channel": {0: "batch"},
        "cache_last_time": {0: "batch"},
        "cache_last_channel_len": {0: "batch"},
        "outputs": {0: "batch", 2: "out_frames"},
        "encoded_lengths": {0: "batch"},
        "cache_last_channel_next": {0: "batch"},
        "cache_last_time_next": {0: "batch"},
        "cache_last_channel_len_next": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)
```

Then re-run §2 verification on `models/encoder.onnx`. If outputs match, proceed
to §3.

## 6. What to commit

```
.gitignore  →  add models/ (binaries, don't commit)

scripts/
  inspect_export.py       # §1
  verify_onnx.py          # §2
  convert_fp16.py         # §3a  (Mac)
  quantize_int8.py        # §3b  (Linux/Win, dynamic)
  quantize_int8_static.py # §3c  (better)
  export_with_cache.py    # §5   (fallback if default export fails)
```

The ONNX file itself (~200-500 MB) shouldn't be in git. Re-generate it on
first deploy; the scripts above are reproducible.

## Stop and assess before Chapter 03

After this chapter you should have:
- `models/encoder.fp16.onnx` (Mac) or `models/encoder.int8.onnx` (other)
- Verification script passes
- Spot-check WER on 3-5 Vietnamese clips within budget

If any of these failed, **don't go to Chapter 03 yet**. Either patch the
issue here or drop to Chapter 04 (Plan B).
