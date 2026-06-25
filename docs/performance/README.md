# Performance docs — making streaming ASR real-time on Mac CPU

Goal: bring the Nemotron-3.5 streaming Conformer from **RTF ≈ 1.55** (slower
than real-time on a quiet M-series Mac) down to **RTF < 0.6** so the live
transcript stays in step with the speaker and the translator has headroom.

> Hardware target: macOS Apple Silicon, no CUDA. On a Linux/CUDA box this
> whole problem disappears — see the platform notes inside each chapter.

## Three lanes, ranked by effort vs return

| | what it does | effort | typical RTF on M-series |
|---|---|---|---|
| **Lane 1** — ONNX + FP16 + CoreML (ANE) | export the encoder to ONNX in FP16, run it through onnxruntime with the CoreML execution provider so the Apple Neural Engine actually executes the conformer attention | 1-2 days | **0.3-0.5** |
| **Lane 2** — ONNX + INT8 + CPU EP | export to ONNX, dynamic INT8 quantization, run through onnxruntime's CPU provider | ½-1 day | 0.6-0.9 |
| **Lane 3** — swap to a smaller specialist model | replace Nemotron with VietASR (68M) or Parakeet-CTC-0.6B-Vi | 1-3 days (depending on integration depth) | < 0.3 |

**Why these aren't ranked by RTF alone**: Lane 1 keeps your existing Vietnamese
quality intact — Lane 3 gives the biggest speedup but you're trading model
behavior you've already tuned around. Pick Lane 1 first; fall back to Lane 3
only if export fails or RTF stays above 1.

## What changes between lanes

Both ONNX lanes share the same architectural insight: **only the encoder needs
optimization**. The model has three parts and they're not equal:

| Part | Params | % of wall time | Action |
|---|---|---|---|
| Preprocessor (mel spectrogram) | trivial | ~5% | Leave in PyTorch |
| **Encoder (Conformer)** | **~500M** | **~85%** | **This is the only part worth exporting** |
| RNNT decoder (LSTM predictor + joint) | ~50M | ~10% | Leave in PyTorch (it's small + the LSTM keeps streaming state) |

The decoder is also the part PyTorch MPS silently breaks (the LSTM op returns
garbage on MPS — see `stream_translate.py`'s `USE_MPS` comment). Keeping the
decoder in CPU PyTorch sidesteps that bug entirely.

## Decision tree

```
START
  │
  ├─ Have you measured baseline RTF + WER? ────── No ──► Chapter 01
  │                                                       │
  │                              ┌────── Yes ─────────────┘
  │                              ▼
  │   Is your baseline RTF already < 0.8?
  │      │                       │
  │      ├── Yes ─► done, no optimization needed
  │      └── No ──► continue
  │                              ▼
  ├─ On macOS Apple Silicon? ──── Yes ──► Lane 1: Chapter 02 → 03
  │   No (Linux/Win CPU) ─────────────►   Lane 2: Chapter 02 → 03
  │                              │
  ├─ Did the ONNX export work without cache I/O patching? ─── No ──► Chapter 04 (Plan B)
  │                              │
  ├─ Did RTF drop below 0.8 after integration? ─── No ──► Chapter 04 (Plan B)
  │                              │
  └─ Did WER regress > 5% (relative)? ─── Yes ──► drop INT8 step OR Chapter 04
```

## Honest caveats up front

1. **NeMo's streaming Conformer export is not a one-liner.** The encoder uses
   a cache state (4 tensors) that the default `model.encoder.export()` does
   not round-trip correctly. You'll likely need to subclass and override
   `forward_for_export`. If that doesn't fit in a day, drop to Lane 3.

2. **Apple Neural Engine only runs FP16.** If you quantize to INT8 expecting
   CoreML acceleration, the engine falls back to CPU and you lose most of the
   speedup. On Mac, FP16 is the right choice **even though INT8 sounds faster**.

3. **Vietnamese is a tone language.** INT8 dynamic quantization can degrade
   tone-sensitive transcription. Always re-measure WER on Vietnamese clips
   after quantization, not just on FLEURS/LibriSpeech.

4. **numpy ↔ torch overhead is real.** Each `conformer_stream_step` will
   serialize 4 cache tensors in and out of the ONNX session. Budget 5-20 ms
   per chunk for this — if your post-ONNX encoder forward is < 100 ms, that's
   10-20% overhead you can't escape without bigger surgery.

5. **`requirements.txt` will grow.** onnxruntime + onnxruntime-extensions ≈ 30 MB.
   Pin both, including `onnxruntime-coreml` if it's a separate package on your
   platform.

## What you should always do first

**Lane 0** — measure baseline (Chapter 01). 30 minutes. Without these numbers
nothing later means anything. The Vietnamese WER you record here is also the
gate that decides whether Lane 1's INT8 step is acceptable.

Then read Chapter 02. The decision between FP16+CoreML and INT8+CPU happens
there, not before.
