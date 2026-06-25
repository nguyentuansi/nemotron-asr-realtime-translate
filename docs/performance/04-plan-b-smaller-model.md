# Step 04 — Plan B: swap to a smaller Vietnamese-specialist model

**When to read this**: only if Chapter 02 or 03 failed:
- NeMo's encoder export refused to round-trip cache I/O even with the patched
  `forward_for_export` (Chapter 02 §5)
- ONNX integration shipped but RTF stayed > 1.0
- WER drift after quantization is unacceptable for Vietnamese
- You want a "step zero" Vietnamese performance win without touching ONNX
  at all

**Time**: 1-3 days depending on which model and how deep you go.

**Output**: `stream_translate.sh` running on a different ASR model, with
the streaming loop adapted accordingly.

## 1. The three credible candidates

All three are commercial-OK and include Vietnamese. Numbers are from each
model's published evals; **always re-measure on your `audio/bench/` clips**
before committing.

| Model | Params | Streaming | License | Vi WER (own benchmarks) | Notes |
|---|---|---|---|---|---|
| **`nvidia/parakeet-ctc-0.6b-Vietnamese`** | 600M | Limited (CTC, no built-in cache-aware streaming) | NVIDIA Open Model | 9.30 avg WER (own bench) | Same family as Nemotron, CTC decoder is faster than RNNT, code-switching support |
| **`VinAIResearch/PhoWhisper-large`** | ~1.5B | No (Whisper batch) | Apache 2.0 (model), CC-BY (data) | SOTA on VLSP/VIVOS/CommonVoice-Vi | Largest of the three; offline-only — needs streaming hack |
| **VietASR (arxiv 2505.21527)** | 68M | Yes, native 320/640/1280 ms chunks | research code (check before commercial use) | beats Whisper-large-v3 on Vi | Smallest by 9×; research code, not a HF release |

## 2. Decision shortcut

```
Do you need TRUE streaming (commits during speech)?
├─ Yes ──► VietASR (if you can ship research code)
│           └─ or PhoWhisper with a streaming wrapper (complex)
└─ No (committing after end-of-utterance is OK)
   ├─ Best Vi accuracy: PhoWhisper-large
   └─ Best speed/quality balance: Parakeet-CTC-Vi
```

For a real-time UI like `stream_web.sh`, **streaming is what makes it feel
live**. Don't downgrade to non-streaming unless you're rebuilding the UX
around batch transcription (e.g. press-to-talk, send-to-server).

## 3. Lane 3a — VietASR (smallest, fastest, research code)

### 3a.1 Get the code

The model isn't on HuggingFace at time of writing — it's published as code +
checkpoints alongside the arxiv paper. Pull from the authors' repo (search
"VietASR" on GitHub; the paper is arXiv:2505.21527).

### 3a.2 Wire into stream_translate.py

VietASR exposes a streaming API but **not** through NeMo's `conformer_stream_step`.
Plan a small adapter class:

```python
# vietasr_adapter.py
class VietASRStreamingAdapter:
    """Match the subset of the NeMo streaming API the app uses, on top of VietASR."""

    def __init__(self, checkpoint_path):
        from vietasr import StreamingModel  # placeholder import
        self.model = StreamingModel.load(checkpoint_path)
        self.cache = self.model.init_cache()

    def feed_chunk(self, audio_chunk: "np.ndarray") -> str:
        # Returns the *delta* text since last call.
        text, self.cache = self.model.step(audio_chunk, self.cache)
        return text

    def reset(self):
        self.cache = self.model.init_cache()
```

Then in `stream_translate.py`, gate on a `--model {nemotron,vietasr}` flag and
route accordingly. The main loop changes only at the per-chunk call.

### 3a.3 Expected gains

- 68M params on CPU → RTF probably 0.1-0.2
- ANE / MPS likely just works (smaller model, fewer custom ops)
- WER could be **better** on Vi-only audio, **worse** on code-switched (no
  multilingual prompt)

### 3a.4 Risks

- License + provenance: research code may have unclear commercial terms
- No prompt-conditioning — if you ever wanted multilingual code-switching,
  you'd need to layer that yourself
- API stability: it's a research artifact, not a maintained library

## 4. Lane 3b — Parakeet-CTC-0.6B-Vietnamese (NVIDIA, commercial-OK)

### 4b.1 Why it's faster despite same params

Parakeet uses **CTC** decoding, not RNN-T. The RNNT decoder runs an autoregressive
loop per output token; CTC is a single forward pass. For streaming, CTC is
cheaper per chunk.

### 4b.2 Streaming caveat

`parakeet-ctc-0.6b-Vietnamese` was trained for full-utterance transcription,
not cache-aware streaming. Two options:

1. **Buffered batched streaming**: accumulate 4-8 seconds, transcribe, commit.
   Lower latency than full-utterance but not truly live.
2. **Manual chunked CTC**: feed 2-4 second windows with overlap, deduplicate
   outputs. Hacky.

Either way, the streaming experience won't match Nemotron's cache-aware design.
Decide if that tradeoff is worth the speed.

### 4b.3 Integration

NeMo loads it identically to Nemotron:

```python
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/parakeet-ctc-0.6b-Vietnamese", map_location="cpu"
).eval()
```

But the decoding path is different — `transcribe()` works out of the box;
`conformer_stream_step` does not exist on CTC models. You'd refactor
`stream_translate.py`'s inner loop to use buffered-batch transcription.

## 5. Lane 3c — PhoWhisper-large (highest Vi accuracy, no streaming)

Best for **after-the-fact transcription** workflows: record, transcribe, edit.
Wrong shape for a live UI.

If you go this route, the right product change is:
- Drop the live web UI
- Build a press-to-talk recorder that streams audio to a backend
- Transcribe + translate after the user releases the button
- Show results in 0.5-2 s after release

That's a different app. Not a Plan B for the current one.

## 6. Hybrid: faster-whisper for English, Nemotron for Vietnamese

If your usage is mostly English with occasional Vietnamese, faster-whisper
on CPU is RTF < 0.3 and good enough for the streaming UI when wrapped with
buffered batching. Keep Nemotron available behind a flag for Vietnamese
sessions. Don't bother if Vietnamese is the primary language.

## 7. Recommended order to evaluate

If Chapter 03 failed and you got here:

1. **Re-read Chapter 02 §1** — did you really hit a hard wall in NeMo's
   exporter, or was it fixable? Going from "stuck" to "model swap" is a big
   architectural step.
2. **Measure Parakeet-CTC-Vi** on your `audio/bench/` clips with `transcribe()`.
   Compare WER + CER to Nemotron's baseline. If it's competitive AND the
   buffered-streaming UX is acceptable, this is the lowest-risk swap.
3. **Skip VietASR unless** you've confirmed the license fits and you have
   bandwidth for research-code integration.
4. **PhoWhisper only if** the UX shift to non-live is OK.

## 8. Whatever you pick, re-run Chapter 01's benches

The bench/measure_rtf.py and bench/measure_wer.py scripts are model-agnostic
once you write a thin adapter for the new model's API. Re-run, commit JSON
files, compare side-by-side with Nemotron baseline.

If you ship a Plan B model, **keep Nemotron available behind a flag** for at
least one release — Vietnamese ASR quality is the kind of thing users don't
notice for a week and then file an angry bug about.
