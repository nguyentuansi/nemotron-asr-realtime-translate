# Step 01 — measure baseline RTF + WER

**Why this is first**: every claim later ("2× speedup", "WER didn't regress")
is a comparison against this number. Without a baseline, you're guessing
whether your optimization helped or made things worse on Vietnamese.

**Time**: 30-60 minutes.

**Output**:
- `bench/rtf_baseline.json` — per-chunk timing + overall RTF
- `bench/wer_baseline.json` — WER + CER on your held-out Vietnamese clips
- 10-20 short Vietnamese audio clips with ground-truth transcripts in
  `audio/bench/`

## 1. What you're measuring

| Metric | What it tells you | Target after optimization |
|---|---|---|
| **RTF** (real-time factor) = wall_seconds / audio_seconds | < 1.0 means streaming keeps up; > 1.0 means latency builds | < 0.6 with headroom |
| **chunk_ms** = ms to process one 560 ms chunk | the actual unit the streaming loop cares about | < 350 ms |
| **WER** = word error rate vs ground truth | how often the model is wrong (words) | within +5% relative of baseline |
| **CER** = character error rate | for Vietnamese, this catches tone-mark errors that WER masks | within +5% relative of baseline |

Track all four. WER alone can hide diacritic regressions; CER alone can hide
word-substitution regressions.

## 2. Record the eval set

Same idea as `docs/training/01-baseline.md`: short clips, representative of
real use. Smaller than training's because we're measuring perf, not generalization.

```bash
mkdir -p audio/bench
# Record 10-20 clips of 3-8 seconds each, save as 16 kHz mono wav.
# Use whatever recorder you have — Voice Memos works, then convert:
#   ffmpeg -i memo.m4a -ac 1 -ar 16000 audio/bench/clip_01.wav
```

Pick utterances with:
- Names + numbers ("Nguyễn", "hai bảy bốn") — Nemotron is weak here
- Mid-sentence pauses — exercises the silence VAD
- A few code-switched lines ("hello, xin chào") — common in real use

Write the ground truth alongside:

```
audio/bench/
  clip_01.wav
  clip_01.txt          # "Xin chào, hôm nay trời rất đẹp."
  clip_02.wav
  clip_02.txt
  ...
```

## 3. Measure WER + CER

```bash
./.venv/bin/pip install jiwer
```

`bench/measure_wer.py`:

```python
import json
from pathlib import Path
import jiwer

HERE = Path(__file__).resolve().parent.parent

import sys, os, logging
sys.path.insert(0, str(HERE))
os.environ["HF_HOME"] = str(HERE / ".hf-cache")
logging.getLogger("nemo_logger").setLevel(logging.WARNING)
logging.getLogger("nv_one_logger").setLevel(logging.ERROR)
import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nl
nl.setLevel(logging.ERROR)

LANG = "vi-VN"
m = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu"
).eval()
m.set_inference_prompt(LANG)

clips = sorted((HERE / "audio" / "bench").glob("*.wav"))
manifest = HERE / "bench" / "_eval.json"
manifest.parent.mkdir(exist_ok=True)
import soundfile as sf
with manifest.open("w") as fp:
    for wav in clips:
        info = sf.info(str(wav))
        fp.write(json.dumps({
            "audio_filepath": str(wav), "duration": info.duration,
            "text": "", "lang": LANG,
        }) + "\n")

out = m.transcribe(audio=[str(manifest)], batch_size=1, target_lang=LANG, verbose=False)
hyps = [getattr(o, "text", str(o)).strip() for o in out]
refs = [(wav.with_suffix(".txt")).read_text().strip() for wav in clips]

# jiwer normalizes case + punctuation by default — disable for tone-strict CER.
wer = jiwer.wer(refs, hyps)
cer = jiwer.cer(refs, hyps)
print(f"WER={wer*100:.2f}%  CER={cer*100:.2f}%  N={len(clips)}")

(HERE / "bench" / "wer_baseline.json").write_text(json.dumps({
    "wer": wer, "cer": cer, "n_clips": len(clips),
    "pairs": list(zip(refs, hyps)),
}, ensure_ascii=False, indent=2))
```

Run:
```bash
./.venv/bin/python bench/measure_wer.py
```

**Acceptable baseline**: WER 10-25% is normal for a multilingual model on
short Vietnamese clips. If you see > 40%, the model isn't actually loading
the right prompt — re-check `set_inference_prompt('vi-VN')`.

## 4. Measure RTF (real-time factor) on the streaming path

Critical: measure the **streaming** path (`conformer_stream_step`), not the
**file** path (`transcribe`). They have different cost.

`bench/measure_rtf.py`:

```python
import json, time, sys, os, logging
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ["HF_HOME"] = str(HERE / ".hf-cache")
logging.getLogger("nemo_logger").setLevel(logging.WARNING)
logging.getLogger("nv_one_logger").setLevel(logging.ERROR)

import numpy as np, soundfile as sf, torch
import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nl; nl.setLevel(logging.ERROR)
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

m = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu"
).eval()
m.set_inference_prompt("vi-VN")
m.encoder.set_default_att_context_size([70, 6])  # 560 ms chunks

SR = 16000
CHUNK = int((1 + 6) * 0.08 * SR)  # 8960 samples
cfg = m.encoder.streaming_cfg
results = []

for wav in sorted((HERE / "audio" / "bench").glob("*.wav")):
    audio, sr = sf.read(str(wav)); assert sr == SR
    cache_lc, cache_lt, cache_lcl = m.encoder.get_initial_cache_state(
        batch_size=1, device="cpu")
    buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=True)
    prev_hyp, pred_out = None, None
    step = 0; t0 = time.time()
    pos = 0
    while pos < len(audio):
        buf.append_audio(audio[pos:pos+CHUNK].astype(np.float32),
                         stream_id=(-1 if buf.buffer is None else 0))
        pos += CHUNK
        while True:
            if buf.buffer is None: break
            req = cfg.chunk_size[0] if step == 0 else cfg.chunk_size[1]
            if buf.buffer.size(-1) - buf.buffer_idx < req: break
            try:
                ca, cl = next(iter(buf))
            except StopIteration: break
            with torch.inference_mode():
                r = m.conformer_stream_step(
                    processed_signal=ca, processed_signal_length=cl,
                    cache_last_channel=cache_lc, cache_last_time=cache_lt,
                    cache_last_channel_len=cache_lcl,
                    keep_all_outputs=False,
                    previous_hypotheses=prev_hyp, previous_pred_out=pred_out,
                    drop_extra_pre_encoded=(0 if step == 0 else cfg.drop_extra_pre_encoded),
                    return_transcription=True,
                )
            pred_out, _, cache_lc, cache_lt, cache_lcl, prev_hyp = r
            step += 1
    elapsed = time.time() - t0
    audio_dur = len(audio) / SR
    rtf = elapsed / audio_dur
    chunk_ms = elapsed * 1000 / max(step, 1)
    results.append({"file": wav.name, "audio_s": audio_dur,
                    "elapsed_s": elapsed, "rtf": rtf,
                    "n_chunks": step, "chunk_ms": chunk_ms})
    print(f"{wav.name:30s} {audio_dur:5.2f}s -> {elapsed:5.2f}s  RTF={rtf:.2f}  chunk={chunk_ms:.0f}ms")

avg_rtf = sum(r["rtf"] for r in results) / len(results)
avg_chunk = sum(r["chunk_ms"] for r in results) / len(results)
print(f"\nAVG RTF={avg_rtf:.2f}  AVG chunk_ms={avg_chunk:.0f}")
(HERE / "bench" / "rtf_baseline.json").write_text(
    json.dumps({"avg_rtf": avg_rtf, "avg_chunk_ms": avg_chunk,
                "per_file": results}, indent=2))
```

Run:
```bash
./.venv/bin/python bench/measure_rtf.py
```

Expect avg RTF ≈ 1.4-1.6 on M1/M2 CPU, ~0.6-0.9 on M2/M3 Pro+. Anything below
0.8 and Chapter 02 is optional — you're already real-time.

## 5. Sanity-check the numbers before moving on

| Symptom | Probable cause | Action |
|---|---|---|
| WER > 50% on Vietnamese | wrong prompt set, or `vi-VN` not in this model's prompt dict | check `model.prompt_dictionary` keys |
| RTF varies wildly between clips | thermal throttling or other heavy processes | close everything, re-run |
| `chunk_ms` < `audio_chunk_ms` (560) but WER weird | streaming buffer isn't accumulating across chunks correctly | check encoder cache shape on entry to step 1 |
| RTF < 0.5 on baseline CPU | this machine is faster than expected — Chapter 02 may not be worth it | go straight to using the app |

Commit the two JSON files. Now you have a measurable target for Chapter 02.
