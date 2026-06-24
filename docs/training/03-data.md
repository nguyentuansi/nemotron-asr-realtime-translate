# Step 03 — collect + prepare training data

**This is the longest step. Plan for ~80% of total project time to live here.**

The model's final quality is bounded by data quality, not by architecture or
training tricks. Two hours on this step well-spent saves you a month of
chasing why training "didn't help."

**Time**: 1-3 weeks (mostly recording + manual verification).

**Output**: `audio/train/manifest.json` + `audio/val/manifest.json`, both in
NeMo Lhotse-compatible format.

## How much data do you actually need?

| audio hours | what to expect on diacritics + accent |
|---|---|
| **< 2 h** | basically no measurable gain on Nemotron-3.5 — too small to overcome pre-training bias |
| **5-10 h** | first detectable WER drop (10-15% relative). LoRA fits comfortably. |
| **10-30 h** | sweet spot for Lane 3 on this hardware. 20-40% relative gain. |
| **30-100 h** | diminishing returns from LoRA — at this point you want full encoder-decoder fine-tuning, which doesn't fit your 6 GB GPU without offload |
| **> 100 h** | retrain a smaller model from scratch may make sense |

**Start with 10 hours of your own voice.** That's the minimum useful step
and the maximum that fits comfortably into a 1-2 week collection sprint.

## Sources, ranked by quality for your use case

### Source A — record yourself (recommended, best quality)

Highest signal. The model adapts to YOUR voice, mic, room. Plan:

- **10 hours of audio = ~3-4 hours of recording sessions** (with retakes and
  pauses)
- Read a mix of:
  - news articles (formal vocabulary)
  - conversational scripts (filler words, hesitations)
  - the specific words you saw errors on (`máy/mấy/mây`, `cậu/câu/cầu`,
    `lây/lấy/lầy`, etc.) — repeat 50+ times each across different contexts
  - prompts from your real use case (meetings? dictation? subtitling?)
- Record at **16 kHz mono**, raw PCM or FLAC
- One sentence per file, 2-15 s duration

### Source B — Common Voice Vietnamese (free, mixed quality)

```bash
# Download the latest CV bundle for Vietnamese (~3 GB):
# https://commonvoice.mozilla.org/en/datasets → Vietnamese → latest
# Unpack to ./data/common_voice_vi/
```

Pros: ~50-100 h of audio with transcripts. Free.
Cons: 50+ different speakers, mixed quality, many recordings are poor.

Use this as a **supplement** to your own data, not as the only source.
Filter to only the `validated.tsv` rows and apply audio quality screening
(remove clips with SNR < 15 dB).

### Source C — VLSP-2020 ASR challenge data

Vietnamese academic benchmark, ~250 h. Requires email request for access via
the VLSP organizers. High quality but generic — best as a "stabilization"
dataset to prevent catastrophic forgetting.

## Recording workflow

If you're going with Source A, here's a battery-included flow.

### 1. Prepare a prompt list

Write or scrape ~500-1000 Vietnamese sentences covering:
- everyday topics (weather, food, work)
- your domain (meetings, news, technical content)
- the **specific confusable words** you saw errors on

```
audio/train/prompts/
├── prompts_day1.txt    # 50 sentences
├── prompts_day2.txt    # 50 sentences
└── ...
```

### 2. Record using a small helper

The `mic.sh` interactive mode from the existing project records one
utterance at a time. Adapt it to "show prompt, record, save with id":

```python
# scripts/record_dataset.py — minimal prompt-driven recorder.
import argparse, json, os, sys, time
from pathlib import Path
import alsaaudio, numpy as np
import soundfile as sf

SR = 16000
OUT = Path("audio/train/raw")
OUT.mkdir(parents=True, exist_ok=True)

def record(seconds):
    pcm = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL,
                        device="default", channels=1, rate=SR,
                        format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=1024)
    chunks, t0 = [], time.time()
    while time.time() - t0 < seconds:
        n, data = pcm.read()
        if n > 0:
            chunks.append(np.frombuffer(data, dtype=np.int16))
    pcm.close()
    return np.concatenate(chunks).astype(np.float32) / 32768.0

ap = argparse.ArgumentParser()
ap.add_argument("prompts_file")
ap.add_argument("--max-secs", type=float, default=15.0)
args = ap.parse_args()

with open(args.prompts_file, encoding="utf-8") as fp:
    prompts = [l.strip() for l in fp if l.strip()]

next_id = max([int(p.stem.split("_")[-1])
               for p in OUT.glob("*.wav")] + [0]) + 1

for prompt in prompts:
    print(f"\n[{next_id:04d}] {prompt}")
    input("press ENTER to start (or Ctrl-C to quit) ")
    audio = record(args.max_secs)
    wav = OUT / f"clip_{next_id:04d}.wav"
    txt = OUT / f"clip_{next_id:04d}.txt"
    sf.write(str(wav), audio, SR, subtype="PCM_16")
    txt.write_text(prompt, encoding="utf-8")
    print(f"  saved {wav.name} ({len(audio)/SR:.1f}s)")
    keep = input("  keep? [Y/n] ").strip().lower()
    if keep == "n":
        wav.unlink(); txt.unlink()
        print("  discarded")
    else:
        next_id += 1
```

```bash
.venv/bin/python scripts/record_dataset.py audio/train/prompts/prompts_day1.txt
```

### 3. Verify every transcript

> **This is the step that determines your model's quality.**

Listen back to each recording while reading the transcript. Fix:
- diacritic typos (`máy` vs `mấy`)
- missing words (skipped, mumbled, ran together)
- wrong words you actually said vs the prompt
- punctuation that you naturally paused for

Tools:
- `aplay audio/train/raw/clip_0001.wav` to play
- Just edit the `.txt` files in your editor of choice

**Budget ~2× the recording time for verification.** It's tedious; it's
non-negotiable.

## Build the manifests

```python
# scripts/build_train_manifest.py
import json, random, os
from pathlib import Path
import soundfile as sf

random.seed(42)

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "audio" / "train" / "raw"
TRAIN_OUT = HERE / "audio" / "train" / "manifest.json"
VAL_OUT = HERE / "audio" / "val" / "manifest.json"
VAL_OUT.parent.mkdir(parents=True, exist_ok=True)

entries = []
for wav in sorted(RAW.glob("*.wav")):
    txt = wav.with_suffix(".txt")
    if not txt.exists():
        continue
    info = sf.info(str(wav))
    if info.samplerate != 16000 or info.channels != 1:
        print(f"WARN {wav.name}: sr={info.samplerate} ch={info.channels}, skipping")
        continue
    dur = info.frames / info.samplerate
    if dur < 0.5 or dur > 20.0:
        print(f"SKIP {wav.name}: duration {dur:.1f}s out of range")
        continue
    entries.append({
        "audio_filepath": str(wav),
        "duration": dur,
        "text": txt.read_text(encoding="utf-8").strip(),
        "lang": "vi-VN",
    })

random.shuffle(entries)
n_val = max(50, len(entries) // 10)
val, train = entries[:n_val], entries[n_val:]

def write(path, items):
    with path.open("w", encoding="utf-8") as fp:
        for it in items:
            fp.write(json.dumps(it, ensure_ascii=False) + "\n")

write(TRAIN_OUT, train)
write(VAL_OUT, val)
print(f"train: {len(train)} ({sum(e['duration'] for e in train)/3600:.2f} h)")
print(f"val:   {len(val)} ({sum(e['duration'] for e in val)/3600:.2f} h)")
```

```bash
.venv/bin/python scripts/build_train_manifest.py
```

Expected output:

```
train: 1450 (9.32 h)
val:   161  (1.04 h)
```

## Quality gates before moving on

Don't proceed until all of these are true:

- [ ] At least **5 hours** in `train` (10+ is better)
- [ ] At least **50 utterances** in `val`
- [ ] **Zero** validation utterances overlap with training (same prompt OK,
      same audio file NOT)
- [ ] **Zero** files at `sr ≠ 16000` or `channels ≠ 1` in either manifest
- [ ] Every clip is between 0.5 s and 20 s
- [ ] You've spot-checked at least **20 random** clips and confirmed
      transcripts match the audio (sample: `head -20 audio/train/manifest.json
      | jq -r 'select(.) | .audio_filepath'`)
- [ ] No `\t` or unusual whitespace in transcripts (`grep -P '\\t' audio/train/manifest.json`
      should print nothing)

→ Next: **[04-finetune.md](04-finetune.md)**
