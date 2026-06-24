# Step 01 — measure baseline WER

**Why this is first**: you can't decide "is fine-tuning worth it?" without a
number. Any % improvement claim later is meaningless without a baseline. This
step also doubles as the held-out test set you'll use to evaluate every later
change.

**Time**: 2-4 hours (mostly recording + transcribing 30-60 utterances).

**Output**: `audio/eval/manifest.json` + a `baseline_wer.txt` number.

## 1. Install the WER tool

```bash
.venv/bin/pip install jiwer
```

`jiwer` computes Word Error Rate (WER) and Character Error Rate (CER). For
Vietnamese, CER is the more meaningful number because the tone marks are
single characters — a `máy` vs `mấy` flip costs 1 char (CER) but 1 word (WER).
We'll track both.

## 2. Record ~30 minutes of held-out audio

Pick utterances **representative of how you'll actually use the system** —
news reading, Q&A, conversational, whatever your real use case is. Diversity
matters more than volume here; this is the test set, not the training set.

Target: **50-100 utterances**, each 2-15 seconds. Total ~20-30 min of audio.

```bash
mkdir -p audio/eval

# Use the existing mic test in single-take mode, save each utterance as a wav
# (or use any recorder — `arecord`, Audacity, your phone). 16 kHz mono.
```

Suggested folder layout:

```
audio/eval/
├── 001.wav
├── 001.txt        ← reference transcript (UTF-8, with proper diacritics)
├── 002.wav
├── 002.txt
└── ...
```

**Quality of reference transcripts is the whole game.** If the references have
the same diacritic errors the model makes, you can't measure improvement.
Type them yourself, no auto-transcription, double-check tone marks.

## 3. Build a NeMo manifest from the recordings

```python
# scripts/build_eval_manifest.py
import json, glob, soundfile as sf
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # project root
EVAL = HERE / "audio" / "eval"
OUT = EVAL / "manifest.json"

with OUT.open("w", encoding="utf-8") as fp:
    for wav in sorted(EVAL.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"skip {wav.name}: no transcript")
            continue
        info = sf.info(str(wav))
        if info.samplerate != 16000 or info.channels != 1:
            print(f"WARN {wav.name}: sr={info.samplerate} ch={info.channels} (want 16k mono)")
        fp.write(json.dumps({
            "audio_filepath": str(wav),
            "duration": info.frames / info.samplerate,
            "text": txt.read_text(encoding="utf-8").strip(),
            "lang": "vi-VN",
        }, ensure_ascii=False) + "\n")
print(f"wrote {OUT}")
```

```bash
.venv/bin/python scripts/build_eval_manifest.py
```

## 4. Run the baseline evaluation

```python
# scripts/eval_wer.py — measure baseline WER + CER on the held-out set.
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))

import jiwer
import torch
import nemo.collections.asr as nemo_asr

MANIFEST = HERE / "audio" / "eval" / "manifest.json"
LANG = "vi-VN"

print(f"[load] nemotron-3.5-asr-streaming-0.6b ...")
t0 = time.time()
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu",
)
print(f"[load] {time.time()-t0:.1f}s")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    model = model.half().to("cuda").eval()
else:
    model = model.eval()

refs, hyps = [], []
with MANIFEST.open() as fp:
    for line in fp:
        if line.strip():
            refs.append(json.loads(line)["text"])

out = model.transcribe(audio=[str(MANIFEST)], batch_size=4, target_lang=LANG)
hyps = [getattr(o, "text", o).rstrip() for o in out]

# Some Nemotron outputs end with the lang tag, e.g. "Xin chào. <vi-VN>" — strip
# for fair comparison against the references.
import re
hyps = [re.sub(r"<[a-zA-Z-]+>\s*$", "", h).rstrip() for h in hyps]

print(f"\n[wer] computing over {len(refs)} utterances")
wer = jiwer.wer(refs, hyps)
cer = jiwer.cer(refs, hyps)
print(f"[wer] WER = {wer*100:.2f}%")
print(f"[wer] CER = {cer*100:.2f}%")

# Per-utterance diff for the worst offenders.
print("\n[wer] worst 10:")
per_utt = [(jiwer.wer([r], [h]), r, h) for r, h in zip(refs, hyps)]
per_utt.sort(reverse=True)
for w, r, h in per_utt[:10]:
    print(f"  WER={w*100:5.1f}%  ref={r!r}")
    print(f"                  hyp={h!r}")

(HERE / "baseline_wer.txt").write_text(
    f"WER={wer*100:.2f}\nCER={cer*100:.2f}\nN={len(refs)}\n"
)
```

```bash
.venv/bin/python scripts/eval_wer.py
```

You should see something like:

```
[wer] WER = 17.42%
[wer] CER = 8.71%
```

Don't trust a baseline measured on fewer than ~30 utterances — variance is
too high to tell signal from noise.

## 5. Decide where to go next

| baseline WER | recommendation |
|---|---|
| **< 8%** | the model is already very good for you. Lane 3 fine-tuning may not be worth 2-4 weeks. Just Lane 1 (diacritic post-processing) and done. |
| **8-15%** | Lane 1 first. Re-measure. If residual >10%, Lane 3 fine-tuning is justified. |
| **> 15%** | Lane 1 + Lane 3 both. Lane 2 (LM rescoring) optional. |
| **CER < 5% but WER > 12%** | classic diacritic-confusion signature — Lane 1 will probably crush this. |

Save your baseline number — you'll compare every subsequent change against it.

→ Next: **[02-quick-wins.md](02-quick-wins.md)**
