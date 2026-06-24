# Step 05 — evaluate + deploy the fine-tuned checkpoint

Goal: confirm the new model is actually better than the baseline, then swap it
into the live demos. Roll back cleanly if it regresses anywhere.

**Time**: 1-2 hours.

**Output**: `stream_web.sh` / `stream_translate.sh` running on the LoRA model,
with a documented A/B vs the baseline.

## 1. Final WER on the held-out set

Run `eval_wer.py` from Step 01, but pointed at the new checkpoint:

```python
# scripts/eval_wer_finetuned.py — same as eval_wer.py, two lines changed.
import nemo.collections.asr as nemo_asr

# OLD:  model = nemo_asr.models.ASRModel.from_pretrained(
#           model_name="nvidia/nemotron-3.5-asr-streaming-0.6b", ...)
# NEW:
model = nemo_asr.models.ASRModel.restore_from(
    str(HERE / "models" / "nemotron-vi-lora.nemo"),
    map_location="cpu",
)
```

```bash
.venv/bin/python scripts/eval_wer_finetuned.py | tee logs/eval-finetuned.txt
```

Compare the numbers:

```bash
echo "BASELINE:"; cat baseline_wer.txt
echo "AFTER LANE 1:"; cat baseline_wer_lane1.txt   # if you ran Step 02
echo "AFTER LORA:"; tail -3 logs/eval-finetuned.txt
```

Decision matrix:

| outcome | what to do |
|---|---|
| WER dropped ≥ 10% relative AND no per-utterance regression > 2× baseline | **deploy** |
| WER dropped < 5% relative | **don't deploy**. Go back to Step 03 (more data) or Step 04 (more epochs / higher LoRA rank). |
| Avg WER dropped but some utterances got dramatically WORSE | adapter overfit. Try lower epoch checkpoint, lower LR, or more diverse training data. |
| WER on Vietnamese got better but English/other languages collapsed | catastrophic forgetting. Add a small held-out English set to validation; use the checkpoint from BEFORE val_wer on English crossed threshold. |

## 2. Per-utterance regression report

The aggregate WER number hides regressions. Check explicitly:

```python
# scripts/compare_wer.py
import json, jiwer
from pathlib import Path

base = json.loads(Path("logs/eval-baseline.json").read_text())
new = json.loads(Path("logs/eval-finetuned.json").read_text())

regressions = []
improvements = []
for utt_id, ref in base.items():
    if utt_id not in new:
        continue
    w_base = jiwer.wer([ref["ref"]], [ref["hyp"]])
    w_new = jiwer.wer([ref["ref"]], [new[utt_id]["hyp"]])
    if w_new > w_base + 0.05:
        regressions.append((w_new - w_base, ref["ref"], ref["hyp"], new[utt_id]["hyp"]))
    elif w_new < w_base - 0.05:
        improvements.append((w_base - w_new, ref["ref"], ref["hyp"], new[utt_id]["hyp"]))

regressions.sort(reverse=True)
improvements.sort(reverse=True)

print(f"\n=== {len(regressions)} regressions ===")
for delta, ref, old_hyp, new_hyp in regressions[:10]:
    print(f"  +{delta*100:.1f}%  ref={ref!r}")
    print(f"           base hyp={old_hyp!r}")
    print(f"           new  hyp={new_hyp!r}")

print(f"\n=== {len(improvements)} improvements (top 10) ===")
for delta, ref, old_hyp, new_hyp in improvements[:10]:
    print(f"  -{delta*100:.1f}%  ref={ref!r}")
    print(f"           base hyp={old_hyp!r}")
    print(f"           new  hyp={new_hyp!r}")
```

If `regressions` is non-empty, look at the patterns. If they're all on
audio that's out-of-distribution vs your training set (different mic,
heavy background noise, accent shift), that's expected — your model is now
specialized.

## 3. Wire the new checkpoint into the live demos

There are two scripts that load the ASR model: `stream_translate.py`
(terminal) and `stream_web.py` (browser). Both have a `load_asr()` function.

Add a CLI flag rather than hard-coding the path — keeps the baseline
recoverable:

```python
# In stream_web.py and stream_translate.py, replace load_asr():
def load_asr(checkpoint: str | None = None):
    import nemo.collections.asr as nemo_asr
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    t0 = time.time()
    if checkpoint:
        print(f"[asr]  loading {checkpoint} ...", flush=True)
        model = nemo_asr.models.ASRModel.restore_from(checkpoint, map_location="cpu")
    else:
        print(f"[asr]  loading nvidia/nemotron-3.5-asr-streaming-0.6b ...", flush=True)
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu",
        )
    print(f"[asr]  loaded in {time.time()-t0:.1f}s", flush=True)
    if torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1":
        torch.cuda.empty_cache()
        model = model.to("cuda").eval()
        print(f"[asr]  fp32 weights on GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB",
              flush=True)
    else:
        model = model.eval()
    return model

# In the argparse:
ap.add_argument("--checkpoint", default=None,
                help="path to .nemo fine-tuned checkpoint; default = baseline pretrained")

# In main():
model = load_asr(args.checkpoint)
```

## 4. A/B run

```bash
# baseline
./stream_web.sh --lang vi-VN --port 8765

# fine-tuned
./stream_web.sh --lang vi-VN --checkpoint models/nemotron-vi-lora.nemo --port 8766
```

Open both in separate browser tabs. Speak the same sentences into the mic
twice (gap of >2 s between to allow VAD to commit both clean) and compare the
source transcripts side by side. Confirm the diacritic fixes are real.

## 5. Promote to default (optional)

Once you're happy:

```bash
# Symlink so all scripts default to the fine-tuned model
mkdir -p models
ln -sf nemotron-vi-lora.nemo models/asr-default.nemo

# Add a default in stream_web.py / stream_translate.py:
ap.add_argument("--checkpoint", default=str(HERE / "models" / "asr-default.nemo"),
                help="path to .nemo fine-tuned checkpoint; pass '' for baseline pretrained")
```

To go back to baseline: `./stream_web.sh --checkpoint ""`.

## 6. Document the gain

Write a `MODEL.md` or update the project README with:

```
## Active ASR model

| field | value |
|---|---|
| base | nvidia/nemotron-3.5-asr-streaming-0.6b |
| adapter | LoRA rank-32 on all Conformer blocks + prompt_kernel |
| training data | 9.32 h Vietnamese, recorded 2026-MM-DD..MM-DD |
| baseline WER (vi-VN, our eval set) | 17.42% |
| WER after Lane 1 (post-process) | 11.85% |
| WER after Lane 3 (LoRA) | 9.31% |
| evaluation set | audio/eval/manifest.json (52 utterances, 21 min) |
```

Future-you (or the next contributor) will need this when they wonder "wait,
what model is actually live?" three months from now.

## 7. Iterate

Lane 3 is not "do it once and you're done." Re-record an evaluation pass
every ~3 months — voice changes, mic setup drifts, model degrades on new
patterns. If the new WER on your held-out set has crept up by > 2 absolute
points, you're due for another fine-tuning round with fresh data.
