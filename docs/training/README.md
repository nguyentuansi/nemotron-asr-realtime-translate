# Training docs — improving Vietnamese ASR accuracy

Goal: reduce the diacritic-confusion errors we see in production (`máy/mấy`,
`rước/trước`, `lây/lấy`, etc.) by adapting `nvidia/nemotron-3.5-asr-streaming-0.6b`
to **your voice and your speaking domain**.

> Hardware target: this machine — RTX 2060 (6 GB VRAM). All steps below are
> sized for that ceiling. If you're on something bigger, the same recipes work
> with larger batches.

## Three lanes, ranked by effort vs return

| | what it does | effort | typical gain on the diacritic errors |
|---|---|---|---|
| **Lane 1** — post-process | run committed ASR text through a Vietnamese diacritic restorer (`pyvi` / `underthesea`) before display + translation | ½ day | **60-80%** of diacritic errors fixed |
| **Lane 2** — LM rescoring | beam-search the RNNT decoder + an external Vietnamese n-gram LM (KenLM) | 2-5 days (mostly corpus collection) | **20-40%** overall WER, helps with disambiguation |
| **Lane 3** — fine-tune | LoRA adapters on the encoder + your own Vietnamese audio (~10 h) | 2-4 weeks | **30-50%** on diacritics, **15-30%** overall |

Lane 1 fixes the most-visible errors with the least effort. Lane 3 wins on
non-diacritic mistakes that no post-processor can catch (wrong word entirely,
your specific accent quirks, microphone characteristics).

**You should always do Lane 1 first.** Then measure (Step 01 below) and decide
if the residual is worth Lane 2 or Lane 3.

## Workflow (Lane 3, full fine-tuning path)

| step | doc | what you do | gate / output |
|---|---|---|---|
| **01** | [01-baseline.md](01-baseline.md) | measure current WER on held-out audio | a baseline number to beat |
| **02** | [02-quick-wins.md](02-quick-wins.md) | wire in Lane 1 post-processing + (optional) Lane 2 KenLM | new WER number |
| **03** | [03-data.md](03-data.md) | collect ~10 h of audio, transcribe + verify, build NeMo manifests | `train.json` + `val.json` |
| **04** | [04-finetune.md](04-finetune.md) | LoRA fine-tune with frozen encoder backbone | `nemotron-vi-lora.nemo` |
| **05** | [05-deploy.md](05-deploy.md) | swap the checkpoint into the live demos, A/B vs baseline | shipped model |

Each doc is self-contained — exact commands, file paths, what success looks
like, what failure looks like, and what to try next when something doesn't
work.

## Read first

- [What `training_mode` in the config actually was](#footnote-training-mode)
  (it's a leftover from NVIDIA's pretraining config, not a runtime switch)
- The model is an [RNNT-based FastConformer with prompt conditioning](../../model_readme.md#model-architecture-amp-training)
- Cache-aware streaming inference details are in [model_readme.md](../../model_readme.md)

---

### Footnote: training_mode

When you load the pretrained model you see logs like:

```
prompt_field: target_lang
prompt_dictionary: { en-US: 0, ..., vi-VN: 33, ..., auto: 101 }
num_prompts: 128
subsampling_factor: 8
training_mode: true
```

That `training_mode: true` lives inside the train-data config that ships with
the checkpoint and describes **how NVIDIA originally trained the model**. It's
not a runtime mode you can flip to "improve" inference. The way to improve the
model on your data is the Lane 3 workflow in this folder.
