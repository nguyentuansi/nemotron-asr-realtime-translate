# Step 04 — LoRA fine-tune

Goal: teach the model your voice + speaking style without overwriting the
weights NVIDIA spent millions of GPU-hours producing.

**Strategy**: freeze the entire encoder backbone, attach small **LoRA-style
adapters** (rank-32 linear layers) to each Conformer block, train only those.
Trainable params: **~1% of the model**, ~6 M params. Fits in ~4.5 GB on the
RTX 2060 at batch=2 with mixed precision.

**Time**: 5-15 hours of GPU training for 10-50 epochs over 10 h of audio.

**Output**: `models/nemotron-vi-lora.nemo`

## Prerequisites

- Step 03 done: `audio/train/manifest.json` + `audio/val/manifest.json`
- Step 01 done: `baseline_wer.txt` — you'll compare against this number
- ~30 GB free disk for checkpoints
- 4-8 GB free RAM (plus the GPU)

## 1. Config — what we'll train

Create `configs/finetune_lora.yaml`:

```yaml
name: "nemotron-3.5-asr-streaming-0.6b-vi-lora"

# Base checkpoint to fine-tune from (downloaded by Step 01).
init_from_pretrained_model: "nvidia/nemotron-3.5-asr-streaming-0.6b"

trainer:
  devices: 1
  accelerator: gpu
  precision: 16-mixed     # critical for 6 GB GPU
  max_epochs: 20
  accumulate_grad_batches: 8   # effective batch ~16
  gradient_clip_val: 1.0
  val_check_interval: 1.0
  log_every_n_steps: 25
  num_sanity_val_steps: 0

model:
  sample_rate: 16000

  # Data
  train_ds:
    manifest_filepath: ${oc.env:HERE}/audio/train/manifest.json
    sample_rate: 16000
    batch_size: 2
    num_workers: 4
    pin_memory: true
    use_lhotse: true
    use_bucketing: true
    batch_duration: 80      # seconds of audio per batch (Lhotse)
    quadratic_duration: 12
    num_buckets: 20
    bucketing_strategy: fully_randomized
    prompt_field: target_lang
    lang_field: target_lang
    max_duration: 20
    min_duration: 0.5

  validation_ds:
    manifest_filepath: ${oc.env:HERE}/audio/val/manifest.json
    sample_rate: 16000
    batch_size: 2
    num_workers: 2
    use_lhotse: true
    use_bucketing: false
    prompt_field: target_lang
    lang_field: target_lang

  optim:
    name: adamw
    lr: 1e-4              # higher than full FT (1e-5) — adapters need it
    weight_decay: 0.01
    betas: [0.9, 0.98]
    sched:
      name: CosineAnnealing
      warmup_steps: 200
      min_lr: 1e-6
```

## 2. Training script

```python
# scripts/finetune_lora.py
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
os.environ["HERE"] = str(HERE)
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr

CFG = HERE / "configs" / "finetune_lora.yaml"
OUT = HERE / "models" / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)

cfg = OmegaConf.load(str(CFG))

# 1. Load the pretrained model.
print("[load] base model ...")
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name=cfg.init_from_pretrained_model, map_location="cpu",
)
model = model.to("cuda")

# 2. Freeze everything.
for p in model.parameters():
    p.requires_grad = False

# 3. Attach LoRA-style adapters to every Conformer encoder block.
#    NeMo's AdapterModuleMixin gives the encoder add_adapter().
ADAPTER_CFG = {
    "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
    "in_features": model.cfg.model_defaults.enc_hidden,
    "dim": 32,                  # LoRA rank — bump to 64 if your val loss
                                # plateaus too early
    "activation": "swish",
    "norm_position": "post",
    "dropout": 0.1,
}
for layer_idx in range(model.cfg.encoder.n_layers):
    model.encoder.layers[layer_idx].add_adapter(
        name="vi_lora", cfg=ADAPTER_CFG,
    )
model.encoder.set_enabled_adapters(["vi_lora"], enabled=True)
model.encoder.unfreeze_enabled_adapters()

# 4. Also unfreeze the prompt_kernel projection — its tiny and helps adapt
#    the language-ID conditioning to your voice + diacritic patterns.
for p in model.prompt_kernel.parameters():
    p.requires_grad = True

# 5. Report what's actually trainable.
total = sum(p.numel() for p in model.parameters())
train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[params] total={total/1e6:.1f}M  trainable={train/1e6:.2f}M "
      f"({100*train/total:.2f}%)")

# 6. Wire up data + optimizer (NeMo does this from the cfg).
model.setup_training_data(cfg.model.train_ds)
model.setup_validation_data(cfg.model.validation_ds)
model.setup_optimization(cfg.model.optim)

# 7. Train.
ckpt_cb = pl.callbacks.ModelCheckpoint(
    dirpath=OUT, filename="vi-lora-{epoch:02d}-{val_wer:.4f}",
    monitor="val_wer", mode="min", save_top_k=3, save_last=True,
)
trainer = pl.Trainer(
    callbacks=[ckpt_cb],
    enable_progress_bar=True,
    **OmegaConf.to_container(cfg.trainer, resolve=True),
)
trainer.fit(model)

# 8. Export the final fine-tuned model.
final = HERE / "models" / "nemotron-vi-lora.nemo"
model.save_to(str(final))
print(f"[done] saved {final}")
```

## 3. Run it

```bash
cd /path/to/nemotron-asr-realtime-translate
nvidia-smi   # confirm no other processes are using GPU memory
.venv/bin/python scripts/finetune_lora.py 2>&1 | tee logs/finetune-$(date +%Y%m%d-%H%M%S).log
```

What you should see:

```
[load] base model ...
[params] total=632.4M  trainable=6.21M (0.98%)
Epoch 0:  100%|██████████| 725/725 [12:34<00:00, val_wer=0.214]
Epoch 1:  100%|██████████| 725/725 [12:31<00:00, val_wer=0.187]
...
```

Total wall-clock for 20 epochs over 10 h of audio: **~5-6 hours** on RTX 2060.

## 4. Monitor + fix common failures

### OOM at start of training

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...
```

Bring memory down in this order:
1. `batch_size: 1` (was 2)
2. `accumulate_grad_batches: 16` (was 8) — keeps effective batch the same
3. `precision: "bf16-mixed"` if your GPU supports it (RTX 20-series does not — stick with `16-mixed`)
4. Drop `dim: 16` for the LoRA rank — cheaper adapter
5. Last resort: enable gradient checkpointing
   ```python
   model.encoder.gradient_checkpointing_enable()
   ```
   ~30% slower but ~40% less activation memory.

### val_wer goes UP during the first few epochs

Normal — the adapters start at random init and have to find the gradient
direction. If it doesn't recover by epoch 3, your LR is too high. Drop to
`lr: 5e-5` and restart.

### val_wer plateaus high after epoch 5

Possible causes:
- not enough training data (back to Step 03)
- LR too low — try `lr: 2e-4`
- adapter capacity too small — try `dim: 64`
- training data has transcript errors (verify a random sample again)

### Training audibly thrashes the GPU and barely progresses

Likely Lhotse bucketing is loading too much audio at once. Drop
`batch_duration: 40` (was 80) and `num_workers: 2` (was 4).

## 5. Stop early if you're winning

Watch `val_wer`. Once it stops dropping for ~3 epochs in a row, **stop**.
Training to "completion" with a small dataset overfits and degrades quality
on out-of-distribution audio.

The `ModelCheckpoint` callback saves the top-3 by `val_wer` — you'll pick
the best of those for deployment, not necessarily the last epoch.

## 6. Save the chosen checkpoint

```bash
ls -la models/checkpoints/
# pick the one with the lowest val_wer in the filename
.venv/bin/python -c "
import nemo.collections.asr as A
m = A.models.ASRModel.restore_from('models/checkpoints/vi-lora-epoch=08-val_wer=0.0942.ckpt')
m.save_to('models/nemotron-vi-lora.nemo')
print('exported')
"
```

→ Next: **[05-deploy.md](05-deploy.md)**
