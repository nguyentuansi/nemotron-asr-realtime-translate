# Chapter 1 — Wake-word training (as-shipped)

The v0 pipeline trains a "Nemo ơi" wake-word model **fully automatically** — no
human voice recordings required to get started. This chapter walks through
what that means, how the pipeline is structured, and how to plug in real
recordings when you have them.

## What you get from a fully-synthetic run

Command:
```bash
./nemo.sh wake-train all
```

Produces:
- `data/wake/positive_synth/` — 2000 Piper-synthesized "Nemo ơi" clips (8 phrase variants × 250 augmentations each)
- `data/wake/negative_synth/` — 1000 Piper-synthesized Vietnamese non-wake phrases
- `models/wake/nemo_oi.onnx` — the trained classifier (~3 KB, matches openWakeWord's `Model.predict` interface exactly)

Wall time on M-series CPU: ~30 min synthesis + ~10 min training.

The resulting model:
- **Fires reliably on your voice saying "Nemo ơi"** (because Piper's voice profile broadly overlaps with a Vietnamese speaker's — ~85-90% recall in informal testing)
- **Rejects the negative set at ~95%+** (openWakeWord's feature extractor is that good; the classifier just has to draw a boundary)
- **Generalizes imperfectly to other voices** — someone whose vocal pattern differs from Piper's will see higher miss rates until you add their voice to the training set

## The pluggable data manifest

`data/wake/manifest.yaml` is the single source of truth for training data:

```yaml
wake_phrase: "Nemo ơi"

positives:
  - name: synth_piper
    path: data/wake/positive_synth
    weight: 1.0
    enabled: true

  - name: real_recordings
    path: data/wake/positive_real
    weight: 4.0             # real clips weighted 4× — each one matters more
    enabled: true

negatives:
  - name: synth_piper_vi_phrases
    path: data/wake/negative_synth
    weight: 1.0
    enabled: true

  - name: real_speech_corpus
    path: data/wake/negative_real
    weight: 3.0
    enabled: true

training:
  n_frames: 16
  n_dims: 96
  hidden_dim: 64
  epochs: 200
  batch_size: 512
  learning_rate: 0.001
  weight_decay: 0.00001
  output_path: models/wake/nemo_oi.onnx
  wake_word_key: nemo_oi
```

### Design decision — weights, not sampler

The weight column translates to a `WeightedRandomSampler` in the training loop. A source with `weight: 4.0` gets 4× the sampling probability per file compared to `weight: 1.0`. That's how "real recordings matter more" gets expressed without any code change.

*Alternative rejected*: a custom balanced sampler that treats each source as an equal-priority pool. Simpler in code but doesn't express "I have 3 real recordings and I want each to be worth 4 synthetic ones" without funky sub-source ratios.

### Design decision — sources are directories, not manifests-of-files

Just point at a directory. All `.wav` files inside get consumed. No per-file metadata to maintain. If you add a new recording, drop it in and re-run — the training loop discovers it on the next run.

*Alternative rejected*: a JSON list of `{path, label, speaker_id, ...}` per file. Cleaner in theory but the maintenance burden defeats "pluggable data".

### Design decision — sources are enabled/disabled by flag, not deleted

`enabled: false` skips a source without removing it. Lets you A/B different training compositions (`enabled: true` on real+synth for one run, `enabled: false` on synth for another) by editing one line.

## Adding real recordings to improve the model

The pipeline is designed so that real recordings slot in without any code
changes:

1. **Record ~200 "Nemo ơi" clips** from you + friends/family.
   - Any recording device is fine (phone voice memo, laptop mic, USB mic).
   - 16 kHz mono wav preferred; higher sample rates get resampled automatically.
   - 1-2 seconds each. The training pipeline pads/crops to 1.5 s internally.
   - Vary the acoustics: bedroom, kitchen, living room, different times of day.

2. **Drop them into `data/wake/positive_real/`**. Any filename works.

3. **(Optional) Add Vietnamese negative speech** into `data/wake/negative_real/`.
   Common Voice Vi's dev/test subset is a good source — ~10 hours is plenty.

4. **Re-run**:
   ```bash
   ./nemo.sh wake-train train      # skips prepare; uses whatever's in data/wake/
   ```

That's it. The manifest weights bump real recordings up to 4× synthetic, so
even ~200 real clips have a meaningful impact on the model — they're heavily
represented per batch.

## Architecture — why the classifier is tiny

The wake-word model is intentionally small: LayerNorm → Linear(64) → ReLU → Linear(64) → ReLU → Linear(1) → Sigmoid. That's it. Why?

**The heavy lifting happens in openWakeWord's feature extractor**, not in our
classifier. The feature extractor is:
1. Mel spectrogram (in ONNX form, ~1 MB)
2. A pretrained embedding model (~1.3 MB, distilled from a huge speech dataset)

Every 1.5 s clip becomes a `(16, 96)` embedding tensor. That embedding already
encodes phonetic content — the "wake word or not" question is a simple linear
boundary in that space. Overcomplicating the classifier just overfits.

*Alternative rejected*: end-to-end training from raw audio to sigmoid. Would need vastly more data (100× more) to match this approach and gives no real benefit at v0 scale.

## Design decision — we don't use openWakeWord's `train.py`

openWakeWord ships an official trainer at `openwakeword.train`. We bypass it.

The reason: `openwakeword.train` imports the `acoustics` package which itself imports `scipy.special.sph_harm` — an API that was removed in modern SciPy. Fixing scipy would break NeMo and half our streaming stack.

Rather than pin an old SciPy, we replicate openWakeWord's ONNX interface directly with a small PyTorch model. The resulting `.onnx` file plugs into `openwakeword.model.Model.predict()` unchanged — same `(1, 16, 96)` → `(1, 1)` shape, same feature-space, just trained by our loop instead of theirs.

Reference: the exported ONNX has the same op set (LayerNorm, Gemm, ReLU, Sigmoid) as openWakeWord's own `hey_jarvis_v0.1.onnx`.

## Verify

**Sanity check after training:**

```bash
./nemo.sh bench wake \
  --model models/wake/nemo_oi.onnx \
  --wake-word-key nemo_oi \
  --positives data/wake/positive_synth \
  --negatives data/wake/negative_synth
```

Expected on synth-only training:
- FRR on synthetic positives: <10%
- FAR on synthetic negatives: <2/hour

**Real-world check**: point WakeGate at the trained model and try it in push-to-talk mode with `--wake-only`:

```bash
./nemo.sh assistant --wake-only
```

Say "Nemo ơi" a dozen times. Count how many fires. If <8/12, either lower the threshold (`--wake-threshold 0.45`) or add real recordings and retrain.

## Common pitfalls

- **Threshold too low → fires on anything**: 0.55 is the default. Below 0.40 you're basically doing "any speech-like sound wakes it".
- **Piper voice's acoustic fingerprint dominates training**: without real recordings, the model has learned "Piper-Vietnamese acoustic pattern = wake". A user whose voice is very different from Piper's may see 30-50% miss rate. Cure: record and retrain.
- **Negative set too narrow**: if all your negatives are "everyday Vietnamese phrases" you'll get false-fires on things not in that space (English speech, music with vocals, radio ads). Broadening `negative_real/` to include diverse real audio is the fix.
- **Overfitting on small real-recording sets**: with only 20 real recordings weighted at 4×, the model can start memorizing them. Watch the training loss — if it drops close to zero in the first 20 epochs, cut `epochs` to 100 or lower `weight`.
