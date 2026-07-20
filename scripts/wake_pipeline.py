"""Wake-word training pipeline — synthesis, augmentation, features, training, export.

See docs/assistant/00-build-story.md Chapter 2 for the design rationale.

This module is intentionally functional (not object-oriented) — every stage
is a top-level function that reads from disk and writes to disk. That lets
each stage be tested / re-run in isolation, and lets the CLI wrapper compose
them.

Public functions:
    synthesize_positives(voice, out_dir, n)
    synthesize_negatives(voice, out_dir, n)
    load_manifest(path) -> dict
    build_dataset(manifest) -> (positive_feats, negative_feats)
    train_classifier(pos_feats, neg_feats, cfg) -> torch.nn.Module
    export_onnx(model, path, cfg)

Why not use openWakeWord's official training script:
    openwakeword.train imports 'acoustics' which needs a scipy API removed in
    modern scipy. Fixing scipy would break NeMo + the rest of the ASR stack.
    Instead we replicate openWakeWord's classifier shape and train it directly
    with PyTorch — the resulting ONNX plugs into openWakeWord's Model.predict
    unchanged, because it has the exact same (1, 16, 96) → (1, 1) interface.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import yaml

LOG = logging.getLogger("wake_pipeline")

# Piper voice model. Auto-downloaded by TTSSpeaker if missing.
PIPER_VOICE = Path("models/piper/vi_VN-vais1000-medium.onnx")

SAMPLE_RATE = 16000

# 2.0 s at 16 kHz — this is the length that produces EXACTLY 16 embedding
# frames (matching openWakeWord's classifier input shape). Empirically
# verified: AudioFeatures.get_embedding_shape(2.0) == (16, 96).
# 1.5s (a naive first guess) gives 9 frames → had to zero-pad → model
# learned a bogus "half-zero-frames" pattern that never appears at inference.
CLIP_SAMPLES = int(2.0 * SAMPLE_RATE)


# =====================================================================
# Stage 1 — synthesize positives / negatives with Piper + augmentation
# =====================================================================

# Wake phrase variants — all sound like "Nemo ơi" to a Vietnamese speaker
# but Piper renders each slightly differently, giving natural variation.
_WAKE_VARIANTS = [
    "Nemo ơi",
    "Nemo ơi.",
    "Nemo ơi,",
    "Nê mô ơi",
    "Nê-mô ơi",
    "Nemo ơi!",
    "Nemo ơi?",
    "Nemo, ơi",
]

# Vietnamese phrases that are NOT the wake word. Piper synthesizes each →
# our "close but not wake" negative set. Mix of everyday sentences, numbers,
# household references, and words phonetically similar to "Nemo" / "ơi" so
# the classifier learns the boundary explicitly.
_NEGATIVE_PHRASES = [
    # Everyday
    "Xin chào bạn", "Chúc bạn ngủ ngon", "Cảm ơn nhiều nhé", "Đi đâu đấy",
    "Ăn cơm chưa", "Về nhà thôi", "Trời hôm nay đẹp quá", "Mẹ ơi",
    "Bố ơi", "Bà ơi", "Em ơi", "Con ơi", "Bé ơi", "Chị ơi",
    # Numbers / dates
    "Một hai ba bốn năm", "Sáu bảy tám chín mười", "Hôm nay là thứ hai",
    "Ngày mai họp lúc mười giờ", "Chín tháng chạp năm hai không hai sáu",
    # Common commands to other assistants
    "Hey Siri", "Alexa", "Ok Google", "Bật đèn", "Tắt đèn phòng ngủ",
    # Vietnamese-specific words with 'ơ' or 'ê' vowels
    "Người phụ nữ", "Cơm rau canh", "Ngày mưa", "Mê cung", "Kê khai",
    "Nê tường vôi trắng", "Nê máy giặt", "Kem tươi", "Nghê ngợi",
    # Random news-style phrases
    "Giá xăng dầu tăng", "Hôm qua có mưa lớn", "Đội tuyển bóng đá thắng",
    "Kinh tế tăng trưởng ổn định", "Học sinh nghỉ hè", "Đêm nay trời quang",
    # Sentences with the exact syllables "ne", "mo", "oi" but not together
    "Nếu mà chị đi", "Mo ước của tôi", "Ơi các bạn", "Nề nếp gia đình",
    "Không nê ai cả", "Ba mô hình khác nhau",
]


def _piper_synth(voice, text: str) -> tuple[np.ndarray, int]:
    """Synthesize `text` via Piper. Returns (float32 audio in [-1,1], sample_rate)."""
    chunks = []
    sr = 0
    for ac in voice.synthesize(text):
        chunks.append(ac.audio_int16_array.astype(np.float32) / 32768.0)
        sr = ac.sample_rate
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return audio, sr


def _resample(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """Resample audio to `tgt_sr`. Uses scipy.signal.resample_poly (fast, decent quality)."""
    if src_sr == tgt_sr:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(src_sr, tgt_sr)
    return resample_poly(audio, tgt_sr // g, src_sr // g).astype(np.float32)


def _augment(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply one random augmentation combination to a float32 clip.

    We compose 0-2 transforms from a fixed set. Each transform is deliberately
    small so combinations don't destroy the signal — the goal is 'the same
    wake phrase, said a bit differently'.

    Transforms:
      - Gain (±3 dB)
      - Time stretch (0.9-1.1×) via simple resample trick
      - Pitch shift approximation (±1 semitone via resample+trim)
      - Additive noise (white, -25 to -15 dB SNR)
      - Silence padding on either side
    """
    ops = []

    # Gain
    if rng.random() < 0.7:
        gain_db = rng.uniform(-3, 3)
        audio = audio * (10 ** (gain_db / 20))
        ops.append(f"gain{gain_db:+.1f}")

    # Time stretch — resample + trim/pad back to original length. Cheap approx.
    if rng.random() < 0.5:
        rate = rng.uniform(0.9, 1.1)
        new_len = int(len(audio) / rate)
        if new_len > 0:
            audio = _resample(audio, len(audio), new_len)
            ops.append(f"speed{rate:.2f}")

    # White noise mix
    if rng.random() < 0.6:
        snr_db = rng.uniform(15, 30)
        sig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
        noise_rms = sig_rms / (10 ** (snr_db / 20))
        noise = rng.gauss(0, noise_rms) * np.ones(len(audio))
        # Actually use per-sample noise, not a constant — a scalar RNG.gauss returns 1 value
        rng_np = np.random.default_rng(rng.randint(0, 2**32 - 1))
        noise = (rng_np.standard_normal(len(audio)) * noise_rms).astype(np.float32)
        audio = audio + noise
        ops.append(f"snr{snr_db:.0f}dB")

    # Clip to prevent overflow after augmentation
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    return audio


def _pad_or_crop_to(audio: np.ndarray, target_samples: int, rng: random.Random) -> np.ndarray:
    """Pad with silence or crop to exactly target_samples. Puts the audio at
    a random position within the padded window so the model doesn't learn
    'wake word always starts at frame 0'."""
    if len(audio) >= target_samples:
        # Crop from a random start
        start = rng.randint(0, len(audio) - target_samples)
        return audio[start:start + target_samples]
    # Pad — random position within window
    pad_total = target_samples - len(audio)
    pad_left = rng.randint(0, pad_total)
    pad_right = pad_total - pad_left
    return np.concatenate([
        np.zeros(pad_left, dtype=np.float32),
        audio,
        np.zeros(pad_right, dtype=np.float32),
    ])


def _write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save as 16-bit PCM wav — the format openWakeWord expects."""
    sf.write(str(path), audio, sr, subtype="PCM_16")


def _load_piper() -> "PiperVoice":
    """Load the Piper voice, downloading if needed. Lazy import."""
    from piper.voice import PiperVoice
    if not PIPER_VOICE.exists():
        # Reuse TTSSpeaker's download logic instead of duplicating.
        from tts_speaker import TTSSpeaker
        LOG.info("piper voice missing — downloading via TTSSpeaker")
        t = TTSSpeaker(voice_model=PIPER_VOICE)
        t._ensure_voice()   # side effect: download
    return PiperVoice.load(str(PIPER_VOICE))


def synthesize_positives(out_dir: Path, n: int, seed: int = 42) -> int:
    """Generate `n` synthetic positive clips of 'Nemo ơi' at 16 kHz mono.

    Each clip is one wake-phrase variant + random augmentation, padded/cropped
    to CLIP_SAMPLES (1.5 s). Files: `out_dir/synth_{idx:05d}.wav`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    voice = _load_piper()

    # Cache Piper output per variant — synthesis is the slow part, augmentation
    # is fast, so we synthesize each variant ONCE and augment many times.
    LOG.info("synthesizing %d variants of the wake phrase", len(_WAKE_VARIANTS))
    base_audio = {}
    for variant in _WAKE_VARIANTS:
        audio, sr = _piper_synth(voice, variant)
        base_audio[variant] = _resample(audio, sr, SAMPLE_RATE)

    LOG.info("writing %d augmented positive clips → %s", n, out_dir)
    t0 = time.time()
    for i in range(n):
        variant = _WAKE_VARIANTS[i % len(_WAKE_VARIANTS)]
        raw = base_audio[variant].copy()
        aug = _augment(raw, rng)
        final = _pad_or_crop_to(aug, CLIP_SAMPLES, rng)
        _write_wav(out_dir / f"synth_{i:05d}.wav", final)
        if (i + 1) % 200 == 0:
            LOG.info("  %d/%d in %.1fs", i + 1, n, time.time() - t0)
    LOG.info("done in %.1fs", time.time() - t0)
    return n


def synthesize_negatives(out_dir: Path, n: int, seed: int = 43) -> int:
    """Generate `n` synthetic negative clips — Vietnamese phrases that aren't
    the wake word, plus phonetically-close distractors.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    voice = _load_piper()

    # Cache per unique phrase (much fewer than n)
    LOG.info("synthesizing %d unique negative phrases", len(_NEGATIVE_PHRASES))
    base_audio = {}
    for phrase in _NEGATIVE_PHRASES:
        audio, sr = _piper_synth(voice, phrase)
        base_audio[phrase] = _resample(audio, sr, SAMPLE_RATE)

    LOG.info("writing %d augmented negative clips → %s", n, out_dir)
    t0 = time.time()
    for i in range(n):
        phrase = _NEGATIVE_PHRASES[i % len(_NEGATIVE_PHRASES)]
        raw = base_audio[phrase].copy()
        aug = _augment(raw, rng)
        final = _pad_or_crop_to(aug, CLIP_SAMPLES, rng)
        _write_wav(out_dir / f"synth_{i:05d}.wav", final)
        if (i + 1) % 200 == 0:
            LOG.info("  %d/%d in %.1fs", i + 1, n, time.time() - t0)
    LOG.info("done in %.1fs", time.time() - t0)
    return n


# =====================================================================
# Stage 2 — manifest loading + data gathering
# =====================================================================

def load_manifest(path: Path = Path("data/wake/manifest.yaml")) -> dict:
    """Load the training manifest. Fails clearly if missing/invalid."""
    if not path.exists():
        raise FileNotFoundError(
            f"manifest not found: {path}. "
            f"Run `./nemo.sh wake-train prepare` to create the default one."
        )
    return yaml.safe_load(path.read_text())


def _gather_wavs(sources: list[dict]) -> list[tuple[Path, float]]:
    """Return a list of (wav_path, weight) tuples from all enabled sources.

    weight is repeated per file in the source — so a source with weight=3 and
    100 files contributes 300 (path, 1.0) entries. Simplifies downstream
    weighted sampling.
    """
    result: list[tuple[Path, float]] = []
    for src in sources:
        if not src.get("enabled", True):
            continue
        d = Path(src["path"])
        if not d.exists():
            LOG.warning("source path missing: %s (skipping)", d)
            continue
        wavs = list(d.glob("*.wav"))
        if not wavs:
            LOG.warning("no wavs in %s (skipping)", d)
            continue
        weight = float(src.get("weight", 1.0))
        LOG.info("source %s: %d wavs × weight %.1f", src.get("name", d.name), len(wavs), weight)
        result.extend((w, weight) for w in wavs)
    return result


# =====================================================================
# Stage 3 — feature extraction using openWakeWord's AudioFeatures
# =====================================================================

def _extract_features(wav_paths: list[Path], n_frames: int) -> np.ndarray:
    """Extract (N, n_frames, 96) features from wavs using openWakeWord's
    AudioFeatures (melspec + embedding). Uses openWakeWord's own model files
    so our output matches theirs bit-for-bit.
    """
    from openwakeword.utils import AudioFeatures
    af = AudioFeatures(inference_framework="onnx")

    # Load audio into (N, N_samples) int16 for embed_clips.
    audios = []
    for p in wav_paths:
        audio, sr = sf.read(str(p))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = _resample(audio.astype(np.float32), sr, SAMPLE_RATE)
        # Pad or crop to exactly CLIP_SAMPLES
        rng = random.Random(hash(str(p)) & 0xFFFFFFFF)
        audio = _pad_or_crop_to(audio.astype(np.float32), CLIP_SAMPLES, rng)
        audios.append((audio * 32767).astype(np.int16))

    x = np.stack(audios)
    LOG.info("extracting features from %d clips (input shape %s)...", len(audios), x.shape)
    feats = af.embed_clips(x, batch_size=64)
    # AudioFeatures returns (N, frames, 96). Center-crop or pad the frame dim.
    N, F, D = feats.shape
    LOG.info("features raw shape: %s", feats.shape)
    if F < n_frames:
        pad = np.zeros((N, n_frames - F, D), dtype=feats.dtype)
        feats = np.concatenate([feats, pad], axis=1)
    elif F > n_frames:
        start = (F - n_frames) // 2
        feats = feats[:, start:start + n_frames]
    LOG.info("features clipped to %s", feats.shape)
    return feats


def build_dataset(manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    """From the manifest, gather all wavs, extract features, return
    (positive_features, negative_features) as float32 arrays.

    Weights in the manifest translate to sample duplication so a real-recording
    source with weight=4 contributes 4× samples per file. Cheap way to get
    weighted training without a custom sampler.
    """
    cfg = manifest["training"]
    n_frames = cfg["n_frames"]

    pos_sources = _gather_wavs(manifest["positives"])
    neg_sources = _gather_wavs(manifest["negatives"])
    if not pos_sources:
        raise RuntimeError("no positive wavs found — run `./nemo.sh wake-train prepare`?")
    if not neg_sources:
        raise RuntimeError("no negative wavs found — run `./nemo.sh wake-train prepare`?")

    # Extract features per unique file (not per weighted copy). We'll expand
    # by weight at the training-batch level.
    pos_paths = [p for p, _ in pos_sources]
    neg_paths = [p for p, _ in neg_sources]
    pos_weights = np.array([w for _, w in pos_sources], dtype=np.float32)
    neg_weights = np.array([w for _, w in neg_sources], dtype=np.float32)

    pos_feats = _extract_features(pos_paths, n_frames)
    neg_feats = _extract_features(neg_paths, n_frames)

    return pos_feats, pos_weights, neg_feats, neg_weights


# =====================================================================
# Stage 4 — classifier + training loop
# =====================================================================

class WakeClassifier:
    """Small feedforward classifier matching openWakeWord's ONNX shape.

    Not an nn.Module for scaffolding — see build_torch_model() for the real
    PyTorch construction (lazy import so this module is cheap to load).
    """


def build_torch_model(n_frames: int = 16, n_dims: int = 96, hidden: int = 64):
    """Return a torch.nn.Module: (batch, 16, 96) → (batch, 1) sigmoid probability.

    Architecture matches openWakeWord's published models' op set:
    LayerNorm → Flatten → Linear+ReLU → Linear+ReLU → Linear+Sigmoid.
    """
    import torch.nn as nn
    import torch

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(n_dims)
            self.fc1 = nn.Linear(n_frames * n_dims, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, 1)

        def forward(self, x):
            # x: (batch, n_frames, n_dims)
            x = self.ln(x)
            x = x.flatten(1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            return x

    return Net()


def train_classifier(pos_feats: np.ndarray, pos_weights: np.ndarray,
                     neg_feats: np.ndarray, neg_weights: np.ndarray,
                     cfg: dict) -> "torch.nn.Module":
    """Train the classifier on positive + negative features.

    Uses BCELoss + Adam. Balances positives/negatives per batch via a
    WeightedRandomSampler that respects manifest weights.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    # Features → tensors + label 1 for positives, 0 for negatives
    Xp = torch.from_numpy(pos_feats.astype(np.float32))
    Yp = torch.ones(len(Xp), 1)
    Xn = torch.from_numpy(neg_feats.astype(np.float32))
    Yn = torch.zeros(len(Xn), 1)

    X = torch.cat([Xp, Xn])
    Y = torch.cat([Yp, Yn])

    # Per-sample weights = per-source weight (from manifest). We combine with
    # a class-balance factor so positives (typically far fewer) are drawn as
    # often as negatives despite being rarer in the dataset.
    n_pos = len(Xp)
    n_neg = len(Xn)
    per_sample_w = torch.zeros(n_pos + n_neg)
    per_sample_w[:n_pos] = torch.from_numpy(pos_weights) * (n_neg / max(n_pos, 1))
    per_sample_w[n_pos:] = torch.from_numpy(neg_weights)

    sampler = WeightedRandomSampler(
        weights=per_sample_w.tolist(),
        num_samples=cfg["batch_size"] * 200,   # 200 batches per epoch
        replacement=True,
    )
    loader = DataLoader(TensorDataset(X, Y), batch_size=cfg["batch_size"], sampler=sampler)

    model = build_torch_model(cfg["n_frames"], cfg["n_dims"], cfg["hidden_dim"])
    opt = torch.optim.Adam(model.parameters(),
                            lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])
    loss_fn = torch.nn.BCELoss()

    LOG.info("training: %d pos + %d neg samples, %d epochs, batch=%d",
             n_pos, n_neg, cfg["epochs"], cfg["batch_size"])
    model.train()
    for epoch in range(cfg["epochs"]):
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % max(1, cfg["epochs"] // 10) == 0:
            LOG.info("  epoch %3d/%d  loss=%.4f", epoch + 1, cfg["epochs"], total_loss / n_batches)

    # Eval on the full train set — cheap sanity check
    model.eval()
    with torch.no_grad():
        pred_pos = model(Xp).squeeze().numpy()
        pred_neg = model(Xn).squeeze().numpy()
    LOG.info("  train FRR@0.55 = %.1f%%  train FAR@0.55 = %.1f%%",
             float(np.mean(pred_pos < 0.55) * 100),
             float(np.mean(pred_neg >= 0.55) * 100))

    return model


# =====================================================================
# Stage 5 — ONNX export
# =====================================================================

def export_onnx(model, out_path: Path, cfg: dict) -> None:
    """Export the trained classifier to ONNX at (1, n_frames, n_dims) → (1, 1)
    so openWakeWord's Model.predict can load it unchanged.
    """
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, cfg["n_frames"], cfg["n_dims"])
    model.eval()

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["x.1"],
        output_names=[cfg["wake_word_key"]],
        dynamic_axes={"x.1": {0: "batch"}, cfg["wake_word_key"]: {0: "batch"}},
        opset_version=17,
    )
    LOG.info("exported %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)
