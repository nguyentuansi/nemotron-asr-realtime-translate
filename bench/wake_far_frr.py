"""Benchmark WakeGate false-accept and false-reject rates.

Usage:
    python bench/wake_far_frr.py \\
        --model models/wake/nemo_oi.onnx \\
        --positives audio/wake_test/positive/ \\
        --negatives audio/wake_test/negative/ \\
        --threshold 0.55

Expects a directory of positive wavs (each contains someone saying "Nemo ơi")
and a directory of negative wavs (Vietnamese speech that isn't "Nemo ơi",
including hard negatives like "Bé ơi", "Bố ơi", TV background, etc.).

Reports:
    FRR — fraction of positive wavs where the wake gate did NOT fire.
          Lower is better. v0 target < 10%; v1 target < 5%.

    FAR — fraction of negative-wav seconds during which the wake gate fired.
          Lower is better. v0 target < 2/hour; v1 target < 1/hour.

Notes:
- We feed each wav in 1024-sample chunks (same rate MicProducer emits) so the
  measurement matches production timing.
- Between wavs we reset the gate to clear pre-roll + cooldown — otherwise
  cooldown from the previous file could hide a legitimate fire.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wake_gate import WakeGate

CHUNK_SAMPLES = 1024
SR = 16000


def _load_wav(path: Path) -> np.ndarray:
    """Load a wav as float32 mono at 16 kHz. Downmix stereo, resample if needed."""
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import scipy.signal as sps
        audio = sps.resample_poly(audio, SR, sr)
    return audio.astype(np.float32)


def _feed(gate: WakeGate, audio: np.ndarray) -> tuple[bool, float]:
    """Feed audio through the gate in 1024-sample chunks.

    Returns (fired_at_least_once, max_score_seen). max_score is useful for
    threshold tuning even when the gate didn't fire.
    """
    max_score = 0.0
    fired = False
    for i in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[i:i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            # Zero-pad the final short chunk so predict() sees a full frame.
            pad = np.zeros(CHUNK_SAMPLES - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])
        ev = gate.process(chunk)
        # We can't peek at the internal score cheaply — we know it's ≥threshold
        # when ev fires, else <threshold. So max_score is a boolean floor.
        if ev is not None:
            fired = True
            max_score = max(max_score, ev.score)
    return fired, max_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to nemo_oi.onnx")
    ap.add_argument("--positives", required=True, help="dir of 'Nemo ơi' wavs")
    ap.add_argument("--negatives", required=True, help="dir of non-wake wavs")
    ap.add_argument("--wake-word-key", default="nemo_oi",
                    help="output label of the openWakeWord model")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--report", default="bench/wake_far_frr.json",
                    help="write results here")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)

    pos_dir = Path(args.positives)
    neg_dir = Path(args.negatives)

    pos_wavs = sorted(pos_dir.glob("*.wav"))
    neg_wavs = sorted(neg_dir.glob("*.wav"))

    if not pos_wavs:
        sys.exit(f"no positive wavs in {pos_dir}")
    if not neg_wavs:
        sys.exit(f"no negative wavs in {neg_dir}")

    print(f"positives: {len(pos_wavs)}  negatives: {len(neg_wavs)}  "
          f"threshold: {args.threshold}\n")

    # --- FRR: positives ---
    print("=== FRR (positives — want ALL to fire) ===")
    frr_misses = 0
    frr_details = []
    for p in pos_wavs:
        gate = WakeGate(model_path=args.model,
                        wake_word_key=args.wake_word_key,
                        threshold=args.threshold)
        audio = _load_wav(p)
        fired, score = _feed(gate, audio)
        status = "FIRE" if fired else "MISS"
        if not fired:
            frr_misses += 1
        print(f"  {status:4s}  {p.name:30s}  dur={len(audio)/SR:.2f}s")
        frr_details.append({"file": p.name, "fired": fired, "duration_s": len(audio)/SR})

    frr = frr_misses / len(pos_wavs) if pos_wavs else 0.0
    print(f"\n  FRR = {frr*100:.1f}%  ({frr_misses}/{len(pos_wavs)} missed)")

    # --- FAR: negatives ---
    print("\n=== FAR (negatives — want NONE to fire) ===")
    total_neg_secs = 0.0
    far_fires = 0
    far_details = []
    for n in neg_wavs:
        gate = WakeGate(model_path=args.model,
                        wake_word_key=args.wake_word_key,
                        threshold=args.threshold)
        audio = _load_wav(n)
        dur = len(audio) / SR
        total_neg_secs += dur
        fired, _ = _feed(gate, audio)
        status = "FIRE!" if fired else "ok"
        if fired:
            far_fires += 1
        print(f"  {status:5s}  {n.name:30s}  dur={dur:.2f}s")
        far_details.append({"file": n.name, "fired": fired, "duration_s": dur})

    far_per_hour = (far_fires / total_neg_secs) * 3600 if total_neg_secs else 0.0
    print(f"\n  FAR = {far_per_hour:.2f}/hour  "
          f"({far_fires} fires over {total_neg_secs:.1f}s = "
          f"{total_neg_secs/3600:.2f}h of negatives)")

    # --- Summary ---
    print("\n=== SLO check ===")
    v0_frr_ok = frr < 0.10
    v0_far_ok = far_per_hour < 2.0
    print(f"  v0 FRR target (<10%):     {'PASS' if v0_frr_ok else 'FAIL'}  "
          f"got {frr*100:.1f}%")
    print(f"  v0 FAR target (<2/hour):  {'PASS' if v0_far_ok else 'FAIL'}  "
          f"got {far_per_hour:.2f}/hour")

    report = {
        "threshold": args.threshold,
        "model": str(args.model),
        "wake_word_key": args.wake_word_key,
        "positives": {
            "count": len(pos_wavs),
            "missed": frr_misses,
            "frr": frr,
            "details": frr_details,
        },
        "negatives": {
            "count": len(neg_wavs),
            "total_seconds": total_neg_secs,
            "fires": far_fires,
            "far_per_hour": far_per_hour,
            "details": far_details,
        },
        "slo": {
            "v0_frr_ok": v0_frr_ok,
            "v0_far_ok": v0_far_ok,
        },
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nreport → {args.report}")


if __name__ == "__main__":
    main()
