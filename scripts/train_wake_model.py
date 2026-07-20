"""CLI entry point for training the 'Nemo ơi' wake-word model.

Subcommands:
    prepare      Synthesize 2000 positives + 1000 negatives via Piper Vi.
                 (~15-30 min depending on your CPU. Idempotent — safe to re-run.)
    train        Read data/wake/manifest.yaml, extract features from all
                 enabled sources, train the classifier, export the ONNX.
                 (~5-15 min on M-series CPU.)
    all          prepare + train, in that order.

Config lives in data/wake/manifest.yaml — edit to change data sources or
hyperparameters. The manifest is intentionally pluggable: drop real
recordings into data/wake/positive_real/ and rerun `train`; no code
change needed.

Usage:
    ./nemo.sh wake-train prepare
    ./nemo.sh wake-train train
    ./nemo.sh wake-train all

Design note:
    We don't use openwakeword.train — that module needs the 'acoustics' Python
    package which uses a scipy API removed in modern scipy. Fixing scipy would
    break the rest of the NeMo/ASR stack. Instead we replicate openWakeWord's
    ONNX shape via a small PyTorch classifier + direct training loop. Result
    plugs into openwakeword.model.Model unchanged.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

# Quiet the ambient noise from NeMo/OneLogger even though this script
# doesn't use them directly — imports may transitively pull them.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Number of clips to generate. Tunable. 2000 pos + 1000 neg gives a solid
# baseline for synthetic-only training; more helps marginally.
N_POSITIVES = 2000
N_NEGATIVES = 1000


def cmd_prepare(args) -> None:
    """Generate synthetic positives + negatives into data/wake/."""
    from scripts.wake_pipeline import synthesize_positives, synthesize_negatives

    pos_dir = HERE / "data" / "wake" / "positive_synth"
    neg_dir = HERE / "data" / "wake" / "negative_synth"

    print(f"\n=== Synthetic positives ({N_POSITIVES} clips) ===")
    print(f"    → {pos_dir}")
    synthesize_positives(pos_dir, N_POSITIVES)

    print(f"\n=== Synthetic negatives ({N_NEGATIVES} clips) ===")
    print(f"    → {neg_dir}")
    synthesize_negatives(neg_dir, N_NEGATIVES)

    # Ensure real-recording directories exist so users know where to drop wavs.
    (HERE / "data" / "wake" / "positive_real").mkdir(parents=True, exist_ok=True)
    (HERE / "data" / "wake" / "negative_real").mkdir(parents=True, exist_ok=True)

    print(f"\nDone.")
    print(f"To improve the model, drop real recordings into:")
    print(f"  data/wake/positive_real/   ('Nemo ơi' from you + friends/family)")
    print(f"  data/wake/negative_real/   (Vietnamese speech — Common Voice VI works)")
    print(f"then re-run: ./nemo.sh wake-train train\n")


def cmd_train(args) -> None:
    """Read manifest, extract features, train, export."""
    from scripts.wake_pipeline import (
        load_manifest, build_dataset, train_classifier, export_onnx,
    )

    manifest = load_manifest(HERE / "data" / "wake" / "manifest.yaml")
    cfg = manifest["training"]

    print(f"\n=== Building dataset ===")
    pos_feats, pos_w, neg_feats, neg_w = build_dataset(manifest)
    print(f"positive features: {pos_feats.shape}  weights={pos_w.tolist()[:5]}...")
    print(f"negative features: {neg_feats.shape}  weights={neg_w.tolist()[:5]}...")

    print(f"\n=== Training ===")
    model = train_classifier(pos_feats, pos_w, neg_feats, neg_w, cfg)

    print(f"\n=== Exporting ONNX ===")
    out_path = HERE / cfg["output_path"]
    export_onnx(model, out_path, cfg)
    print(f"Model → {out_path}")

    print(f"\nSmoke test:")
    print(f"  ./nemo.sh assistant --wake-model {cfg['output_path']} \\")
    print(f"                       --wake-word-key {cfg['wake_word_key']}")
    print(f"\nOr run the FAR/FRR benchmark:")
    print(f"  ./nemo.sh bench wake \\")
    print(f"    --model {cfg['output_path']} \\")
    print(f"    --wake-word-key {cfg['wake_word_key']} \\")
    print(f"    --positives data/wake/positive_synth/ \\")
    print(f"    --negatives data/wake/negative_synth/\n")


def cmd_all(args) -> None:
    cmd_prepare(args)
    cmd_train(args)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(name)-14s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train", "all"])
    args = ap.parse_args()

    dispatch = {"prepare": cmd_prepare, "train": cmd_train, "all": cmd_all}
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
