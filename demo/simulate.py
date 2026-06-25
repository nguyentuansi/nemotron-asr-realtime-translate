"""End-to-end pipeline simulation against a pre-recorded WAV.

Stand-in for the mic-driven demo until the real one is recorded. Runs the same
two models the live UI uses — Nemotron ASR for Vi-VN, NLLB-200 for translation
— against `audio/demo_vi.wav`, then writes a transcript to
`demo/simulated-output.txt`.

Usage:
    .venv/bin/python demo/simulate.py
    .venv/bin/python demo/simulate.py audio/my-clip.wav --src vi-VN --tgt en-US
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
if os.environ.get("NEMO_VERBOSE") != "1":
    logging.getLogger("nemo_logger").setLevel(logging.WARNING)
    logging.getLogger("nv_one_logger").setLevel(logging.ERROR)

sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", default=str(ROOT / "audio" / "demo_vi.wav"))
    ap.add_argument("--src", default="vi-VN", help="ASR source lang (default: vi-VN)")
    ap.add_argument("--tgt", default="en-US", help="Translation target lang (default: en-US)")
    ap.add_argument("--translator-dir", default=str(ROOT / "nllb-200-distilled-600M-ct2-int8"))
    ap.add_argument("--out", default=str(HERE / "simulated-output.txt"))
    args = ap.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        sys.exit(f"audio not found: {audio_path}")

    import torch
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}")

    print("[load] importing nemo (slow, first time only)…")
    t0 = time.time()
    import nemo.collections.asr as nemo_asr
    print(f"[load] nemo imported in {time.time()-t0:.1f}s")

    # Manifest: NeMo's transcribe() reads `lang` from each cut for prompt conditioning.
    manifest = ROOT / "audio" / "_sim_manifest.json"
    with manifest.open("w") as fp:
        fp.write(json.dumps({
            "audio_filepath": str(audio_path),
            "duration": 100000,
            "text": "",
            "lang": args.src,
        }) + "\n")

    print(f"[asr ] loading nvidia/nemotron-3.5-asr-streaming-0.6b on cpu…")
    t0 = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
        map_location="cpu",
    ).eval()
    print(f"[asr ] model loaded in {time.time()-t0:.1f}s")

    print(f"[asr ] transcribing {audio_path.name} (lang={args.src})…")
    t0 = time.time()
    out = model.transcribe(audio=[str(manifest)], batch_size=1, target_lang=args.src)
    asr_secs = time.time() - t0
    src_text = getattr(out[0], "text", out[0]).strip()
    print(f"[asr ] done in {asr_secs:.2f}s  →  {src_text!r}")

    print(f"[tx  ] loading NLLBTranslator from {args.translator_dir}…")
    from translator import NLLBTranslator
    t0 = time.time()
    tx = NLLBTranslator(args.translator_dir, device="cpu", compute_type="int8")
    print(f"[tx  ] translator loaded in {time.time()-t0:.1f}s")

    print(f"[tx  ] translating {args.src} → {args.tgt}…")
    t0 = time.time()
    tgt_text = tx.translate(src_text, args.src, args.tgt)
    tx_secs = time.time() - t0
    print(f"[tx  ] done in {tx_secs:.2f}s  →  {tgt_text!r}")

    summary = (
        "=== Simulated demo transcript ==========================================\n"
        f"  audio:        {audio_path.name}\n"
        f"  source lang:  {args.src}\n"
        f"  target lang:  {args.tgt}\n"
        f"  ASR time:     {asr_secs:.2f}s\n"
        f"  Translator:   {tx_secs:.2f}s (NLLB-200-distilled-600M int8)\n"
        "------------------------------------------------------------------------\n"
        f"  [{args.src}]  {src_text}\n"
        f"  [{args.tgt}]  {tgt_text}\n"
        "========================================================================\n"
    )
    print()
    print(summary)
    Path(args.out).write_text(summary)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
