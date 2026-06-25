"""Record N seconds of mic via the shim, save a wav, transcribe with the
production ASR model. Prints audio stats + transcript.

If this prints your words back to you, mic capture works end-to-end and any
remaining issue is in the streaming pipeline (stream_translate.py /
stream_web.py). If transcript is empty, the captured audio isn't reaching the
model in a recognizable form.

    ./.venv/bin/python mic_to_asr_test.py --lang vi-VN --secs 8
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import logging
if os.environ.get("NEMO_VERBOSE") != "1":
    logging.getLogger("nemo_logger").setLevel(logging.WARNING)
    logging.getLogger("nv_one_logger").setLevel(logging.ERROR)

try:
    import alsaaudio
except ImportError:
    import alsa_shim as alsaaudio

SR = 16000


def record(secs: float, device: str = "default") -> np.ndarray:
    pcm = alsaaudio.PCM(
        type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL,
        device=device, channels=1, rate=SR,
        format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=1024,
    )
    print(f"\n>>> RECORDING {secs:.0f}s — SPEAK NOW")
    for i in range(3, 0, -1):
        print(f"    ...{i}")
        time.sleep(1)
    print("    GO")
    chunks = []
    t0 = time.time()
    while time.time() - t0 < secs:
        n, data = pcm.read()
        if n > 0:
            chunks.append(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
    pcm.close()
    print(">>> STOP\n")
    return np.concatenate(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="vi-VN", help="prompt language (vi-VN, en-US, ...)")
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--device", default="default")
    args = ap.parse_args()

    audio = record(args.secs, args.device)
    secs = len(audio) / SR
    peak = float(np.abs(audio).max())
    mean = float(np.abs(audio).mean())
    voiced = float((np.abs(audio) > 0.05).mean())
    print(f"[audio]  {secs:.2f}s  peak={peak:.3f}  mean_abs={mean:.4f}  voiced_frac={voiced*100:.1f}%")
    if peak < 0.02:
        print("[audio]  WARNING: very quiet — check mic permission in System Settings")

    wav = HERE / "audio" / "mic_capture.wav"
    sf.write(str(wav), audio, SR)
    manifest = HERE / "audio" / "_mic_manifest.json"
    with manifest.open("w") as fp:
        fp.write(json.dumps({
            "audio_filepath": str(wav),
            "duration": secs,
            "text": "",
            "lang": args.lang,
        }) + "\n")
    print(f"[wav]    saved {wav}")

    print("[asr]    loading model (first run takes ~30s on CPU)...")
    import nemo.collections.asr as nemo_asr
    if os.environ.get("NEMO_VERBOSE") != "1":
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(logging.ERROR)
    t0 = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu"
    ).eval()
    model.set_inference_prompt(args.lang)
    print(f"[asr]    loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    out = model.transcribe(audio=[str(manifest)], batch_size=1,
                           target_lang=args.lang, verbose=False)
    text = getattr(out[0], "text", str(out[0]))
    print(f"[asr]    transcribed in {time.time()-t0:.1f}s")
    print(f"\n>>> TRANSCRIPT [{args.lang}]:\n    {text!r}\n")
    if not text.strip():
        print("DIAGNOSIS: empty transcript. Possible causes:")
        print("  - audio is silent or mostly noise (peak should be > 0.1 when speaking)")
        print("  - language mismatch (try the other --lang)")
        print("  - mic was muted in macOS or another app has exclusive access")
    else:
        print("DIAGNOSIS: mic + ASR work end-to-end. Any issue in")
        print("./stream_translate.sh is in the streaming pipeline, not the audio path.")


if __name__ == "__main__":
    main()
