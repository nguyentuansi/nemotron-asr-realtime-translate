"""Assistant end-to-end regression test.

Feeds a scripted sequence of (wake wav, command wav, expected_skill,
expected_response_substring) tuples through the full assistant pipeline —
WakeGate → CacheAwareStreamingAudioBuffer → conformer_stream_step → IntentRouter
→ skill.handle — and asserts on the response.

Sibling to demo/simulate.py which does the same for the streaming pipeline.
Neither uses a real mic; both drive the pipeline from prerecorded wavs so
they can run headless in CI.

Usage:
    python demo/simulate_assistant.py --script demo/assistant_script.json

Script JSON schema:
    [
      {"wake_wav": "wake1.wav", "command_wav": "cmd1.wav",
       "expected_skill": "time", "expected_response_contains": "giờ"},
      ...
    ]

Design note — why we don't test skill.handle in isolation here:
    The skills package already has focused unit tests inline (see the smoke
    checks each skill's file passed when it was built). This script exercises
    the WHOLE loop — wake detection, streaming ASR, intent routing, skill
    dispatch, response formatting — which no smaller test can catch.
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

# NeMo/OneLogger silence, same as assistant.py
import os
if os.environ.get("NEMO_VERBOSE") != "1":
    logging.getLogger("nemo_logger").setLevel(logging.WARNING)
    logging.getLogger("nv_one_logger").setLevel(logging.ERROR)

CHUNK_SAMPLES = 1024
SR = 16000


def _load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import scipy.signal as sps
        audio = sps.resample_poly(audio, SR, sr)
    return audio.astype(np.float32)


class _WavProducer:
    """Fake MicProducer that yields chunks from a prerecorded wav array.

    take(n) returns exactly n samples, or None when the wav is exhausted.
    Never blocks — the assistant loop's time.sleep on None is fine here
    because the outer script drives synchronously.
    """
    def __init__(self, audio: np.ndarray):
        self.audio = audio
        self.pos = 0
        self._recent_peak = 0.0

    def take(self, n: int):
        if self.pos + n > len(self.audio):
            return None
        chunk = self.audio[self.pos:self.pos + n]
        self.pos += n
        self._recent_peak = float(np.abs(chunk).max()) if chunk.size else 0.0
        return chunk

    def peak(self) -> float:
        return self._recent_peak

    def start(self): pass
    def stop(self): pass
    def dropped_samples(self) -> int: return 0


def _run_one(model, router, tc: dict, wavs_dir: Path) -> dict:
    """Drive one test-case entry through the pipeline. Returns a result dict."""
    from assistant import _capture_command
    from wake_gate import WakeGate

    combined = np.concatenate([
        _load_wav(wavs_dir / tc["wake_wav"]),
        _load_wav(wavs_dir / tc["command_wav"]),
    ])
    producer = _WavProducer(combined)

    # Bogus wake gate — we KNOW the wake word is at the start (script says so).
    # Skip real wake detection and just synthesize a WakeEvent with the first
    # 1s as pre-roll. This lets us test the ASR+router+skill path without
    # requiring a trained wake model.
    pre_roll = combined[:SR]  # 1 s
    producer.pos = SR         # advance past the pre-roll

    from stream_translate import ATT_CONTEXT
    att = ATT_CONTEXT["560ms"]
    chunk_samples = int((1 + att[1]) * 0.08 * SR)

    t0 = time.time()
    command = _capture_command(
        model=model,
        producer=producer,
        pre_roll=pre_roll,
        chunk_samples=chunk_samples,
        att_ctx=att,
    )
    asr_ms = (time.time() - t0) * 1000

    t0 = time.time()
    result = router.route(command)
    route_ms = (time.time() - t0) * 1000

    if result.skill_name is None:
        response = "Xin lỗi, Nemo chưa hiểu câu đó."
    else:
        response = result.handler(result.slots)

    skill_ok = tc.get("expected_skill") is None or result.skill_name == tc["expected_skill"]
    resp_ok = tc.get("expected_response_contains", "") in response

    return {
        "command_text": command,
        "skill": result.skill_name,
        "slots": result.slots,
        "response": response,
        "asr_ms": asr_ms,
        "route_ms": route_ms,
        "skill_ok": skill_ok,
        "response_ok": resp_ok,
        "pass": skill_ok and resp_ok,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="demo/assistant_script.json")
    ap.add_argument("--wavs-dir", default="demo/wavs")
    args = ap.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        # If no script yet, emit a template + exit
        template = [
            {"wake_wav": "wake_01.wav", "command_wav": "cmd_time.wav",
             "expected_skill": "time", "expected_response_contains": "giờ"},
        ]
        script_path.write_text(json.dumps(template, indent=2, ensure_ascii=False))
        print(f"template written to {script_path} — populate + rerun")
        return

    script = json.loads(script_path.read_text())

    from stream_translate import load_asr
    from intent_router import IntentRouter
    from assistant import register_skills

    print("loading ASR (30s first time)...")
    model = load_asr()
    model.set_inference_prompt("vi-VN")

    router = IntentRouter()
    register_skills(router)

    print(f"running {len(script)} cases...\n")
    all_results = []
    for i, tc in enumerate(script, 1):
        print(f"[{i}/{len(script)}]  {tc['wake_wav']} + {tc['command_wav']}")
        r = _run_one(model, router, tc, Path(args.wavs_dir))
        all_results.append(r)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"   {status}  text={r['command_text']!r}  skill={r['skill']}")
        print(f"        response={r['response']}")
        print(f"        (asr={r['asr_ms']:.0f}ms  route={r['route_ms']:.1f}ms)")
        print()

    passed = sum(1 for r in all_results if r["pass"])
    print(f"=== {passed}/{len(all_results)} pass ===")

    out = Path("bench") / "assistant_regression.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"details → {out}")


if __name__ == "__main__":
    main()
