"""Measure RTF with the ONNX encoder routed in via wrap_encoder_with_onnx.

BENCH_TAG names the run (default 'onnx'). ONNX_PROVIDER constrains providers,
e.g. ONNX_PROVIDER=cpu disables CoreML.
"""
import os, sys, json, time, logging
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
logging.getLogger("nemo_logger").setLevel(logging.WARNING)
logging.getLogger("nv_one_logger").setLevel(logging.ERROR)

import numpy as np, soundfile as sf, torch
import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nl; nl.setLevel(logging.ERROR)
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
import onnxruntime as ort
from onnx_encoder import wrap_encoder_with_onnx, pick_providers

TAG = os.environ.get("BENCH_TAG", "onnx")
LANG = os.environ.get("BENCH_LANG", "en-US")
ONNX_PROVIDER = os.environ.get("ONNX_PROVIDER", "auto")
SR = 16000
CHUNK = int((1 + 6) * 0.08 * SR)

print(f"[bench] tag={TAG} lang={LANG} provider={ONNX_PROVIDER}")
m = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu"
).eval()
m.set_inference_prompt(LANG)
m.encoder.set_default_att_context_size([56, 6])  # matches stream_translate ATT_CONTEXT["560ms"]

if ONNX_PROVIDER == "cpu":
    providers = ["CPUExecutionProvider"]
elif ONNX_PROVIDER == "coreml":
    providers = [("CoreMLExecutionProvider", {
        "MLComputeUnits": "CPUAndNeuralEngine", "ModelFormat": "MLProgram"
    }), "CPUExecutionProvider"]
else:
    providers = pick_providers()

wrap_encoder_with_onnx(m, HERE / "models" / "encoder.onnx", providers=providers)
print(f"[bench] providers in session: {m.encoder.providers_used}")
cfg = m.encoder.streaming_cfg


def bench_file(wav_path: Path) -> dict:
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != SR:
        import scipy.signal as sps
        audio = sps.resample_poly(audio, SR, sr)
    audio = audio.astype(np.float32)
    cache_lc, cache_lt, cache_lcl = m.encoder.get_initial_cache_state(batch_size=1, device="cpu")
    buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=True)
    prev_hyp, pred_out = None, None
    step = 0; pos = 0; chunk_times = []
    t_start = time.time()
    while pos < len(audio):
        buf.append_audio(audio[pos:pos+CHUNK],
                         stream_id=(-1 if buf.buffer is None else 0))
        pos += CHUNK
        while True:
            if buf.buffer is None: break
            req = cfg.chunk_size[0] if step == 0 else cfg.chunk_size[1]
            if buf.buffer.size(-1) - buf.buffer_idx < req: break
            try:
                ca, cl = next(iter(buf))
            except StopIteration: break
            ct0 = time.time()
            with torch.inference_mode():
                r = m.conformer_stream_step(
                    processed_signal=ca, processed_signal_length=cl,
                    cache_last_channel=cache_lc, cache_last_time=cache_lt,
                    cache_last_channel_len=cache_lcl,
                    keep_all_outputs=False,
                    previous_hypotheses=prev_hyp, previous_pred_out=pred_out,
                    drop_extra_pre_encoded=(0 if step == 0 else cfg.drop_extra_pre_encoded),
                    return_transcription=True,
                )
            chunk_times.append((time.time() - ct0) * 1000)
            pred_out, _, cache_lc, cache_lt, cache_lcl, prev_hyp = r
            step += 1
    elapsed = time.time() - t_start
    audio_dur = len(audio) / SR
    return {
        "file": wav_path.name,
        "audio_s": round(audio_dur, 3),
        "elapsed_s": round(elapsed, 3),
        "rtf": round(elapsed / audio_dur, 3),
        "n_chunks": step,
        "avg_chunk_ms": round(sum(chunk_times) / max(len(chunk_times), 1), 1),
        "p50_chunk_ms": round(float(np.percentile(chunk_times, 50)), 1) if chunk_times else 0,
        "p95_chunk_ms": round(float(np.percentile(chunk_times, 95)), 1) if chunk_times else 0,
    }


candidates = []
bench_dir = HERE / "audio" / "bench"
if bench_dir.exists():
    candidates += sorted(bench_dir.glob("*.wav")) + sorted(bench_dir.glob("*.flac"))
candidates += [HERE / "audio" / "sample1.flac", HERE / "audio" / "sample2.flac",
               HERE / "audio" / "mic_capture.wav"]
candidates = [p for p in candidates if p.exists()]
print(f"[bench] benching {len(candidates)} files (one warmup run discarded per file)\n")
results = []
for p in candidates:
    bench_file(p)  # warmup — first call has session init + JIT
    r = bench_file(p)
    results.append(r)
    print(f"  {r['file']:30s} {r['audio_s']:5.2f}s -> {r['elapsed_s']:5.2f}s  "
          f"RTF={r['rtf']:.2f}  chunks={r['n_chunks']:3d}  "
          f"chunk_ms p50={r['p50_chunk_ms']:5.1f} p95={r['p95_chunk_ms']:5.1f}")

avg_rtf = round(sum(r["rtf"] for r in results) / len(results), 3)
avg_chunk = round(sum(r["avg_chunk_ms"] for r in results) / len(results), 1)
overall = {
    "tag": TAG, "lang": LANG, "providers": m.encoder.providers_used,
    "n_files": len(results), "avg_rtf": avg_rtf, "avg_chunk_ms": avg_chunk,
    "per_file": results,
}
out_path = HERE / "bench" / f"rtf_{TAG}.json"
out_path.write_text(json.dumps(overall, indent=2))
print(f"\n[bench] avg_rtf={avg_rtf}  avg_chunk_ms={avg_chunk}")
print(f"[bench] wrote {out_path}")
