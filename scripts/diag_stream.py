"""Smoke-test the streaming loop on sample1.flac.

Feeds the file as chunks of raw audio into the same conformer_stream_step()
path stream_demo.py uses, prints each intermediate transcript, and reports
the final result. If the final text matches the known smoke_test output,
the streaming pipeline is correct.
"""
import os, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import soundfile as sf
import torch

LANG = "en-US"
ATT_CTX = [56, 6]  # 560ms chunks

import nemo.collections.asr as nemo_asr
t0 = time.time()
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
    map_location="cpu",
)
print(f"[load] {time.time()-t0:.1f}s")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    model = model.to("cuda").eval()  # fp32; streaming path doesn't auto-cast like transcribe()
else:
    model = model.eval()

model.encoder.set_default_att_context_size(ATT_CTX)
model.set_inference_prompt(LANG)

from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

dev = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype

cache_last_channel, cache_last_time, cache_last_channel_len = (
    model.encoder.get_initial_cache_state(batch_size=1, device=dev)
)
streaming_buffer = CacheAwareStreamingAudioBuffer(model=model, online_normalization=True)

# Read audio
audio, sr = sf.read(str(HERE / "audio" / "sample1.flac"))
audio = audio.astype(np.float32)
print(f"[in] sample1.flac sr={sr} dur={len(audio)/sr:.2f}s")

# Feed exactly one chunk's worth of audio per step. chunk_size is in 80ms encoder
# frames; right_context=6 means chunk_size=7 frames = 7*80ms = 560ms = 8960 samples.
chunk_secs = (1 + ATT_CTX[1]) * 0.08
chunk_samples = int(chunk_secs * sr)
print(f"[cfg] chunk_secs={chunk_secs} chunk_samples={chunk_samples} "
      f"streaming_cfg.chunk_size={model.encoder.streaming_cfg.chunk_size}")

previous_hypotheses = None
pred_out_stream = None
step = 0
last_text = ""

print(f"[cfg] shift_size={model.encoder.streaming_cfg.shift_size} "
      f"pre_encode_cache_size={model.encoder.streaming_cfg.pre_encode_cache_size} "
      f"drop_extra_pre_encoded={model.encoder.streaming_cfg.drop_extra_pre_encoded}")

def required_remaining(step_num):
    """Frames the buffer must have ahead of buffer_idx for the next iter yield to be a full chunk."""
    cs = model.encoder.streaming_cfg.chunk_size
    if step_num == 0 and isinstance(cs, list):
        return cs[0]
    return cs[1] if isinstance(cs, list) else cs


for i in range(0, len(audio) - chunk_samples + 1, chunk_samples):
    new_audio = audio[i: i + chunk_samples]
    sid = -1 if streaming_buffer.buffer is None else 0
    streaming_buffer.append_audio(new_audio, stream_id=sid)
    print(f"  appended idx={i} buffer.size={streaming_buffer.buffer.size(-1)} buffer_idx={streaming_buffer.buffer_idx}")
    # Only run iter while a full chunk is available — the buffer's iter yields
    # short chunks when partial data remains, which produces garbage.
    while streaming_buffer.buffer.size(-1) - streaming_buffer.buffer_idx >= required_remaining(step):
        gen = iter(streaming_buffer)
        try:
            chunk_audio, chunk_lengths = next(gen)
        except StopIteration:
            break
        drop_extra_pre_encoded = (
            model.encoder.streaming_cfg.drop_extra_pre_encoded if step != 0 else 0
        )
        if chunk_audio.dtype != model_dtype:
            chunk_audio = chunk_audio.to(model_dtype)
        with torch.inference_mode():
            result = model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=pred_out_stream,
                drop_extra_pre_encoded=drop_extra_pre_encoded,
                return_transcription=True,
            )
        (
            pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = result
        step += 1
        hyp = transcribed_texts[0] if transcribed_texts else None
        text = getattr(hyp, "text", "") if hyp is not None else ""
        if text != last_text:
            print(f"  step={step:3d} t={(i+chunk_samples)/sr:5.2f}s  {text!r}")
            last_text = text

print(f"\n[final] {last_text!r}")
print(f"[steps] {step}")
