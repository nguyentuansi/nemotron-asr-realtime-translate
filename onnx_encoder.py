"""Drop-in replacement for nemo_asr ConformerEncoder, backed by ONNX Runtime.

Only the forward() pass is routed through ONNX. Auxiliary methods
(get_initial_cache_state, set_default_att_context_size, streaming_cfg, ...)
delegate to the original PyTorch encoder — they're called rarely and aren't
the bottleneck.

Usage:
    from onnx_encoder import wrap_encoder_with_onnx
    model = nemo_asr.models.ASRModel.from_pretrained(...).eval()
    wrap_encoder_with_onnx(model, "models/encoder.onnx", providers=...)

On macOS the CoreML execution provider can route ops through the Apple Neural
Engine (FP16 only) or GPU. On Linux/Windows the CPU EP with INT8 weights is
the typical path. See docs/performance/03-integrate-and-bench.md.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

# NeMo's CacheAwareStreamingAudioBuffer does isinstance(encoder, StreamingEncoder).
# Our wrapper must satisfy that check; subclass it (it's an ABC with abstract
# methods that we delegate via __getattr__).
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder


class ONNXEncoderWrapper(nn.Module, StreamingEncoder):
    """Wraps an ONNX session to look like a NeMo ConformerEncoder.

    Subclasses StreamingEncoder so the streaming buffer's isinstance check
    passes. Inherits nn.Module so PyTorch parameter walks don't crash. Any
    attribute we don't override is fetched from the wrapped original encoder
    via __getattr__ (covers _feat_in, pre_encode, etc.).
    """

    def __init__(self, original_encoder: nn.Module, onnx_path: str | Path, providers):
        super().__init__()
        self._orig = original_encoder
        # ORT session init can throw if a non-CPU EP rejects the graph (e.g.
        # CoreML on a Conformer attention op). Retry CPU-only on failure so we
        # still get the ONNX speedup (which is huge even on CPU EP) instead of
        # falling back to PyTorch entirely.
        try:
            self._sess = ort.InferenceSession(str(onnx_path), providers=providers)
        except Exception as e:
            non_cpu = [p for p in providers
                       if (isinstance(p, str) and p != "CPUExecutionProvider")
                       or (isinstance(p, tuple) and p[0] != "CPUExecutionProvider")]
            if non_cpu:
                print(f"[onnx_encoder] EP init failed ({type(e).__name__}); "
                      f"retrying CPU-only", flush=True)
                self._sess = ort.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"])
            else:
                raise
        self._output_names = [o.name for o in self._sess.get_outputs()]
        self.providers_used = self._sess.get_providers()

    # --- delegated to original PyTorch encoder ---
    def set_default_att_context_size(self, *a, **kw):
        return self._orig.set_default_att_context_size(*a, **kw)

    def get_initial_cache_state(self, *a, **kw):
        return self._orig.get_initial_cache_state(*a, **kw)

    # StreamingEncoder abstract; original already implements it.
    def setup_streaming_params(self, *a, **kw):
        return self._orig.setup_streaming_params(*a, **kw)

    @property
    def streaming_cfg(self):
        return self._orig.streaming_cfg

    # Forward attribute lookup for anything else stream_step touches.
    def __getattr__(self, name):
        # nn.Module's __getattr__ raises AttributeError for missing; chain.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig, name)

    # --- the hot path: route through ONNX ---
    def forward(self, audio_signal=None, length=None,
                cache_last_channel=None, cache_last_time=None,
                cache_last_channel_len=None, **kwargs):
        feeds = {
            "audio_signal": audio_signal.detach().cpu().numpy(),
            "length": length.detach().cpu().numpy().astype(np.int64),
            "cache_last_channel": cache_last_channel.detach().cpu().numpy(),
            "cache_last_time": cache_last_time.detach().cpu().numpy(),
            "cache_last_channel_len": cache_last_channel_len.detach().cpu().numpy().astype(np.int64),
        }
        outs = self._sess.run(self._output_names, feeds)
        return tuple(torch.from_numpy(o) for o in outs)


def pick_providers():
    """Order providers by speed: CUDA -> CPU.

    CoreML is intentionally OFF by default — the Nemotron-3.5 streaming
    Conformer's attention graph hits an axis-rank error during CoreML EP
    initialization (axis=3 out of [-3,2]). The session falls back to CPU
    EP anyway via the wrapper's retry, but the C++ stderr noise is ugly.
    Set ONNX_USE_COREML=1 to attempt CoreML (will likely fail gracefully).
    The CPU EP alone already gives ~8x speedup over the PyTorch encoder.
    """
    import os as _os
    avail = ort.get_available_providers()
    chain = []
    if _os.environ.get("ONNX_USE_COREML") == "1" and "CoreMLExecutionProvider" in avail:
        chain.append(("CoreMLExecutionProvider", {
            "MLComputeUnits": "CPUAndNeuralEngine",
            "ModelFormat": "MLProgram",
        }))
    if "CUDAExecutionProvider" in avail:
        chain.append("CUDAExecutionProvider")
    chain.append("CPUExecutionProvider")
    return chain


def wrap_encoder_with_onnx(model, onnx_path, providers=None):
    """Mutate `model` so encoder forward routes through ONNX. Returns model."""
    if providers is None:
        providers = pick_providers()
    model.encoder = ONNXEncoderWrapper(model.encoder, onnx_path, providers)
    return model
