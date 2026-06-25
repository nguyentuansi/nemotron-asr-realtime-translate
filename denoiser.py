"""Real-time speech denoiser wrapper around sherpa-onnx + GTCRN.

GTCRN is a 48k-parameter recurrent network (Group Temporal Convolutional
Recurrent Network) that strips stationary + non-stationary noise (fan, AC,
keyboard, room hum) before audio reaches the ASR.

Why this over DeepFilterNet:
  - DF's pypi pkg pins numpy<2.0 and uses removed torchaudio.backend APIs;
    incompatible with our NeMo + torchaudio 2.11 stack
  - sherpa-onnx (Apache 2.0) ships Python 3.13 wheels, works with numpy 2.x,
    and exposes the GTCRN streaming API directly
  - ~40 ms denoise per 560 ms chunk on M-series CPU (RTF ~0.07)

Model file: models/gtcrn_simple.onnx  (~520 KB, download once)

    curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/\
speech-enhancement-models/gtcrn_simple.onnx -o models/gtcrn_simple.onnx
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np


class DenoiseStream:
    """Stateful per-stream wrapper. Not thread-safe — one stream per producer."""

    def __init__(self, model_path: str | Path, num_threads: int = 1):
        import sherpa_onnx
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"GTCRN model not found: {model_path}. Download with:\n"
                f"  curl -L -o {model_path} "
                f"https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                f"speech-enhancement-models/gtcrn_simple.onnx"
            )
        cfg = sherpa_onnx.OnlineSpeechDenoiserConfig()
        cfg.model.gtcrn.model = str(model_path)
        cfg.model.provider = "cpu"
        cfg.model.num_threads = num_threads
        cfg.validate()
        self._sd = sherpa_onnx.OnlineSpeechDenoiser(cfg)
        self.sample_rate = self._sd.sample_rate
        self._lock = threading.Lock()

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Run one chunk through the denoiser. Returns the denoised samples
        produced (variable size due to internal buffering; can be empty)."""
        if chunk.size == 0:
            return chunk
        with self._lock:
            r = self._sd(chunk.astype(np.float32, copy=False), self.sample_rate)
        if not r.samples:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(r.samples, dtype=np.float32)

    def flush(self) -> np.ndarray:
        with self._lock:
            r = self._sd.flush()
        if not r.samples:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(r.samples, dtype=np.float32)
