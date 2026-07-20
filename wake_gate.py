"""WakeGate — always-on "Nemo ơi" wake-word detector for the Vietnamese assistant.

See docs/assistant/00-build-story.md Chapter 2 for the design rationale.

Consumes the same 1024-sample chunks that MicProducer emits (16 kHz mono float32).
Runs an openWakeWord ONNX model on each chunk. Maintains a rolling pre-roll
buffer so that when the wake fires, the ASR gets ~1 s of audio BEFORE the wake
instant — the first word of a command often overlaps with the wake phrase.

Public shape:

    from wake_gate import WakeGate

    gate = WakeGate(
        model_path="models/wake/nemo_oi.onnx",
        wake_word_key="nemo_oi",     # matches the model's output label
        threshold=0.55,
        preroll_s=1.0,
        cooldown_s=1.5,
    )
    for chunk in mic_stream:            # 1024 samples, float32 [-1,1]
        ev = gate.process(chunk)
        if ev is not None:
            hand_off_to_asr(ev.pre_roll)
            gate.reset()                 # clear pre-roll & cooldown so the
                                         # NEXT command starts fresh
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

LOG = logging.getLogger("wake_gate")


@dataclass
class WakeEvent:
    """Emitted by WakeGate.process() when the wake phrase is detected."""
    pre_roll: np.ndarray       # last preroll_s of audio, float32, 16 kHz mono
    score: float               # openWakeWord confidence at the fire instant, 0..1
    timestamp: float           # time.time() at detection


class WakeGate:
    """Streaming wake-word detector. NOT thread-safe; call process() from one thread.

    Why single-threaded: the class holds mutable state (pre-roll ring buffer,
    cooldown timestamp). Callers already funnel mic audio through one consumer
    thread, so there's no reason to add locks. Keeps testing dead simple —
    unit tests feed pre-recorded chunks synchronously.
    """

    # If the local model file is missing, we try to download it from here.
    # Set to None to disable auto-download (raise instead).
    # TODO: publish nemo_oi.onnx to a release + fill this URL.
    DEFAULT_MODEL_URL: Optional[str] = None

    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_path: str | Path,
        wake_word_key: str = "nemo_oi",
        threshold: float = 0.55,
        preroll_s: float = 1.0,
        cooldown_s: float = 1.5,
        inference_framework: str = "onnx",
        model_url: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        # openWakeWord returns predictions keyed by the model's internal label,
        # not by the file name. Our trained model exports a "nemo_oi" head; we
        # let callers override in case they load a different model for testing.
        self.wake_word_key = wake_word_key
        self.threshold = threshold
        self.preroll_samples = int(preroll_s * self.SAMPLE_RATE)
        self.cooldown_s = cooldown_s
        self.inference_framework = inference_framework
        self.model_url = model_url or self.DEFAULT_MODEL_URL

        # --- ring-buffer state ---
        # Pre-roll is a fixed-size numpy array. We keep a write cursor; on each
        # chunk we overwrite in place, wrapping. When wake fires, we roll it
        # back into linear order before handing to ASR.
        # np.zeros allocates once at construction; no allocs in the hot path.
        self._preroll = np.zeros(self.preroll_samples, dtype=np.float32)
        self._preroll_pos = 0            # next write index into _preroll
        self._preroll_filled = False     # false until _preroll has wrapped once

        self._last_fire_at = 0.0         # wall-clock of last WakeEvent
        self._model = None               # lazy-loaded openwakeword.Model

    # --------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------

    def process(self, chunk: np.ndarray) -> Optional[WakeEvent]:
        """Feed one 1024-sample float32 chunk. Returns a WakeEvent if the
        model's score for wake_word_key crosses `threshold` AND we're not in
        the post-fire cooldown, else None.
        """
        # Ensure the model is loaded exactly once, on the first real chunk.
        # Lazy load matters because __init__ shouldn't do I/O — that would
        # break unit tests that construct a WakeGate with a bogus path.
        if self._model is None:
            self._ensure_model()

        # Always update pre-roll first, even in cooldown. That way when the
        # cooldown expires, the ring buffer already contains recent audio.
        self._push_preroll(chunk)

        # Cooldown check comes after pre-roll update but BEFORE model.predict.
        # Skipping predict during cooldown saves ~2-4% CPU on the idle path.
        if (time.time() - self._last_fire_at) < self.cooldown_s:
            return None

        # openWakeWord.predict wants int16 samples, not float32 (silent gotcha:
        # if you feed float32 in [-1,1], the melspec sees near-zero amplitude and
        # scores are always ~0). MicProducer gives us float32, so convert here.
        chunk_i16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
        scores = self._model.predict(chunk_i16)
        score = float(scores.get(self.wake_word_key, 0.0))
        if score < self.threshold:
            return None

        # Fire! Snapshot the pre-roll into linear order for the caller. We
        # give the caller a COPY, not a view, because they'll pass it to
        # the ASR streaming buffer which mutates its input.
        preroll_snapshot = self._snapshot_preroll()
        self._last_fire_at = time.time()
        LOG.info("wake fired: score=%.3f (threshold=%.2f)", score, self.threshold)
        return WakeEvent(
            pre_roll=preroll_snapshot,
            score=score,
            timestamp=self._last_fire_at,
        )

    def reset(self) -> None:
        """Clear pre-roll buffer + cool-down. Call this AFTER the ASR consumed
        the wake event's pre_roll, so the next command starts from silence
        rather than leftover audio from the previous command.
        """
        self._preroll[:] = 0.0
        self._preroll_pos = 0
        self._preroll_filled = False
        # We keep _last_fire_at unchanged: the reset happens naturally when
        # cooldown_s elapses. Clearing it here would let a spurious second
        # wake fire mid-command handoff.

    # --------------------------------------------------------------------
    # Private: pre-roll ring buffer
    # --------------------------------------------------------------------

    def _push_preroll(self, chunk: np.ndarray) -> None:
        """Append `chunk` into the ring buffer at current write cursor,
        wrapping if necessary. In-place; no allocation.
        """
        n = chunk.shape[0]
        end = self._preroll_pos + n
        if end <= self.preroll_samples:
            # Fits in one span — the common case.
            self._preroll[self._preroll_pos:end] = chunk
            self._preroll_pos = end % self.preroll_samples
            if end == self.preroll_samples:
                self._preroll_filled = True
        else:
            # Wraps around the end of the buffer.
            first = self.preroll_samples - self._preroll_pos
            self._preroll[self._preroll_pos:] = chunk[:first]
            self._preroll[:n - first] = chunk[first:]
            self._preroll_pos = n - first
            self._preroll_filled = True

    def _snapshot_preroll(self) -> np.ndarray:
        """Return a linear-order copy of the pre-roll buffer."""
        if not self._preroll_filled:
            # Buffer not fully populated yet — return only the valid prefix
            # to avoid feeding leading zeros to the ASR.
            return self._preroll[:self._preroll_pos].copy()
        # Buffer full and wrapped: concatenate the two halves in order.
        return np.concatenate([
            self._preroll[self._preroll_pos:],
            self._preroll[:self._preroll_pos],
        ])

    # --------------------------------------------------------------------
    # Private: model loading + auto-download
    # --------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the openWakeWord model on first predict. Downloads it first
        if we have a URL and the file is missing.
        """
        if not self.model_path.exists():
            if self.model_url:
                LOG.info("wake model missing at %s — downloading from %s",
                         self.model_path, self.model_url)
                self._download_model()
            else:
                raise FileNotFoundError(
                    f"Wake model not found: {self.model_path}.\n\n"
                    "This is expected on a fresh clone — training the 'Nemo ơi' model\n"
                    "requires real Vietnamese voice samples that only you can provide.\n\n"
                    "Three ways to unblock yourself:\n"
                    "  1. Push-to-talk (no wake model needed):\n"
                    "       ./nemo.sh assistant --no-wake\n"
                    "  2. Use a community openWakeWord model for smoke-testing:\n"
                    "       ./nemo.sh assistant --wake-model /path/to/hey_jarvis.onnx \\\n"
                    "                           --wake-word-key hey_jarvis\n"
                    "  3. Train the real 'Nemo ơi' model:\n"
                    "       ./nemo.sh wake-train prepare\n"
                    "       # then record ~200 'Nemo ơi' clips + a VI negative corpus\n"
                    "       ./nemo.sh wake-train train"
                )

        # Lazy import so the module is importable even if openwakeword isn't
        # installed (e.g. running unit tests that don't touch process()).
        from openwakeword.model import Model as OWWModel

        self._model = OWWModel(
            wakeword_models=[str(self.model_path)],
            inference_framework=self.inference_framework,
        )
        LOG.info("wake gate loaded model=%s framework=%s",
                 self.model_path.name, self.inference_framework)

    def _download_model(self) -> None:
        """Fetch DEFAULT_MODEL_URL into self.model_path atomically.

        Atomic = download to a .tmp sibling, then rename. Avoids leaving a
        half-written file if the process is killed mid-download.
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.model_path.with_suffix(self.model_path.suffix + ".tmp")
        urllib.request.urlretrieve(self.model_url, str(tmp))
        tmp.rename(self.model_path)
