"""TTSSpeaker — Vietnamese text-to-speech for the assistant's responses.

See docs/assistant/00-build-story.md Chapter 4 for design notes.

Wraps Piper (open-source neural TTS) with a Vietnamese voice model
(vi_VN-vais1000-medium). Auto-downloads the voice on first construction.

Public shape:

    from tts_speaker import TTSSpeaker

    tts = TTSSpeaker(voice_model="models/piper/vi_VN-vais1000-medium.onnx")
    tts.speak("Bây giờ là 8 giờ 30")    # blocks by default
    tts.speak("Đã đặt báo thức", blocking=False)  # background
    tts.stop()                           # cancel current speech (barge-in)

Why an English-loanword preprocessor:
    Piper Vietnamese was trained on Vietnamese text. When asked to say "CPU"
    it mispronounces it as one Vietnamese word ("sai cấu"). Documented in
    demo/sample-output.txt:25-31. The "spell" workaround rewrites all-caps
    tokens as their Vietnamese letter-name spelling so Piper says
    "xê pê u" instead. Same idea for common camelCase brand names — split
    into words so each half is at least pronounceable.
"""
from __future__ import annotations

import logging
import re
import threading
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

LOG = logging.getLogger("tts_speaker")


# Vietnamese letter-name spellings for the English alphabet. Used by the
# "spell" preprocessor. These are how a Vietnamese person spells English
# letters out loud (e.g. saying an acronym).
_VI_LETTER_NAMES = {
    "a": "a", "b": "bê", "c": "xê", "d": "đê", "e": "e", "f": "ép",
    "g": "gờ", "h": "hát", "i": "i", "j": "gi", "k": "ca", "l": "el",
    "m": "em", "n": "en", "o": "o", "p": "pê", "q": "quy", "r": "rờ",
    "s": "ét", "t": "tê", "u": "u", "v": "vê", "w": "vê kép",
    "x": "ích", "y": "y", "z": "dét",
}

# Match all-uppercase Latin tokens (2..6 chars, like CPU, USB, HTML, iOS).
# Bounded on both sides by word boundaries so we don't chop apart something
# like MRIexam. 2-6 range is a pragmatic compromise — longer than 6 is
# probably a real word that should be spoken as-is.
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6})\b")

# Match camelCase / PascalCase (e.g. "MacBook", "iPhone", "GitHub"). We split
# these on the internal uppercase boundary so each fragment is at least
# individually pronounceable.
_CAMEL_RE = re.compile(r"\b([a-z]+|[A-Z][a-z]+)([A-Z][a-z]+)+\b")


class TTSSpeaker:
    """Piper-backed Vietnamese TTS.

    NOT thread-safe for concurrent speak() calls. The caller (assistant.py)
    serializes: one command → one response → one speak.
    """

    DEFAULT_VOICE_URL = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx"
    )
    DEFAULT_CONFIG_URL = DEFAULT_VOICE_URL + ".json"

    def __init__(
        self,
        voice_model: str | Path,
        english_workaround: str = "spell",   # "spell" | "off"
    ) -> None:
        self.voice_model = Path(voice_model)
        self.english_workaround = english_workaround
        self._voice = None                    # lazy PiperVoice instance
        self._stop_event = threading.Event()
        self._current_thread: Optional[threading.Thread] = None

    # --------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------

    def speak(self, text: str, blocking: bool = True) -> None:
        """Synthesize + play the text. If blocking=True, return when playback
        finishes; if False, return immediately and speak in a background thread.
        Calling speak() while a previous non-blocking speak is still playing
        cancels the previous one (barge-in style).
        """
        if not text or not text.strip():
            return

        # Cancel any in-flight speech first — user just asked to say something
        # else, we honor that immediately.
        if self._current_thread is not None and self._current_thread.is_alive():
            self.stop()
            self._current_thread.join(timeout=0.5)

        self._stop_event.clear()

        processed = self._preprocess_for_vi(text)
        LOG.debug("speak: %r -> %r", text, processed)

        if blocking:
            self._speak_impl(processed)
            return

        self._current_thread = threading.Thread(
            target=self._speak_impl,
            args=(processed,),
            daemon=True,
        )
        self._current_thread.start()

    def stop(self) -> None:
        """Cancel current speech. Used for barge-in ("Nemo ơi, dừng lại")."""
        self._stop_event.set()
        # Also cut whatever sounddevice is currently playing.
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    # --------------------------------------------------------------------
    # Private: synthesis + playback
    # --------------------------------------------------------------------

    def _speak_impl(self, text: str) -> None:
        """Actually synthesize + play. Runs on the caller's thread when
        blocking=True, on the background thread when non-blocking.
        """
        self._ensure_voice()

        # PiperVoice.synthesize() returns an iterable of AudioChunk. Each has
        # an int16 numpy array and knows its own sample rate. For v0 we
        # collect all chunks first, then play once — simple and correct.
        # v1 can move to streaming playback for a ~200 ms latency win.
        chunks = []
        sample_rate = 0
        for ac in self._voice.synthesize(text):
            if self._stop_event.is_set():
                return
            chunks.append(ac.audio_int16_array)
            sample_rate = ac.sample_rate

        if not chunks or sample_rate == 0:
            return

        audio = np.concatenate(chunks)

        # sounddevice.play(...) starts async playback; wait() blocks until done.
        # Between them we periodically check the stop_event so barge-in is
        # snappy — a full playback of a 5s response shouldn't cost 5s to cancel.
        try:
            import sounddevice as sd
            sd.play(audio, sample_rate)
            # Poll for stop while playing; sd.wait() alone would block us from
            # honoring stop() until the buffer drains naturally.
            while sd.get_stream().active:
                if self._stop_event.is_set():
                    sd.stop()
                    return
                self._stop_event.wait(timeout=0.05)
        except Exception as e:
            LOG.warning("audio playback failed: %s", e)

    def _preprocess_for_vi(self, text: str) -> str:
        """Rewrite English tokens for Vietnamese pronunciation.

        "spell" mode:
          - All-caps 2-6 letter tokens → space-separated Vietnamese letter names.
            "CPU" → "xê pê u".
          - camelCase → split on internal uppercase.
            "MacBook" → "Mac Book". Piper handles each half slightly better
            than the joined form.

        "off" mode: pass through untouched.
        """
        if self.english_workaround == "off":
            return text

        def _spell_acronym(match: re.Match) -> str:
            letters = match.group(1)
            return " ".join(_VI_LETTER_NAMES.get(c.lower(), c) for c in letters)

        def _split_camel(match: re.Match) -> str:
            # Insert a space before each internal uppercase letter.
            # "MacBook" → "Mac Book"; "iPhone" → "i Phone"
            token = match.group(0)
            return re.sub(r"(?<!^)(?=[A-Z])", " ", token)

        # camelCase first, then acronyms. Order matters: if we did acronyms
        # first we'd turn "MacBookCPU" into an unrecognizable mess.
        out = _CAMEL_RE.sub(_split_camel, text)
        out = _ACRONYM_RE.sub(_spell_acronym, out)
        return out

    # --------------------------------------------------------------------
    # Private: model loading + auto-download
    # --------------------------------------------------------------------

    def _ensure_voice(self) -> None:
        """Load PiperVoice on first speak(). Auto-download if missing."""
        if self._voice is not None:
            return

        if not self.voice_model.exists():
            LOG.info("piper voice missing at %s — downloading", self.voice_model)
            self._download_voice()

        # Piper's config file lives next to the .onnx with the same stem plus
        # ".json". PiperVoice.load auto-finds it if we don't pass config_path.
        from piper.voice import PiperVoice
        self._voice = PiperVoice.load(str(self.voice_model))
        LOG.info("tts loaded voice=%s", self.voice_model.name)

    def _download_voice(self) -> None:
        """Fetch voice + config into models/piper/ atomically."""
        self.voice_model.parent.mkdir(parents=True, exist_ok=True)

        # Download .onnx
        tmp = self.voice_model.with_suffix(self.voice_model.suffix + ".tmp")
        urllib.request.urlretrieve(self.DEFAULT_VOICE_URL, str(tmp))
        tmp.rename(self.voice_model)

        # Download companion .onnx.json config next to it
        cfg_path = self.voice_model.with_suffix(self.voice_model.suffix + ".json")
        if not cfg_path.exists():
            tmp_cfg = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
            urllib.request.urlretrieve(self.DEFAULT_CONFIG_URL, str(tmp_cfg))
            tmp_cfg.rename(cfg_path)
