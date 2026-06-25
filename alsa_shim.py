"""sounddevice-backed shim for the subset of pyalsaaudio this repo uses.

Loaded as a fallback on platforms where pyalsaaudio isn't available (macOS,
Windows). Mirrors only the surface area the scripts actually call:
  PCM_FORMAT_S16_LE, PCM_CAPTURE, PCM_NORMAL, ALSAAudioError
  PCM(type=, mode=, device=, channels=, rate=, format=, periodsize=)
  pcm.read() -> (frames, bytes)
  pcm.close()
  pcms(direction) -> [device_name, ...]
"""
import queue

import numpy as np
import sounddevice as sd

PCM_FORMAT_S16_LE = "S16_LE"
PCM_CAPTURE = "capture"
PCM_NORMAL = "normal"


class ALSAAudioError(Exception):
    pass


class PCM:
    def __init__(self, type=PCM_CAPTURE, mode=PCM_NORMAL, device="default",
                 channels=1, rate=16000, format=PCM_FORMAT_S16_LE,
                 periodsize=1024):
        if type != PCM_CAPTURE:
            raise ALSAAudioError("alsa_shim only supports PCM_CAPTURE")
        if format != PCM_FORMAT_S16_LE:
            raise ALSAAudioError(f"alsa_shim only supports S16_LE (got {format})")
        self.channels = channels
        self.rate = rate
        self.periodsize = periodsize
        self._queue: "queue.Queue[bytes]" = queue.Queue()
        sd_device = None if device in (None, "", "default") else device

        def _cb(indata, frames, time_info, status):
            self._queue.put(bytes(indata))

        try:
            self._stream = sd.RawInputStream(
                samplerate=rate,
                channels=channels,
                dtype="int16",
                blocksize=periodsize,
                device=sd_device,
                callback=_cb,
            )
            self._stream.start()
        except Exception as e:
            raise ALSAAudioError(str(e)) from e

    def read(self):
        try:
            data = self._queue.get(timeout=1.0)
        except queue.Empty:
            return 0, b""
        frames = len(data) // (2 * self.channels)
        return frames, data

    def close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


def pcms(direction=PCM_CAPTURE):
    try:
        return [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    except Exception:
        return []
