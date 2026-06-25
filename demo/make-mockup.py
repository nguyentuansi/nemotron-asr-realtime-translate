"""Render an animated terminal mockup of the demo for placeholder use.

Output: demo/demo.gif (~3 MB, ~14 s loop, 960x540, 15 fps).

This is a MOCKUP — it shows what the real demo would look like. Until a real
recording is captured per demo/README.md, this file occupies the README's
hero slot so the page isn't broken.

The mockup is watermarked "MOCKUP" in the corner so it's never confused with
the real recording.

Run:
    .venv/bin/python demo/make-mockup.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "demo.gif"

# Canvas
W, H = 960, 540
BG = (24, 26, 31)          # terminal background
CHROME = (40, 44, 52)      # title bar
DIM = (110, 118, 130)      # subdued text (loading, hints)
TEXT = (220, 224, 230)     # default text
PROMPT = (148, 226, 213)   # teal prompt
USER = (240, 240, 240)     # what the user typed
LISTENING = (255, 196, 96) # listening indicator
PARTIAL_VI = (180, 188, 200)   # partial (uncommitted) Vi — dim
COMMIT_VI = (132, 220, 198)    # committed Vi line — teal/green
PARTIAL_EN = (200, 200, 200)
COMMIT_EN = (255, 218, 121)    # committed English — warm yellow
STATS = (148, 226, 213)
WATERMARK = (90, 92, 98)

# Animation timing (15 fps)
FPS = 15
FRAMES = []

# Load fonts — DejaVu Sans Mono covers Vietnamese diacritics
def font(size, bold=False):
    if bold:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", size)
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)

F_TITLE = font(13)
F_BODY  = font(18)
F_BODY_B = font(18, bold=True)
F_SMALL = font(12)

# Content
VI_1 = "Đây là demo hệ thống dịch nói thời gian thực."
EN_1 = "This is a real-time speech translation demo."
VI_2 = "Tất cả chạy trên CPU, không cần internet."
EN_2 = "Everything runs on the CPU — no internet needed."

CMD = "./stream_translate.sh --lang vi-VN --target-lang en-US"
LOAD_LINES = [
    "[env]  torch=2.12.0  cuda=True",
    "[load] Nemotron-3.5-asr-streaming-0.6b … 28.4s",
    "[load] NLLB-200-distilled-600M int8 …  2.1s",
    "[asr]  ONNX encoder ready (RTF 0.20)",
]


def base():
    """Return a fresh frame with terminal chrome + watermark."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # title bar
    d.rectangle([0, 0, W, 28], fill=CHROME)
    d.ellipse([10, 8, 22, 20], fill=(252, 96, 92))
    d.ellipse([28, 8, 40, 20], fill=(254, 188, 46))
    d.ellipse([46, 8, 58, 20], fill=(40, 200, 64))
    d.text((W // 2 - 80, 6), "nemotron-asr — vi → en", fill=DIM, font=F_TITLE)
    # corner watermark
    d.text((W - 96, 6), "MOCKUP", fill=WATERMARK, font=F_TITLE)
    return img, d


def draw_lines(d, lines, x0=20, y0=44, lh=24):
    """Draw a list of (text, color, font) tuples top-to-bottom."""
    for i, (text, color, fnt) in enumerate(lines):
        d.text((x0, y0 + i * lh), text, fill=color, font=fnt)


def push(img, n=1):
    for _ in range(n):
        FRAMES.append(img.copy())


def cursor(d, x, y, fnt, on=True):
    if on:
        d.text((x, y), "▌", fill=TEXT, font=fnt)


def make():
    # --- 0. empty prompt, cursor blink (1s) ---
    for i in range(FPS):
        img, d = base()
        line_y = 44
        d.text((20, line_y), "$ ", fill=PROMPT, font=F_BODY)
        cursor(d, 44, line_y, F_BODY, on=(i // 4) % 2 == 0)
        FRAMES.append(img)

    # --- 1. type the command (1.5s) ---
    for i in range(1, len(CMD) + 1):
        img, d = base()
        d.text((20, 44), "$ ", fill=PROMPT, font=F_BODY)
        d.text((44, 44), CMD[:i], fill=USER, font=F_BODY)
        cursor(d, 44 + len(CMD[:i]) * 11, 44, F_BODY)
        # vary speed: pause briefly at spaces
        n = 2 if CMD[i - 1] != " " else 3
        push(img, n)

    # hold full command
    img, d = base()
    d.text((20, 44), "$ ", fill=PROMPT, font=F_BODY)
    d.text((44, 44), CMD, fill=USER, font=F_BODY)
    push(img, 6)

    # --- 2. loading lines appear one at a time (2s) ---
    shown_loads = []
    for line in LOAD_LINES:
        shown_loads.append(line)
        img, d = base()
        d.text((20, 44), "$ ", fill=PROMPT, font=F_BODY)
        d.text((44, 44), CMD, fill=USER, font=F_BODY)
        for j, ll in enumerate(shown_loads):
            d.text((20, 76 + j * 26), ll, fill=DIM, font=F_BODY)
        push(img, 6)

    # --- 3. listening indicator (0.5s) ---
    for i in range(8):
        img, d = base()
        d.text((20, 44), "$ ", fill=PROMPT, font=F_BODY)
        d.text((44, 44), CMD, fill=USER, font=F_BODY)
        for j, ll in enumerate(LOAD_LINES):
            d.text((20, 76 + j * 26), ll, fill=DIM, font=F_BODY)
        blink = "●" if (i // 2) % 2 == 0 else "○"
        d.text((20, 196), f"{blink} Listening  vi-VN → en-US", fill=LISTENING, font=F_BODY_B)
        FRAMES.append(img)

    # --- 4. Vi partial 1 grows letter-by-letter (3s, ~3 chars/frame) ---
    def render_state(*, vi_partial="", vi_committed=None, en_partial="", en_committed=None, show_listen=True, stats=False):
        img, d = base()
        d.text((20, 44), "$ ", fill=PROMPT, font=F_BODY)
        d.text((44, 44), CMD, fill=USER, font=F_BODY)
        for j, ll in enumerate(LOAD_LINES):
            d.text((20, 76 + j * 26), ll, fill=DIM, font=F_BODY)
        if show_listen:
            d.text((20, 196), "● Listening  vi-VN → en-US", fill=LISTENING, font=F_BODY_B)

        y = 232
        if vi_committed:
            for line in vi_committed:
                d.text((20, y), "  vi  ", fill=DIM, font=F_BODY)
                d.text((76, y), line, fill=COMMIT_VI, font=F_BODY)
                y += 28
        if en_committed:
            for line in en_committed:
                d.text((20, y), "  en  ", fill=DIM, font=F_BODY)
                d.text((76, y), line, fill=COMMIT_EN, font=F_BODY_B)
                y += 28
        if vi_partial:
            d.text((20, y), "  vi  ", fill=DIM, font=F_BODY)
            d.text((76, y), vi_partial + "▌", fill=PARTIAL_VI, font=F_BODY)
            y += 28
        if en_partial:
            d.text((20, y), "  en  ", fill=DIM, font=F_BODY)
            d.text((76, y), en_partial + "▌", fill=PARTIAL_EN, font=F_BODY)
            y += 28

        if stats:
            d.text((20, H - 36), "── streaming  ASR RTF 0.20 · NLLB int8 · CPU/laptop · MIT", fill=STATS, font=F_BODY)

        return img

    # Vi partial 1 grows in word-chunks (more realistic than letter-by-letter for ASR)
    vi1_words = VI_1.split()
    accum = ""
    for w in vi1_words:
        accum = (accum + " " + w).strip()
        FRAMES.append(render_state(vi_partial=accum))
        FRAMES.append(render_state(vi_partial=accum))
        FRAMES.append(render_state(vi_partial=accum))

    # commit Vi 1: hold briefly with cursor blink, then commit
    for _ in range(4):
        FRAMES.append(render_state(vi_partial=VI_1))

    # En partial 1 grows
    en1_words = EN_1.split()
    accum_en = ""
    for w in en1_words:
        accum_en = (accum_en + " " + w).strip()
        FRAMES.append(render_state(vi_committed=[VI_1], en_partial=accum_en))
        FRAMES.append(render_state(vi_committed=[VI_1], en_partial=accum_en))

    # commit En 1
    for _ in range(6):
        FRAMES.append(render_state(vi_committed=[VI_1], en_committed=[EN_1]))

    # Vi partial 2 grows
    vi2_words = VI_2.split()
    accum2 = ""
    for w in vi2_words:
        accum2 = (accum2 + " " + w).strip()
        FRAMES.append(render_state(vi_committed=[VI_1], en_committed=[EN_1], vi_partial=accum2))
        FRAMES.append(render_state(vi_committed=[VI_1], en_committed=[EN_1], vi_partial=accum2))
        FRAMES.append(render_state(vi_committed=[VI_1], en_committed=[EN_1], vi_partial=accum2))

    for _ in range(4):
        FRAMES.append(render_state(vi_committed=[VI_1], en_committed=[EN_1], vi_partial=VI_2))

    # En partial 2 grows
    en2_words = EN_2.split()
    accum_en2 = ""
    for w in en2_words:
        accum_en2 = (accum_en2 + " " + w).strip()
        FRAMES.append(render_state(vi_committed=[VI_1, VI_2], en_committed=[EN_1], en_partial=accum_en2))
        FRAMES.append(render_state(vi_committed=[VI_1, VI_2], en_committed=[EN_1], en_partial=accum_en2))

    # commit + stats reveal
    for _ in range(8):
        FRAMES.append(render_state(vi_committed=[VI_1, VI_2], en_committed=[EN_1, EN_2]))
    for _ in range(20):
        FRAMES.append(render_state(vi_committed=[VI_1, VI_2], en_committed=[EN_1, EN_2], show_listen=False, stats=True))


def encode_gif(frames):
    """Frames → GIF via ffmpeg+palettegen (universal, no gifski dependency)."""
    if not frames:
        sys.exit("no frames")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, f in enumerate(frames):
            f.save(td / f"f{i:05d}.png")
        palette = td / "palette.png"
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS),
            "-i", str(td / "f%05d.png"),
            "-vf", "palettegen=stats_mode=full",
            str(palette),
        ])
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS),
            "-i", str(td / "f%05d.png"),
            "-i", str(palette),
            "-lavfi", "paletteuse=dither=bayer:bayer_scale=5",
            "-loop", "0",
            str(OUT),
        ])
    size_mb = OUT.stat().st_size / 1_048_576
    print(f"wrote {OUT}  ({size_mb:.2f} MB, {len(frames)} frames, {len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found")
    make()
    encode_gif(FRAMES)
