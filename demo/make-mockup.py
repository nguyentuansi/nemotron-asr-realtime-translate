"""Render an animated terminal mockup of the demo for placeholder use.

Output: demo/demo.gif (960x540, 15 fps).

Mirrors the actual stream_translate.py UI behaviour: the ASR partial (Vi)
and the in-flight translator draft (EN) both update concurrently in the
"live area", EN trailing Vi by ~1 word because the background translator
thread processes each new partial. On commit (silence/sentence-final),
both lines move into history (`vi` + `  ↳ en`) and a fresh live area
starts below for the next utterance.

Run:
    .venv/bin/python demo/make-mockup.py
"""
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
BG = (24, 26, 31)
CHROME = (40, 44, 52)
DIM = (110, 118, 130)
TEXT = (220, 224, 230)
PROMPT = (148, 226, 213)
USER = (240, 240, 240)
LISTENING = (255, 196, 96)
PARTIAL_VI = (180, 188, 200)   # partial (uncommitted) Vi — dim grey
DRAFT_EN = (160, 168, 180)     # in-flight EN draft — even dimmer
COMMIT_VI = (132, 220, 198)    # committed Vi — teal
COMMIT_EN = (255, 218, 121)    # committed EN — warm yellow
STATS = (148, 226, 213)

FPS = 15
FRAMES = []


def font(size, bold=False):
    # DejaVu Sans Mono has the composed Vietnamese diacritic glyphs (Hack Nerd
    # Font / Noto Mono don't); we use it for body text. Arrows are rendered
    # separately via arrow_font() below because DejaVu lacks U+2933 ⤳.
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono"
    return ImageFont.truetype(f"{path}{'-Bold' if bold else ''}.ttf", size)


def arrow_font(size, bold=False):
    # No common monospace font on Ubuntu has U+2933 ⤳ — checked dejavu/hack/
    # firacode/noto-mono via fontTools.getBestCmap. Math/serif fonts do; we
    # use NotoSansMath (clean math arrow) with DejaVu Serif as the fallback.
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansMath-Regular.ttf",
        f"/usr/share/fonts/truetype/dejavu/DejaVuSerif{'-Bold' if bold else ''}.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return font(size, bold)


F_TITLE = font(13)
F_BODY = font(18)
F_BODY_B = font(18, bold=True)
F_ARROW = arrow_font(20)
F_ARROW_B = arrow_font(20, bold=True)

VI_1 = "Đây là demo hệ thống dịch nói thời gian thực."
EN_1 = "This is a real-time speech translation demo."
VI_2 = "Tất cả chạy trên CPU, không cần internet."
EN_2 = "Everything runs on the CPU, no internet needed."

CMD = "./stream_translate.sh --lang vi-VN --target-lang en-US"
LOAD_LINES = [
    "[env]  torch=2.12.0  cuda=True",
    "[load] Nemotron-3.5-asr-streaming-0.6b … 28.4s",
    "[load] NLLB-200-distilled-600M int8 …  2.1s",
    "[asr]  ONNX encoder ready (RTF 0.20)",
]


def base():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 28], fill=CHROME)
    d.ellipse([10, 8, 22, 20], fill=(252, 96, 92))
    d.ellipse([28, 8, 40, 20], fill=(254, 188, 46))
    d.ellipse([46, 8, 58, 20], fill=(40, 200, 64))
    title = "nemotron-asr-realtime-translate"
    tw = d.textlength(title, font=F_TITLE)
    d.text(((W - tw) // 2, 6), title, fill=DIM, font=F_TITLE)
    return img, d


def push(img, n=1):
    for _ in range(n):
        FRAMES.append(img.copy())


def cursor(d, x, y, fnt, on=True):
    if on:
        d.text((x, y), "▌", fill=TEXT, font=fnt)


def render(*, vi_partial="", en_draft="", history=None, show_listen=True, stats=False):
    """Render one frame.

    history is a list of (vi_committed, en_committed) tuples — scrolled-up
    utterances. vi_partial / en_draft are the currently-live area, both
    rendered together (this is the parallel behaviour: ASR partial AND
    translator draft update concurrently).
    """
    img, d = base()
    d.text((20, 44), "$ ", fill=PROMPT, font=F_BODY)
    d.text((44, 44), CMD, fill=USER, font=F_BODY)
    for j, ll in enumerate(LOAD_LINES):
        d.text((20, 76 + j * 26), ll, fill=DIM, font=F_BODY)
    if show_listen:
        d.text((20, 196), "● Listening  vi-VN → en-US", fill=LISTENING, font=F_BODY_B)

    y = 232
    for vi_c, en_c in history or []:
        d.text((20, y), "  vi  ", fill=DIM, font=F_BODY)
        d.text((76, y), vi_c, fill=COMMIT_VI, font=F_BODY)
        y += 28
        d.text((20, y), "    ", fill=DIM, font=F_BODY)
        d.text((60, y - 2), "↳", fill=DIM, font=F_ARROW)
        d.text((84, y), en_c, fill=COMMIT_EN, font=F_BODY_B)
        y += 28

    # Live area: partial Vi + draft EN, both visible simultaneously
    if vi_partial or en_draft:
        d.text((20, y), "  vi  ", fill=DIM, font=F_BODY)
        d.text((76, y), vi_partial + ("▌" if vi_partial else ""), fill=PARTIAL_VI, font=F_BODY)
        y += 28
        d.text((20, y), "    ", fill=DIM, font=F_BODY)
        d.text((60, y - 2), "⤳", fill=DIM, font=F_ARROW)
        d.text((84, y), en_draft, fill=DRAFT_EN, font=F_BODY)
        y += 28

    if stats:
        d.text((20, H - 36),
               "── streaming  ASR RTF 0.20 · NLLB int8 · vi-VN → en-US · MIT",
               fill=STATS, font=F_BODY)
    return img


def stream_pair(vi_text, en_text, *, lag=2, history=None, hold_after=8):
    """Animate vi_text growing word-by-word with en_text trailing by `lag` words.

    Both lines update in the SAME frame — the translator's draft slot is
    overwritten as new ASR partials arrive, just like the live UI.
    """
    vi_words = vi_text.split()
    en_words = en_text.split()
    steps = len(vi_words) + lag  # extra steps so EN can catch up after Vi finishes
    for i in range(1, steps + 1):
        vi_now = " ".join(vi_words[:min(i, len(vi_words))])
        # EN draft trails Vi by `lag` words, capped at full EN
        en_idx = max(0, min(i - lag, len(en_words)))
        en_now = " ".join(en_words[:en_idx])
        FRAMES.append(render(vi_partial=vi_now, en_draft=en_now, history=history))
        FRAMES.append(render(vi_partial=vi_now, en_draft=en_now, history=history))
    # hold the moment just before commit
    final = render(vi_partial=vi_text, en_draft=en_text, history=history)
    push(final, hold_after)


def make():
    # 0. empty prompt, cursor blink
    for i in range(FPS):
        img, d = base()
        d.text((20, 44), "$ ", fill=PROMPT, font=F_BODY)
        cursor(d, 44, 44, F_BODY, on=(i // 4) % 2 == 0)
        FRAMES.append(img)

    # 1. type command
    for i in range(1, len(CMD) + 1):
        img, d = base()
        d.text((20, 44), "$ ", fill=PROMPT, font=F_BODY)
        d.text((44, 44), CMD[:i], fill=USER, font=F_BODY)
        cursor(d, 44 + len(CMD[:i]) * 11, 44, F_BODY)
        push(img, 2 if CMD[i - 1] != " " else 3)

    img, d = base()
    d.text((20, 44), "$ ", fill=PROMPT, font=F_BODY)
    d.text((44, 44), CMD, fill=USER, font=F_BODY)
    push(img, 6)

    # 2. loading lines (cumulative)
    shown = []
    for line in LOAD_LINES:
        shown.append(line)
        img, d = base()
        d.text((20, 44), "$ ", fill=PROMPT, font=F_BODY)
        d.text((44, 44), CMD, fill=USER, font=F_BODY)
        for j, ll in enumerate(shown):
            d.text((20, 76 + j * 26), ll, fill=DIM, font=F_BODY)
        push(img, 6)

    # 3. listening
    for i in range(8):
        img, d = base()
        d.text((20, 44), "$ ", fill=PROMPT, font=F_BODY)
        d.text((44, 44), CMD, fill=USER, font=F_BODY)
        for j, ll in enumerate(LOAD_LINES):
            d.text((20, 76 + j * 26), ll, fill=DIM, font=F_BODY)
        blink = "●" if (i // 2) % 2 == 0 else "○"
        d.text((20, 196), f"{blink} Listening  vi-VN → en-US", fill=LISTENING, font=F_BODY_B)
        FRAMES.append(img)

    # 4. utterance 1 — Vi + EN draft grow in parallel
    stream_pair(VI_1, EN_1, lag=2, history=[])

    # 5. commit utterance 1 — moves into history, live area clears for next
    history_after_1 = [(VI_1, EN_1)]
    push(render(history=history_after_1), 4)

    # 6. utterance 2 — same parallel growth, with previous as history
    stream_pair(VI_2, EN_2, lag=2, history=history_after_1)

    # 7. commit utterance 2 + final stats banner
    history_after_2 = [(VI_1, EN_1), (VI_2, EN_2)]
    push(render(history=history_after_2), 6)
    push(render(history=history_after_2, show_listen=False, stats=True), 22)


def encode_gif(frames):
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
