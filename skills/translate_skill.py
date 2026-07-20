"""Translate skill — reuses the existing NLLB/EnViT5 translator stack.

See docs/assistant/00-build-story.md Chapter 5 for design notes.

Vietnamese command shapes handled:
    "dịch sang tiếng Anh: <body>"
    "dịch sang tiếng Nhật <body>"          (colon optional)
    "dịch <body> sang tiếng Trung"          (body-first variant)

Registered by assistant.py with named-group patterns that produce two slots:
    slots["target_name"]  — Vietnamese lang name ("tiếng anh")
    slots["body"]         — the Vietnamese source text

We resolve target_name → NLLB code (jpn_Jpan, eng_Latn, ...) via VI_LANG_NAMES,
then call the shared NLLB translator (translator.make_translator).

Why NLLB not EnViT5:
    EnViT5 is vi↔en only. This skill needs to hit any target language, so we
    always route through NLLB. EnViT5 stays reserved for the streaming
    pipeline's --translator envit5 mode where users specifically want it.

Why a lazy singleton translator:
    make_translator() loads a CT2 model (~1.5 GB on disk, ~200 MB int8 RAM).
    We only load it once per assistant process — on the FIRST translate
    command — and reuse the instance for every subsequent translate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


# Vietnamese language name → NLLB target code. Extend by adding rows.
# Names are lowercased before lookup so the caller doesn't have to worry about
# case — the ASR sometimes capitalizes "Anh" and sometimes doesn't.
VI_LANG_NAMES: dict[str, str] = {
    "tiếng anh": "eng_Latn",
    "tiếng nhật": "jpn_Jpan",
    "tiếng trung": "zho_Hans",
    "tiếng đức": "deu_Latn",
    "tiếng pháp": "fra_Latn",
    "tiếng hàn": "kor_Hang",
    "tiếng nga": "rus_Cyrl",
    "tiếng tây ban nha": "spa_Latn",
    "tiếng bồ đào nha": "por_Latn",
    "tiếng ý": "ita_Latn",
    "tiếng thái": "tha_Thai",
    "tiếng indonesia": "ind_Latn",
    "tiếng ả rập": "arb_Arab",
    "tiếng hindi": "hin_Deva",
}

# NLLB model directory. Same path stream_translate.py uses; if the user
# already ran the NLLB conversion this file exists. If not, we return a
# helpful Vietnamese error message rather than crashing.
_NLLB_MODEL_DIR = Path(__file__).resolve().parent.parent / "nllb-200-distilled-600M-ct2-int8"

_translator = None  # lazy singleton


def _get_translator():
    """Load the NLLB translator once. Cached at module scope."""
    global _translator
    if _translator is None:
        # Import here so importing this skill doesn't force loading the
        # translator module + its heavy deps (transformers, ctranslate2).
        from translator import make_translator
        _translator = make_translator("nllb", _NLLB_MODEL_DIR)
    return _translator


def handle(slots: dict) -> str:
    """slots:
        target_name: Vietnamese language name ("tiếng anh")
        body: the text to translate
    Returns the translation as a plain string (which TTS will speak).

    Note: TTS quality on English/other-language output from Piper Vi is limited
    — this is where the English-loanword problem (Chapter 4) matters. The
    translated string is spoken as-is; TTSSpeaker's preprocessing handles
    what it can.
    """
    body = (slots.get("body") or "").strip()
    target_name = (slots.get("target_name") or "").strip().lower()

    if not body:
        return "Bạn chưa nói câu cần dịch."

    if target_name not in VI_LANG_NAMES:
        # Fail gracefully — list a couple of supported languages so the user
        # knows what to try. We could list all 14 but the response would run
        # for 30s of TTS. Two examples is enough hint.
        return (f"Nemo chưa hỗ trợ dịch sang {target_name}. "
                "Hãy thử tiếng Anh hoặc tiếng Nhật.")

    tgt_code = VI_LANG_NAMES[target_name]

    if not _NLLB_MODEL_DIR.exists():
        # Return a helpful message rather than crashing. Users can convert the
        # model with the ct2-transformers-converter command documented in
        # translator.py's NLLBTranslator error path.
        return ("Chưa cài mô hình dịch NLLB. "
                "Vui lòng chạy script chuyển đổi trước khi dùng lệnh dịch.")

    try:
        translator = _get_translator()
    except Exception as e:
        return f"Không tải được mô hình dịch: {type(e).__name__}"

    # translator.translate(text, src_lang, tgt_lang) — we always translate FROM
    # vi-VN because commands arrive already transcribed in Vietnamese by the ASR.
    # Passing the ASR-style code "vi-VN" (translator.py maps it to vie_Latn
    # internally via ASR_TO_NLLB).
    try:
        translated = translator.translate(body, "vi-VN", _asr_code_for(tgt_code))
    except Exception as e:
        return f"Lỗi khi dịch: {type(e).__name__}"

    return translated


def _asr_code_for(nllb_code: str) -> str:
    """Reverse-lookup: given an NLLB code, return the ASR-style code that
    translator.ASR_TO_NLLB would map back to it.

    translator.translate expects ASR-style codes (en-US, ja-JP, ...) because
    those are what the streaming pipeline uses. This helper bridges the gap
    without exposing ASR_TO_NLLB's internal layout.
    """
    # Import at call time to avoid a circular import risk at module load.
    from translator import ASR_TO_NLLB
    for asr_code, nllb in ASR_TO_NLLB.items():
        if nllb == nllb_code:
            return asr_code
    # If our VI_LANG_NAMES has a code that ASR_TO_NLLB doesn't, fall through
    # to the NLLB code itself — NLLBTranslator will error clearly.
    return nllb_code
