"""CTranslate2-backed NLLB-200 translator with HF tokenizer.

Used by stream_translate.py to translate committed ASR utterances. Runs on CPU
by default so the GPU stays owned by the streaming ASR model.

NLLB language codes are not BCP47 — they're like 'eng_Latn', 'vie_Latn',
'spa_Latn'. We map our ASR lang codes (en-US, vi-VN, ...) to NLLB codes here.

Model dir (created by `ct2-transformers-converter`):
    nllb-200-distilled-600M-ct2-int8/
        config.json
        model.bin              # CT2 int8 weights
        shared_vocabulary.json
        tokenizer.json
        sentencepiece.bpe.model
        ...
"""
from __future__ import annotations

import time
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer


# Map ASR lang codes (BCP47-ish, as the ASR model accepts) → NLLB language codes.
# Add rows as you discover new langs from the ASR's prompt_dictionary.
ASR_TO_NLLB = {
    "en-US": "eng_Latn",
    "en":    "eng_Latn",
    "en-GB": "eng_Latn",
    "enGB":  "eng_Latn",
    "vi-VN": "vie_Latn",
    "vi":    "vie_Latn",
    "es-ES": "spa_Latn",
    "es":    "spa_Latn",
    "es-US": "spa_Latn",
    "esES":  "spa_Latn",
    "fr-FR": "fra_Latn",
    "fr":    "fra_Latn",
    "de-DE": "deu_Latn",
    "de":    "deu_Latn",
    "it-IT": "ita_Latn",
    "pt-BR": "por_Latn",
    "pt-PT": "por_Latn",
    "ja-JP": "jpn_Jpan",
    "ko-KR": "kor_Hang",
    "zh-CN": "zho_Hans",
    "zh-ZH": "zho_Hans",
    "ar":    "arb_Arab",
    "ru-RU": "rus_Cyrl",
    "th-TH": "tha_Thai",
    "id-ID": "ind_Latn",
    "ms-MY": "zsm_Latn",
    "hi-IN": "hin_Deva",
}


def asr_lang_to_nllb(lang: str) -> str:
    """Map an ASR lang code to NLLB. Falls back to None if unknown."""
    if lang in ASR_TO_NLLB:
        return ASR_TO_NLLB[lang]
    # Try base part before the dash.
    base = lang.split("-", 1)[0]
    if base in ASR_TO_NLLB:
        return ASR_TO_NLLB[base]
    return None


class NLLBTranslator:
    """Thin wrapper around ctranslate2.Translator + NLLB sentencepiece tokenizer."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 4,
        max_length: int = 256,
        inter_threads: int = 1,
        intra_threads: int = 4,
    ):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Translator model dir not found: {model_dir}. Run:\n"
                f"  ct2-transformers-converter --model facebook/nllb-200-distilled-600M "
                f"--output_dir {model_dir.name} --quantization int8 "
                f"--copy_files tokenizer.json tokenizer_config.json special_tokens_map.json "
                f"sentencepiece.bpe.model"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.translator = ctranslate2.Translator(
            str(model_dir),
            device=device,
            compute_type=compute_type,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
        )
        self.beam_size = beam_size
        self.max_length = max_length

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate one short string. Sub-second on CPU for typical utterances."""
        if not text or not text.strip():
            return ""
        src = asr_lang_to_nllb(src_lang)
        tgt = asr_lang_to_nllb(tgt_lang)
        if src is None or tgt is None:
            raise ValueError(
                f"Unsupported lang pair {src_lang}->{tgt_lang}. Add a mapping to ASR_TO_NLLB."
            )
        self.tokenizer.src_lang = src
        # Encode source: produces ids including the source lang token + EOS.
        source_ids = self.tokenizer(text, return_tensors=None)["input_ids"]
        source_tokens = self.tokenizer.convert_ids_to_tokens(source_ids)
        # Target prefix: the target lang token. NLLB decodes starting from this.
        target_prefix = [tgt]
        results = self.translator.translate_batch(
            [source_tokens],
            target_prefix=[target_prefix],
            beam_size=self.beam_size,
            max_decoding_length=self.max_length,
        )
        out_tokens = results[0].hypotheses[0]
        # Drop the target-lang prefix token we forced.
        if out_tokens and out_tokens[0] == tgt:
            out_tokens = out_tokens[1:]
        out_ids = self.tokenizer.convert_tokens_to_ids(out_tokens)
        return self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()


def _self_test():
    """Smoke test runnable as `python translator.py`. Asserts the CTranslate2 model
    loads and can translate one short Vietnamese sentence to English."""
    here = Path(__file__).resolve().parent
    model_dir = here / "nllb-200-distilled-600M-ct2-int8"
    t = NLLBTranslator(model_dir)
    samples = [
        ("vi-VN", "en-US", "Xin chào, hôm nay trời đẹp quá."),
        ("en-US", "vi-VN", "Hello, the weather is beautiful today."),
        ("vi-VN", "en-US", "A lô a lô một hai ba bốn."),
    ]
    for src, tgt, text in samples:
        t0 = time.time()
        out = t.translate(text, src, tgt)
        dt = time.time() - t0
        print(f"  [{src} -> {tgt}] ({dt*1000:.0f}ms) {text!r}\n    -> {out!r}")


if __name__ == "__main__":
    _self_test()
