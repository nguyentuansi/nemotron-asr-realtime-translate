"""CTranslate2-backed translators with HF tokenizer.

Used by stream_translate.py / stream_web.py to translate committed ASR
utterances. Runs on CPU by default so the GPU stays owned by the streaming ASR.

Two backends are available; pick via make_translator(name, model_dir):

  - "nllb"    NLLBTranslator    — facebook/nllb-200-distilled-600M, 20+ langs,
                                  MIT license. Sentencepiece BPE, lang tokens
                                  like 'eng_Latn' / 'vie_Latn'.
  - "envit5"  EnViT5Translator  — VietAI/envit5-translation, vi<->en only,
                                  OpenRAIL-M (commercial OK). T5, prefix
                                  prompting 'vi: ...' / 'en: ...'. ~3x smaller
                                  than NLLB and reports stronger PhoMT/MTet
                                  BLEU on Vietnamese.

Model dirs are produced by `ct2-transformers-converter`:

    nllb-200-distilled-600M-ct2-int8/
        config.json
        model.bin              # CT2 int8 weights
        tokenizer.json
        sentencepiece.bpe.model
        ...

    envit5-translation-ct2-int8/
        config.json
        model.bin
        tokenizer.json
        spiece.model
        ...
"""
from __future__ import annotations

import time
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer, T5TokenizerFast


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


class EnViT5Translator:
    """T5-based bidirectional Vietnamese<->English translator.

    Uses prefix prompting (the model's training format):
        "vi: <vietnamese text>"  -> English output
        "en: <english text>"     -> Vietnamese output

    Only vi<->en is supported. For other language pairs, use NLLBTranslator.
    """

    SUPPORTED_PAIRS = {("vi", "en"), ("en", "vi")}

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
                f"  ct2-transformers-converter --model VietAI/envit5-translation "
                f"--output_dir {model_dir.name} --quantization int8 "
                f"--copy_files tokenizer.json tokenizer_config.json special_tokens_map.json "
                f"spiece.model"
            )
        # Bypass AutoTokenizer here: the copied tokenizer_config.json pins
        # tokenizer_class="T5Tokenizer" (slow), and the slow T5 path on modern
        # transformers/tokenizers raises
        #   TypeError: argument 'vocab': 'dict' object cannot be converted to 'Sequence'
        # while building its internal Unigram. Loading T5TokenizerFast straight
        # from tokenizer.json avoids that entirely.
        self.tokenizer = T5TokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"))
        self.translator = ctranslate2.Translator(
            str(model_dir),
            device=device,
            compute_type=compute_type,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
        )
        self.beam_size = beam_size
        self.max_length = max_length

    @staticmethod
    def _base(lang: str) -> str:
        return lang.split("-", 1)[0].lower()

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text or not text.strip():
            return ""
        src = self._base(src_lang)
        tgt = self._base(tgt_lang)
        if (src, tgt) not in self.SUPPORTED_PAIRS:
            raise ValueError(
                f"EnViT5 only supports vi<->en (got {src_lang}->{tgt_lang}). "
                "Use NLLBTranslator for other language pairs."
            )
        prompt = f"{src}: {text.strip()}"
        source_ids = self.tokenizer(prompt, return_tensors=None)["input_ids"]
        source_tokens = self.tokenizer.convert_ids_to_tokens(source_ids)
        results = self.translator.translate_batch(
            [source_tokens],
            beam_size=self.beam_size,
            max_decoding_length=self.max_length,
        )
        out_tokens = results[0].hypotheses[0]
        out_ids = self.tokenizer.convert_tokens_to_ids(out_tokens)
        out = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        # EnViT5 sometimes echoes the target tag in its output ("en: foo"). Strip.
        low = out.lower()
        for prefix in ("en:", "vi:"):
            if low.startswith(prefix):
                out = out[len(prefix):].lstrip()
                break
        return out


def make_translator(name: str, model_dir, **kwargs):
    """Factory: 'nllb' -> NLLBTranslator, 'envit5' -> EnViT5Translator."""
    key = (name or "").lower().replace("_", "-")
    if key in ("nllb", "nllb-200"):
        return NLLBTranslator(model_dir, **kwargs)
    if key in ("envit5", "envit5-translation"):
        return EnViT5Translator(model_dir, **kwargs)
    raise ValueError(f"Unknown translator '{name}'. Choices: nllb, envit5.")


def _self_test():
    """Smoke test runnable as `python translator.py [nllb|envit5]`. Asserts the
    CTranslate2 model loads and can translate a few Vietnamese<->English clips."""
    import sys
    here = Path(__file__).resolve().parent
    name = sys.argv[1] if len(sys.argv) > 1 else "nllb"
    model_dir = {
        "nllb": here / "nllb-200-distilled-600M-ct2-int8",
        "envit5": here / "envit5-translation-ct2-int8",
    }[name]
    t = make_translator(name, model_dir)
    samples = [
        ("vi-VN", "en-US", "Xin chào, hôm nay trời đẹp quá."),
        ("en-US", "vi-VN", "Hello, the weather is beautiful today."),
        ("vi-VN", "en-US", "A lô a lô một hai ba bốn."),
    ]
    for src, tgt, text in samples:
        t0 = time.time()
        out = t.translate(text, src, tgt)
        dt = time.time() - t0
        print(f"  [{name}] [{src} -> {tgt}] ({dt*1000:.0f}ms) {text!r}\n    -> {out!r}")


if __name__ == "__main__":
    _self_test()
