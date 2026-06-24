# Step 02 — quick wins (Lane 1 + optional Lane 2)

Goal: capture the cheap improvements before committing to weeks of fine-tuning.

**Time**: ½ day (Lane 1), +2-5 days if you do Lane 2 too.

**Output**: a measurable WER drop relative to Step 01's baseline.

## Lane 1 — Vietnamese diacritic restorer (do this)

The most-visible errors in our production logs are tone-mark / diacritic
confusions on short words: `máy ↔ mấy`, `rước ↔ trước`, `lây ↔ lấy`,
`cậu ↔ câu`. These are not ASR-level failures in the sense that the encoder
got the phoneme stream wrong — it's mostly that the model picked the wrong
diacritic among acoustically-similar candidates. A Vietnamese NLP toolkit
trained on millions of well-formed sentences can fix most of these from
text context alone.

Two libraries; pick one:

| | size | speed | notes |
|---|---|---|---|
| **`underthesea`** | ~150 MB (includes models) | ~50-100 ms / sentence | More polished API, broader NLP toolkit. Recommended. |
| **`pyvi`** | ~30 MB | ~10-30 ms / sentence | Lighter, faster, fewer features. |

### Install

```bash
.venv/bin/pip install underthesea
```

### Wire it into the live demos

Add a tiny helper module that the streaming scripts call before display and
before translation:

```python
# vi_postprocess.py — Vietnamese transcript clean-up applied to every commit.
from underthesea import text_normalize

def clean_vi(text: str) -> str:
    """Restore standard Vietnamese diacritics + punctuation spacing.
    Conservative — only fixes high-confidence cases; leaves model output
    alone when ambiguous."""
    if not text or not text.strip():
        return text
    normalized = text_normalize(text)
    return normalized
```

Patch `commit_utterance` in `stream_translate.py` and `stream_web.py`:

```python
from vi_postprocess import clean_vi

def commit_utterance(raw_text_now, reason):
    ...
    finalized = strip_lang_tags(new_raw).strip()
    if not finalized or not has_real_content(finalized):
        ...
        return
    finalized = clean_vi(finalized)   # <-- new line, before submit_final
    ...
```

That's it. Re-run **Step 01's `eval_wer.py`** after adding `clean_vi` to the
hypothesis-cleanup pass:

```python
hyps = [clean_vi(h) for h in hyps]
```

Typical result: WER drops 20-40%, CER drops 30-50%, on a Vietnamese-heavy
test set.

### When Lane 1 won't help

`underthesea` works on a per-sentence context; it can't fix:
- whole wrong words (`câu nói tử` for `cậu nói thử` — both are valid surface
  forms with different meanings)
- missing words the ASR skipped
- punctuation that the model didn't emit at all

Those need Lane 2 or Lane 3.

## Lane 2 — KenLM rescoring (optional)

If after Lane 1 your WER is still > 10% and the residual errors look like
"the model picked a word that's grammatically wrong in context", an external
n-gram LM helps. NeMo's RNNT decoder supports beam search + LM fusion.

### 1. Install KenLM

```bash
sudo apt install -y build-essential cmake libboost-system-dev \
    libboost-thread-dev libboost-program-options-dev libboost-test-dev \
    libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
.venv/bin/pip install https://github.com/kpu/kenlm/archive/master.zip
```

### 2. Collect Vietnamese text corpus

Aim for ≥ 100 MB of clean Vietnamese text. Sources, easiest first:

| source | size | notes |
|---|---|---|
| **Vietnamese Wikipedia dump** | ~500 MB raw, ~150 MB cleaned | `wget https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2` then run `wikiextractor` |
| **OSCAR Vietnamese subset** | ~10 GB | Web-crawled, noisier |
| **VnExpress / Tuoi Tre news scrape** | as much as you want | Domain-matched if you read news |
| **Your own transcripts** (next step's data) | up to ~10 MB | Best style match, but small |

Concatenate everything into one UTF-8 text file `vi_corpus.txt`, one
sentence per line.

### 3. Train the n-gram LM

```bash
mkdir -p models/lm
lmplz -o 5 --skip_symbols < vi_corpus.txt > models/lm/vi_5gram.arpa
build_binary models/lm/vi_5gram.arpa models/lm/vi_5gram.bin
```

`-o 5` = 5-gram. For 100 MB of training text, this produces a ~300-500 MB
binary LM. Memory-mapped at decode time, so fine on 6 GB GPU.

### 4. Switch the streaming demo to beam search + LM rescoring

```python
# Done once before the streaming loop starts.
from omegaconf import OmegaConf
decoding_cfg = model.cfg.decoding
decoding_cfg.strategy = "alsd"      # alignment-length-synchronous beam decoder
decoding_cfg.beam = OmegaConf.create({
    "beam_size": 4,
    "kenlm_path": str(HERE / "models/lm/vi_5gram.bin"),
    "ngram_lm_alpha": 0.5,           # LM weight; tune in 0.2–1.0
    "beam_alpha": 0.0,
    "beam_beta": 1.5,                # word-insertion bonus
})
model.change_decoding_strategy(decoding_cfg=decoding_cfg)
```

> **Heads-up**: this trades latency for accuracy. Beam=4 RNNT decoding is
> ~2-3× slower per chunk than greedy. On the 560 ms chunk, that's still
> well under real-time on RTX 2060, but each chunk takes ~150 ms decode
> instead of ~50 ms. The streaming demo will *feel* slightly less snappy.

Re-run `eval_wer.py`. Typical gain on top of Lane 1: another 10-20% relative.

### When Lane 2 isn't worth it

- You don't have access to a Vietnamese text corpus matching your domain.
- The latency hit (beam search) isn't acceptable for your use case.
- Errors are mostly acoustic-level, not lexical — Lane 3 is the answer for
  those, not Lane 2.

## Decision after Step 02

Re-measure:

```bash
.venv/bin/python scripts/eval_wer.py
```

Compare against `baseline_wer.txt`:

| current WER | next step |
|---|---|
| **< 5%** | done. Ship Lane 1 + (optionally) Lane 2 and skip fine-tuning. |
| **5-10%** | optional fine-tuning. Decide based on how much the residual errors hurt your use case. |
| **> 10%** | proceed to Lane 3 — Steps 03 → 05. |

→ Next: **[03-data.md](03-data.md)**
