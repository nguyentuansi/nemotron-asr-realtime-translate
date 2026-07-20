# nemotron-asr-realtime-translate

<!--
  demo/demo.gif is currently a programmatic mockup rendered by
  demo/make-mockup.py — it faithfully mirrors stream_translate.py's live UI
  (parallel ASR partial + translator draft, then commit). Replace it with a
  real screen recording before any public launch — see demo/README.md.
-->
![Demo: Vietnamese speech → English translation, live on a MacBook CPU](demo/demo.gif)

**Offline Vietnamese voice AI assistant, built on Nemotron-3.5 ASR.**
Wake with **"Nemo ơi"**, speak a command, get a spoken answer. Everything
runs locally on a MacBook — no accounts, no cloud, no API keys.
The same pipeline also drives real-time streaming speech translation across
**19 source × 200 target languages** when you want that instead.

## Get running in one command

```bash
./nemo.sh assistant           # always-on: "Nemo ơi, ..."   (needs trained wake model)
./nemo.sh ptt                 # push-to-talk: press ENTER each turn (no wake model)
./nemo.sh stream              # streaming ASR + translation UI (no assistant)
./nemo.sh                     # interactive menu of everything
```

## What the assistant can do (v0)

Vietnamese speech in, Vietnamese speech out — with rule-based intent routing
(no LLM in v0, deferred to v2 to keep everything local + fast):

| Skill | Example command | Spoken response |
|---|---|---|
| **Time / date** | "Nemo ơi, mấy giờ rồi?" | "Bây giờ là tám giờ ba mươi phút sáng" |
| **Alarms & timers** | "Nemo ơi, đặt báo thức 6 giờ sáng" | "Đã đặt báo thức lúc sáu giờ sáng" |
| **Translation** | "Nemo ơi, dịch sang tiếng Anh: xin chào bạn" | "Hello, friend" *(14 target languages)* |
| **Home Assistant** | "Nemo ơi, tắt đèn phòng khách" | "Đã tắt đèn phòng khách" |
| **Help / intro** | "Nemo ơi, bạn có thể làm gì?" | *(introduces the four skill families with examples)* |

First-run setup for Home Assistant (URL + long-lived token + Vietnamese entity aliases):
```bash
./nemo.sh setup
```

## Wake-word training is pluggable

The wake-word model ships trained on **synthetic-only data** (2000 Piper-Vi
"Nemo ơi" clips + augmentation + 1000 synthetic Vietnamese negatives).
Fires reliably on the author's voice; generalizes imperfectly to others.

To improve it, drop real recordings into `data/wake/positive_real/` and
retrain — the manifest weights real clips 4× per file so even ~50 shift
the model meaningfully:

```bash
./nemo.sh wake-train all      # ~30-60 min, fully automated, uses your manifest
```

Full design walkthrough:
- **[docs/assistant/00-build-story.md](docs/assistant/00-build-story.md)** — 8-chapter design story covering wake gate, intent router, TTS, skills, main loop, verification
- **[docs/assistant/01-wake-word-training.md](docs/assistant/01-wake-word-training.md)** — training pipeline design + how to add real recordings

## Stack at a glance

| Layer | Choice | Why |
|---|---|---|
| **Wake word** | **openWakeWord** ONNX classifier (custom-trained "Nemo ơi") | ~5% CPU idle 24/7; auto-download of the trained model + pluggable retraining from `data/wake/manifest.yaml` |
| **Streaming ASR** | **Nemotron-3.5-asr-streaming-0.6b** (NVIDIA, NeMo) | **19 production languages**, cache-aware streaming Conformer with prompt conditioning per language |
| **ASR runtime** | **ONNX Runtime (CPU EP)** | Custom export with cache I/O → 8× faster than PyTorch on CPU (RTF 1.6 → 0.20) |
| **Intent routing** | Rule-based regex matcher (`intent_router.py`) | Deterministic; `_llm_fallback` seam declared for v2 |
| **Skills** | Plain Python `handle(slots) -> str` modules | Trivial to extend — see `skills/*.py` for the four v0 examples |
| **Text-to-speech** | **Piper** with `vi_VN-vais1000-medium` voice | Neural TTS, ~60 MB, ~200 ms first-audio latency; English-loanword spelling workaround built in |
| **Translation (general)** | **NLLB-200-distilled-600M** (Meta) via **CTranslate2 int8** | **200 languages** — powers the translate skill + the streaming translation UI |
| **Translation (Vi↔En specialist)** | **EnViT5** (VietAI) via CTranslate2 int8 | SOTA on PhoMT/MTet, OpenRAIL (commercial OK), 3× smaller than NLLB — default for `--lang vi-VN` |
| **Noise suppression** | **GTCRN** via sherpa-onnx (opt-in `--denoise`) | ~40 ms/chunk; feeds the silence VAD without touching ASR audio |
| **Web UI** | **FastAPI** + **uvicorn[standard]** + WebSocket | Multi-client browser display of the streaming pipeline |
| **Audio capture** | **alsaaudio** (Linux) / **sounddevice** shim (macOS, Windows) | Native ALSA where available, cross-platform fallback elsewhere |
| **Runtime** | **Python 3.13** + **PyTorch 2.12** + **NeMo `@main`** | NeMo pinned to git main for the Nemotron streaming class |
| **First-run setup** | `_bootstrap.sh` auto-creates `.venv`, installs from `requirements.txt` | Just `./nemo.sh` and it boots |
| **Project license** | **MIT** | See [LICENSE](LICENSE); each model has its own license, all commercial-friendly |

End-to-end assistant flow: mic → WakeGate (openWakeWord ONNX) → on wake, hand pre-roll + subsequent chunks to CacheAwareStreamingAudioBuffer → ONNX-routed Conformer encoder → RNNT decoder → silence-VAD commit → IntentRouter → skill.handle() → Piper TTS → speaker. ~1-2 s wake-to-first-audio on M-series CPU.

## Also: real-time streaming translation

The same pipeline powers a standalone streaming ASR + translation UI without
the assistant / wake word — useful for live captioning, meetings, or piping
translations onto a screen:

```bash
./nemo.sh stream                                             # vi → en, terminal
./nemo.sh web                                                 # same, browser UI at :8765
./nemo.sh stream --lang es-ES --target-lang en-US --translator nllb
./nemo.sh stream --lang en-US --target-lang ja-JP --translator nllb
./nemo.sh stream --lang hi-IN --target-lang arb_Arab --translator nllb
```

`--translator nllb` switches to NLLB-200 for non-Vi↔En pairs. Any of Nemotron's
19 source languages × NLLB's 200 target languages works out of the box.
Vi↔En stays the well-tuned foundation: EnViT5 beats NLLB on PhoMT/MTet, the
silence/commit tuning is calibrated for tone-language speech.

## Why this exists

In 2026 every mainstream voice assistant (Google, Siri, Alexa) still treats
Vietnamese as a second-class citizen — cloud-only, tone-mark handling that's
mediocre at best, forced account signup, and audio leaving your household.

This repo is the open, offline, Vietnamese-first alternative. Same architecture
Google Assistant uses (wake word → ASR → intent → skill → TTS) but every
component runs on your laptop. If you speak Vietnamese and want a voice
assistant that actually understands you, without giving Google your kitchen
conversations, this is for you.

## One entry point for everything

Every mode of the project runs through **`./nemo.sh`**:

```bash
./nemo.sh                         # interactive menu
./nemo.sh --help                  # list every command
./nemo.sh stream                  # ASR + translation, terminal UI
./nemo.sh web                     # same, browser at :8765
./nemo.sh assistant               # voice assistant ("Nemo ơi")
./nemo.sh setup                   # first-run wizard for the assistant
./nemo.sh mic-test 5              # 5-second mic → ASR round-trip
./nemo.sh bench rtf               # ASR real-time-factor benchmark
./nemo.sh bench wake              # wake-word FAR/FRR benchmark
./nemo.sh smoke                   # smoke test on bundled audio (no mic)
./nemo.sh regress                 # end-to-end assistant regression
./nemo.sh wake-train prepare      # generate synthetic wake-word data
```

Any flags after the subcommand pass through to the underlying script, e.g.
`./nemo.sh stream --translator nllb --lang vi-VN` or
`./nemo.sh assistant --no-tts`. The individual wrappers (`stream_translate.sh`,
`assistant.sh`, …) keep working standalone; `nemo.sh` just dispatches.

## Quick start

```bash
git clone https://github.com/nguyentuansi/nemotron-asr-realtime-translate.git
cd nemotron-asr-realtime-translate
./nemo.sh
```

That's it. First run does ~10 minutes of one-time setup:

1. Auto-creates a Python 3.13 venv (`_bootstrap.sh` finds the best Python in your `PATH`)
2. Installs torch, NeMo (from `@main` — Nemotron-3.5 isn't on PyPI NeMo yet), ctranslate2, fastapi, sherpa-onnx
3. Downloads the Nemotron ASR model (~2.4 GB) into `.hf-cache/`
4. Loads the Vietnamese ↔ English translator (run the conversion once with `ct2-transformers-converter` — instructions appear if the directory is missing)

Subsequent runs start in ~30 seconds (model load time).

### Run with noise suppression

If you're in a noisy room (fan, AC, traffic), the silence-VAD has trouble detecting when you've actually paused:

```bash
./nemo.sh stream --denoise         # or ./nemo.sh assistant --denoise
```

This uses GTCRN to give the VAD a clean reference signal so pauses commit reliably. The ASR still sees raw audio so Vietnamese tones aren't damaged. GTCRN auto-downloads on first use (~520 KB).

### Run the web UI (streaming-translation mode)

```bash
./nemo.sh web
# open http://127.0.0.1:8765 in a browser
```

Same pipeline as `./nemo.sh stream`, different display. Useful for screen-sharing, captioning meetings, or running ASR on a beefy machine and viewing from a phone on the same network. The assistant does not yet have a web UI — that's on the v1 list.

## Features

Assistant mode:
- **Vietnamese wake word** — custom-trained openWakeWord ("Nemo ơi"), ~5% idle CPU, pluggable retraining from `data/wake/manifest.yaml`
- **Four v0 skills** — time/date, alarms & timers, translation (14 target languages), Home Assistant (lights/switches/scenes)
- **Rule-based intent routing** — deterministic regex matcher with named-slot capture; `_llm_fallback` seam declared for v2
- **Piper TTS in Vietnamese** — `vi_VN-vais1000-medium`, neural voice, ~200 ms first-audio latency; English-loanword spelling workaround built in
- **Push-to-talk fallback** — `./nemo.sh ptt` needs no wake model, works on a fresh clone
- **First-run setup wizard** — `./nemo.sh setup` writes `~/.config/nemo-assistant.yaml` (HA URL, long-lived token, VN aliases)

Streaming-translation mode (foundation the assistant is built on):
- **Wide language coverage** — 19 source ASR languages × 200 NLLB target languages = thousands of pairs, all swappable with two CLI flags
- **Bidirectional Vi ↔ En specialist** built in (~275 M params int8 EnViT5) as the polished foundation example
- **Streaming, not batched** — partial transcripts appear chunk-by-chunk (560 ms cadence), commits land on natural sentence boundaries
- **CPU-real-time** on Apple Silicon via ONNX Runtime (encoder exported with cache-state I/O, ~8× speedup over PyTorch)
- **Drop-in CUDA support** — if `torch.cuda.is_available()`, the ASR runs on GPU automatically
- **Cross-platform mic capture** — native ALSA on Linux, `sounddevice` shim everywhere else (`alsa_shim.py`)
- **Session recording + structured logs** — every run writes `logs/audio-<ts>.wav` (raw mic) + `logs/stream-<ts>.log` (per-chunk debug) for offline replay

## Architecture

```
   mic (16 kHz mono S16)
      │
      ▼
  ┌─────────────────────────────────────────────┐
  │ MicProducer (background thread)              │
  │   - alsa_shim or alsaaudio capture           │
  │   - write RAW to logs/audio-<ts>.wav         │
  │   - optional: GTCRN denoise -> VAD peak only │
  │   - bounded ring buffer (--max-buffer-secs)  │
  └─────────────────────────────────────────────┘
      │ float32 PCM, 16 kHz mono
      ▼
  ┌─────────────────────────────────────────────┐
  │ Streaming ASR loop (main thread)             │
  │   - 560 ms chunks                            │
  │   - CacheAwareStreamingAudioBuffer (NeMo)    │
  │   - encoder.forward via ONNX Runtime         │
  │   - RNNT decoder in PyTorch (stateful LSTM)  │
  │   - cache_last_channel / time / len threaded │
  │     through every chunk                      │
  └─────────────────────────────────────────────┘
      │ partial + final hypotheses
      ▼
  ┌─────────────────────────────────────────────┐
  │ Commit logic                                 │
  │   - silence VAD (peak<threshold for Ns)      │
  │   - end-of-utterance <lang> tag              │
  │   - sentence-final punctuation               │
  └─────────────────────────────────────────────┘
      │ committed source-language text
      ▼
  ┌─────────────────────────────────────────────┐
  │ Translator worker (background thread)        │
  │   - CTranslate2 int8 (EnViT5 or NLLB)        │
  │   - FINAL queue (committed text) drained 1st │
  │   - DRAFT slot (latest partial) for live UX  │
  └─────────────────────────────────────────────┘
      │ English translation
      ▼
   terminal display  |  WebSocket -> browser
```

## Performance

Measured on M-series MacBook Pro, CPU only, default `560 ms` chunks:

| Stage | PyTorch CPU | ONNX CPU (default) |
|---|---:|---:|
| ASR avg chunk time | 921 ms | **114 ms** |
| ASR RTF | 1.64 | **0.20** |
| Translator (EnViT5 int8) | 80-200 ms / utterance | same |
| Denoiser (GTCRN, optional) | — | +30-40 ms / chunk |
| End-to-end RTF | 1.6+ (lagging) | ~0.25 (real-time) |

See [`docs/performance/`](docs/performance/) for the full ONNX export + integration recipe if you want to reproduce or extend the optimization.

## Common flags (streaming-translation mode)

```bash
# Default — Vi -> En with EnViT5
./nemo.sh stream

# Different language pair
./nemo.sh stream --lang en-US --target-lang vi-VN
./nemo.sh stream --lang ja-JP --target-lang en-US --translator nllb

# Transcription-only (no translation)
./nemo.sh stream --no-translate

# Noisy environment
./nemo.sh stream --denoise

# Tight latency (faster commits, more fragmentation)
./nemo.sh stream --silence-secs 0.6 --max-utterance-secs 4

# Dictation mode (longer pauses tolerated, fewer commits)
./nemo.sh stream --silence-secs 3.0

# Bypass the ONNX encoder (slower; use if ONNX crashes on your CPU)
NO_ONNX=1 ./nemo.sh stream

# Quiet down NeMo's import-time chatter (default), or restore it
NEMO_VERBOSE=1 ./nemo.sh stream

# Save audio + log to specific paths
./nemo.sh stream --record-audio my-session.wav --log-file my-session.log
```

All flags discoverable via `./nemo.sh stream --help`. Assistant-mode flags
(`--no-wake`, `--wake-model`, `--log-file`, `--no-tts`) via `./nemo.sh assistant --help`.

## Project layout

```
nemotron-asr-realtime-translate/
├── nemo.sh               umbrella launcher — dispatches to every mode + interactive menu
│
│   # Assistant mode
├── assistant.py          main event loop: wake → ASR → intent → skill → TTS
├── assistant.sh          wrapper for assistant.py (routes --setup to wizard)
├── assistant_setup.py    first-run wizard → ~/.config/nemo-assistant.yaml
├── wake_gate.py          openWakeWord ONNX classifier + pre-roll ring buffer
├── intent_router.py      regex-based intent matcher, LLM-fallback seam for v2
├── tts_speaker.py        Piper Vi TTS + English-loanword spelling workaround
├── skills/               plug-in skill modules
│   ├── time_skill.py
│   ├── alarm_skill.py
│   ├── translate_skill.py
│   ├── home_assistant_skill.py
│   └── help_skill.py
├── data/wake/manifest.yaml   pluggable wake-word training data sources
├── scripts/wake_pipeline.py  synth + train + ONNX-export the wake model
│
│   # Streaming-translation mode
├── stream_translate.sh   wrapper: ./stream_translate.py via _bootstrap.sh
├── stream_translate.py   terminal UI, streaming ASR + translation
├── stream_web.sh         wrapper: ./stream_web.py
├── stream_web.py         FastAPI/WebSocket UI, same pipeline
│
│   # Shared foundation
├── translator.py         NLLBTranslator + EnViT5Translator + factory
├── onnx_encoder.py       ONNX Runtime wrapper for the Conformer encoder
├── denoiser.py           sherpa-onnx GTCRN wrapper (optional --denoise)
├── alsa_shim.py          sounddevice fallback for non-Linux platforms
├── _bootstrap.sh         sourced by every wrapper; ensures .venv exists
├── requirements.txt      pinned deps incl. NeMo @ git main
│
├── audio/                bundled LibriSpeech samples + your bench wavs
├── bench/                RTF / WER + wake FAR/FRR measurement scripts
├── demo/                 recording protocol + Vi script + mp4→gif helper +
│                           demo/simulate_assistant.py end-to-end regress
├── docs/
│   ├── assistant/        8-chapter assistant build story + wake-word training
│   ├── training/         Vietnamese fine-tuning workflow (5 chapters)
│   └── performance/      ONNX + INT8 + CoreML perf workflow (5 chapters)
├── logs/                 per-session debug logs + raw audio recordings
└── models/               ONNX encoder + GTCRN + wake model (auto-generated)
```

## Documentation

- [`docs/assistant/`](docs/assistant/) — Assistant build story (why + how), wake-word training pipeline, skill authoring.
- [`docs/training/`](docs/training/) — Improving Vietnamese ASR accuracy: baseline measurement, post-processing diacritic restoration, KenLM rescoring, LoRA fine-tuning, deploy.
- [`docs/performance/`](docs/performance/) — CPU performance: ONNX encoder export with cache I/O, FP16/INT8 trade-offs, integration, benchmark, fallback to smaller models.

## License

This project: **MIT** ([LICENSE](LICENSE)).

The models you load through it have their own licenses — all are commercially usable, with one caveat:

| Component | License | Commercial use? |
|---|---|---|
| This code | MIT | Yes |
| Nemotron-3.5 ASR | NVIDIA Open Model License | Yes (read the NVIDIA OML terms) |
| EnViT5 (default translator) | OpenRAIL-M | Yes (with use-case restrictions — no surveillance, harassment, etc.) |
| NLLB-200 (alternate translator) | MIT | Yes |
| GTCRN denoiser | MIT (via [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn)) | Yes |
| sherpa-onnx runtime | Apache 2.0 | Yes |
| NeMo toolkit | Apache 2.0 | Yes |

If you swap the translator for **vinai-translate-vi2en-v2** instead of the default EnViT5: it is **AGPL-3.0**. Any network-accessible service calling it must release its server source — typically a deal-breaker for SaaS. Use EnViT5 (the default) for commercial work.

## Credits

- **NVIDIA NeMo team** — Nemotron-3.5 streaming ASR model + the streaming framework this is built on
- **Meta AI** — NLLB-200 translation model
- **VietAI** — EnViT5 Vietnamese translator and the MTet corpus that trained it
- **VinAI** — PhoMT corpus and reference work that made Vietnamese ASR/NMT measurable
- **k2-fsa / Xiaobin Rong** — sherpa-onnx + GTCRN noise suppression
- **The Vietnamese NLP community** — for keeping low-resource speech research alive

## Contributing

Bug reports, language-pair extensions, and benchmark numbers from your machine are welcome. The codebase is small (~3 kloc) and deliberately approachable — every file is < 800 lines and the wrappers are 5 lines each.

If you ship a quality improvement (better translator, smaller/faster ASR, real-time GPU benchmarks), please open a PR with before/after `bench/rtf_*.json` and `bench/wer_*.json` files so we can compare apples to apples.
