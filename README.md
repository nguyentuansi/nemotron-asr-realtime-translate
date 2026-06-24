# nemotron-asr

Local sandbox for [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) — multilingual cache-aware streaming ASR (40 languages, 19 production-ready including Vietnamese).

## Layout

```
nemotron-asr/
├── .venv/          Python 3.11 venv (torch+cu124, nemo_toolkit[asr]) — ~6 GB
├── .hf-cache/      Hugging Face model cache (~1-2 GB after first run)
├── audio/          Test clips (LibriSpeech samples from the model card)
├── model_readme.md Pinned copy of the model card
├── smoke_test.py   Load model + transcribe sample1.flac, sample2.flac
└── run.sh          Wrapper: activate venv, set HF_HOME, run smoke_test.py
```

Everything project-local. `~/.cache/pip` may hold extra wheels but isn't required for runtime.

## Usage

```bash
./run.sh
```

Or manually:

```bash
source .venv/bin/activate
export HF_HOME="$PWD/.hf-cache"
python smoke_test.py
```

## Streaming inference

For the full cache-aware streaming pipeline, see NeMo's reference script:
`examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`
in the [NeMo repo](https://github.com/NVIDIA/NeMo).
