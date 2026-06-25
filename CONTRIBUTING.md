# Contributing

Bug reports, language-pair extensions, performance work, and benchmark numbers
from your machine are all welcome.

## Scope

The codebase is small (~3 kloc Python) and deliberately approachable. Every
file is under 800 lines, the wrappers are 5 lines each, and there's no plugin
system to learn. Read `stream_translate.py` end-to-end before opening a
non-trivial PR — it's the spine of the project.

## Setup

```bash
git clone https://github.com/nguyentuansi/nemotron-asr-realtime-translate.git
cd nemotron-asr-realtime-translate
./stream_translate.sh         # first run bootstraps the venv (~10 min)
```

After that, work in the existing `.venv/`:

```bash
.venv/bin/python smoke_test.py
.venv/bin/python -m pytest    # if you add tests
```

## What we accept

**Yes, please:**

- Quality improvements — better translator, smaller/faster ASR, real-time GPU
  benchmarks — with before/after `bench/rtf_*.json` and `bench/wer_*.json` so
  reviewers can compare apples to apples
- New specialist translators wired through `make_translator` in
  `translator.py` (the EnViT5 path is the reference pattern)
- Performance numbers from your hardware in `docs/performance/`
- Documentation fixes, typos, broken-link reports
- Bug reports with reproducible audio (a short `.wav` and the command line is
  enough)

**Probably no:**

- Wholesale rewrites that change the runtime stack (PyTorch → JAX, ONNX →
  CoreML-only, etc.). Open an issue first to discuss
- New dependencies without a measurable payoff in the bench numbers
- Cloud-API integrations — this project's identity is on-device, on-laptop,
  no API keys

## Style

- Match the existing code. No new abstractions unless they earn their keep
- Keep `requirements.txt` minimal and pinned
- No comments explaining what well-named code already says; add a one-line
  comment only when the *why* would surprise a reader

## Benchmarks

When you claim a speedup or accuracy improvement, attach the numbers:

```bash
.venv/bin/python bench/measure_rtf_onnx.py --out bench/rtf_your-change.json
```

Put the before/after JSON files in your PR. "Feels faster" doesn't merge.

## License

By contributing you agree your work is released under this project's MIT
license. Each model the project loads keeps its own license — don't add a
component whose terms are incompatible with MIT distribution (e.g. AGPL,
no-commercial).
