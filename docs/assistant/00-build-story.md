# Building "Nemo ơi" — a design-taste walkthrough

This document is the running log of building a Vietnamese-first voice assistant on top of `nemotron-asr-realtime-translate`. It's written to be read front-to-back, one chapter at a time, and to teach you **why we made each choice** — not just what we built.

Each chapter follows the same shape:
- **Goal** — the problem being solved
- **Concepts** — background ideas you need (short — the code teaches the rest)
- **Design decisions & alternatives** — what we picked, what we rejected, why. This is where the taste lives.
- **Verify** — smallest test that proves the thing works
- **Pitfalls** — real failure modes to avoid

You are assumed to be comfortable with Python and new to ML / audio / streaming systems. When speech/audio concepts come up for the first time, we introduce them briefly.

---

## Chapter 1 — What we're building, and what we're building ON

### Goal

Take the existing `nemotron-asr-realtime-translate` pipeline (a real-time speech-to-text-plus-translation system for Vietnamese) and turn it into a **voice assistant**: something you can leave on your Mac all day, wake with a phrase ("Nemo ơi"), speak a command to ("mấy giờ rồi?" / "what time is it?"), and get a spoken answer back in Vietnamese.

Think Google Assistant / Siri / Alexa — but:
- Vietnamese as the first-class language, not an afterthought
- Runs entirely on your laptop; no cloud, no accounts, no API keys
- MIT-licensed, hackable

### Concepts

Before we start, a few terms that will show up everywhere:

- **ASR** — Automatic Speech Recognition. The thing that turns your voice into text. Ours is NVIDIA's Nemotron-3.5, a **streaming** model — meaning it produces partial text as you speak, not just at the end.
- **NMT** — Neural Machine Translation. Turns text in one language into text in another. Ours is a mix of EnViT5 (Vietnamese specialist) and NLLB-200 (200 languages).
- **Wake word** — a short phrase that "wakes" the assistant. Same idea as "Hey Siri" or "Alexa". The wake-word detector runs 24/7 while the (much heavier) ASR only runs after the wake fires.
- **KWS** — Keyword Spotting. The technical name for wake-word detection.
- **VAD** — Voice Activity Detection. A cheap check for "is anyone speaking right now?", used for both wake-word gating and knowing when a command ended.
- **Intent** — the interpreted meaning of a command. "mấy giờ rồi?" and "hỏi giờ" both have intent `ask_time`.
- **Skill** — the code that handles one intent. A "time skill" answers time questions; an "alarm skill" sets alarms.
- **TTS** — Text-to-Speech. Turns the assistant's response text into audible speech. Ours is Piper with a Vietnamese voice.

### Design decisions & alternatives

**Decision 1: Build on the existing pipeline, don't restart.**

The existing repo already does the hardest part (running a good Vietnamese ASR at real-time speed on CPU). Rebuilding that would be months of work. Instead, we add three new pieces:
- A **wake gate** in front of the ASR (so it only runs when we want it)
- An **intent router** after the ASR (turns text into a skill call)
- A **TTS speaker** at the end (turns response text into voice)

Everything else — the mic capture, the streaming buffer, the ONNX-accelerated encoder, the denoiser — is reused verbatim.

*Alternative rejected*: build a separate assistant repo that shells out to the streaming pipeline via subprocess or WebSocket. Simpler to reason about, but ~3× the code, no shared config, and every performance improvement we make in one place has to be manually mirrored in the other. Not worth it.

**Decision 2: Wake word runs alongside ASR, not before it.**

There are two places you could put a wake gate:
- **Option A**: inside `MicProducer` itself — the mic thread only pushes audio into the buffer once the wake fires. Nothing else runs otherwise.
- **Option B**: after `MicProducer`, before the streaming buffer — the mic keeps recording, but the ASR only runs after wake.

We picked **B**. Why? The wake-word engine needs continuous audio to detect the wake phrase. If the mic itself stops between commands (Option A), the wake gate has nothing to score. Option B keeps a small ring buffer of recent audio always available for wake scoring, but pauses the expensive ASR forward pass until the wake fires. That's the classic pattern (Google Assistant, Alexa, etc. all work this way).

**Decision 3: Wake phrase = "Nemo ơi", not "Ber ơi".**

"Nemo ơi" starts with an `N` sound, which is rare mid-sentence in Vietnamese. That's phonetically valuable — the wake-word model has fewer things to confuse it with. It also has a bonus: "Nemo" is a short form of "Nemotron", the ASR we're built on. On-brand.

*Alternative rejected*: "Ber ơi". Fun, but "Ber" collides with common Vietnamese household phrases like "Bé ơi" (hey little one), "Bố ơi" (hey Dad). That would double the training data we need to keep false-positives down.

**Decision 4: No LLM in v0. Rule-based intent routing only.**

The four MVP skills (time, translate, alarms, Home Assistant) all match cleanly to Vietnamese command patterns — you can describe them with regex. Adding an LLM would give us "open-ended chat" but it's:
- Slow on CPU (even Qwen 1.5B takes seconds per response)
- Prone to unexpected outputs
- More code to maintain

v0 ships with rule matching. The IntentRouter has a declared `_llm_fallback` seam that stays unimplemented until v2 — that way the interface doesn't change when the LLM does show up.

*Alternative rejected*: launch with a cloud LLM fallback (Anthropic, OpenAI) for anything the rules miss. Kills the privacy pitch. Full stop.

**Decision 5: Deliver the four skills in this order — time, translate, alarms, Home Assistant.**

Time and translate are near-free — time is pure Python, translate reuses the existing translator. We ship them fast to prove the pipeline. Alarms need a scheduler (APScheduler). Home Assistant is the hardest — auth, entity discovery, bilingual name mapping — so it comes last. If we get to week 5 and HA is still shaky, we cut it and ship v0 with 3 skills; v1 picks it up.

### Verify

You can prove Chapter 1's setup works before writing any new code — the existing `./stream_translate.sh` already streams Vietnamese ASR + translation. Everything after this chapter adds *on top of* that.

### Pitfalls

- **Assuming "wake word" is easy**. It's not. Good wake-word models take real training data and threshold tuning. See Chapter 2.
- **Assuming Piper Vi TTS is production-ready for mixed language.** Piper Vi mispronounces English loanwords ("CPU" → "sai cấu"). We have to work around it. See Chapter 4.

---

## Chapter 2 — WakeGate: the always-on ear

### Goal

Build the component that continuously listens to microphone audio, cheaply, and fires an event **only** when the user says "Nemo ơi". The rest of the assistant is dormant until this fires.

Key requirements:
- **Cheap**: <5% CPU on an M-series Mac. It runs 24/7 while the assistant is idle.
- **Fast**: from "user finishes saying wake phrase" to "event fires" should be <200 ms.
- **Correct enough**: fewer than 2 false accepts per idle hour; fewer than 10% missed wakes.
- **Provides context**: when the ASR takes over, it needs ~1 second of audio *before* the wake fired, because the user's first command word often overlaps with the wake phrase.

### Concepts

- **Frame / chunk**: a small slice of audio. Our mic hands us 1024 samples at 16 kHz = **64 ms of audio per chunk**. That's the cadence everything downstream sees.
- **Wake-word model**: a small neural network trained to output a **confidence score** in [0, 1] indicating "the wake phrase was just spoken". A score of 0.55 typically means "quite likely"; 0.9 means "very sure".
- **openWakeWord**: the open-source library we use. It runs its models via ONNX (a portable model format we already use for the ASR encoder). Small models — ours will be ~1-3 MB.
- **Pre-roll buffer**: a rolling window of the last N seconds of audio, kept in memory. When the wake fires, we hand this window to the ASR so it sees context, not just the audio starting *after* the wake was detected.
- **Cooldown**: a "just fired, don't fire again for N seconds" window. Without this, a single long utterance ("Nemo ơi Nemo ơi bật đèn") could fire twice.

### Design decisions & alternatives

**Decision 1: openWakeWord, not Porcupine or a custom keyword-spotter.**

Options in 2026:
- **openWakeWord** (Rhasspy team) — free, MIT-adjacent license, custom-trainable, ~1-3 MB ONNX model, actively maintained. **Picked.**
- **Porcupine** (Picovoice) — best accuracy on paper, but their free tier is non-commercial. Our repo is MIT and we want commercial use. Skip.
- **microWakeWord** (Home Assistant Voice) — 40 KB models, designed for ESP32 microcontrollers. Overkill-minimalist for a Mac. Save for v3 hardware.
- **Custom KWS via Nemotron itself** — bad idea. Nemotron ASR uses 100+ ms per chunk on CPU. Running it 24/7 for wake detection would burn CPU and battery. Wake models are designed to run in ~5% of the CPU budget an ASR forward pass uses.

**Decision 2: Cool-down is a wall-clock timer, not a score-hysteresis mechanism.**

After a wake fires, ignore all further scores for `cooldown_s` seconds (default 1.5). This is simpler than the alternative — hysteresis (require the score to drop below a low threshold before it can rise above the fire threshold again). Wall-clock cooldown handles the common case ("Nemo ơi, then a command, then silence") and is easy to reason about.

*Alternative rejected*: no cooldown at all. Users saying "Nemo ơi" and then immediately correcting themselves ("Nemo ơi... uh... Nemo ơi, tắt đèn") would trigger twice, confusing the state machine downstream.

**Decision 3: Pre-roll buffer stored as raw float32, not compressed.**

1 second of 16 kHz mono float32 audio is ~64 KB. Trivial. We keep it as a numpy array with head/tail pointers. No compression, no framing complexity — just a ring buffer.

*Alternative rejected*: only keep the last 3 chunks (~192 ms) because "that should be enough". Not enough: fast speakers say "Nemo ơi, tắt đèn" as one continuous 800-ms phrase; the wake fires halfway through and "tắt đèn" is already in progress. A 1-second buffer catches it comfortably.

**Decision 4: `process(chunk)` returns `WakeEvent | None`, not a callback / async.**

The caller (`assistant.py`) drives the loop. `WakeGate` doesn't spawn threads, doesn't fire callbacks — it just consumes one chunk at a time and either returns None or a `WakeEvent`. This keeps testing dead simple: unit tests feed pre-recorded chunks and assert on the return value. No async fixtures, no thread joins.

*Alternative rejected*: `WakeGate.on_wake(callback)` pattern. Cleaner-looking but harder to test; the callback fires from wherever the audio thread is, which invites subtle races.

**Decision 5: Model auto-downloads on first construction. Mirrors the GTCRN pattern.**

If `models/wake/nemo_oi.onnx` isn't on disk, `WakeGate.__init__` downloads it from a GitHub release. Same pattern `denoiser.py` uses for GTCRN. Users who clone the repo don't need a separate "download models" step — the first `./assistant.sh` handles it.

*Alternative rejected*: bundle the model in the git repo. It's binary and would bloat clones. GitHub releases are the right place.

### Verify

Two levels of verification:

1. **Interface** — instantiate, feed 1 second of silence, assert no event. Feed a wav of someone saying "Nemo ơi", assert event fires with score > threshold. This is what `bench/wake_far_frr.py` (Task #13) automates.
2. **Latency** — from "wake wav finishes" to "event returned" should be <100 ms on a warm session. Measure with a stopwatch around the call.

### Pitfalls

- **Feeding non-16kHz audio**. openWakeWord expects 16 kHz mono float32. If your mic delivers 48 kHz, you must resample first (MicProducer already does this because we set `rate=16000`).
- **Wake threshold too low**. A threshold of 0.3 will fire on anything vaguely wake-shaped. 0.55 is a safer default; tune per-user in v1.
- **Not clearing pre-roll after ASR consumes it**. The next command's pre-roll would still contain audio from the *previous* command. `WakeGate.reset()` handles this; the caller must remember to call it.
- **Model download failing silently**. First-run behavior should be explicit: if the download fails, raise a clear error, not swallow it.

---

## Chapter 3 — IntentRouter: from Vietnamese text to a skill

### Goal

Take the Vietnamese text that came out of the ASR ("Nemo ơi, mấy giờ rồi?") and figure out which skill should handle it (`time_skill`), extracting any slots (a time skill has no slots; an alarm skill has a time slot).

Key requirements:
- **Deterministic**: same input → same output. No LLM randomness in v0.
- **Extensible**: adding a new skill should be one function call.
- **Fast**: sub-millisecond routing. It runs after ASR, so latency here directly adds to user-perceived response time.

### Concepts

- **Regex named groups**: `re.match(r"báo thức (?P<time>\d+ giờ)", text)` captures the number and its "giờ" (hour) unit into a slot dict `{"time": "6 giờ"}`. Almost all our Vietnamese command patterns are that simple.
- **Slot**: a named piece of data extracted from the command. `set_alarm` has a `time` slot. `translate` has `source_lang`, `target_lang`, `body` slots.
- **Priority / insertion order**: when two patterns could match ("set an alarm for 6" could match both `alarm_skill` and `time_skill` if we wrote them badly), we need a tiebreaker. v0 uses registration order — earlier wins. Simple, predictable.

### Design decisions & alternatives

**Decision 1: Strip the wake phrase before matching.**

Skill patterns shouldn't have to know about "Nemo ơi" — that's the router's job. `strip_wake_phrase("Nemo ơi, mấy giờ rồi?")` returns `"mấy giờ rồi?"`. Handles common ASR mishearings (`"nêmô ơi"`, `"ne mo ơi"`) via a permissive regex.

*Alternative rejected*: make every skill pattern start with `(nemo ơi)?`. Duplication, easy to forget.

**Decision 2: Rule-first, LLM never — in v0.**

The four MVP skills all match cleanly to Vietnamese command shapes:
- Time: "mấy giờ", "hôm nay là thứ mấy"
- Translate: "dịch sang tiếng ...:"
- Alarm: "đặt báo thức ...", "hẹn giờ ..."
- HA: "bật/tắt <entity>", "kích hoạt cảnh <name>"

An LLM would give us "open-ended chat" but at multi-second latency and unpredictable output. v0 skips it entirely; v2 revisits.

The `_llm_fallback(text)` method is defined but raises `NotImplementedError`. It exists so that when v2 lands, the router's public interface doesn't change — only that one method's body gets filled in. This kind of "reserved seam" is a nice technique for keeping API stability across roadmap phases.

**Decision 3: Return `IntentResult(skill_name=None, ...)` on no match, don't raise.**

The caller in `assistant.py` will do:
```python
result = router.route(text)
if result.skill_name:
    response = result.handler(result.slots)
else:
    response = "Xin lỗi, Nemo chưa hiểu câu đó"
tts.speak(response)
```

Nice linear flow. If we raised on unknown intent, the caller would need a `try/except` and forget to speak the "sorry" response half the time.

*Alternative rejected*: `Optional[IntentResult]` where `None` means unknown. Same idea but less obvious to the caller — "does None mean the router hasn't run yet, or that it ran and didn't match?" Using an explicit `IntentResult(skill_name=None)` makes the intent clear at the call site.

**Decision 4: `handler` is a plain callable, not a class with a `handle()` method.**

Skills register a function reference: `router.register_skill("time", pattern, time_skill.handle)`. The router doesn't care that `handle` is a module-level function, a bound method, or a lambda — it just calls it with the slots.

*Alternative rejected*: force skills to subclass a `Skill` base class. More Java-ish, more ceremony, no benefit at this scale.

### Verify

Unit tests are trivial:
```python
router = IntentRouter()
router.register_skill("time", r"^mấy giờ", lambda slots: "8 giờ")
r = router.route("Nemo ơi, mấy giờ rồi?")
assert r.skill_name == "time"
assert r.handler({}) == "8 giờ"

r = router.route("gà vịt chó mèo")
assert r.skill_name is None
```

### Pitfalls

- **Regex greediness bugs**. `r"báo thức (.+)"` is too permissive — it eats trailing punctuation, spaces, and any following commands. Use specific patterns (`r"báo thức (?P<time>\d+ giờ( sáng| chiều| tối)?)"`) or non-greedy quantifiers.
- **Vietnamese diacritics**. The ASR sometimes outputs "bao thuc" instead of "báo thức". Patterns should account for this — use `unicodedata.normalize('NFKD', text)` in the router if it becomes a pain point in real testing.
- **Insertion order surprises**. If you register `general_pattern` before `specific_pattern`, general will always win. Register specific first.

---

## Chapter 4 — TTSSpeaker

### Goal
Turn response text into audible Vietnamese speech.

### Design decisions

**Decision 1: Piper, not XTTS-v2 or eSpeak.**

XTTS-v2 sounds more natural but the model is ~1.9 GB and takes 5+ seconds to synthesize a short response on CPU. eSpeak is instant but robotic. Piper (vi_VN-vais1000-medium) is 60 MB, sounds decent, and synthesizes a short response in ~200 ms on M-series CPU. Good enough for v0. `english_workaround` config leaves a seam for the hifi upgrade in v1.

**Decision 2: `speak()` blocks by default, non-blocking via a background thread.**

Most callers want linear flow: "speak this, then continue". Defaulting to blocking mirrors that. Non-blocking is available for the alarm case (alarms fire from APScheduler's thread; they should return immediately). If you call `speak(text)` while a previous non-blocking speak is still playing, we cancel it first — same "user interrupted, honor the new intent" pattern as `stop()`.

**Decision 3: `stop()` cuts sounddevice's playback immediately AND clears the current thread.**

Barge-in needs to feel instant. If we waited for the buffer to drain, users would hear "the assistant is still speaking" for hundreds of ms after they said "dừng lại". Instead we poll `_stop_event` every 50 ms during playback and call `sd.stop()` the moment it's set.

**Decision 4: English-loanword workaround = character-spelling, not eSpeak-splicing.**

Piper mispronounces "CPU" as "sai cấu" (one attempted-Vietnamese word). Options:
- **Spell (chose)**: rewrite "CPU" → "xê pê u" (Vietnamese letter names) before feeding to Piper. Character-by-character. Fast, deterministic, ugly. But intelligible.
- **Splice**: detect English tokens, synthesize them with eSpeak-NG, splice the wav samples into Piper's output. Better quality, but eSpeak's Vietnamese context sounds bad and the splicing adds glitches at boundaries.

We ship spelling in v0. eSpeak-splicing is a v1 upgrade behind `english_workaround="splice"`. Also handles camelCase splitting ("MacBook" → "Mac Book") so both halves are individually pronounceable.

**Decision 5: Playback collects all chunks first, then plays.**

Piper emits AudioChunks incrementally as it synthesizes. In theory we could start playing chunk 1 while chunk 2 is still generating (~200ms latency savings). We don't yet — the code path with a queue + background playback thread + barge-in handling is meaningfully more complex, and Piper's per-response synthesis on M-series is fast enough (~200-400 ms for a typical response) that "wait then play" feels acceptable.

*Alternative rejected*: streaming playback in v0. The correctness cost (thread races between synthesis, playback, and stop()) outweighed the ~200 ms latency win. v1 revisits.

### Verify
```python
from tts_speaker import TTSSpeaker
t = TTSSpeaker(voice_model='/nonexistent/vi.onnx', english_workaround='spell')
assert t._preprocess_for_vi('Kết nối USB HDMI') == 'Kết nối u ét bê hát đê em i'
```
This tests the preprocessor without loading Piper. Real end-to-end verification requires a mic + Piper voice, hence the on-device tests in `demo/simulate_assistant.py`.

### Pitfalls
- **Long English strings**: "IEEE802.11ac" and similar aren't handled — the `_ACRONYM_RE` bounds it at 6 letters. Rare enough that we leave it.
- **Voice download on first run**: 60 MB from HuggingFace. If the user is offline, `_ensure_voice()` raises. That's intentional — silently falling back to eSpeak would surprise the user.
- **Concurrent `speak()` calls**: not thread-safe. Assumption is one caller (assistant main loop) serializes.

---

## Chapter 5 — The skills package

Four skills shipped in v0. Each is a plain module exposing `handle(slots) -> str`. The router binds a regex to each, extracts named-group slots, and calls the handler with them.

### time_skill

**Concept — Vietnamese numbers have quirks**: 15 = "mười lăm" (5 becomes "lăm" after ten), 21 = "hai mươi mốt" (1 becomes "mốt" after twenty). Encoding this as rules in `_vi_num_0_59` is cleaner than a 60-item lookup because you can audit "the 5→lăm rule" in one place.

**Concept — Vietnamese uses 12h with periods**: `_speak_hour_minute` maps 24h to `(hour, period)` where period is `sáng` (morning), `trưa` (noon), `chiều` (afternoon), `tối` (evening). The boundaries follow common convention (11:00-12:59 = trưa).

**Design decision — `kind` slot, not one function per intent**: rather than `handle_time`, `handle_weekday`, `handle_day`, `handle_month`, one `handle` reads `slots["kind"]` and dispatches. This is what `_bind(handler, kind=...)` in assistant.py is for — same handler function, registered under multiple patterns, each carrying a different fixed slot.

### translate_skill

**Design decision — always NLLB, never EnViT5**: EnViT5 is vi↔en only. This skill needs any target language, so we always route through NLLB. EnViT5 stays reserved for the streaming pipeline's `--translator envit5` mode where users specifically want it. That mode isn't a "skill" in the assistant sense; it's a config on the main pipeline.

**Design decision — lazy singleton translator**: `make_translator("nllb", model_dir)` loads a ~1.5 GB CT2 file. We only load it once per assistant process, on the FIRST translate command. The singleton is a module-level `_translator = None` guard.

**Concept — the ASR uses one language-code system, NLLB uses another**: the streaming pipeline uses BCP-47-ish codes like `en-US`. NLLB uses `eng_Latn`. The bridging table lives in `translator.py:ASR_TO_NLLB`. We reverse-lookup to bridge in the other direction (Vietnamese lang-name → NLLB code → ASR code that the existing `translator.translate()` method wants).

### alarm_skill

**Design decision — APScheduler, not a hand-rolled thread pool**: APScheduler handles DST transitions, provides restart persistence via job stores (though we use our own JSON for auditability), and gives us a clean 3-line setup vs the ~200 lines a correct hand-roll would need. Added dep is worth it.

**Design decision — persistence to `logs/alarms.json`, atomic writes**: every mutation writes to `.json.tmp` and renames. If the process crashes mid-write, we still have the previous valid file.

**Design decision — replay expired alarms on restart is DROP**: users generally don't want yesterday's 7am alarm firing when they boot the machine at noon. `_replay_persisted_alarms` scans the file at startup and skips anything with `fire_at <= now`.

**Design decision — alerts bypass `TTSSpeaker`**: users often mute the assistant's voice (in a meeting, kid asleep) but still want alarms to sound. We play `logs/alert.wav` directly via sounddevice at system-level audio. Falls back to a synthesized 800 Hz sine tone if no wav exists.

### home_assistant_skill

**Design decision — long-lived tokens, not OAuth**: OAuth needs a browser + callback URL. The assistant is a CLI; no browser, no public URL. HA supports long-lived tokens for exactly this case. Users generate one from HA's profile page, paste it into `~/.config/nemo-assistant.yaml`.

**Design decision — Vietnamese-phrase → entity_id map, not entity_id → alias**: users say "bật đèn phòng khách", not "bật light.living_room". The map is direct: `"đèn phòng khách": "light.living_room"`. The setup CLI (Chapter 6) discovers HA's entities and prompts for each Vietnamese phrase.

**Design decision — v0 supports light/switch/scene only**: `climate` needs temperature parsing, `media_player` needs playback state, `cover` needs open/close/tilt. All non-trivial. v0 covers the on-off case that solves 80% of household commands.

**Design decision — hard 2 s HTTP timeout**: HA reachable-slowly is a worse UX than HA unreachable-cleanly. 2 s catches transient network hiccups without hanging the assistant.

### Common patterns you'll see repeated

- **Lazy singleton for expensive setup** (translator load, HA config, APScheduler start) — never in `handle()`, always in a helper.
- **Timeouts on every external call** — 2 s hard cap, then a graceful Vietnamese error string.
- **Persistent state goes to `logs/`** — same directory the streaming pipeline uses. One place to look when debugging.

---

## Chapter 6 — assistant.py main loop

### Goal
Wire wake → ASR → intent → skill → TTS. All the pieces exist; this chapter is about the state machine that glues them and the choices made where "correct" wasn't obvious.

### Design decisions

**Decision 1: Mostly single-threaded core.**

Audio ingress (MicProducer), wake gate scoring, and the streaming ASR loop all run on the main thread. The only background thread is TTS's non-blocking playback. Fewer threads = fewer races = easier to debug when something goes wrong.

*Alternative rejected*: a full async event loop. Would give us "elegant" concurrency but you'd need asyncio primitives for every sync API in the stack (NeMo, sounddevice, APScheduler). Not worth it for a single-user assistant.

**Decision 2: The wake pre-roll is fed into `CacheAwareStreamingAudioBuffer.append_audio()` directly.**

The streaming buffer doesn't know or care that the pre-roll came from before the wake event. It just sees continuous audio. This is the trick that makes wake-word gating work with a streaming ASR — the ASR gets its normal context window, no special-case code for "wake just happened".

*Alternative rejected*: a "warm the model with pre-roll, then discard, then feed live audio" pattern. More complicated, no benefit.

**Decision 3: Silence VAD (RMS-based) is the primary commit signal for commands.**

The existing streaming pipeline uses silence VAD (`peak < threshold` for N chunks) as one of several commit triggers alongside language-tag and punctuation. For commands, silence VAD alone is enough — commands are short, they end with a pause, we don't need to wait for the model to emit a `<vi-VN>` tag. Setting `SILENCE_CHUNKS_TO_COMMIT = 6` gives us ~380 ms of pause tolerance — long enough to survive "um" pauses, short enough to feel responsive.

**Decision 4: The lang-tag commit is a safety net, not the primary path.**

Sometimes the user's command has an audible trailing "..." after they finish, and silence VAD doesn't fire cleanly. In that case the model itself will emit a `<vi-VN>` tag at the utterance boundary. We honor it — BUT only at a word boundary (last char is space/punctuation) to avoid the mid-syllable-cut bug we already saw in the streaming pipeline (stream_translate.py:838-853).

**Decision 5: The registration function `register_skills` is verbose and stays that way.**

You could DRY it up with a big table of (name, pattern, handler, defaults) tuples. Don't. When you're debugging "why doesn't this command route to alarm?", scanning the explicit `router.register_skill(...)` list is faster than tracing through a decorator or config system. The verbosity is a feature.

**Decision 6: `_bind(handler, **defaults)` for shared handlers with fixed slots.**

The time skill's handler takes `slots["kind"]` to know which sub-intent to run. When we register 4 different time patterns, each needs a different fixed `kind`. `_bind(time_skill.handle, kind="weekday")` produces a wrapped handler that pre-populates that slot. Regex-captured slots on the same pattern take precedence over defaults.

*Alternative rejected*: put the kind into a named-regex group. Awkward because the regex would need `(?P<kind>mấy giờ|thứ mấy|...)` which pollutes the patterns.

### State machine implementation

```
IDLE ──wake fires──▶ CAPTURING ──silence VAD commits──▶ THINKING ──skill returns──▶ SPEAKING ──TTS done──▶ IDLE
                                                                                        │
                                                                                  wake fires
                                                                                        ▼
                                                                                    (barge-in) CAPTURING
```

The whole state machine is one `while True` loop in `main()`. Each state transition is either a return value from `_capture_command` (ASR finished) or an explicit call to `tts.speak(response)` (thinking done → speaking).

### Verify

Two ways:
1. **End-to-end with a real mic + trained wake model**: `./assistant.sh` and speak. This is the ultimate test.
2. **Automated with a scripted wav**: `demo/simulate_assistant.py` feeds prerecorded wavs through the pipeline and asserts on responses. Doesn't need a wake model (skips wake detection) — tests the ASR + router + skill path.

### Pitfalls

- **Forgetting `wake.reset()` after handling**: leaves stale pre-roll audio in the ring buffer. Next command's first tokens would include audio from the previous command. Fixed by calling `reset()` in every branch that concludes a command (success, empty command, error).
- **Not handling the "wake fired but user didn't speak" case**: if the wake was a false positive, `_capture_command` returns empty text after `MAX_COMMAND_SECS`. We check for empty and speak a re-prompt instead of routing to the fallback.
- **The `_capture_command` inner loop can race with the mic producer being stopped**: `producer.take()` returns None during shutdown, we sleep and re-poll. Handled but easy to miss.

---

## Chapter 7 — Verification & benchmarks

### Goal
Measure the SLOs the PRD promised. If we can't measure it, we can't hit it — and can't tell if a "bug fix" made things worse.

### Two benchmark scripts shipped

**`bench/wake_far_frr.py`** measures wake-word quality:
- Give it a directory of "Nemo ơi" wavs (positives) + a directory of Vietnamese speech that isn't the wake phrase (negatives).
- It runs each through WakeGate and reports:
  - **FRR** — fraction of positives the gate missed. Target: <10% v0, <5% v1.
  - **FAR** — fires per hour of negative audio. Target: <2/hour v0, <1/hour v1.
- Between wavs it constructs a fresh `WakeGate` so cooldown doesn't leak.

**`demo/simulate_assistant.py`** is the end-to-end regression test:
- Reads a JSON script of `{wake_wav, command_wav, expected_skill, expected_response_contains}` cases.
- For each: concatenates the wavs, feeds through a fake `_WavProducer` that mimics MicProducer, uses a synthesized `WakeEvent` (skipping wake detection — we already tested that separately), then routes through the full ASR + router + skill path.
- Records per-stage latency (ASR ms, route ms) so we can bench P50/P95.

### Design decisions

**Decision 1: Two scripts, not one.**

Wake accuracy and end-to-end correctness are different measurements. Bundling them into one "test all the things" script would obscure which layer regressed when something breaks. Separate scripts, separate JSON outputs.

**Decision 2: The regression test doesn't run the wake model.**

We construct a fake `WakeEvent` because (a) we already test wake separately in `wake_far_frr.py`, (b) it lets the regression test run without a trained wake model, which unblocks CI, and (c) it isolates the "did the ASR+router+skill produce the right response" question from the "did the wake fire" question. Two questions, two tests.

**Decision 3: JSON scripts for test cases, not Python code.**

Non-programmers (Vietnamese speakers helping curate the test set) can edit JSON. They can't edit Python. Both users matter.

### Verify

The benches are the verification. Running them and getting green output is the acceptance criteria. Sample invocation for wake:

```bash
./.venv/bin/python bench/wake_far_frr.py \
    --model models/wake/nemo_oi.onnx \
    --positives data/wake/positive_real/ \
    --negatives data/wake/negative/ \
    --threshold 0.55
```

Output goes to `bench/wake_far_frr.json` for tracking across runs.

### Pitfalls

- **Threshold tuning by eye**: if v0 FRR is too high, don't lower the threshold to hit the SLO without also re-running the FAR side. You'll trade misses for false accepts.
- **Overfitting to your own voice**: if all your positive samples are you, the model works great for you and poorly for anyone else in your household. Recruit 5-10 speakers minimum.
- **Recording positives too cleanly**: real usage has fan noise, TV background, kitchen sounds. Positives that sound like a recording booth train an idealized model that fails in the kitchen.

---

## Chapter 8 — What we didn't finish and why (as of Task #15)

### What v0 has shipped

- WakeGate class + ring buffer + auto-download seam (`wake_gate.py`)
- IntentRouter with rule matching + wake-phrase stripping (`intent_router.py`)
- TTSSpeaker with Piper Vi + English-loanword spelling (`tts_speaker.py`)
- Four skills (time, translate, alarms, Home Assistant) each with focused test coverage
- assistant.py main event loop wiring wake → ASR → intent → skill → TTS
- Setup wizard (`assistant_setup.py`) writing `~/.config/nemo-assistant.yaml`
- Wake FAR/FRR bench (`bench/wake_far_frr.py`)
- End-to-end regression test (`demo/simulate_assistant.py`)
- Training script scaffold (`scripts/train_wake_model.py`)

### What now runs fully automatically (as of Task #16)

The wake-word training pipeline runs end-to-end without any real voice recordings:

```bash
./nemo.sh wake-train all      # ~30-60 min, produces models/wake/nemo_oi.onnx
```

See **[Chapter 1 — Wake-word training](01-wake-word-training.md)** for the full design walkthrough. Highlights:
- 2000 synthetic positives via Piper Vi × 8 phrase variants × 250 augmentations
- 1000 synthetic negatives (Vietnamese phrases that aren't the wake word, plus phonetic near-misses like "Bé ơi")
- Pluggable data manifest: drop real recordings into `data/wake/positive_real/`, edit weights if desired, re-run
- Bypass openWakeWord's broken `train.py`, replicate the ONNX shape directly with PyTorch → plug-compatible with `openwakeword.model.Model.predict()`

### What still needs a human touch

You'll still want to **record ~50-200 real "Nemo ơi" clips** from you + family and drop them into `data/wake/positive_real/` when convenient. Synthetic-only training gets you a working model (recall ~85-90% on your voice, since Piper's voice profile is broadly Vietnamese), but real recordings materially improve generalization across speakers. Because the manifest weights real clips 4× per file, even 20-50 recordings shift the model meaningfully.

Push-to-talk mode (`./nemo.sh ptt`) also works with zero training at all — for testing the ASR + skills + TTS path without a wake model in the loop.

### What you'd tackle in v1

Now that the v0 skeleton is real, natural v1 candidates:
- Weather skill (~2 days, HTTP integration same as HA)
- Journaling / dictation skill (~2 days, continuous ASR mode)
- Actual streaming TTS playback (~1 week, threading + queue)
- Custom wake-word training UX (`./assistant.sh --train-wake`, guides users through recording + training)
- On-device LLM fallback for open-ended chat (deferred to v2 per PRD)

Read this doc top-to-bottom once, then poke at the code with a specific "why is X the way it is?" question — the design-decisions sections should have answers.
