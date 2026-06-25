# Demo scripts

Pick one. Read aloud at a natural pace — don't slow down for the ASR, it's
designed for normal speech and rushing/over-enunciating actually hurts WER.

Each script is timed for ~25–30 seconds spoken. The English column is the
expected translation, NOT a subtitle to read.

---

## Script A — Pitch (formal, demos the product itself)

**Vietnamese**

> Đây là demo trực tiếp của hệ thống dịch nói thời gian thực. Tất cả đều
> chạy trên CPU của máy MacBook, không cần internet, không cần API key.
> Mô hình Nemotron của NVIDIA nhận diện tiếng nói, sau đó EnViT5 dịch sang
> tiếng Anh. Độ trễ dưới một giây cho mỗi câu.

**Expected English (rough)**

> This is a live demo of a real-time speech translation system. Everything
> runs on a MacBook CPU, no internet, no API key. NVIDIA's Nemotron model
> handles speech recognition, then EnViT5 translates to English. Latency is
> under one second per sentence.

Why this script: matches the README pitch, ~28s, mixes proper nouns
(Nemotron, NVIDIA, EnViT5, MacBook) and ordinary Vietnamese sentences — good
stress test of mixed-domain ASR.

---

## Script B — Story (casual, demos tone + naturalness)

**Vietnamese**

> Sáng nay tôi đi cà phê với bạn. Quán đông người, nhạc to, nhưng cà phê
> rất ngon. Chúng tôi ngồi nói chuyện hai tiếng đồng hồ. Bạn tôi kể về
> chuyến đi Đà Nẵng tuần trước. Cuối cùng tôi về nhà đi làm việc.

**Expected English (rough)**

> This morning I went for coffee with a friend. The shop was crowded and
> loud, but the coffee was great. We sat and talked for two hours. My friend
> told me about her trip to Da Nang last week. Finally I went home to work.

Why this script: ~25s, conversational tone, exercises proper-name handling
(Đà Nẵng) and natural pacing.

---

## Script C — Technical (demos domain robustness)

**Vietnamese**

> Mô hình này dùng kiến trúc Conformer với cache streaming. Bộ mã hóa được
> xuất sang ONNX để chạy nhanh hơn tám lần trên CPU. Real-time factor đạt
> không phẩy hai trên MacBook Pro M2. Toàn bộ pipeline mở mã nguồn theo
> giấy phép MIT.

**Expected English (rough)**

> This model uses a Conformer architecture with streaming cache. The encoder
> is exported to ONNX to run eight times faster on CPU. The real-time factor
> reaches zero point two on a MacBook Pro M2. The entire pipeline is open
> source under MIT license.

Why this script: ~25s, dense with technical loanwords (Conformer, ONNX, CPU,
MIT) — the hardest test, but the most impressive if it works clean.

---

## Recommended take order

1. Warm-up read of **Script A** to set mic level + check terminal layout.
2. Record **Script A** as the primary candidate (matches README pitch).
3. Record **Script C** as a backup (the tech crowd on HN will react to the
   loanword handling).
4. Pick the cleaner take after watching both.

If a take has one bad word but everything else is good — ship it. Don't chase
perfection; a slightly imperfect transcription is more honest than a polished
one and shows the user what to actually expect.
