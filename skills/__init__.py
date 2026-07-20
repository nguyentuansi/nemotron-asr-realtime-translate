"""Skills package — one module per intent handler.

Each skill exposes a `handle(slots: dict) -> str` function (or coroutine, in
v2+). The IntentRouter binds a regex pattern to that function; when the pattern
matches an incoming command, the router calls `handle(slots)` and speaks the
returned string via TTSSpeaker.

Skills should be small, testable, and free of side effects on the assistant's
audio path. Anything that talks to the network or filesystem should time out
fast (< 2 s) and return a graceful error string on failure.

Registered skills (v0):
    time_skill                — "mấy giờ", "hôm nay là thứ mấy"
    translate_skill           — "dịch sang tiếng Anh: ..."
    alarm_skill               — "đặt báo thức 6 giờ sáng"
    home_assistant_skill      — "tắt đèn phòng khách"
"""
