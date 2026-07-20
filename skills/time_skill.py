r"""Time / date / day-of-week skill.

Matches Vietnamese time-and-date questions, returns a Vietnamese response
string. Zero external dependencies. See docs/assistant/00-build-story.md
Chapter 5 for design notes.

Patterns handled (registered by assistant.py):
    ^\s*mấy giờ.*                              → current time     (kind=time)
    ^\s*(bây giờ là )?mấy giờ.*                → current time     (kind=time)
    ^\s*hôm nay là (thứ|ngày) mấy               → weekday|day     (kind=weekday|day)
    ^\s*hôm nay là ngày (mấy|bao nhiêu).*       → day-of-month    (kind=day)
    ^\s*tháng (mấy|bao nhiêu).*                 → month            (kind=month)

Response examples:
    "Bây giờ là 8 giờ 32 phút sáng"
    "Hôm nay là thứ ba"
    "Hôm nay là ngày 25 tháng 6"
    "Bây giờ là tháng 6"
"""
from __future__ import annotations

from datetime import datetime


# Vietnamese day-of-week names. datetime.weekday() returns 0=Monday..6=Sunday,
# which conveniently maps 1:1 to Vietnamese "thứ hai" (2nd) through "chủ nhật".
_VI_WEEKDAYS = [
    "thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật",
]

# Vietnamese digit words. Used for 0..9 directly; larger numbers built up
# via _vi_num_0_59.
_VI_DIGITS = [
    "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín",
]


def _vi_num_0_59(n: int) -> str:
    """Convert 0..59 to Vietnamese spoken form.

    Vietnamese numbers have a few quirks worth encoding here rather than in a
    lookup table:
      - 10 = "mười", 20 = "hai mươi", 30 = "ba mươi"  (tens use "mươi" not "mười")
      - 11 = "mười một", 15 = "mười lăm"  (5 becomes "lăm" after "mười")
      - 21 = "hai mươi mốt"  (1 becomes "mốt" after "mươi")
      - 25 = "hai mươi lăm"  (5 stays as "lăm" after any "mươi")
    These rules apply the same way from 20 through 59.
    """
    if not (0 <= n <= 59):
        raise ValueError(f"_vi_num_0_59 only handles 0..59, got {n}")
    if n < 10:
        return _VI_DIGITS[n]
    if n == 10:
        return "mười"
    if n < 20:
        # 11..19 = "mười" + unit
        unit = n - 10
        if unit == 5:
            return "mười lăm"
        return f"mười {_VI_DIGITS[unit]}"
    # 20..59: tens + "mươi" + optional unit
    tens = n // 10
    unit = n % 10
    base = f"{_VI_DIGITS[tens]} mươi"
    if unit == 0:
        return base
    if unit == 1:
        return f"{base} mốt"     # "hai mươi mốt" not "hai mươi một"
    if unit == 5:
        return f"{base} lăm"     # "hai mươi lăm" not "hai mươi năm"
    return f"{base} {_VI_DIGITS[unit]}"


def _speak_hour_minute(hour_24: int, minute: int) -> str:
    """Format an H:M in 24h into Vietnamese-natural spoken form.

    Vietnamese uses 12-hour speech with period modifiers (sáng/trưa/chiều/tối),
    even in formal contexts. The period boundaries follow common convention:
      04:00-10:59 = sáng   (morning)
      11:00-12:59 = trưa   (noon)
      13:00-17:59 = chiều  (afternoon)
      18:00-03:59 = tối    (evening/night)
    """
    # Period
    if 4 <= hour_24 <= 10:
        period = "sáng"
    elif 11 <= hour_24 <= 12:
        period = "trưa"
    elif 13 <= hour_24 <= 17:
        period = "chiều"
    else:
        period = "tối"

    # 12-hour clock: 0→12, 13→1, ...
    if hour_24 == 0:
        h12 = 12
    elif hour_24 > 12:
        h12 = hour_24 - 12
    else:
        h12 = hour_24

    hour_word = _vi_num_0_59(h12)
    if minute == 0:
        return f"{hour_word} giờ {period}"
    minute_word = _vi_num_0_59(minute)
    return f"{hour_word} giờ {minute_word} phút {period}"


def handle(slots: dict) -> str:
    """Dispatch on slots['kind']:
      "time"    → current wall time
      "weekday" → today's day-of-week
      "day"     → today's day-of-month + month
      "month"   → current month
    Falls back to "time" if kind is missing (safer default).
    """
    now = datetime.now()
    kind = slots.get("kind", "time")

    if kind == "weekday":
        return f"Hôm nay là {_VI_WEEKDAYS[now.weekday()]}"

    if kind == "day":
        # "Hôm nay là ngày 25 tháng 6" — day and month both as digits,
        # never spoken as words (natural Vietnamese usage for dates).
        return f"Hôm nay là ngày {now.day} tháng {now.month}"

    if kind == "month":
        return f"Bây giờ là tháng {now.month}"

    # Default: current time
    return f"Bây giờ là {_speak_hour_minute(now.hour, now.minute)}"
