"""Help / intro skill — answers "what can you do?".

Vietnamese command shapes that route here:
    "bạn có thể làm gì"
    "bạn giúp được gì"
    "trợ giúp"
    "help"
    "danh sách lệnh"
    "làm được gì"

Rather than reciting every command, we give a curated short list of what
each of the four v0 skills can do, spoken naturally. Users pick up patterns
from these examples faster than from a formal enumeration.

Design decision — hard-coded response, not generated:
    We could have this skill introspect the IntentRouter and describe every
    registered pattern. Fine idea but noisy — the router has 14 patterns
    covering 4 conceptual skills. Users don't want to hear all 14. A
    hand-curated response for each skill category is clearer.

Design decision — end with a "try this" example:
    Users are more likely to try a concrete command after they hear one
    than to invent their own. The response ends with "Ví dụ: ..." (Example: ...).
"""
from __future__ import annotations


_INTRO = (
    "Nemo có thể trả lời bốn loại câu hỏi. "
    "Thứ nhất, hỏi giờ và ngày — ví dụ 'mấy giờ rồi' hoặc 'hôm nay là thứ mấy'. "
    "Thứ hai, đặt báo thức và hẹn giờ — ví dụ 'đặt báo thức sáu giờ sáng' hoặc 'hẹn giờ năm phút'. "
    "Thứ ba, dịch tiếng — ví dụ 'dịch sang tiếng Anh xin chào bạn'. "
    "Thứ tư, điều khiển đèn nếu bạn dùng Home Assistant — ví dụ 'bật đèn phòng khách'. "
    "Nói 'Nemo ơi' rồi thử một câu lệnh xem sao."
)


def handle(slots: dict) -> str:
    return _INTRO
