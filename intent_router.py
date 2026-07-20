"""IntentRouter — rule-based intent classifier for the Vietnamese assistant.

See docs/assistant/00-build-story.md Chapter 3 for the design rationale.

v0 policy: rule matching only. No LLM. When no rule matches, the caller speaks
"Xin lỗi, Nemo chưa hiểu câu đó" and moves on. The `_llm_fallback` seam is
declared for v2 but not implemented.

Public shape:

    from intent_router import IntentRouter, IntentResult

    router = IntentRouter()
    router.register_skill(
        name="time",
        pattern=r"^(mấy giờ|hôm nay là (thứ|ngày) mấy)",
        handler=time_skill.handle,
    )
    result = router.route("Nemo ơi, mấy giờ rồi?")
    if result.skill_name is not None:
        response = result.handler(result.slots)
    else:
        response = "Xin lỗi, Nemo chưa hiểu câu đó"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# Wake-phrase strip patterns. Kept permissive to survive ASR mishearings —
# the ASR occasionally emits "nêmô ơi" or "ne mo ơi" instead of "nemo ơi".
# Order doesn't matter (all alternatives ORed together in one regex).
_WAKE_PREFIXES = [
    r"^\s*nemo\s+ơi[,\s]*",
    r"^\s*nêmô\s+ơi[,\s]*",
    r"^\s*ne\s*mo\s+ơi[,\s]*",
    r"^\s*nê\s*mô\s+ơi[,\s]*",
]
_WAKE_RE = re.compile("|".join(_WAKE_PREFIXES), re.IGNORECASE)


def strip_wake_phrase(text: str) -> str:
    """Remove a leading "Nemo ơi" (with common ASR variants) from the text.

    Skills should never see the wake phrase in their input — it's the router's
    job. Idempotent (running it twice is safe).
    """
    return _WAKE_RE.sub("", text).strip()


@dataclass
class IntentResult:
    """Returned by IntentRouter.route().

    skill_name is None when nothing matched — the caller uses that as the
    "speak the fallback" signal. This is cleaner than raising because it
    keeps the caller's flow linear (no try/except around every route()).
    """
    skill_name: Optional[str]
    slots: dict = field(default_factory=dict)
    handler: Optional[Callable] = None
    matched_text: str = ""      # what the router saw AFTER stripping the wake phrase
    confidence: float = 0.0     # 1.0 for exact-rule matches; reserved for future LLM


@dataclass
class _RegisteredSkill:
    """Internal record for one registered skill."""
    name: str
    pattern: re.Pattern
    handler: Callable


class IntentRouter:
    """Rule-first intent router with a declared but unimplemented LLM fallback.

    Patterns are matched in registration order — earlier registrations win.
    Register your MOST-SPECIFIC patterns first, general fallbacks last.
    """

    def __init__(self) -> None:
        self._skills: list[_RegisteredSkill] = []

    def register_skill(
        self,
        name: str,
        pattern: str | re.Pattern,
        handler: Callable,
    ) -> None:
        """Register a skill. Patterns are checked in insertion order.

        `pattern` can be a raw string (compiled with re.IGNORECASE) or a
        pre-compiled Pattern (used as-is, so the caller controls flags).
        """
        if isinstance(pattern, str):
            compiled = re.compile(pattern, re.IGNORECASE)
        else:
            compiled = pattern
        self._skills.append(_RegisteredSkill(name=name, pattern=compiled, handler=handler))

    def route(self, text: str) -> IntentResult:
        """Match the text against registered skills.

        Steps:
        1. Strip the wake phrase (skills don't see "Nemo ơi").
        2. Try each registered pattern in insertion order.
        3. First match wins — extract named groups as slots, return.
        4. Nothing matched → IntentResult(skill_name=None).
        """
        stripped = strip_wake_phrase(text)

        for skill in self._skills:
            m = skill.pattern.search(stripped)
            if m is not None:
                # groupdict() gives us named capture groups as a dict. Regex
                # without named groups → empty dict. Either way, downstream
                # skills receive a well-formed slots parameter.
                slots = {k: v for k, v in m.groupdict().items() if v is not None}
                return IntentResult(
                    skill_name=skill.name,
                    slots=slots,
                    handler=skill.handler,
                    matched_text=stripped,
                    confidence=1.0,     # rule match is a hard 1.0 by convention
                )

        # Nothing matched. The caller reads skill_name is None and speaks the
        # Vietnamese fallback "Xin lỗi, Nemo chưa hiểu câu đó".
        return IntentResult(
            skill_name=None,
            slots={},
            handler=None,
            matched_text=stripped,
            confidence=0.0,
        )

    def _llm_fallback(self, text: str) -> IntentResult:
        """v2 seam. Not implemented in v0.

        Kept in place so v2's addition doesn't require a public-interface
        change. When we do implement, this will call a local LLM (Qwen2.5-1.5B
        via llama.cpp), classify the intent, and either dispatch to a known
        skill or return a "generic_chat" IntentResult that a chat skill handles.
        """
        raise NotImplementedError("LLM fallback deferred to v2")
