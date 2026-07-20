"""Alarms & timers skill.

See docs/assistant/00-build-story.md Chapter 5 for design notes.

Vietnamese command shapes handled (registered by assistant.py):
    "đặt báo thức 6 giờ sáng"        → alarm at 06:00 tomorrow (or today if <06:00)
    "đặt báo thức 6 giờ chiều"       → alarm at 18:00
    "đặt báo thức 6 giờ"              → alarm at 06:00 (assume morning if <12)
    "hẹn giờ 5 phút"                 → timer 5 minutes from now
    "hẹn giờ 90 giây"                → timer 90 seconds from now
    "hủy báo thức"                    → cancel all pending alarms
    "còn báo thức không"              → count pending alarms

State model:
    - APScheduler BackgroundScheduler runs the timer thread
    - Every alarm is persisted to logs/alarms.json across restarts
    - On restart we replay logs/alarms.json; already-expired alarms are dropped

Alert playback bypasses TTSSpeaker (users often mute the assistant's voice
but still expect alarms to sound). We play logs/alert.wav via sounddevice
at system audio.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

LOG = logging.getLogger("alarm_skill")

_ROOT = Path(__file__).resolve().parent.parent
ALARMS_FILE = _ROOT / "logs" / "alarms.json"
ALERT_WAV = _ROOT / "logs" / "alert.wav"    # user-supplied (falls back to a sine tone)


# --------------------------------------------------------------------
# Slot-parsing helpers — turn "6 giờ sáng" or "5 phút" into a datetime
# --------------------------------------------------------------------

_VI_PERIODS = {
    "sáng": "am",
    "trưa": "am",      # 11-13 range — treat as morning (12=noon)
    "chiều": "pm",
    "tối": "pm",
    "đêm": "pm",
}


def _parse_clock_time(spec: str) -> Optional[datetime]:
    """Parse "6 giờ sáng" / "13 giờ" / "6 giờ 30 phút chiều" → next-occurrence datetime.

    If the target hour is earlier today, schedule it for TOMORROW (users saying
    "6 giờ sáng" at 8am plainly mean tomorrow morning).
    """
    import re
    # hours ('giờ') is required; minutes ('phút') and period suffix are optional.
    m = re.match(
        r"(?P<h>\d{1,2})\s*giờ(?:\s*(?P<m>\d{1,2})\s*phút)?"
        r"(?:\s*(?P<period>sáng|trưa|chiều|tối|đêm))?",
        spec.strip(),
        re.IGNORECASE,
    )
    if not m:
        return None

    hour = int(m.group("h"))
    minute = int(m.group("m") or 0)
    period = m.group("period")

    # Period normalization: chiều/tối/đêm means PM, so 6 chiều = 18:00.
    if period and _VI_PERIODS.get(period.lower()) == "pm" and hour < 12:
        hour += 12
    # 12 sáng = midnight in colloquial Vietnamese; 12 chiều/trưa = noon.
    if hour == 24:
        hour = 0

    now = datetime.now()
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def _parse_relative(spec: str) -> Optional[datetime]:
    """Parse "5 phút" / "90 giây" / "2 giờ" → future datetime.

    Returned time is relative to _now_ so this is meaningful only at call-time.
    """
    import re
    m = re.match(r"(?P<n>\d+)\s*(?P<unit>giây|phút|giờ)", spec.strip(), re.IGNORECASE)
    if not m:
        return None
    n = int(m.group("n"))
    unit = m.group("unit").lower()
    delta = {
        "giây": timedelta(seconds=n),
        "phút": timedelta(minutes=n),
        "giờ": timedelta(hours=n),
    }[unit]
    return datetime.now() + delta


# --------------------------------------------------------------------
# Scheduler singleton — starts on first use, persists across handle() calls
# --------------------------------------------------------------------

_scheduler = None
_alarms: dict[str, dict] = {}         # id → {"fire_at": ISO, "label": str, "kind": "clock"|"timer"}
_lock = threading.Lock()


def _get_scheduler():
    """Return the APScheduler BackgroundScheduler (starting it on first call)."""
    global _scheduler
    if _scheduler is None:
        from apscheduler.schedulers.background import BackgroundScheduler
        _scheduler = BackgroundScheduler()
        _scheduler.start()
        _replay_persisted_alarms()
        LOG.info("alarm scheduler started")
    return _scheduler


def _persist_alarms() -> None:
    """Write current alarms atomically. Called after every mutation."""
    ALARMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ALARMS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_alarms, ensure_ascii=False, indent=2))
    tmp.rename(ALARMS_FILE)


def _replay_persisted_alarms() -> None:
    """Re-arm any alarms that survive across a restart. Expired ones dropped."""
    global _alarms
    if not ALARMS_FILE.exists():
        return
    try:
        raw = json.loads(ALARMS_FILE.read_text())
    except Exception as e:
        LOG.warning("could not read %s: %s", ALARMS_FILE, e)
        return

    now = datetime.now()
    kept = {}
    for aid, spec in raw.items():
        try:
            fire_at = datetime.fromisoformat(spec["fire_at"])
        except Exception:
            continue
        if fire_at <= now:
            LOG.info("dropping expired alarm %s (was %s)", aid, spec.get("label"))
            continue
        _schedule_one(aid, fire_at, spec.get("label", ""))
        kept[aid] = spec
    _alarms = kept
    _persist_alarms()


def _schedule_one(alarm_id: str, fire_at: datetime, label: str) -> None:
    """Register a one-shot APScheduler job."""
    sched = _get_scheduler()
    sched.add_job(
        _fire_alarm,
        trigger="date",
        run_date=fire_at,
        args=[alarm_id, label],
        id=alarm_id,
        replace_existing=True,
    )


def _fire_alarm(alarm_id: str, label: str) -> None:
    """Play the alert wav + remove the alarm from the persistent list."""
    LOG.info("alarm fired: id=%s label=%s", alarm_id, label)
    _play_alert()
    with _lock:
        _alarms.pop(alarm_id, None)
        _persist_alarms()


def _play_alert() -> None:
    """Play logs/alert.wav via sounddevice. If missing, generate a 3s sine
    tone at 800 Hz — simple + audible, no external asset required.
    """
    try:
        import sounddevice as sd
        if ALERT_WAV.exists():
            import soundfile as sf
            audio, sr = sf.read(str(ALERT_WAV), dtype="float32")
        else:
            # Generated sine tone: 800 Hz for 3 seconds at 16 kHz.
            sr = 16000
            t = np.arange(0, 3.0, 1.0 / sr, dtype=np.float32)
            audio = 0.3 * np.sin(2 * np.pi * 800 * t)
            # Fade in/out 50ms so it's not a hard click.
            fade = int(0.05 * sr)
            audio[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            audio[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        sd.play(audio, sr, blocking=False)
    except Exception as e:
        LOG.warning("could not play alert: %s", e)


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------

def handle(slots: dict) -> str:
    """Dispatch on slots["op"]:
        "set"    — with slots["time_spec"] + slots["kind"] ("clock" | "timer")
        "cancel" — cancel all alarms
        "count"  — how many alarms pending
    """
    op = slots.get("op", "set")

    if op == "cancel":
        with _lock:
            n = len(_alarms)
            sched = _get_scheduler()
            for aid in list(_alarms.keys()):
                try:
                    sched.remove_job(aid)
                except Exception:
                    pass
            _alarms.clear()
            _persist_alarms()
        return f"Đã hủy {n} báo thức" if n else "Không có báo thức nào để hủy"

    if op == "count":
        return f"Bạn có {len(_alarms)} báo thức" if _alarms else "Không có báo thức nào"

    # op == "set"
    kind = slots.get("kind", "clock")
    time_spec = slots.get("time_spec", "")

    if kind == "timer":
        fire_at = _parse_relative(time_spec)
    else:
        fire_at = _parse_clock_time(time_spec)

    if fire_at is None:
        return "Nemo không hiểu thời gian bạn nói. Hãy thử '6 giờ sáng' hoặc '5 phút'."

    aid = f"alarm-{int(time.time() * 1000)}"
    with _lock:
        _schedule_one(aid, fire_at, time_spec)
        _alarms[aid] = {"fire_at": fire_at.isoformat(), "label": time_spec, "kind": kind}
        _persist_alarms()

    # Human-friendly confirmation. Reuse time_skill's speech format for
    # clock alarms; timers we describe relatively.
    if kind == "timer":
        return f"Đã hẹn giờ {time_spec}"
    from skills.time_skill import _speak_hour_minute
    when = _speak_hour_minute(fire_at.hour, fire_at.minute)
    return f"Đã đặt báo thức lúc {when}"


# For testability / bench: expose the internal state readable.
def _snapshot_alarms() -> dict:
    with _lock:
        return dict(_alarms)
