"""Home Assistant control skill.

See docs/assistant/00-build-story.md Chapter 5 for design notes.

Vietnamese command shapes (registered by assistant.py):
    "bật đèn phòng khách"                    → light.turn_on / switch.turn_on
    "tắt đèn phòng khách"                    → light.turn_off / switch.turn_off
    "kích hoạt cảnh xem phim"                → scene.turn_on
    "trạng thái đèn phòng khách"              → state check

Config (~/.config/nemo-assistant.yaml, loaded via load_config()):

    home_assistant:
      url: "http://homeassistant.local:8123"
      token: "eyJ0eXAi..."                # long-lived token
      entity_aliases:                     # Vietnamese phrase → HA entity_id
        "đèn phòng khách": "light.living_room"
        "đèn bếp": "light.kitchen"
        "cảnh xem phim": "scene.movie_night"

v0 domains: light, switch, scene. climate/media/cover/lock deferred to v1.

Timeouts: every HTTP call hard-caps at 2 s. HA unreachable → graceful
Vietnamese error, never a hang.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("home_assistant_skill")

CONFIG_FILE = Path.home() / ".config" / "nemo-assistant.yaml"

# Supported domain → (turn-on service, turn-off service). None on the off side
# means "no off action" (scenes are one-shot activations).
_DOMAIN_SERVICES = {
    "light": ("turn_on", "turn_off"),
    "switch": ("turn_on", "turn_off"),
    "scene": ("turn_on", None),
}

# HTTP timeout for HA REST calls. Non-negotiable — anything slower ruins UX.
_HA_TIMEOUT_S = 2.0

# Lazy singleton for parsed config.
_config: Optional[dict] = None


def _load_config() -> dict:
    """Load ~/.config/nemo-assistant.yaml (cached). Returns {} on missing/bad."""
    global _config
    if _config is not None:
        return _config
    try:
        import yaml
        _config = yaml.safe_load(CONFIG_FILE.read_text()) or {}
    except FileNotFoundError:
        LOG.info("no config at %s — HA skill will refuse commands", CONFIG_FILE)
        _config = {}
    except Exception as e:
        LOG.warning("could not load %s: %s", CONFIG_FILE, e)
        _config = {}
    return _config


def _get_ha_settings() -> Optional[dict]:
    """Return {"url", "token", "aliases"} if HA is configured, else None."""
    cfg = _load_config().get("home_assistant") or {}
    url = cfg.get("url")
    token = cfg.get("token")
    if not url or not token:
        return None
    return {
        "url": url.rstrip("/"),
        "token": token,
        "aliases": cfg.get("entity_aliases") or {},
    }


def _call_service(domain: str, service: str, entity_id: str, url: str, token: str) -> bool:
    """POST /api/services/{domain}/{service} with {"entity_id": entity_id}.

    Returns True on 2xx, False on any error. Hard 2 s timeout.
    """
    import requests
    endpoint = f"{url}/api/services/{domain}/{service}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(
            endpoint,
            headers=headers,
            json={"entity_id": entity_id},
            timeout=_HA_TIMEOUT_S,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        LOG.warning("HA call failed: %s %s: %s", domain, service, e)
        return False


def _get_state(entity_id: str, url: str, token: str) -> Optional[str]:
    """GET /api/states/{entity_id} → state string ('on', 'off', ...), or None."""
    import requests
    endpoint = f"{url}/api/states/{entity_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(endpoint, headers=headers, timeout=_HA_TIMEOUT_S)
        r.raise_for_status()
        return r.json().get("state")
    except Exception as e:
        LOG.warning("HA state read failed: %s", e)
        return None


def discover_entities(url: str, token: str) -> list[dict]:
    """GET /api/states, filter to light/switch/scene. Called by the setup CLI.

    Each result is {"entity_id", "friendly_name", "domain"}.
    """
    import requests
    endpoint = f"{url}/api/states"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(endpoint, headers=headers, timeout=5.0)
    r.raise_for_status()
    out = []
    for state in r.json():
        eid = state.get("entity_id", "")
        domain = eid.split(".", 1)[0] if "." in eid else ""
        if domain not in _DOMAIN_SERVICES:
            continue
        out.append({
            "entity_id": eid,
            "friendly_name": state.get("attributes", {}).get("friendly_name", eid),
            "domain": domain,
        })
    return out


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------

def handle(slots: dict) -> str:
    """Dispatch on slots["op"]:
        "turn_on"  — with slots["alias"]
        "turn_off" — with slots["alias"]
        "activate" — with slots["alias"] (for scenes)
        "state"    — with slots["alias"] (read only)
    """
    settings = _get_ha_settings()
    if settings is None:
        return ("Chưa cài Home Assistant. "
                "Hãy chạy assistant.sh với --setup để cài đặt.")

    op = slots.get("op", "turn_on")
    alias = (slots.get("alias") or "").strip().lower()

    if not alias:
        return "Nemo không biết bạn muốn điều khiển thiết bị nào."

    # Resolve Vietnamese phrase to HA entity_id
    entity_id = settings["aliases"].get(alias)
    if entity_id is None:
        return f"Nemo không tìm thấy thiết bị '{alias}' trong danh sách."

    domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
    if domain not in _DOMAIN_SERVICES:
        return f"Nemo chưa hỗ trợ loại thiết bị '{domain}'."

    if op == "state":
        state = _get_state(entity_id, settings["url"], settings["token"])
        if state is None:
            return "Không đọc được trạng thái. Hãy kiểm tra Home Assistant."
        # Human-friendly state phrasing
        state_vi = {
            "on": "đang bật", "off": "đang tắt",
            "unavailable": "không phản hồi", "unknown": "trạng thái chưa xác định",
        }.get(state, state)
        return f"{alias.capitalize()} {state_vi}"

    # Turn on / turn off / activate → service call
    on_svc, off_svc = _DOMAIN_SERVICES[domain]
    if op == "turn_off":
        if off_svc is None:
            return f"Không thể tắt {alias} — đây là một cảnh."
        service = off_svc
    else:
        # "turn_on" and "activate" both use the on service.
        service = on_svc

    ok = _call_service(domain, service, entity_id,
                       settings["url"], settings["token"])
    if not ok:
        return "Không kết nối được Home Assistant."

    # Concise Vietnamese confirmation. Different verbs per domain feel more
    # natural than one generic "OK".
    verbs = {
        ("light", "turn_on"): "Đã bật",
        ("light", "turn_off"): "Đã tắt",
        ("switch", "turn_on"): "Đã bật",
        ("switch", "turn_off"): "Đã tắt",
        ("scene", "turn_on"): "Đã kích hoạt",
    }
    verb = verbs.get((domain, service), "Đã thực hiện")
    return f"{verb} {alias}"
