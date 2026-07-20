"""assistant_setup.py — first-run configuration wizard for the assistant.

Interactive CLI that walks the user through creating ~/.config/nemo-assistant.yaml.
Called via `./assistant.sh --setup`. Splits into a separate module because it
imports pyyaml + the HA discovery helpers, which we don't need at every launch.

Config schema (written to ~/.config/nemo-assistant.yaml):

    wake:
      threshold: 0.55
      cooldown_s: 1.5

    tts:
      voice: "vi_VN-vais1000-medium"
      english_workaround: "spell"      # or "off"

    silence_vad:
      threshold: 0.15
      chunks_to_commit: 6

    home_assistant:
      url: "http://homeassistant.local:8123"
      token: "..."
      entity_aliases:
        "đèn phòng khách": "light.living_room"
        ...

Users can re-run the wizard any time; it re-reads any existing config as
defaults and only overwrites what they change.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import yaml

CONFIG_FILE = Path.home() / ".config" / "nemo-assistant.yaml"


DEFAULT_CONFIG: dict[str, Any] = {
    "wake": {"threshold": 0.55, "cooldown_s": 1.5},
    "tts": {"voice": "vi_VN-vais1000-medium", "english_workaround": "spell"},
    "silence_vad": {"threshold": 0.15, "chunks_to_commit": 6},
    "home_assistant": {"url": "", "token": "", "entity_aliases": {}},
}


def _load_existing() -> dict:
    """Load the existing config if any, else return DEFAULT_CONFIG copy."""
    if not CONFIG_FILE.exists():
        return {k: dict(v) if isinstance(v, dict) else v
                for k, v in DEFAULT_CONFIG.items()}
    try:
        raw = yaml.safe_load(CONFIG_FILE.read_text()) or {}
    except Exception as e:
        print(f"Không đọc được {CONFIG_FILE}: {e}. Bắt đầu từ đầu.")
        raw = {}
    # Merge with defaults so newly-added fields have sane starting values
    merged = {k: (raw.get(k, v) if isinstance(v, dict) else raw.get(k, v))
              for k, v in DEFAULT_CONFIG.items()}
    return merged


def _prompt(question: str, default: Optional[str] = None) -> str:
    """Prompt with a default. Empty input → default. Trims whitespace."""
    if default:
        s = input(f"{question} [{default}]: ").strip()
    else:
        s = input(f"{question}: ").strip()
    return s or (default or "")


def _prompt_bool(question: str, default: bool = True) -> bool:
    d = "y/N" if not default else "Y/n"
    s = input(f"{question} [{d}]: ").strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "yes", "1", "true", "có", "đúng")


def _write_config(cfg: dict) -> None:
    """Atomically write the config file to ~/.config/nemo-assistant.yaml."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_FILE.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    tmp.rename(CONFIG_FILE)
    print(f"\nĐã lưu {CONFIG_FILE}")


def _setup_home_assistant(current: dict) -> dict:
    """Interactive Home Assistant configuration.

    Steps:
    1. Ask for URL + token
    2. Attempt entity discovery
    3. For each entity found, prompt for a Vietnamese phrase (default = friendly_name)
    4. Return the updated ha_config dict
    """
    print("\n--- Home Assistant ---")
    url = _prompt("URL (ví dụ http://homeassistant.local:8123)", current.get("url", ""))
    if not url:
        print("(bỏ qua Home Assistant)")
        return current

    token = _prompt("Long-lived access token", current.get("token", ""))
    if not token:
        print("(bỏ qua Home Assistant — cần token)")
        return current

    aliases = dict(current.get("entity_aliases") or {})

    try:
        from skills.home_assistant_skill import discover_entities
        entities = discover_entities(url.rstrip("/"), token)
    except Exception as e:
        print(f"Không kết nối được HA: {e}")
        print("Đã lưu URL + token — bạn có thể sửa aliases sau.")
        return {"url": url, "token": token, "entity_aliases": aliases}

    if not entities:
        print("Không tìm thấy đèn/công tắc/cảnh nào.")
        return {"url": url, "token": token, "entity_aliases": aliases}

    print(f"\nTìm thấy {len(entities)} thiết bị. Nhập tên tiếng Việt cho mỗi cái,")
    print("hoặc bỏ trống để bỏ qua thiết bị đó.\n")

    for e in entities:
        # If we already have a Vietnamese alias for this entity, show it
        existing_alias = next(
            (v_alias for v_alias, eid in aliases.items() if eid == e["entity_id"]),
            None,
        )
        default = existing_alias or e["friendly_name"].lower()
        prompt = f"  {e['entity_id']} ({e['friendly_name']})"
        alias = _prompt(prompt, default)
        if alias:
            # Remove any previous entry pointing at this entity to avoid dupes
            for old_alias in [a for a, eid in aliases.items() if eid == e["entity_id"]]:
                aliases.pop(old_alias, None)
            aliases[alias.lower()] = e["entity_id"]

    return {"url": url, "token": token, "entity_aliases": aliases}


def run() -> None:
    """Interactive wizard entry point."""
    print("=== Cài đặt Nemo ơi ===\n")
    cfg = _load_existing()

    # Wake threshold
    print("--- Wake word ---")
    cur = cfg["wake"].get("threshold", 0.55)
    val = _prompt("Ngưỡng phát hiện (0-1, càng thấp càng nhạy)", str(cur))
    try:
        cfg["wake"]["threshold"] = float(val)
    except ValueError:
        print(f"(giá trị không hợp lệ, giữ {cur})")

    # TTS voice choice (only Piper Vi in v0; leave the seam)
    print("\n--- Text-to-speech ---")
    voice = cfg["tts"].get("voice", "vi_VN-vais1000-medium")
    print(f"Giọng đọc: {voice} (v0 chỉ hỗ trợ 1 giọng)")

    workaround = cfg["tts"].get("english_workaround", "spell")
    val = _prompt("Xử lý tiếng Anh trong câu (spell/off)", workaround)
    if val in ("spell", "off"):
        cfg["tts"]["english_workaround"] = val

    # Home Assistant
    if _prompt_bool("\nCấu hình Home Assistant?", default=bool(cfg["home_assistant"].get("url"))):
        cfg["home_assistant"] = _setup_home_assistant(cfg["home_assistant"])

    _write_config(cfg)
    print("\nBạn có thể chạy ./assistant.sh để dùng ngay.")


if __name__ == "__main__":
    try:
        run()
    except (KeyboardInterrupt, EOFError):
        print("\n(hủy)")
        sys.exit(1)
