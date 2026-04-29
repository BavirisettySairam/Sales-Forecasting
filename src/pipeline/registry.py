from __future__ import annotations

import json
from pathlib import Path

from src.utils.logger import logger

_REGISTRY_FILE = Path("models") / "registry.json"


def _load_registry() -> dict:
    if _REGISTRY_FILE.exists():
        with open(_REGISTRY_FILE) as f:
            return json.load(f)
    return {"models": [], "champion": None}


def _save_registry(reg: dict) -> None:
    _REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_REGISTRY_FILE, "w") as f:
        json.dump(reg, f, indent=2, default=str)


def save_model(
    name: str,
    version: str,
    path: str,
    metrics: dict,
    is_champion: bool = False,
    state: str | None = None,
) -> None:
    reg = _load_registry()
    entry = {
        "name": name,
        "version": version,
        "path": path,
        "metrics": metrics,
        "is_champion": is_champion,
        "state": state,
    }
    reg["models"] = [
        m for m in reg["models"] if not (m["name"] == name and m["version"] == version)
    ]
    reg["models"].append(entry)

    if is_champion:
        reg["champion"] = {"name": name, "version": version, "path": path}
        for m in reg["models"]:
            if m["name"] != name or m["version"] != version:
                m["is_champion"] = False

    _save_registry(reg)
    logger.info("Model registered", name=name, version=version, champion=is_champion)


def load_model(name: str, version: str | None = None) -> dict | None:
    reg = _load_registry()
    candidates = [m for m in reg["models"] if m["name"] == name]
    if not candidates:
        return None
    if version:
        candidates = [m for m in candidates if m["version"] == version]
    return candidates[-1] if candidates else None


def get_champion() -> dict | None:
    reg = _load_registry()
    if not reg.get("champion"):
        return None
    champ = reg["champion"]
    return load_model(champ["name"], champ["version"])


def list_models() -> list[dict]:
    return _load_registry().get("models", [])
