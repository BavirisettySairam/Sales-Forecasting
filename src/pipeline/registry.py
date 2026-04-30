from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from src.utils.logger import logger

_REGISTRY_FILE = Path("models") / "registry.json"
_LOCK_FILE = _REGISTRY_FILE.with_suffix(".lock")


def _state_key(state: str | None) -> str:
    return "__global__" if state is None else state.strip().title()


@contextmanager
def _registry_lock(timeout_seconds: float = 30.0):
    """Small cross-process lock for concurrent per-state training writes."""
    _REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(_LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            if time.monotonic() - start > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for registry lock: {_LOCK_FILE}")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.remove(_LOCK_FILE)
        except FileNotFoundError:
            pass


def _load_registry() -> dict:
    if _REGISTRY_FILE.exists():
        with open(_REGISTRY_FILE) as f:
            reg = json.load(f)
            reg.setdefault("models", [])
            reg.setdefault("champion", None)
            reg.setdefault("champions", {})
            return reg
    return {"models": [], "champion": None, "champions": {}}


def _save_registry(reg: dict) -> None:
    _REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _REGISTRY_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2, default=str)
    os.replace(tmp, _REGISTRY_FILE)


def save_model(
    name: str,
    version: str,
    path: str | None,
    metrics: dict,
    is_champion: bool = False,
    state: str | None = None,
) -> None:
    with _registry_lock():
        reg = _load_registry()
        state_key = _state_key(state)
        entry = {
            "name": name,
            "version": version,
            "path": path,
            "metrics": metrics,
            "is_champion": is_champion,
            "state": state.strip().title() if state else None,
        }
        reg["models"] = [
            m
            for m in reg["models"]
            if not (
                m["name"] == name
                and m["version"] == version
                and _state_key(m.get("state")) == state_key
            )
        ]
        reg["models"].append(entry)

        if is_champion:
            champion_ref = {
                "name": name,
                "version": version,
                "path": path,
                "state": entry["state"],
            }
            if state is None:
                reg["champion"] = champion_ref
            else:
                reg.setdefault("champions", {})[state_key] = champion_ref

            for m in reg["models"]:
                if _state_key(m.get("state")) != state_key:
                    continue
                m["is_champion"] = m["name"] == name and m["version"] == version

        _save_registry(reg)
    logger.info("Model registered", name=name, version=version, champion=is_champion)


def load_model(
    name: str, version: str | None = None, state: str | None = None
) -> dict | None:
    reg = _load_registry()
    candidates = [m for m in reg["models"] if m["name"] == name]
    if not candidates:
        return None
    if version:
        candidates = [m for m in candidates if m["version"] == version]
    if state is not None:
        state_key = _state_key(state)
        candidates = [m for m in candidates if _state_key(m.get("state")) == state_key]
    return candidates[-1] if candidates else None


def get_champion(state: str | None = None) -> dict | None:
    reg = _load_registry()
    if state is not None:
        state_key = _state_key(state)
        champ = reg.get("champions", {}).get(state_key)
        if not champ:
            return None
        return load_model(champ["name"], champ["version"], state=state)

    if not reg.get("champion"):
        return None
    champ = reg["champion"]
    return load_model(champ["name"], champ["version"], state=None)


def list_models(state: str | None = None) -> list[dict]:
    models = _load_registry().get("models", [])
    if state is None:
        return models
    state_key = _state_key(state)
    return [m for m in models if _state_key(m.get("state")) == state_key]


def list_trained_states() -> list[str]:
    return sorted(
        {
            m["state"].strip().title()
            for m in _load_registry().get("models", [])
            if m.get("state")
        }
    )
