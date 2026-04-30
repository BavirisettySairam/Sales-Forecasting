"""Helpers for reading the nested training YAML consistently."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_FORECAST_HORIZON = 8
MODEL_NAMES = ("sarima", "prophet", "xgboost", "lightgbm", "lstm")


def preprocessing_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return preprocessing settings from either nested or flat config."""
    return deepcopy(config.get("preprocessing", config))


def feature_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return feature settings from either nested or flat config."""
    cfg = deepcopy(config.get("features", {}))
    for key in ("lag_periods", "rolling_windows", "rolling_stats", "holiday_country"):
        if key in config:
            cfg[key] = deepcopy(config[key])
    return cfg


def model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten feature/model sections into the shape forecasters expect."""
    cfg = feature_config(config)
    models_section = config.get("models", {})
    for name in MODEL_NAMES:
        if name in models_section:
            cfg[name] = deepcopy(models_section[name])
        elif name in config:
            cfg[name] = deepcopy(config[name])
    return cfg


def training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return training settings from either nested or flat config."""
    return deepcopy(config.get("training", config))


def forecast_horizon(config: dict[str, Any], override: int | None = None) -> int:
    if override is not None:
        return int(override)
    cfg = training_config(config)
    return int(cfg.get("forecast_horizon", DEFAULT_FORECAST_HORIZON))


def cv_folds(config: dict[str, Any], override: int | None = None) -> int:
    if override is not None:
        return int(override)
    cfg = training_config(config)
    return int(cfg.get("cv_folds", 3))


def cv_min_train_weeks(config: dict[str, Any], default: int = 30) -> int:
    cfg = training_config(config)
    return int(cfg.get("cv_min_train_weeks", default))


def enabled_models(config: dict[str, Any], available: list[str]) -> list[str]:
    """Return enabled model names, preserving registry order."""
    models_section = config.get("models", {})
    if not models_section:
        return available
    return [
        name for name in available if models_section.get(name, {}).get("enabled", True)
    ]


def configured_data_path(config: dict[str, Any], fallback: str = "data.csv") -> str:
    data_section = config.get("data", {})
    raw_path = data_section.get("raw_path") if isinstance(data_section, dict) else None
    if raw_path and Path(raw_path).exists():
        return str(raw_path)
    return fallback


def discover_states(data_path: str) -> list[str]:
    """Discover region/state names from the raw dataset."""
    df = pd.read_csv(data_path)
    state_col = next(
        (c for c in df.columns if c.strip().lower() == "state"),
        df.columns[0],
    )
    return sorted(df[state_col].dropna().astype(str).str.strip().unique().tolist())
