"""
Preprocessing orchestrator.

Flow:
  load raw CSV
    → validate raw schema (Pandera)
    → clean (dedup, fill gaps, impute, outliers, weekly aggregation)
    → validate clean schema (Pandera)
    → save processed CSV (optional)
    → return weekly DataFrame
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.preprocessing.cleaner import clean, load_raw
from src.preprocessing.validator import validate_clean, validate_raw


def run(
    raw_path: str,
    config: dict,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Args:
        raw_path:    Path to the raw CSV file.
        config:      Dict from training_config.yaml['preprocessing'].
        output_path: If provided, saves the cleaned weekly CSV here.

    Returns:
        Weekly-aggregated, validated DataFrame ready for feature engineering.
    """
    # ------------------------------------------------------------------ #
    # 1. Load
    # ------------------------------------------------------------------ #
    logger.info(f"[Pipeline] Loading raw data from '{raw_path}'")
    df = load_raw(raw_path)
    logger.info(f"[Pipeline] Raw rows loaded: {len(df)}")

    # ------------------------------------------------------------------ #
    # 2. Validate raw schema
    # ------------------------------------------------------------------ #
    logger.info("[Pipeline] Validating raw schema")
    try:
        validate_raw(df)
        logger.info("[Pipeline] Raw schema valid")
    except Exception as exc:
        logger.warning(f"[Pipeline] Raw schema validation warnings: {exc}")
        # Non-fatal: log but continue; cleaner handles most format issues

    # ------------------------------------------------------------------ #
    # 3. Clean
    # ------------------------------------------------------------------ #
    fill_method = config.get("fill_method", "interpolate")
    outlier_method = config.get("outlier_method", "iqr")
    outlier_threshold = float(config.get("outlier_threshold", 1.5))

    df_clean = clean(
        df,
        fill_method=fill_method,
        outlier_method=outlier_method,
        outlier_threshold=outlier_threshold,
    )
    logger.info(f"[Pipeline] After cleaning: {len(df_clean)} weekly rows, {df_clean['state'].nunique()} states")

    # ------------------------------------------------------------------ #
    # 4. Validate clean schema
    # ------------------------------------------------------------------ #
    logger.info("[Pipeline] Validating clean schema")
    try:
        validate_clean(df_clean)
        logger.info("[Pipeline] Clean schema valid")
    except Exception as exc:
        logger.error(f"[Pipeline] Clean schema validation FAILED: {exc}")
        raise

    # ------------------------------------------------------------------ #
    # 5. Save (optional)
    # ------------------------------------------------------------------ #
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        logger.info(f"[Pipeline] Saved processed data to '{output_path}'")

    # ------------------------------------------------------------------ #
    # 6. Summary log
    # ------------------------------------------------------------------ #
    date_range = f"{df_clean['date'].min().date()} → {df_clean['date'].max().date()}"
    states = sorted(df_clean["state"].unique().tolist())
    logger.info(
        f"[Pipeline] Summary | dates: {date_range} | states ({len(states)}): {states[:5]}{'...' if len(states) > 5 else ''}"
    )

    return df_clean
