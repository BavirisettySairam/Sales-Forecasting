from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import logger


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else float("inf")
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))
    return {"mape": round(mape, 4), "rmse": round(rmse, 4), "mae": round(mae, 4)}


def time_series_cv(
    model_cls,
    model_config: dict,
    data: pd.DataFrame,
    target_col: str = "total",
    n_splits: int = 5,
    horizon: int = 12,
    min_train_size: int = 52,
) -> dict[str, float]:
    """
    Expanding window cross-validation (no future leakage).
    Each fold trains on all data up to the split point and evaluates on the next `horizon` weeks.
    Returns averaged MAPE, RMSE, MAE across folds.
    """
    n = len(data)
    fold_metrics: list[dict[str, float]] = []

    step = max((n - min_train_size - horizon) // n_splits, 1)

    for fold in range(n_splits):
        train_end = min_train_size + fold * step
        test_end = train_end + horizon

        if test_end > n:
            break

        train = data.iloc[:train_end]
        test = data.iloc[train_end:test_end]

        try:
            forecaster = model_cls(model_config)
            forecaster.fit(train, target_col)
            forecast_df = forecaster.predict(horizon)
            actual = test[target_col].values[: len(forecast_df)]
            predicted = forecast_df["predicted_value"].values[: len(actual)]
            metrics = calculate_metrics(actual, predicted)
            fold_metrics.append(metrics)
            logger.debug("CV fold done", fold=fold + 1, **metrics)
        except Exception as exc:
            logger.warning("CV fold failed", fold=fold + 1, error=str(exc))

    if not fold_metrics:
        return {"mape": float("inf"), "rmse": float("inf"), "mae": float("inf"), "n_folds": 0}

    avg = {
        "mape": round(float(np.mean([m["mape"] for m in fold_metrics])), 4),
        "rmse": round(float(np.mean([m["rmse"] for m in fold_metrics])), 4),
        "mae": round(float(np.mean([m["mae"] for m in fold_metrics])), 4),
        "n_folds": len(fold_metrics),
    }
    logger.info("CV complete", **avg)
    return avg
