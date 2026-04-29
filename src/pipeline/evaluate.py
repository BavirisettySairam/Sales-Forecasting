from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import logger


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mask = actual != 0
    mape = (
        float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
        if mask.any()
        else float("inf")
    )
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
    min_train_size: int = 110,
) -> dict[str, float]:
    """
    Expanding-window CV split by unique calendar dates (not row index).
    Multi-state data: trains on all states up to the cutoff date, evaluates
    on the national aggregate (sum across states) for the test window.
    """
    sorted_dates = sorted(data["date"].unique())
    n_dates = len(sorted_dates)

    if n_dates < min_train_size + horizon:
        logger.warning(
            "Not enough dates for CV",
            n_dates=n_dates,
            required=min_train_size + horizon,
        )
        return {"mape": float("inf"), "rmse": float("inf"), "mae": float("inf"), "n_folds": 0}

    step = max((n_dates - min_train_size - horizon) // n_splits, 1)
    fold_metrics: list[dict[str, float]] = []

    for fold in range(n_splits):
        train_end_idx = min_train_size + fold * step
        test_end_idx = train_end_idx + horizon
        if test_end_idx > n_dates:
            break

        cutoff = sorted_dates[train_end_idx - 1]
        test_end = sorted_dates[test_end_idx - 1]

        train = data[data["date"] <= cutoff].copy()
        test = data[(data["date"] > cutoff) & (data["date"] <= test_end)].copy()

        try:
            forecaster = model_cls(model_config)
            forecaster.fit(train, target_col)
            forecast_df = forecaster.predict(horizon)

            # Aggregate actual values to national level (sum across states per date)
            actual = (
                test.groupby("date")[target_col].sum().sort_index().values
            )
            predicted = forecast_df["predicted_value"].values
            n = min(len(actual), len(predicted))
            metrics = calculate_metrics(actual[:n], predicted[:n])
            fold_metrics.append(metrics)
            logger.info("CV fold done", fold=fold + 1, cutoff=str(cutoff)[:10], **metrics)
        except Exception as exc:
            logger.warning("CV fold failed", fold=fold + 1, error=str(exc))
            logger.exception("CV fold traceback")

    if not fold_metrics:
        return {
            "mape": float("inf"),
            "rmse": float("inf"),
            "mae": float("inf"),
            "n_folds": 0,
        }

    avg = {
        "mape": round(float(np.mean([m["mape"] for m in fold_metrics])), 4),
        "rmse": round(float(np.mean([m["rmse"] for m in fold_metrics])), 4),
        "mae": round(float(np.mean([m["mae"] for m in fold_metrics])), 4),
        "n_folds": len(fold_metrics),
    }
    logger.info("CV complete", **avg)
    return avg
