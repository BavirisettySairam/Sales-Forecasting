"""Training orchestrator — run as a script or import run_training()."""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.pipeline.evaluate import time_series_cv
from src.pipeline.registry import save_model
from src.pipeline.select import rank_models, select_best_model
from src.preprocessing.pipeline import run as run_pipeline
from src.utils.logger import logger

MODEL_REGISTRY: dict[str, type] = {}


def _register_models():
    global MODEL_REGISTRY
    from src.models.sarima_model import SARIMAForecaster
    from src.models.prophet_model import ProphetForecaster
    from src.models.xgboost_model import XGBoostForecaster
    from src.models.lightgbm_model import LightGBMForecaster
    from src.models.lstm_model import LSTMForecaster

    MODEL_REGISTRY = {
        "sarima": SARIMAForecaster,
        "prophet": ProphetForecaster,
        "xgboost": XGBoostForecaster,
        "lightgbm": LightGBMForecaster,
        "lstm": LSTMForecaster,
    }


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_training(
    data_path: str,
    config_path: str = "config/training_config.yaml",
    models_to_run: list[str] | None = None,
    state_filter: str | None = None,
    output_dir: str = "models",
    horizon: int = 12,
    cv_splits: int = 5,
    skip_cv: bool = False,
) -> dict:
    _register_models()

    config = _load_config(config_path)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    logger.info("Training run started", run_id=run_id, version=version)

    # Preprocess
    clean_df = run_pipeline(data_path, config=config)

    # Filter by state if requested
    if state_filter and "state" in clean_df.columns:
        clean_df = clean_df[clean_df["state"] == state_filter]
        logger.info("State filter applied", state=state_filter, rows=len(clean_df))

    if len(clean_df) < 52:
        raise ValueError(f"Insufficient data after filtering: {len(clean_df)} rows")

    target_models = models_to_run or list(MODEL_REGISTRY.keys())
    cv_results: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}

    for model_name in target_models:
        if model_name not in MODEL_REGISTRY:
            logger.warning("Unknown model skipped", model=model_name)
            continue

        model_cls = MODEL_REGISTRY[model_name]
        logger.info("Fitting model", model=model_name)

        try:
            if skip_cv:
                cv_results[model_name] = {"mape": 0.0, "rmse": 0.0, "mae": 0.0, "n_folds": 0}
            else:
                cv_results[model_name] = time_series_cv(
                    model_cls=model_cls,
                    model_config=config,
                    data=clean_df,
                    target_col="total",
                    n_splits=cv_splits,
                    horizon=horizon,
                )

            # Fit on full data for final model
            forecaster = model_cls(config)
            forecaster.fit(clean_df, "total")
            fitted_models[model_name] = forecaster
            logger.info("Model fitted", model=model_name)

        except Exception as exc:
            logger.error("Model training failed", model=model_name, error=str(exc))
            cv_results[model_name] = {"mape": float("inf"), "rmse": float("inf"), "mae": float("inf"), "n_folds": 0}

    if not fitted_models:
        raise RuntimeError("All models failed to train")

    champion_name = select_best_model({k: v for k, v in cv_results.items() if k in fitted_models})
    rankings = rank_models({k: v for k, v in cv_results.items() if k in fitted_models})

    # Save all fitted models and register them
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, forecaster in fitted_models.items():
        is_champ = model_name == champion_name
        model_path = str(out_dir / f"{model_name}_{version}")
        forecaster.save(model_path)
        save_model(
            name=model_name,
            version=version,
            path=model_path,
            metrics=cv_results[model_name],
            is_champion=is_champ,
            state=state_filter,
        )

    logger.info(
        "Training complete",
        run_id=run_id,
        champion=champion_name,
        version=version,
        rankings=[f"{r['model']}(MAPE={r['mape']})" for r in rankings],
    )

    return {
        "run_id": run_id,
        "version": version,
        "champion": champion_name,
        "rankings": rankings,
        "cv_results": cv_results,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument("--data", required=True, help="Path to raw CSV data")
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--models", nargs="+", help="Model names to train (default: all)")
    parser.add_argument("--state", default=None, help="Filter to a single state")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation (faster)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    result = run_training(
        data_path=args.data,
        config_path=args.config,
        models_to_run=args.models,
        state_filter=args.state,
        output_dir=args.output_dir,
        horizon=args.horizon,
        cv_splits=args.cv_splits,
        skip_cv=args.skip_cv,
    )
    print(f"\nChampion: {result['champion']} (version {result['version']})")
    for r in result["rankings"]:
        print(f"  #{r['rank']} {r['model']}: MAPE={r['mape']:.2f}% RMSE={r['rmse']:.2f}")
    sys.exit(0)
