"""Training orchestrator — run as a script or import run_training()."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.config.training import (
    configured_data_path,
    cv_folds,
    cv_min_train_weeks,
    discover_states,
    enabled_models,
    forecast_horizon,
    model_config,
    preprocessing_config,
)
from src.pipeline.evaluate import (
    calculate_metrics,
    time_series_cv,
    train_val_test_split,
)
from src.pipeline.registry import save_model
from src.pipeline.select import rank_models, select_best_model
from src.preprocessing.pipeline import run as run_pipeline
from src.utils.logger import logger

MODEL_REGISTRY: dict[str, type] = {}


def _register_models():
    global MODEL_REGISTRY
    from src.models.lightgbm_model import LightGBMForecaster
    from src.models.lstm_model import LSTMForecaster
    from src.models.prophet_model import ProphetForecaster
    from src.models.sarima_model import SARIMAForecaster
    from src.models.xgboost_model import XGBoostForecaster

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
    horizon: int | None = None,
    cv_splits: int | None = None,
    skip_cv: bool = False,
) -> dict:
    _register_models()

    raw_config = _load_config(config_path)
    pre_cfg = preprocessing_config(raw_config)
    model_cfg = model_config(raw_config)
    horizon = forecast_horizon(raw_config, horizon)
    cv_splits = cv_folds(raw_config, cv_splits)
    min_train_weeks = cv_min_train_weeks(raw_config)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    state_label = state_filter.strip().title() if state_filter else None
    logger.info(
        "Training run started",
        run_id=run_id,
        version=version,
        state=state_label,
        horizon=horizon,
    )

    # Preprocess
    clean_df = run_pipeline(data_path, config=pre_cfg)

    # Filter by state if requested
    if state_label and "state" in clean_df.columns:
        clean_df = clean_df[clean_df["state"].str.strip().str.title() == state_label]
        logger.info("State filter applied", state=state_label, rows=len(clean_df))

    n_dates = (
        len(clean_df["date"].unique()) if "date" in clean_df.columns else len(clean_df)
    )
    if n_dates < 52:
        raise ValueError(f"Insufficient data after filtering: {n_dates} unique dates")

    # ── 70:15:15 chronological split ──────────────────────────────────────────
    # train (70%) + val (15%) used for CV and model selection.
    # test (15%) held out for final honest evaluation — never seen during fitting.
    # Production model is then re-fitted on ALL data for best forecasting.
    train_df, val_df, test_df = train_val_test_split(clean_df)
    dev_df = pd.concat([train_df, val_df]).sort_values("date").reset_index(drop=True)

    target_models = models_to_run or enabled_models(
        raw_config, list(MODEL_REGISTRY.keys())
    )
    cv_results: dict[str, dict] = {}
    successful_models: set[str] = set()

    for model_name in target_models:
        if model_name not in MODEL_REGISTRY:
            logger.warning("Unknown model skipped", model=model_name)
            continue

        model_cls = MODEL_REGISTRY[model_name]
        logger.info("Fitting model", model=model_name)

        try:
            # ── Step 1: CV on train+val (dev set = 80%) ──────────────────────
            if skip_cv:
                cv_results[model_name] = {
                    "mape": float("inf"),
                    "rmse": float("inf"),
                    "mae": float("inf"),
                    "n_folds": 0,
                }
            else:
                cv_results[model_name] = time_series_cv(
                    model_cls=model_cls,
                    model_config=model_cfg,
                    data=dev_df,
                    target_col="total",
                    n_splits=cv_splits,
                    horizon=horizon,
                    min_train_size=min_train_weeks,
                )

            # ── Step 2: Fit on dev set, evaluate on held-out test (20%) ─────
            dev_forecaster = model_cls(model_cfg)
            dev_forecaster.fit(dev_df, "total")

            if len(test_df) > 0:
                test_dates = sorted(test_df["date"].unique())
                test_horizon = len(test_dates)
                fc_df = dev_forecaster.predict(test_horizon)
                actual_test = (
                    test_df.groupby("date")["total"].sum().sort_index().values
                )
                n = min(len(actual_test), len(fc_df))
                test_m = calculate_metrics(
                    actual_test[:n], fc_df["predicted_value"].values[:n]
                )
                cv_results[model_name]["test_mape"] = test_m["mape"]
                cv_results[model_name]["test_rmse"] = test_m["rmse"]
                cv_results[model_name]["test_mae"] = test_m["mae"]
                logger.info(
                    "Test-set evaluation",
                    model=model_name,
                    test_dates=test_horizon,
                    **test_m,
                )

            successful_models.add(model_name)

        except Exception as exc:
            logger.exception("Model training failed", model=model_name, error=str(exc))
            cv_results[model_name] = {
                "mape": float("inf"),
                "rmse": float("inf"),
                "mae": float("inf"),
                "n_folds": 0,
            }

    if not successful_models:
        raise RuntimeError("All models failed to train")

    # Select champion by CV MAPE (most robust), break ties with test_mape if available
    valid_cv = {k: v for k, v in cv_results.items() if k in successful_models}
    champion_name = select_best_model(valid_cv)
    rankings = rank_models(valid_cv)

    # Register every candidate's metrics, then fit/save only the full-data champion.
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in successful_models:
        save_model(
            name=model_name,
            version=version,
            path=None,
            metrics=cv_results[model_name],
            is_champion=False,
            state=state_label,
        )

    champion_cls = MODEL_REGISTRY[champion_name]
    champion_model = champion_cls(model_cfg)
    champion_model.fit(clean_df, "total")
    state_slug = f"_{_slug_state(state_label)}" if state_label else ""
    champion_path = str(out_dir / f"{champion_name}{state_slug}_{version}")
    champion_model.save(champion_path)
    save_model(
        name=champion_name,
        version=version,
        path=champion_path,
        metrics=cv_results[champion_name],
        is_champion=True,
        state=state_label,
    )
    logger.info(
        "Champion production model fitted on full regional dataset",
        model=champion_name,
        state=state_label,
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
        "state": state_label,
        "horizon": horizon,
        "rankings": rankings,
        "cv_results": cv_results,
    }


def _slug_state(state: str | None) -> str:
    if not state:
        return "global"
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in state).strip("_")


def run_training_all_states(
    data_path: str | None = None,
    config_path: str = "config/training_config.yaml",
    models_to_run: list[str] | None = None,
    states: list[str] | None = None,
    output_dir: str = "models",
    horizon: int | None = None,
    cv_splits: int | None = None,
    skip_cv: bool = False,
) -> dict:
    raw_config = _load_config(config_path)
    data_path = data_path or configured_data_path(raw_config)
    target_states = states or discover_states(data_path)

    results: list[dict] = []
    failures: list[dict] = []
    for i, state in enumerate(target_states, 1):
        state_label = state.strip().title()
        logger.info(
            "Training state",
            state=state_label,
            index=i,
            total=len(target_states),
        )
        try:
            results.append(
                run_training(
                    data_path=data_path,
                    config_path=config_path,
                    models_to_run=models_to_run,
                    state_filter=state_label,
                    output_dir=output_dir,
                    horizon=horizon,
                    cv_splits=cv_splits,
                    skip_cv=skip_cv,
                )
            )
        except Exception as exc:
            logger.exception("State training failed", state=state_label, error=str(exc))
            failures.append({"state": state_label, "error": str(exc)})

    return {
        "states_requested": len(target_states),
        "states_succeeded": len(results),
        "states_failed": len(failures),
        "results": results,
        "failures": failures,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument("--data", required=True, help="Path to raw CSV data")
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument(
        "--models", nargs="+", help="Model names to train (default: all)"
    )
    parser.add_argument("--state", default=None, help="Filter to a single state")
    parser.add_argument(
        "--states",
        nargs="+",
        help="Train specific states only (space-separated, e.g. --states California Texas)",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Train a separate champion model for every state in the dataset",
    )
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--cv-splits", type=int, default=None)
    parser.add_argument(
        "--skip-cv", action="store_true", help="Skip cross-validation (faster)"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    if args.all_states or args.states:
        result = run_training_all_states(
            data_path=args.data,
            config_path=args.config,
            models_to_run=args.models,
            states=args.states or None,
            output_dir=args.output_dir,
            horizon=args.horizon,
            cv_splits=args.cv_splits,
            skip_cv=args.skip_cv,
        )
        for r in result["results"]:
            print(f"[{r['state']}] Champion: {r['champion']}")
        if result["failures"]:
            print("\nFailures:")
            for failure in result["failures"]:
                print(f"  {failure['state']}: {failure['error']}")
        print("\nAll states done.")
        if result["failures"]:
            sys.exit(1)
    else:
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
            print(
                f"  #{r['rank']} {r['model']}: "
                f"MAPE={r['mape']:.2f}% RMSE={r['rmse']:.2f}"
            )
    sys.exit(0)
