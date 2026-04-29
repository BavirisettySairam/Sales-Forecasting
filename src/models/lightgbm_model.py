from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src.features.engineering import get_feature_columns
from src.models.base import BaseForecaster
from src.utils.logger import logger

optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBMForecaster(BaseForecaster):
    def __init__(self, config: dict) -> None:
        super().__init__("lightgbm", config)
        self._feature_cols: list[str] = []
        self._model_lower: lgb.LGBMRegressor | None = None
        self._model_upper: lgb.LGBMRegressor | None = None
        self._last_known: pd.DataFrame | None = None

    @staticmethod
    def _ensure_feature_columns(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        if "state" not in df.columns:
            df["state"] = "national"
        if "category" not in df.columns:
            df["category"] = "all"
        if "date" not in df.columns:
            raise ValueError("DataFrame must have a 'date' column or DatetimeIndex")
        return df

    def _build_Xy(self, data: pd.DataFrame, target_col: str):
        from src.features.engineering import create_features

        df = self._ensure_feature_columns(data)
        featured = create_features(df, self.config)
        self._feature_cols = get_feature_columns(featured)
        X = featured[self._feature_cols]  # keep DataFrame so feature names are consistent
        y = featured[target_col].values
        return X, y

    def _objective(self, trial: optuna.Trial, X, y):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        }
        from sklearn.metrics import make_scorer, mean_absolute_percentage_error
        from sklearn.model_selection import cross_val_score

        model = lgb.LGBMRegressor(
            **params, objective="regression", random_state=42, verbose=-1
        )
        scores = cross_val_score(
            model, X, y, cv=3, scoring=make_scorer(mean_absolute_percentage_error)
        )
        return scores.mean()

    def fit(self, train_data: pd.DataFrame, target_col: str = "total") -> None:
        cfg = self.config.get("lightgbm", {})
        n_trials = cfg.get("n_trials", 50)

        X, y = self._build_Xy(train_data, target_col)
        self._last_known = train_data.copy()

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(
            lambda t: self._objective(t, X, y),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best = study.best_params
        logger.info("LightGBM Optuna done", best_mape=study.best_value, params=best)

        alpha = cfg.get("quantile_alpha", 0.95)
        self.model = lgb.LGBMRegressor(
            **best, objective="regression", random_state=42, verbose=-1
        )
        self.model.fit(X, y)

        self._model_lower = lgb.LGBMRegressor(
            **best, objective="quantile", alpha=1 - alpha, random_state=42, verbose=-1
        )
        self._model_lower.fit(X, y)

        self._model_upper = lgb.LGBMRegressor(
            **best, objective="quantile", alpha=alpha, random_state=42, verbose=-1
        )
        self._model_upper.fit(X, y)

        self.is_fitted = True

    def _recursive_predict(
        self, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from src.features.engineering import create_features

        history = self._ensure_feature_columns(self._last_known)
        preds, lowers, uppers = [], [], []

        for _ in range(horizon):
            featured = create_features(history, self.config)
            x_row = featured[self._feature_cols].iloc[[-1]].values
            p = float(self.model.predict(x_row)[0])
            lo = float(self._model_lower.predict(x_row)[0])
            hi = float(self._model_upper.predict(x_row)[0])
            preds.append(max(p, 0))
            lowers.append(max(lo, 0))
            uppers.append(max(hi, 0))

            next_date = history["date"].iloc[-1] + pd.offsets.Week(weekday=0)
            new_row = pd.DataFrame(
                {
                    "date": [next_date],
                    "total": [p],
                    "state": history["state"].iloc[-1],
                    "category": history["category"].iloc[-1],
                }
            )
            history = pd.concat([history, new_row], ignore_index=True)

        return np.array(preds), np.array(lowers), np.array(uppers)

    def predict(self, horizon: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        if isinstance(self._last_known.index, pd.DatetimeIndex):
            last_date = self._last_known.index[-1]
        else:
            last_date = self._last_known["date"].iloc[-1]

        dates = pd.date_range(
            start=last_date + pd.offsets.Week(weekday=0), periods=horizon, freq="W-MON"
        )
        preds, lowers, uppers = self._recursive_predict(horizon)

        return pd.DataFrame(
            {
                "date": dates,
                "predicted_value": preds,
                "lower_bound": lowers,
                "upper_bound": uppers,
            }
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "model_lower": self._model_lower,
                "model_upper": self._model_upper,
                "feature_cols": self._feature_cols,
                "last_known": self._last_known,
            },
            path,
        )
        logger.info("LightGBM saved", path=path)

    def load(self, path: str) -> None:
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self._model_lower = artifact["model_lower"]
        self._model_upper = artifact["model_upper"]
        self._feature_cols = artifact["feature_cols"]
        self._last_known = artifact["last_known"]
        self.is_fitted = True
        logger.info("LightGBM loaded", path=path)
