import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pmdarima import auto_arima

from src.models.base import BaseForecaster
from src.utils.logger import logger


class SARIMAForecaster(BaseForecaster):
    def __init__(self, config: dict) -> None:
        super().__init__("sarima", config)
        self._train_series: pd.Series | None = None

    def fit(self, train_data: pd.DataFrame, target_col: str = "total") -> None:
        cfg = self.config.get("sarima", {})
        # Aggregate across states → one value per date (SARIMA is univariate)
        if "date" in train_data.columns and "state" in train_data.columns and train_data["state"].nunique() > 1:
            self._train_series = (
                train_data.groupby("date")[target_col].sum().sort_index()
            )
        else:
            self._train_series = train_data[target_col].copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = auto_arima(
                self._train_series,
                seasonal=True,
                m=cfg.get("seasonal_period", 52),
                stepwise=cfg.get("stepwise", True),
                suppress_warnings=True,
                error_action="ignore",
                information_criterion=cfg.get("information_criterion", "aic"),
                max_p=cfg.get("max_p", 3),
                max_q=cfg.get("max_q", 3),
                max_P=cfg.get("max_P", 2),
                max_Q=cfg.get("max_Q", 2),
                D=cfg.get("D", 1),
                trace=False,
            )

        self.is_fitted = True
        logger.info(
            "SARIMA fitted",
            order=str(self.model.order),
            seasonal=str(self.model.seasonal_order),
        )

    def predict(self, horizon: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        alpha = self.config.get("sarima", {}).get("alpha", 0.05)
        fc, ci = self.model.predict(
            n_periods=horizon, return_conf_int=True, alpha=alpha
        )

        if isinstance(self._train_series.index, pd.DatetimeIndex):
            last_date = self._train_series.index[-1]
            dates = pd.date_range(
                start=last_date + pd.offsets.Week(weekday=0),
                periods=horizon,
                freq="W-MON",
            )
        else:
            dates = pd.date_range(
                start=pd.Timestamp("today"), periods=horizon, freq="W-MON"
            )

        return pd.DataFrame(
            {
                "date": dates,
                "predicted_value": np.maximum(fc, 0),
                "lower_bound": np.maximum(ci[:, 0], 0),
                "upper_bound": np.maximum(ci[:, 1], 0),
            }
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "train_series": self._train_series}, path)
        logger.info("SARIMA saved", path=path)

    def load(self, path: str) -> None:
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self._train_series = artifact["train_series"]
        self.is_fitted = True
        logger.info("SARIMA loaded", path=path)
