import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.utils.logger import logger


class ProphetForecaster(BaseForecaster):
    def __init__(self, config: dict) -> None:
        super().__init__("prophet", config)
        self._last_date: pd.Timestamp | None = None

    def _make_prophet(self):
        # Import here to suppress Prophet's stdout noise at module level
        import prophet  # noqa: F401

        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        os.environ.setdefault("CMDSTAN_VERBOSITY", "0")

        import holidays as hol
        from prophet import Prophet

        cfg = self.config.get("prophet", {})

        us_holidays = pd.DataFrame(
            [
                {"holiday": name, "ds": pd.Timestamp(date)}
                for date, name in hol.US(years=range(2015, 2030)).items()
            ]
        )

        return Prophet(
            yearly_seasonality=cfg.get("yearly_seasonality", True),
            weekly_seasonality=cfg.get("weekly_seasonality", False),
            daily_seasonality=False,
            seasonality_mode=cfg.get("seasonality_mode", "multiplicative"),
            interval_width=cfg.get("interval_width", 0.95),
            holidays=us_holidays,
        )

    def fit(self, train_data: pd.DataFrame, target_col: str = "total") -> None:
        df = (
            train_data.reset_index()
            if isinstance(train_data.index, pd.DatetimeIndex)
            else train_data.copy()
        )
        date_col = "date" if "date" in df.columns else df.columns[0]
        # Aggregate across states → one observation per date (Prophet is univariate)
        prophet_df = (
            df.groupby(date_col)[target_col]
            .sum()
            .reset_index()
            .rename(columns={date_col: "ds", target_col: "y"})
        )
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        self._last_date = prophet_df["ds"].max()
        self.model = self._make_prophet()

        import logging as _logging

        _logging.getLogger("prophet").setLevel(_logging.ERROR)
        _logging.getLogger("cmdstanpy").setLevel(_logging.ERROR)
        self.model.fit(prophet_df)

        self.is_fitted = True
        logger.info("Prophet fitted", last_date=str(self._last_date))

    def predict(self, horizon: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        future = self.model.make_future_dataframe(
            periods=horizon, freq="W-MON", include_history=False
        )
        forecast = self.model.predict(future)

        return pd.DataFrame(
            {
                "date": forecast["ds"].values,
                "predicted_value": np.maximum(forecast["yhat"].values, 0),
                "lower_bound": np.maximum(forecast["yhat_lower"].values, 0),
                "upper_bound": np.maximum(forecast["yhat_upper"].values, 0),
            }
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "last_date": self._last_date}, path)
        logger.info("Prophet saved", path=path)

    def load(self, path: str) -> None:
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self._last_date = artifact["last_date"]
        self.is_fitted = True
        logger.info("Prophet loaded", path=path)
