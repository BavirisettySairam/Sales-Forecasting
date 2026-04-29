from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class every forecasting model must implement."""

    def __init__(self, name: str, config: dict) -> None:
        self.name = name
        self.config = config
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, target_col: str = "total") -> None:
        """Train the model on train_data."""

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
            date             – forecast date (weekly Monday-anchored)
            predicted_value  – point forecast
            lower_bound      – 95% CI lower
            upper_bound      – 95% CI upper
        One row per week, `horizon` rows total.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialise model artifact(s) to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Deserialise model artifact(s) from disk."""

    def get_name(self) -> str:
        return self.name
