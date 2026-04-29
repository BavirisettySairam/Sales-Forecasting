"""Tests for all 5 model implementations — fit, predict shape, CI, save/load."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.models.lightgbm_model import LightGBMForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.sarima_model import SARIMAForecaster
from src.models.xgboost_model import XGBoostForecaster

HORIZON = 4
EXPECTED_COLS = {"date", "predicted_value", "lower_bound", "upper_bound"}


@pytest.fixture
def train_df():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-06", periods=60, freq="W-MON")
    vals = 100_000 + np.arange(60) * 500 + rng.normal(0, 5000, 60)
    return pd.DataFrame(
        {
            "date": dates,
            "total": np.maximum(vals, 0),
            "state": "national",
            "category": "all",
        }
    )


@pytest.fixture
def train_df_indexed(train_df):
    return train_df.set_index("date")


@pytest.fixture
def ml_config():
    return {
        "lag_periods": [1, 2, 4],
        "rolling_windows": [4],
        "rolling_stats": ["mean"],
        "holiday_country": "US",
        "xgboost": {"n_trials": 3, "quantile_alpha": 0.95},
        "lightgbm": {"n_trials": 3, "quantile_alpha": 0.95},
    }


@pytest.fixture
def lstm_config():
    return {
        "lstm": {
            "sequence_length": 8,
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "mc_passes": 5,
        },
    }


@pytest.fixture
def sarima_config():
    return {
        "sarima": {
            "seasonal_period": 4,
            "stepwise": True,
            "max_p": 2,
            "max_q": 2,
            "max_P": 1,
            "max_Q": 1,
            "D": 1,
            "alpha": 0.05,
        }
    }


@pytest.fixture
def prophet_config():
    return {
        "prophet": {
            "interval_width": 0.95,
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            "seasonality_mode": "additive",
        }
    }


def _assert_forecast_shape(fc: pd.DataFrame, horizon: int):
    assert (
        set(fc.columns) >= EXPECTED_COLS
    ), f"Missing columns: {EXPECTED_COLS - set(fc.columns)}"
    assert len(fc) == horizon
    assert (fc["predicted_value"] >= 0).all()
    assert (fc["upper_bound"] >= fc["lower_bound"]).all()


# ── XGBoost ───────────────────────────────────────────────────────────────


def test_xgboost_fit_predict_shape(train_df, ml_config):
    m = XGBoostForecaster(ml_config)
    m.fit(train_df, "total")
    fc = m.predict(HORIZON)
    _assert_forecast_shape(fc, HORIZON)


def test_xgboost_is_fitted_flag(train_df, ml_config):
    m = XGBoostForecaster(ml_config)
    assert not m.is_fitted
    m.fit(train_df, "total")
    assert m.is_fitted


def test_xgboost_predict_before_fit_raises(ml_config):
    m = XGBoostForecaster(ml_config)
    with pytest.raises(RuntimeError):
        m.predict(4)


def test_xgboost_save_load_roundtrip(train_df, ml_config):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "xgb_model")
        m = XGBoostForecaster(ml_config)
        m.fit(train_df, "total")
        fc_before = m.predict(HORIZON)
        m.save(path)

        m2 = XGBoostForecaster(ml_config)
        m2.load(path)
        fc_after = m2.predict(HORIZON)
        assert list(fc_before["predicted_value"]) == pytest.approx(
            list(fc_after["predicted_value"]), rel=1e-4
        )


# ── LightGBM ──────────────────────────────────────────────────────────────


def test_lightgbm_fit_predict_shape(train_df, ml_config):
    m = LightGBMForecaster(ml_config)
    m.fit(train_df, "total")
    fc = m.predict(HORIZON)
    _assert_forecast_shape(fc, HORIZON)


def test_lightgbm_save_load_roundtrip(train_df, ml_config):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "lgb_model")
        m = LightGBMForecaster(ml_config)
        m.fit(train_df, "total")
        fc_before = m.predict(HORIZON)
        m.save(path)

        m2 = LightGBMForecaster(ml_config)
        m2.load(path)
        fc_after = m2.predict(HORIZON)
        assert list(fc_before["predicted_value"]) == pytest.approx(
            list(fc_after["predicted_value"]), rel=1e-4
        )


# ── LSTM ──────────────────────────────────────────────────────────────────


def test_lstm_fit_predict_shape(train_df_indexed, lstm_config):
    m = LSTMForecaster(lstm_config)
    m.fit(train_df_indexed, "total")
    fc = m.predict(HORIZON)
    _assert_forecast_shape(fc, HORIZON)


def test_lstm_ci_bounds_valid(train_df_indexed, lstm_config):
    m = LSTMForecaster(lstm_config)
    m.fit(train_df_indexed, "total")
    fc = m.predict(HORIZON)
    assert (fc["upper_bound"] >= fc["predicted_value"]).all()
    assert (fc["predicted_value"] >= fc["lower_bound"]).all()


def test_lstm_save_load_roundtrip(train_df_indexed, lstm_config):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "lstm_model")
        m = LSTMForecaster(lstm_config)
        m.fit(train_df_indexed, "total")
        m.save(path)

        m2 = LSTMForecaster(lstm_config)
        m2.load(path)
        assert m2.is_fitted
        fc = m2.predict(HORIZON)
        assert len(fc) == HORIZON


# ── Prophet ───────────────────────────────────────────────────────────────


def test_prophet_fit_predict_shape(train_df, prophet_config):
    m = ProphetForecaster(prophet_config)
    m.fit(train_df, "total")
    fc = m.predict(HORIZON)
    _assert_forecast_shape(fc, HORIZON)


def test_prophet_save_load_roundtrip(train_df, prophet_config):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "prophet_model")
        m = ProphetForecaster(prophet_config)
        m.fit(train_df, "total")
        m.save(path)

        m2 = ProphetForecaster(prophet_config)
        m2.load(path)
        assert m2.is_fitted
        fc = m2.predict(HORIZON)
        assert len(fc) == HORIZON


# ── SARIMA ────────────────────────────────────────────────────────────────


def test_sarima_fit_predict_shape(train_df_indexed, sarima_config):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMAForecaster(sarima_config)
        m.fit(train_df_indexed, "total")
        fc = m.predict(HORIZON)
    _assert_forecast_shape(fc, HORIZON)


def test_sarima_save_load_roundtrip(train_df_indexed, sarima_config):
    import warnings

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sarima_model")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SARIMAForecaster(sarima_config)
            m.fit(train_df_indexed, "total")
            fc_before = m.predict(HORIZON)
            m.save(path)

            m2 = SARIMAForecaster(sarima_config)
            m2.load(path)
            fc_after = m2.predict(HORIZON)
        assert list(fc_before["predicted_value"]) == pytest.approx(
            list(fc_after["predicted_value"]), rel=1e-3
        )


# ── Base class contract ───────────────────────────────────────────────────


def test_get_name_returns_model_name(ml_config):
    assert XGBoostForecaster(ml_config).get_name() == "xgboost"
    assert LightGBMForecaster(ml_config).get_name() == "lightgbm"
