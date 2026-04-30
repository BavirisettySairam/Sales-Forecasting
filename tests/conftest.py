"""Shared fixtures for all test modules."""

import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from src.db.models import Base


# ── In-memory SQLite engine ────────────────────────────────────────────────
@pytest.fixture(scope="session")
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture
def db_session(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


# ── Sample weekly DataFrame (3 states × 52 weeks) ─────────────────────────
@pytest.fixture
def sample_weekly_df():
    rng = np.random.default_rng(42)
    rows = []
    states = ["California", "Texas", "New York"]
    dates = pd.date_range("2021-01-04", periods=52, freq="W-MON")
    for state in states:
        base = {"California": 500_000, "Texas": 300_000, "New York": 250_000}[state]
        for date in dates:
            rows.append(
                {
                    "state": state,
                    "date": date,
                    "total": max(0.0, base + rng.normal(0, base * 0.05)),
                    "category": "Beverages",
                }
            )
    return pd.DataFrame(rows)


# ── Sample national weekly series (single-state) ──────────────────────────
@pytest.fixture
def sample_national_df():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-06", periods=80, freq="W-MON")
    vals = 100_000 + np.arange(80) * 500 + rng.normal(0, 5000, 80)
    return pd.DataFrame(
        {
            "date": dates,
            "total": np.maximum(vals, 0),
            "state": "national",
            "category": "all",
        }
    )


# ── Minimal training config ────────────────────────────────────────────────
@pytest.fixture
def base_config():
    return {
        "lag_periods": [1, 2, 4],
        "rolling_windows": [4],
        "rolling_stats": ["mean", "std"],
        "holiday_country": "US",
        "sarima": {
            "seasonal_period": 4,
            "stepwise": True,
            "max_p": 1,
            "max_q": 1,
            "max_P": 1,
            "max_Q": 1,
            "D": 1,
            "alpha": 0.05,
        },
        "prophet": {
            "interval_width": 0.95,
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            "seasonality_mode": "additive",
        },
        "xgboost": {"n_trials": 3, "quantile_alpha": 0.95},
        "lightgbm": {"n_trials": 3, "quantile_alpha": 0.95},
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


# ── FastAPI test client with overridden dependencies ───────────────────────
@pytest.fixture
def api_client(db_session):
    from src.api.dependencies import get_db_dep
    from src.api.main import app

    app.dependency_overrides[get_db_dep] = lambda: db_session
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    app.dependency_overrides.clear()


# ── Valid API key for tests ────────────────────────────────────────────────
VALID_KEY = "forecasting-api-key-2026"
AUTH = {"X-API-Key": VALID_KEY}
