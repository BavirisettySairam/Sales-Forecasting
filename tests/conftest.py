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


# ── Mock Redis ─────────────────────────────────────────────────────────────
@pytest.fixture
def mock_redis():
    store: dict = {}

    redis = MagicMock()
    redis.ping.return_value = True
    redis.get.side_effect = lambda k: store.get(k)
    redis.setex.side_effect = lambda k, ttl, v: store.update({k: v})
    redis.set.side_effect = lambda k, v: store.update({k: v})
    redis.keys.side_effect = lambda pattern: [
        k for k in store if k.startswith(pattern.replace("*", ""))
    ]
    redis.delete.side_effect = lambda *keys: [store.pop(k, None) for k in keys]
    redis.incr.side_effect = (
        lambda k: store.update({k: store.get(k, 0) + 1}) or store[k]
    )
    redis.expire.return_value = True

    pipe = MagicMock()
    pipe.execute.return_value = [0, 0, 1, True]
    pipe.zremrangebyscore.return_value = pipe
    pipe.zadd.return_value = pipe
    pipe.zcard.return_value = pipe
    pipe.expire.return_value = pipe
    redis.pipeline.return_value = pipe

    return redis


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
def api_client(db_session, mock_redis):
    from src.api.dependencies import get_db_dep, get_redis
    from src.api.main import app

    app.dependency_overrides[get_db_dep] = lambda: db_session
    app.dependency_overrides[get_redis] = lambda: mock_redis
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    app.dependency_overrides.clear()


# ── Valid API key for tests ────────────────────────────────────────────────
VALID_KEY = "forecasting-api-key-2026"
AUTH = {"X-API-Key": VALID_KEY}
