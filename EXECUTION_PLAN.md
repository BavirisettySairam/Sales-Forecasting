# Forecasting System — Complete Execution Plan

## Project Overview

Build a production-ready time series forecasting system that predicts next 8 weeks of sales per state. 5 models (SARIMA, Prophet, XGBoost, LightGBM, LSTM), auto model selection, REST API, Streamlit dashboard, PostgreSQL, Redis caching, Docker deployment.

**Dataset:** 8085 rows, columns: State, Date, Total, Category

---

## PHASE 0: Project Scaffolding

### Task 0.1 — Initialize Project Structure

Create the following directory structure exactly:

```
forecasting-system/
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.streamlit
├── Makefile
├── pyproject.toml
├── README.md
├── alembic.ini
├── .gitignore
├── .dockerignore
├── .pre-commit-config.yaml
├── config/
│   └── training_config.yaml
├── secrets/
│   ├── db_password.txt          # contains: forecasting_db_pass_2026
│   ├── redis_password.txt       # contains: forecasting_redis_pass_2026
│   └── api_key.txt              # contains: forecasting-api-key-2026
├── data/
│   └── raw/                     # place the dataset CSV here
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── middleware.py         # CORS, request logging, security headers
│   │   ├── auth.py              # API key authentication
│   │   ├── rate_limiter.py      # Redis-based rate limiting
│   │   ├── dependencies.py      # DB session, Redis connection deps
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── forecast.py      # POST /forecast, GET /forecast/{state}
│   │   │   ├── models.py        # GET /models, GET /models/{state}
│   │   │   ├── health.py        # GET /health
│   │   │   └── retrain.py       # POST /retrain
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── request.py       # ForecastRequest, RetrainRequest
│   │   │   ├── response.py      # StandardResponse, ForecastResponse, ModelResponse
│   │   │   └── errors.py        # ErrorResponse
│   │   └── exceptions.py        # Custom exception handlers
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py           # Missing values, duplicates, date gaps
│   │   ├── validator.py         # Pandera schema definitions
│   │   └── pipeline.py          # Orchestrates: validate → clean → output
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py       # All feature creation logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base class for all models
│   │   ├── sarima_model.py
│   │   ├── prophet_model.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   └── lstm_model.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── train.py             # Main training orchestrator
│   │   ├── evaluate.py          # Cross-validation + metrics
│   │   ├── select.py            # Auto model selection logic
│   │   └── registry.py          # Model save/load/register
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py               # Streamlit main app
│   │   └── pages/
│   │       ├── 01_forecast.py   # Forecast visualization
│   │       ├── 02_model_comparison.py  # Model metrics comparison
│   │       ├── 03_training_history.py  # Training run logs
│   │       └── 04_api_health.py        # API health monitor
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py            # SQLAlchemy ORM models
│   │   ├── session.py           # Engine, session factory
│   │   └── alembic/
│   │       ├── env.py
│   │       └── versions/        # Migration files
│   ├── cache/
│   │   ├── __init__.py
│   │   └── redis_client.py      # Redis connection + cache helpers
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # pydantic-settings based config
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # loguru setup
│       └── response.py          # Standardized response builder
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Fixtures: test DB, test client, sample data
│   ├── test_api.py
│   ├── test_security.py         # Auth, rate limiting, injection tests
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
├── models/                      # Serialized model artifacts (gitignored)
└── notebooks/
    └── eda.ipynb
```

### Task 0.2 — pyproject.toml

```toml
[tool.poetry]
name = "forecasting-system"
version = "1.0.0"
description = "Production-ready time series forecasting system with REST API"
authors = ["Bavirisetty Sairam"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.115.0"
uvicorn = {extras = ["standard"], version = "^0.30.0"}
pydantic = "^2.9.0"
pydantic-settings = "^2.5.0"
sqlalchemy = "^2.0.35"
alembic = "^1.13.0"
psycopg2-binary = "^2.9.9"
redis = "^5.1.0"
pandas = "^2.2.0"
numpy = "^1.26.0"
scikit-learn = "^1.5.0"
statsmodels = "^0.14.0"
pmdarima = "^2.0.4"
prophet = "^1.1.5"
xgboost = "^2.1.0"
lightgbm = "^4.5.0"
torch = "^2.4.0"
pandera = "^0.20.0"
optuna = "^4.0.0"
joblib = "^1.4.0"
loguru = "^0.7.2"
pyyaml = "^6.0.2"
httpx = "^0.27.0"
streamlit = "^1.38.0"
plotly = "^5.24.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.0"
pytest-asyncio = "^0.24.0"
ruff = "^0.6.0"
black = "^24.8.0"
pre-commit = "^3.8.0"

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W"]

[tool.black]
line-length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### Task 0.3 — training_config.yaml

```yaml
data:
  raw_path: "data/raw/dataset.csv"
  date_column: "Date"
  target_column: "Total"
  state_column: "State"
  category_column: "Category"
  date_format: "%Y-%m-%d"

preprocessing:
  fill_method: "interpolate"       # interpolate, ffill, bfill
  outlier_method: "iqr"            # iqr, zscore, none
  outlier_threshold: 1.5

features:
  lag_periods: [1, 7, 14, 30]
  rolling_windows: [7, 14, 30]
  rolling_stats: ["mean", "std"]
  calendar_features: ["dayofweek", "month", "quarter", "weekofyear", "is_month_start", "is_month_end"]
  holiday_country: "US"

training:
  forecast_horizon: 8              # weeks
  cv_folds: 3                      # expanding window folds
  cv_min_train_weeks: 30           # minimum training weeks for first fold
  primary_metric: "mape"           # mape, rmse, mae
  tiebreaker_metric: "rmse"
  random_seed: 42

models:
  sarima:
    enabled: true
    auto_arima: true
    seasonal_period: 52            # weekly seasonality
    max_p: 3
    max_q: 3
    max_P: 2
    max_Q: 2

  prophet:
    enabled: true
    yearly_seasonality: true
    weekly_seasonality: true
    changepoint_prior_scale: 0.05
    seasonality_prior_scale: 10.0

  xgboost:
    enabled: true
    optuna_trials: 50
    param_space:
      n_estimators: [100, 1000]
      max_depth: [3, 10]
      learning_rate: [0.01, 0.3]
      subsample: [0.6, 1.0]
      colsample_bytree: [0.6, 1.0]

  lightgbm:
    enabled: true
    optuna_trials: 50
    param_space:
      n_estimators: [100, 1000]
      max_depth: [3, 10]
      learning_rate: [0.01, 0.3]
      num_leaves: [20, 150]
      subsample: [0.6, 1.0]

  lstm:
    enabled: true
    sequence_length: 30
    hidden_size: 64
    num_layers: 2
    dropout: 0.2
    epochs: 100
    batch_size: 32
    learning_rate: 0.001
    early_stopping_patience: 10

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["http://localhost:8501", "http://streamlit:8501"]

cache:
  ttl_seconds: 86400               # 24 hours
  prefix: "forecast"

database:
  pool_size: 5
  max_overflow: 10
```

### Task 0.4 — Makefile

```makefile
.PHONY: setup train serve test lint clean docker-up docker-down

setup:
	poetry install
	poetry run pre-commit install
	poetry run alembic upgrade head

train:
	poetry run python -m src.pipeline.train --config config/training_config.yaml

serve:
	poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	poetry run streamlit run src/dashboard/app.py --server.port 8501

test:
	poetry run pytest tests/ -v --tb=short

lint:
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/

format:
	poetry run ruff check --fix src/ tests/
	poetry run black src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf models/*.pkl models/*.pt

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v

docker-logs:
	docker-compose logs -f
```

### Task 0.5 — Docker Files

**Dockerfile** (API):
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.8.3
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root --without dev

COPY . .

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile.streamlit**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install poetry==1.8.3
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root --without dev

COPY . .

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**docker-compose.yml**:
```yaml
version: "3.8"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: forecast-api
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql://app:${DB_PASSWORD}@postgres:5432/forecasting
      - REDIS_URL=redis://redis:6379/0
      - ENVIRONMENT=production
    volumes:
      - model-artifacts:/app/models
    secrets:
      - db_password
      - redis_password
      - api_key
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "0.5"
          memory: 1G

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    container_name: forecast-dashboard
    ports:
      - "8501:8501"
    depends_on:
      - api
    environment:
      - API_BASE_URL=http://api:8000
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 2G

  postgres:
    image: postgres:16-alpine
    container_name: forecast-db
    environment:
      POSTGRES_DB: forecasting
      POSTGRES_USER: app
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d forecasting"]
      interval: 5s
      timeout: 3s
      retries: 5
    secrets:
      - db_password
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1G

  redis:
    image: redis:7-alpine
    container_name: forecast-cache
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    volumes:
      - redis-data:/data
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 512M

volumes:
  postgres-data:
  redis-data:
  model-artifacts:

secrets:
  db_password:
    file: ./secrets/db_password.txt
  redis_password:
    file: ./secrets/redis_password.txt
  api_key:
    file: ./secrets/api_key.txt
```

### Task 0.6 — .gitignore

```
__pycache__/
*.pyc
*.pyo
.env
secrets/
models/*.pkl
models/*.pt
models/*.joblib
data/raw/*.csv
data/raw/*.xlsx
*.egg-info/
dist/
build/
.pytest_cache/
.ruff_cache/
notebooks/.ipynb_checkpoints/
```

---

## PHASE 1: Configuration & Utilities

### Task 1.1 — Settings (src/config/settings.py)

Use pydantic-settings to load configuration from environment variables and Docker secrets.

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://app:password@localhost:5432/forecasting"
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"
    cors_origins: list[str] = ["http://localhost:8501"]

    # Paths
    model_artifacts_dir: str = "models"
    training_config_path: str = "config/training_config.yaml"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def load_secrets(self):
        """Load Docker secrets if available."""
        db_secret = Path("/run/secrets/db_password")
        if db_secret.exists():
            password = db_secret.read_text().strip()
            self.database_url = self.database_url.replace("password", password)

settings = Settings()
settings.load_secrets()
```

### Task 1.2 — Logger (src/utils/logger.py)

Configure loguru with structured logging, file rotation, and JSON format for production.

```python
from loguru import logger
import sys

def setup_logger(environment: str = "development"):
    logger.remove()

    if environment == "production":
        logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}", level="INFO", serialize=True)
    else:
        logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="DEBUG", colorize=True)

    logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="DEBUG")

    return logger
```

### Task 1.3 — Response Builder (src/utils/response.py)

Standardized response structure for all API responses.

```python
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

def success_response(data: Any, message: str = "Success") -> dict:
    return {
        "status": "success",
        "data": data,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4())
    }

def error_response(message: str, code: int, details: Optional[Any] = None) -> dict:
    return {
        "status": "error",
        "message": message,
        "code": code,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4())
    }
```

### Task 1.4 — Database Models (src/db/models.py)

SQLAlchemy ORM models for: training_runs, model_metrics, forecasts, api_request_logs.

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()

class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), unique=True, nullable=False)
    state = Column(String(100), nullable=False)
    model_name = Column(String(50), nullable=False)
    avg_mape = Column(Float)
    avg_rmse = Column(Float)
    avg_mae = Column(Float)
    fold_metrics = Column(JSON)             # list of per-fold metrics
    hyperparameters = Column(JSON)
    model_path = Column(Text)
    is_champion = Column(Boolean, default=False)
    training_config = Column(JSON)
    trained_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Forecast(Base):
    __tablename__ = "forecasts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    state = Column(String(100), nullable=False)
    model_name = Column(String(50), nullable=False)
    forecast_date = Column(DateTime, nullable=False)    # the date being predicted
    predicted_value = Column(Float, nullable=False)
    lower_bound = Column(Float)
    upper_bound = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class APIRequestLog(Base):
    __tablename__ = "api_request_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(50), unique=True)
    endpoint = Column(String(200))
    method = Column(String(10))
    state = Column(String(100))
    status_code = Column(Integer)
    response_time_ms = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
```

### Task 1.5 — Database Session (src/db/session.py)

SQLAlchemy async engine and session factory. Create tables on startup if they don't exist. Setup Alembic for migrations.

### Task 1.6 — Redis Client (src/cache/redis_client.py)

```python
import redis
import json
from typing import Optional
from src.config.settings import settings

class RedisClient:
    def __init__(self):
        self.client = redis.from_url(settings.redis_url, decode_responses=True)
        self.ttl = settings.cache_ttl_seconds

    def get_forecast(self, state: str, weeks: int) -> Optional[dict]:
        key = f"forecast:{state.lower()}:{weeks}"
        data = self.client.get(key)
        return json.loads(data) if data else None

    def set_forecast(self, state: str, weeks: int, data: dict) -> None:
        key = f"forecast:{state.lower()}:{weeks}"
        self.client.setex(key, self.ttl, json.dumps(data))

    def invalidate_state(self, state: str) -> None:
        pattern = f"forecast:{state.lower()}:*"
        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)

    def invalidate_all(self) -> None:
        pattern = "forecast:*"
        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)

    def health_check(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False
```

---

## PHASE 2: Data Preprocessing Pipeline

### Task 2.1 — Pandera Validator (src/preprocessing/validator.py)

Define strict schema for the raw dataset:
- State: string, not null
- Date: datetime, not null
- Total: float, >= 0
- Category: string, not null

Also define a post-cleaning schema that enforces no nulls, proper date range, etc.

### Task 2.2 — Data Cleaner (src/preprocessing/cleaner.py)

Implement the following cleaning steps:

1. **Parse dates** — convert Date column to datetime, handle multiple formats
2. **Remove exact duplicates** — drop_duplicates()
3. **Handle duplicate dates per state** — if same state+date appears twice, aggregate (sum Total)
4. **Fill missing dates** — for each state, create a complete date range from min to max date, merge to identify gaps
5. **Handle missing values** — interpolation for numeric (Total), forward-fill for categorical
6. **Outlier detection** — IQR method: flag values below Q1-1.5*IQR or above Q3+1.5*IQR. Log them but don't auto-remove. Cap extreme outliers only.
7. **Aggregate to weekly** — since we're forecasting 8 weeks, resample daily data to weekly (sum of Total per state per week). Use Monday as week start.
8. **Sort** — by state, then by date ascending

### Task 2.3 — Preprocessing Pipeline (src/preprocessing/pipeline.py)

Orchestrator that runs: load raw data → validate raw schema → clean → validate clean schema → aggregate to weekly → save processed data. Log each step with record counts before/after.

---

## PHASE 3: Feature Engineering

### Task 3.1 — Feature Engineering (src/features/engineering.py)

Create features PER STATE (group by state, apply features, never leak across states):

**Lag Features:**
- lag_1, lag_7, lag_14, lag_30 (weeks)

**Rolling Statistics:**
- rolling_mean_7, rolling_mean_14, rolling_mean_30
- rolling_std_7, rolling_std_14, rolling_std_30

**Calendar Features:**
- week_of_year (1-52)
- month (1-12)
- quarter (1-4)
- is_month_start (bool)
- is_month_end (bool)
- year

**Holiday Features:**
- is_holiday (US holidays using `holidays` library)
- days_to_next_holiday
- days_from_last_holiday

**Trend Features:**
- linear_trend (integer index incrementing per week)

**CRITICAL — Data Leakage Prevention:**
- All lag and rolling features must only use past data
- Rolling calculations must use `min_periods=1` and be computed BEFORE the train/test split
- No future information in any feature
- Drop rows where lag features are NaN (beginning of series) ONLY from training data
- For test set, these should be computable from training data

Function signature:
```python
def create_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Input: DataFrame with columns [state, date, total] (weekly aggregated)
    Output: DataFrame with all features added
    """
```

---

## PHASE 4: Model Implementation

### Task 4.0 — Abstract Base Class (src/models/base.py)

```python
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np

class BaseForecaster(ABC):
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, target_col: str = "total") -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Return DataFrame with columns:
        - date
        - predicted_value
        - lower_bound
        - upper_bound
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    def get_name(self) -> str:
        return self.name
```

### Task 4.1 — SARIMA Model (src/models/sarima_model.py)

- Extends BaseForecaster
- Use `pmdarima.auto_arima` for automatic (p,d,q)(P,D,Q,m) selection
- seasonal_period = 52 (weekly data, yearly cycle)
- Confidence intervals from `get_forecast()` method (alpha=0.05 for 95% CI)
- Fit per state independently
- Save with joblib

### Task 4.2 — Prophet Model (src/models/prophet_model.py)

- Extends BaseForecaster
- Rename columns to `ds` and `y` as Prophet requires
- Enable yearly_seasonality, weekly_seasonality from config
- Add US holidays using `prophet.make_holidays_df` or `add_country_holidays('US')`
- Confidence intervals from `yhat_lower` and `yhat_upper`
- Suppress Prophet stdout logging (set `logging.getLogger('prophet').setLevel(logging.WARNING)`)
- Save with joblib (serialize the fitted Prophet object)

### Task 4.3 — XGBoost Model (src/models/xgboost_model.py)

- Extends BaseForecaster
- Uses all engineered features from Phase 3
- Hyperparameter tuning with Optuna (50 trials from config)
- Optuna objective: minimize MAPE on validation fold
- For confidence intervals: train 2 additional quantile regression models (alpha=0.05 and alpha=0.95). XGBoost supports `objective='reg:quantileerror'` with `quantile_alpha` parameter.
- Recursive forecasting for multi-step: predict week t+1, use it as lag for t+2, etc.
- Save with joblib

### Task 4.4 — LightGBM Model (src/models/lightgbm_model.py)

- Same architecture as XGBoost model
- LightGBM-specific params: num_leaves, min_child_samples
- Optuna tuning (50 trials)
- Quantile regression for confidence intervals (`objective='quantile'`, `alpha=0.05` and `0.95`)
- Recursive multi-step forecasting
- Save with joblib

### Task 4.5 — LSTM Model (src/models/lstm_model.py)

- Extends BaseForecaster
- PyTorch implementation (NOT Keras/TensorFlow)
- Architecture:
  ```
  Input → LSTM(hidden=64, layers=2, dropout=0.2) → Linear(64 → 1)
  ```
- Data preparation:
  - Normalize with MinMaxScaler (fit on train only, transform test)
  - Create sequences: sliding window of `sequence_length` weeks → predict next week
  - Use DataLoader with batch_size from config
- Training:
  - Adam optimizer, MSE loss
  - Early stopping (patience=10) monitoring validation loss
  - Learning rate from config
  - Set `torch.manual_seed(42)` for reproducibility
- Confidence intervals via MC Dropout:
  - At inference, keep dropout ON
  - Run 100 forward passes
  - Take mean as prediction, 2.5th and 97.5th percentiles as bounds
- Multi-step: predict one week, append to input sequence, predict next
- Save: `torch.save()` for model, joblib for scaler

---

## PHASE 5: Training Pipeline & Model Selection

### Task 5.1 — Cross-Validation (src/pipeline/evaluate.py)

Implement expanding window time series cross-validation:

```python
def time_series_cv(data: pd.DataFrame, n_folds: int, min_train_weeks: int, horizon: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Returns list of (train_df, test_df) tuples.

    Example with 60 weeks of data, 3 folds, min_train=30, horizon=8:
    Fold 1: train weeks[0:38], test weeks[38:46]
    Fold 2: train weeks[0:46], test weeks[46:54]
    Fold 3: train weeks[0:54], test weeks[54:62]

    The test window always equals the forecast horizon (8 weeks).
    Training set expands with each fold.
    """
```

Metrics calculation:
```python
def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mape": mean_absolute_percentage_error(actual, predicted) * 100,
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "mae": mean_absolute_error(actual, predicted)
    }
```

### Task 5.2 — Model Selection (src/pipeline/select.py)

```python
def select_best_model(results: list[dict]) -> dict:
    """
    Input: list of dicts, each with keys:
        - state, model_name, avg_mape, avg_rmse, avg_mae, fold_metrics

    Logic:
        1. Group by state
        2. For each state, sort by avg_mape ascending
        3. Tiebreaker: lowest avg_rmse
        4. Mark winner as is_champion=True
        5. Return selection results

    Output: dict mapping state -> champion model info
    """
```

### Task 5.3 — Training Orchestrator (src/pipeline/train.py)

Main entry point. This is the most critical file:

```
1. Load config from YAML
2. Run preprocessing pipeline (Phase 2)
3. Run feature engineering (Phase 3)
4. For each state:
    a. Extract state data
    b. Generate CV folds
    c. For each model (SARIMA, Prophet, XGBoost, LightGBM, LSTM):
        i.   For each fold:
             - Train on fold's training data
             - Predict on fold's test data
             - Calculate MAPE, RMSE, MAE
        ii.  Average metrics across folds
        iii. Log results
    d. Select best model for this state
    e. Retrain best model on ALL data (for production forecasting)
    f. Save model artifact to disk
    g. Store all results in PostgreSQL (training_runs table)
    h. Mark champion in DB
5. Invalidate Redis cache
6. Log summary: per-state champion, overall stats
```

CLI interface:
```bash
python -m src.pipeline.train --config config/training_config.yaml
```

Use argparse. Log to stdout and file. Total runtime should be logged.

### Task 5.4 — Model Registry (src/pipeline/registry.py)

Functions:
- `save_model(model, state, model_name)` → saves artifact to `models/{state}/{model_name}.pkl` or `.pt`
- `load_model(state, model_name)` → loads and returns model
- `get_champion(state)` → queries DB for champion model, loads it
- `list_models(state=None)` → returns all trained models with metrics

---

## PHASE 6: REST API

### Task 6.1 — FastAPI Main App (src/api/main.py)

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect DB, Redis, load models into memory
    yield
    # Shutdown: close connections

app = FastAPI(
    title="Forecasting System API",
    description="Production-ready time series forecasting system",
    version="1.0.0",
    lifespan=lifespan
)
```

Register: CORS middleware, request logging middleware, exception handlers, all route routers.

### Task 6.2 — Schemas (src/api/schemas/)

**request.py:**
```python
class ForecastRequest(BaseModel):
    state: str
    weeks: int = Field(default=8, ge=1, le=52)

class RetrainRequest(BaseModel):
    states: Optional[list[str]] = None  # None = retrain all
```

**response.py:**
```python
class ForecastPoint(BaseModel):
    date: str
    predicted_value: float
    lower_bound: float
    upper_bound: float

class ForecastData(BaseModel):
    state: str
    model_used: str
    model_mape: float
    forecast: list[ForecastPoint]

class StandardResponse(BaseModel):
    status: str
    data: Any
    message: str
    timestamp: str
    request_id: str

class ModelInfo(BaseModel):
    state: str
    model_name: str
    is_champion: bool
    avg_mape: float
    avg_rmse: float
    avg_mae: float
    trained_at: str
```

### Task 6.3 — Routes

**health.py:**
- `GET /health` → checks API, DB, Redis connectivity. Returns status of each.

**forecast.py:**
- `POST /forecast` → accepts ForecastRequest, checks Redis cache first, loads champion model, predicts, caches result, stores in DB, returns StandardResponse with ForecastData
- `GET /forecast/{state}` → convenience endpoint, defaults to 8 weeks

**models.py:**
- `GET /models` → list all trained models and their metrics across all states
- `GET /models/{state}` → list all models for a specific state, highlight champion

**retrain.py:**
- `POST /retrain` → triggers retraining pipeline, invalidates cache, returns new champions

### Task 6.4 — Middleware (src/api/middleware.py)

- CORS with configurable origins from settings
- Request ID injection (UUID per request, added to response headers)
- Request/response logging (endpoint, method, status code, response time)
- Log to DB (api_request_logs table) asynchronously

### Task 6.5 — Exception Handlers (src/api/exceptions.py)

Custom exceptions:
- `StateNotFoundException` → 404
- `ModelNotTrainedException` → 503
- `ForecastGenerationError` → 500
- `UnauthorizedError` → 401
- `RateLimitExceededError` → 429
- Generic fallback → 500 with standardized error response

Never expose raw tracebacks to the client. Log full traceback, return clean error message.

### Task 6.6 — API Key Authentication (src/api/auth.py)

API key-based authentication for all endpoints except `/health` and `/docs`.

```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from src.config.settings import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header."
        )
    if api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key."
        )
    return api_key
```

Settings addition:
```python
# In settings.py
api_keys: list[str] = ["forecasting-api-key-2026"]  # loaded from Docker secret in production
```

Apply as a dependency on protected routes:
```python
@router.post("/forecast", dependencies=[Depends(verify_api_key)])
```

Endpoints:
- `/health` → **no auth** (load balancers need this)
- `/docs`, `/redoc` → **no auth in dev**, **disabled in production** (`if settings.environment == "production": app = FastAPI(docs_url=None, redoc_url=None)`)
- Everything else → **requires X-API-Key header**

Store API key in Docker secrets:
```
secrets/
├── db_password.txt
├── redis_password.txt
└── api_key.txt
```

### Task 6.7 — Rate Limiting (src/api/rate_limiter.py)

Redis-based rate limiting using sliding window algorithm.

```python
import redis
import time
from fastapi import Request, HTTPException

class RateLimiter:
    def __init__(self, redis_client: redis.Redis, max_requests: int = 100, window_seconds: int = 60):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window_seconds

    async def check(self, request: Request) -> None:
        # Key: rate_limit:{client_ip} or rate_limit:{api_key}
        client_id = request.headers.get("X-API-Key", request.client.host)
        key = f"rate_limit:{client_id}"
        
        current = self.redis.get(key)
        if current and int(current) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window}s.",
                headers={"Retry-After": str(self.window)}
            )
        
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, self.window)
        pipe.execute()
```

Rate limits (configurable in settings):
- `/forecast` → 100 requests/minute per API key
- `/retrain` → 5 requests/hour per API key (expensive operation)
- `/models`, `/health` → 200 requests/minute

Add rate limit headers to responses:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`

### Task 6.8 — Security Headers Middleware (src/api/middleware.py)

Add security headers to every response:

```python
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
```

### Task 6.9 — Input Sanitization

- Validate `state` parameter against known states from the dataset (whitelist approach):
  ```python
  VALID_STATES = set()  # populated on startup from DB
  
  def validate_state(state: str) -> str:
      state_clean = state.strip().title()
      if state_clean not in VALID_STATES:
          raise StateNotFoundException(f"State '{state}' not found. Valid: {sorted(VALID_STATES)}")
      return state_clean
  ```
- All DB queries through SQLAlchemy ORM (parameterized, never raw SQL strings)
- Pydantic `Field` constraints: `weeks = Field(ge=1, le=52)`, `state = Field(min_length=2, max_length=100)`
- Strip and sanitize all string inputs

---

## PHASE 7: Streamlit Dashboard

### Task 7.1 — Main App (src/dashboard/app.py)

```python
import streamlit as st

st.set_page_config(
    page_title="Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Sales Forecasting System")
st.markdown("Production-ready time series forecasting with automatic model selection")

# Sidebar: state selector, date range, model filter
# Main: summary cards (total states, champion models, avg MAPE)
```

All pages call the FastAPI backend via httpx. Never call models directly from Streamlit.

### Task 7.2 — Forecast Page (src/dashboard/pages/01_forecast.py)

- State dropdown (populated from API)
- Weeks slider (1-52, default 8)
- "Generate Forecast" button
- Plotly chart showing:
  - Historical data (actual sales as solid line)
  - Forecasted values (dashed line, different color)
  - Confidence interval (shaded band)
  - X-axis: dates, Y-axis: sales total
- Below chart: table of forecasted values with dates, predicted, lower, upper bounds

### Task 7.3 — Model Comparison Page (src/dashboard/pages/02_model_comparison.py)

- State selector
- Plotly grouped bar chart: MAPE comparison across all 5 models for selected state
- Table: all models ranked by MAPE with RMSE, MAE columns
- Champion model highlighted with a green badge
- Plotly line chart: overlay all model predictions vs actuals on validation set
- Heatmap: state × model matrix showing MAPE values (which model works best where)

### Task 7.4 — Training History Page (src/dashboard/pages/03_training_history.py)

- Table of all training runs from PostgreSQL
- Filterable by state, model, date range
- Per-fold metrics expandable view
- Training run duration, config used
- Timeline chart of model performance over successive training runs (if retrained multiple times)

### Task 7.5 — API Health Page (src/dashboard/pages/04_api_health.py)

- Hit `GET /health` endpoint, show status of API, DB, Redis
- Green/red indicators
- Recent API request logs (from DB)
- Response time chart (average response time over last N requests)
- Cache hit/miss ratio if trackable

---

## PHASE 8: Testing

### Task 8.1 — Fixtures (tests/conftest.py)

- Create test database (SQLite in-memory for tests, not Postgres)
- Sample dataset fixture (small: 3 states, 52 weeks each)
- FastAPI test client using httpx.AsyncClient
- Mock Redis client

### Task 8.2 — Preprocessing Tests (tests/test_preprocessing.py)

- Test missing date filling
- Test duplicate removal
- Test outlier detection flags correct rows
- Test weekly aggregation sums correctly
- Test Pandera schema rejects bad data (negative totals, null states)

### Task 8.3 — Feature Engineering Tests (tests/test_features.py)

- Test lag features are correct (lag_1 of row N = value of row N-1)
- Test rolling mean calculation
- Test no NaN in features for rows past the warmup period
- Test no data leakage: feature at time T uses only data from T-1 or earlier

### Task 8.4 — Model Tests (tests/test_models.py)

- Test each model can fit on sample data without error
- Test each model returns predictions with correct shape
- Test predictions include lower_bound and upper_bound
- Test model save/load roundtrip (save, load, predict, compare)

### Task 8.5 — API Tests (tests/test_api.py)

- Test `GET /health` returns 200 (no auth required)
- Test `POST /forecast` with valid state returns forecast
- Test `POST /forecast` with invalid state returns 404
- Test `GET /models` returns list
- Test response structure matches StandardResponse schema
- Test CORS headers are present

### Task 8.6 — Security Tests (tests/test_security.py)

- Test `POST /forecast` WITHOUT API key returns 401
- Test `POST /forecast` WITH invalid API key returns 401
- Test `POST /forecast` WITH valid API key returns 200
- Test rate limiting: send 101 requests rapidly, last one returns 429
- Test security headers present in response (X-Content-Type-Options, X-Frame-Options, etc.)
- Test SQL injection attempt in state parameter is rejected (e.g., `state="'; DROP TABLE forecasts; --"`)
- Test state parameter whitelist rejects unknown states
- Test `/docs` is disabled when ENVIRONMENT=production
- Test oversized request body is rejected

---

## PHASE 9: Documentation & Finalization

### Task 9.1 — README.md

Structure:
```markdown
# 📈 Sales Forecasting System

> Production-ready time series forecasting with automatic model selection, REST API, and interactive dashboard.

## Architecture

[Mermaid diagram showing: Data → Preprocessing → Feature Engineering → Model Training (5 models) → Cross-Validation → Auto Selection → Model Registry (PostgreSQL) → FastAPI → Redis Cache → Streamlit Dashboard]

## Tech Stack
[Table of all technologies]

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)

### One-Command Setup
\```bash
git clone <repo>
cd forecasting-system
make docker-up
\```

This starts: API (port 8000), Dashboard (port 8501), PostgreSQL, Redis.

### Train Models
\```bash
make train
\```

### Access
- API Docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Health Check: http://localhost:8000/health

## API Documentation
[Example curl commands for each endpoint]

## Model Selection Methodology
[Explain expanding window CV, MAPE as primary metric, per-state champion selection]

## Project Structure
[Directory tree with descriptions]

## Design Decisions & Tradeoffs
[Why FastAPI, why per-state models, why MAPE, Redis caching strategy, etc.]

## Future Enhancements
- MLflow for experiment tracking
- Apache Airflow for scheduled retraining
- Natural language query interface via LLM
- A/B testing framework for model deployment
- Kubernetes deployment for horizontal scaling
- Prometheus + Grafana monitoring
```

### Task 9.2 — Architecture Diagram

Create a Mermaid diagram in the README showing the full system flow:
```
Raw Data → Preprocessing → Feature Engineering → Training Pipeline
                                                      ↓
                                              Cross-Validation
                                                      ↓
                                              Model Selection
                                                      ↓
                                    PostgreSQL ← Model Registry → File System
                                         ↓
                                      FastAPI ←→ Redis Cache
                                         ↓
                                  Streamlit Dashboard
```

### Task 9.3 — Final Checklist Before Submission

- [ ] `docker-compose up` works from a clean state
- [ ] `make train` completes without errors
- [ ] All 5 models train and produce forecasts
- [ ] API returns proper responses for all endpoints
- [ ] Streamlit dashboard shows all 4 pages correctly
- [ ] Tests pass: `make test`
- [ ] Linting passes: `make lint`
- [ ] CI pipeline passes on GitHub (lint + test + docker build)
- [ ] All feature branches merged into main cleanly
- [ ] README has setup instructions, architecture diagram, API examples
- [ ] No hardcoded paths or credentials
- [ ] .gitignore excludes secrets, data, model artifacts
- [ ] Pre-commit hooks installed and working
- [ ] Video recorded showing: project structure → training → API docs → dashboard

---

## PHASE 10: CI/CD Pipeline

### Task 10.1 — GitHub Actions CI Pipeline

Create `.github/workflows/ci.yml`:

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      - name: Ruff lint
        run: poetry run ruff check src/ tests/
      - name: Black format check
        run: poetry run black --check src/ tests/

  test:
    runs-on: ubuntu-latest
    needs: lint
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: forecasting_test
          POSTGRES_USER: app
          POSTGRES_PASSWORD: test_password
        ports: ["5432:5432"]
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      - name: Run tests
        env:
          DATABASE_URL: postgresql://app:test_password@localhost:5432/forecasting_test
          REDIS_URL: redis://localhost:6379/0
          ENVIRONMENT: test
        run: poetry run pytest tests/ -v --tb=short

  docker-build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: Build API image
        run: docker-compose build api
      - name: Build Streamlit image
        run: docker-compose build streamlit
      - name: Smoke test — containers start
        run: |
          docker-compose up -d
          sleep 10
          curl -f http://localhost:8000/health || exit 1
          docker-compose down

  security-scan:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
          pip install pip-audit safety
      - name: Dependency vulnerability scan (pip-audit)
        run: pip-audit --strict
      - name: Safety check
        run: safety check
        continue-on-error: true  # advisory, don't block CI for now
```

Four stages: **lint → security scan + test (parallel) → docker build + smoke test**.

### Task 10.2 — Pre-commit Hooks

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
```

### Task 10.3 — Update Makefile

Add CI-related targets to the Makefile:

```makefile
ci-lint:
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/

ci-test:
	poetry run pytest tests/ -v --tb=short --junitxml=test-results.xml

ci: ci-lint ci-test
```

### Task 10.4 — Update Final Checklist

Add to the checklist:
- [ ] CI pipeline passes on GitHub (lint + test + docker build)
- [ ] Pre-commit hooks installed and working

---

## GIT BRANCHING WORKFLOW

### Branch Strategy

Use **feature branches off main**. Every phase gets its own branch. Never commit directly to main.

### Rules for Claude Code

**CRITICAL: Include these instructions in EVERY prompt to Claude Code:**

```
Git workflow rules (follow strictly):
1. Before starting work, create a new branch from main:
   git checkout main
   git pull origin main
   git checkout -b feature/phase-[N]-[short-description]

2. Make commits as you go. Commit after each completed task:
   git add -A
   git commit -m "feat: [Task X.Y description]"

3. Use conventional commit messages:
   - feat: new feature
   - fix: bug fix
   - refactor: code restructuring
   - test: adding tests
   - docs: documentation
   - chore: config, dependencies, build

4. When the phase is complete, push the branch:
   git push origin feature/phase-[N]-[short-description]

5. DO NOT merge into main. I will review and merge manually.
```

### Branch Naming Convention

```
feature/phase-0-scaffolding
feature/phase-1-config-utils
feature/phase-2-preprocessing
feature/phase-3-feature-engineering
feature/phase-4-models
feature/phase-5-training-pipeline
feature/phase-6-api
feature/phase-7-streamlit
feature/phase-8-testing
feature/phase-9-docs
feature/phase-10-cicd
```

### Your Review & Merge Process

After Claude Code pushes a branch:

1. Review the diff:
   ```bash
   git diff main..feature/phase-N-description
   ```

2. If it looks good, merge:
   ```bash
   git checkout main
   git merge feature/phase-N-description
   git push origin main
   git branch -d feature/phase-N-description
   ```

3. If changes are needed, tell Claude Code:
   ```
   Switch to branch feature/phase-N-description.
   Fix: [describe the issue].
   Commit with message "fix: [description]" and push.
   ```

4. Only after merge, start the next phase prompt.

### First Prompt (Phase 0) Should Include Git Init

```
First, initialize the git repo:
git init
git checkout -b main
# [after scaffolding is done]
git add -A
git commit -m "chore: initial project scaffolding"
git checkout -b feature/phase-0-scaffolding

Then read EXECUTION_PLAN.md, Phase 0. Implement all tasks.
```

---

## EXECUTION ORDER SUMMARY

```
Phase 0:  Scaffolding (structure, configs, Docker)          ~2 hours
Phase 1:  Config, Logger, DB, Redis, Response builder       ~2 hours
Phase 2:  Preprocessing pipeline                            ~3 hours
Phase 3:  Feature engineering                               ~3 hours
Phase 4:  5 model implementations                           ~10 hours
Phase 5:  Training pipeline & model selection               ~5 hours
Phase 6:  REST API                                          ~4 hours
Phase 7:  Streamlit dashboard                               ~5 hours
Phase 8:  Testing                                           ~3 hours
Phase 9:  Documentation & video                             ~3 hours
Phase 10: CI/CD pipeline                                    ~1 hour
                                                   Total:  ~41 hours
```

Buffer of ~19 hours for debugging, edge cases, and polish.

---

## CLAUDE CODE EXECUTION NOTES

### How to Feed This Plan to Claude Code

1. Save this file as `EXECUTION_PLAN.md` in your project root
2. Feed each phase as a separate prompt
3. Use this prompt template:

```
Git workflow rules (follow strictly):
1. Create branch from main: git checkout main && git pull origin main && git checkout -b feature/phase-[N]-[short-description]
2. Commit after each completed task: git add -A && git commit -m "feat: [Task X.Y description]"
3. Use conventional commits: feat/fix/refactor/test/docs/chore
4. When phase is complete: git push origin feature/phase-[N]-[short-description]
5. DO NOT merge into main. I will review and merge manually.

Now read EXECUTION_PLAN.md, Phase [N]. Implement all tasks in Phase [N] exactly as specified.

Rules:
- Follow the file paths and names exactly as written
- Use type hints on all functions
- Add docstrings to all classes and public functions
- Log key operations with loguru
- Handle errors with try/except, never let exceptions silently pass
- After implementation, run ruff check and fix any issues
- After implementation, run the relevant tests

Do NOT skip any task. Implement fully, no placeholders, no TODOs.
```

4. After each phase, review the branch diff, merge into main, then start next phase.

### Claude Code Performance Tips

1. **Give clear, scoped prompts** — one phase at a time, not the entire plan
2. **Always reference this plan file** — "Read EXECUTION_PLAN.md" at the start of every prompt
3. **Ask it to run tests after each phase** — catches issues early
4. **If it generates placeholder code (pass, TODO, NotImplemented), call it out** — "Implement this fully, no placeholders"
5. **Pin decisions** — "Use XGBoost's reg:quantileerror objective for confidence intervals, not a custom implementation"
6. **If output is cut off**, say "continue from where you stopped" — Claude Code has context limits per response
7. **After Phase 5, do an integration test** — "Run the full training pipeline on the dataset and show me the output logs"
```

---
