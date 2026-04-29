# Sales Forecasting System — Detailed Technical Explanation

**Author:** Bavirisetty Sairam  
**Dataset:** 8,085 rows — columns: State, Date, Total, Category  
**Goal:** Predict next 8 weeks of sales per state using 5 models with automatic champion selection, REST API, Streamlit dashboard, PostgreSQL, Redis, and Docker deployment.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Dataset Analysis](#2-dataset-analysis)
3. [Phase 0 — Project Scaffolding](#3-phase-0--project-scaffolding)
4. [Phase 1 — Configuration & Utilities](#4-phase-1--configuration--utilities)
5. [Phase 2 — Data Preprocessing Pipeline](#5-phase-2--data-preprocessing-pipeline)
6. [Phase 3 — Feature Engineering](#6-phase-3--feature-engineering)
7. [Phase 4 — Model Implementations](#7-phase-4--model-implementations) *(coming)*
8. [Phase 5 — Training Pipeline & Model Selection](#8-phase-5--training-pipeline--model-selection) *(coming)*
9. [Phase 6 — REST API](#9-phase-6--rest-api) *(coming)*
10. [Phase 7 — Streamlit Dashboard](#10-phase-7--streamlit-dashboard) *(coming)*
11. [Phase 8 — Testing](#11-phase-8--testing) *(coming)*
12. [Phase 9 — Documentation](#12-phase-9--documentation) *(coming)*
13. [Phase 10 — CI/CD Pipeline](#13-phase-10--cicd-pipeline)
14. [Design Decisions & Trade-offs](#14-design-decisions--trade-offs)

---

## 1. Architecture Overview

```
Raw CSV (data/raw/dataset.csv)
        │
        ▼
┌─────────────────────────┐
│  Preprocessing Pipeline  │  src/preprocessing/
│  • Validate (Pandera)    │  ├── validator.py
│  • Clean 8 steps         │  ├── cleaner.py
│  • Weekly aggregation    │  └── pipeline.py
└────────────┬────────────┘
             │  weekly DataFrame [state, date, total, category]
             ▼
┌─────────────────────────┐
│  Feature Engineering     │  src/features/
│  • 21 features per state │  └── engineering.py
│  • Zero data leakage     │
└────────────┬────────────┘
             │  feature matrix [state, date, total, category, lag_*, rolling_*, ...]
             ▼
┌─────────────────────────────────────────────────────┐
│  Training Pipeline  (src/pipeline/)                  │
│                                                      │
│  For each STATE:                                     │
│    For each MODEL (SARIMA, Prophet, XGBoost,         │
│                    LightGBM, LSTM):                  │
│      ├── Expanding window cross-validation (3 folds) │
│      ├── MAPE / RMSE / MAE per fold                  │
│      └── Average metrics across folds               │
│    ↓                                                 │
│    Select champion (lowest avg MAPE, RMSE tiebreak)  │
│    Retrain champion on ALL data                      │
│    Save artifact to disk  →  models/{state}/         │
│    Store results in PostgreSQL                        │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
    PostgreSQL       Redis        File System
   (training_runs, (forecast     (model .pkl /
    forecasts,      cache)        .pt files)
    api_logs)
          │
          ▼
┌─────────────────────────┐
│  FastAPI REST API        │  src/api/
│  • API key auth          │  ├── main.py
│  • Rate limiting (Redis) │  ├── auth.py
│  • Security headers      │  ├── rate_limiter.py
│  • CORS middleware       │  ├── middleware.py
│  • POST /forecast        │  └── routes/
│  • GET  /models          │
│  • GET  /health          │
│  • POST /retrain         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Streamlit Dashboard     │  src/dashboard/
│  • Forecast chart        │  ├── app.py
│  • Model comparison      │  └── pages/
│  • Training history      │
│  • API health monitor    │
└─────────────────────────┘
```

**Tech stack summary:**

| Layer | Technology | Reason |
|---|---|---|
| API framework | FastAPI 0.115 | Async, auto OpenAPI docs, Pydantic validation |
| ORM | SQLAlchemy 2.0 | Type-safe DB access, Alembic migrations |
| Database | PostgreSQL 16 | Production-grade, JSON column support |
| Cache | Redis 7 | Sub-millisecond cache, atomic rate limiting |
| Config | pydantic-settings | Env-var + Docker secret loading in one class |
| Logging | loguru | Structured JSON (prod) and colorised (dev) |
| Validation | Pandera | Schema-level DataFrame validation with lazy errors |
| Models | pmdarima, Prophet, XGBoost, LightGBM, PyTorch | One per algorithm family |
| Tuning | Optuna | Bayesian hyperparameter optimisation |
| Dashboard | Streamlit + Plotly | Rapid interactive UI, no frontend build step |
| Containers | Docker + Compose | Reproducible deployment, secrets management |
| CI/CD | GitHub Actions | Lint → test → docker build pipeline |

---

## 2. Dataset Analysis

### Raw file characteristics

| Property | Value |
|---|---|
| Rows | 8,084 (after dedup) |
| Columns | State, Date, Total, Category |
| States | 43 US states |
| Categories | Multiple (e.g. Beverages) |
| Date range | January 2019 → November 2023 |
| Date formats | **Mixed** — some rows use `M/D/YYYY` (e.g. `1/12/2019`), others use `D-MM-YYYY` (e.g. `13-02-2022`) |
| Total format | Comma-separated thousands with surrounding whitespace: `"  109,574,036 "` |

### Key data challenges identified

1. **Mixed date formats** — `pd.to_datetime(..., format="mixed")` required; a fixed format string fails on position 75.
2. **Comma-formatted numbers** — the `Total` column must be stripped of commas and whitespace before casting to float.
3. **Sparse dates** — the raw data is not daily for every state; 68,757 date gaps were found and filled.
4. **Monthly-style entries** — data appears to represent monthly observations that need to be resampled to weekly.
5. **Multiple categories per state** — aggregated per `(state, category)` group before weekly resampling.

After preprocessing: **11,008 weekly rows** across 43 states, date range 2019-01-07 → 2023-11-27, zero nulls.

---

## 3. Phase 0 — Project Scaffolding

### Why a dedicated scaffolding phase?

Setting up the entire directory tree, configuration, and Docker infrastructure before writing any logic ensures:
- All future imports resolve correctly from day one
- No "works on my machine" issues — Docker pins the exact environment
- Secrets are never hardcoded (Docker secrets pattern from the start)
- CI pipeline is wired before any code exists, so every merge is validated

### 3.1 Directory structure

```
forecasting-system/
├── .github/workflows/ci.yml       ← GitHub Actions CI
├── .dockerignore                  ← keep secrets/data out of Docker context
├── .gitignore                     ← keep secrets/data out of git
├── .pre-commit-config.yaml        ← ruff + black run on every commit
├── alembic.ini                    ← DB migration config
├── config/training_config.yaml    ← all model hyperparameters
├── data/raw/                      ← raw CSV lives here (gitignored)
├── docker-compose.yml             ← 4 services: api, streamlit, postgres, redis
├── Dockerfile                     ← API image (non-root user)
├── Dockerfile.streamlit           ← Dashboard image
├── Makefile                       ← developer convenience targets
├── models/                        ← serialised model artifacts (gitignored)
├── notebooks/eda.ipynb            ← exploratory analysis
├── pyproject.toml                 ← single source of deps (Poetry)
├── secrets/                       ← Docker secret files (gitignored)
├── src/                           ← all application code
└── tests/                         ← pytest test suite
```

Every `src/` subdirectory has an `__init__.py` so they are proper Python packages — imports like `from src.config.settings import settings` work anywhere.

### 3.2 pyproject.toml — dependency management

Poetry was chosen over `requirements.txt` because it:
- Locks exact transitive dependency versions (`poetry.lock`)
- Separates dev-only deps (pytest, ruff, black) from runtime deps
- Generates the lock file that Docker uses for reproducible builds

Key dependency groups and why each package was included:

| Package | Why |
|---|---|
| `fastapi`, `uvicorn[standard]` | ASGI server + framework |
| `pydantic`, `pydantic-settings` | Request validation + settings from env |
| `sqlalchemy`, `alembic` | ORM + schema migrations |
| `psycopg2-binary` | PostgreSQL driver (binary = no compile step in Docker) |
| `redis` | Redis client for caching + rate limiting |
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Metrics (MAPE, RMSE, MAE), scalers |
| `statsmodels`, `pmdarima` | SARIMA + `auto_arima` |
| `prophet` | Facebook Prophet |
| `xgboost`, `lightgbm` | Gradient boosting with quantile regression |
| `torch` | LSTM via PyTorch (MC Dropout for CIs) |
| `pandera` | DataFrame schema validation |
| `optuna` | Bayesian hyperparameter search |
| `joblib` | Model serialisation |
| `loguru` | Structured logging |
| `pyyaml` | Parse `training_config.yaml` |
| `httpx` | Async HTTP client for Streamlit → API calls |
| `streamlit`, `plotly` | Dashboard + interactive charts |
| `holidays` | US holiday calendar for feature engineering |

**Note added:** `holidays` was not in the original plan spec but is required by `src/features/engineering.py` for the holiday feature group.

### 3.3 training_config.yaml — centralised model configuration

All hyperparameters live in one YAML file so training can be re-run with different settings without touching code. Structure:

- **`data`** — file paths, column names, date format
- **`preprocessing`** — fill method, outlier detection method and threshold
- **`features`** — lag periods, rolling windows, calendar features, holiday country
- **`training`** — forecast horizon (8 weeks), CV folds (3), primary metric (MAPE)
- **`models`** — per-model switches (`enabled: true/false`) and hyperparameter search spaces
- **`api`** — host, port, CORS origins
- **`cache`** — Redis TTL (24 hours)
- **`database`** — connection pool settings

The `date_format` was corrected from `%Y-%m-%d` (plan default) to `%m/%d/%Y` after inspecting the actual CSV, then later further updated to use `format="mixed"` in the cleaner because the dataset contains two distinct date formats.

### 3.4 Makefile — developer workflow

```makefile
make setup       # poetry install + pre-commit install + alembic upgrade head
make train       # run full training pipeline
make serve       # start FastAPI with hot-reload
make dashboard   # start Streamlit
make test        # pytest with verbose output
make lint        # ruff + black check (no auto-fix)
make format      # ruff + black with auto-fix
make clean       # remove __pycache__ and model artifacts
make docker-up   # build and start all 4 Docker services
make docker-down # stop and remove volumes
make ci-lint     # same as lint (used in GitHub Actions)
make ci-test     # pytest with JUnit XML output for CI
```

### 3.5 Docker setup

**Dockerfile (API service):**
- Base: `python:3.11-slim` — minimal image, no unnecessary OS packages
- Installs `build-essential` for compiling C extensions (e.g. psycopg2)
- Uses Poetry with `virtualenvs.create false` — installs directly into the container Python environment (no nested venv)
- Copies `pyproject.toml` and `poetry.lock` BEFORE copying source code — Docker layer caching means deps are only re-installed when `pyproject.toml` changes, not on every source change
- Creates a non-root `appuser` — running as root in a container is a security risk; if the process is compromised, the attacker has root inside the container
- `--without dev` — dev deps (pytest, ruff, black) are not installed in production

**Dockerfile.streamlit:**
- Same pattern but no `build-essential` (no C extensions needed for dashboard)
- Exposes port 8501 (Streamlit default)

**docker-compose.yml — 4 services:**

1. **`api`** — FastAPI, depends on postgres + redis health checks
2. **`streamlit`** — Dashboard, depends on api
3. **`postgres`** — PostgreSQL 16 Alpine, healthcheck via `pg_isready`
4. **`redis`** — Redis 7 Alpine, healthcheck via `redis-cli ping`

Docker secrets (`db_password`, `redis_password`, `api_key`) are mounted at `/run/secrets/` inside containers. The password is never passed as a plain environment variable — it is read from the secret file at startup. This prevents secrets appearing in `docker inspect` output or process listings.

Resource limits are set on each service to prevent one service starving others on the host machine.

### 3.6 .gitignore design

Critical items excluded from git:
- `secrets/` — credentials must never be committed
- `data/raw/*.csv` — raw data may be proprietary or too large
- `models/*.pkl`, `models/*.pt` — binary model artifacts belong in object storage, not git
- `logs/` — ephemeral runtime files

---

## 4. Phase 1 — Configuration & Utilities

### 4.1 Settings (`src/config/settings.py`)

Uses `pydantic-settings` `BaseSettings` which automatically:
- Reads values from environment variables (uppercase match)
- Falls back to defaults if env var is absent
- Reads from a `.env` file for local development

```python
class Settings(BaseSettings):
    database_url: str = "postgresql://app:password@localhost:5432/forecasting"
    api_keys: list[str] = ["forecasting-api-key-2026"]
    ...
```

The `load_secrets()` method is called immediately after instantiation. It checks `/run/secrets/db_password` — a path that only exists inside Docker containers. If the file is present, the placeholder `"password"` in the database URL is replaced with the real password. The same pattern applies to the API key. This means:
- In **local development**: defaults are used (no Docker secrets path exists)
- In **Docker**: secrets are loaded automatically from mounted files

**Why not read secrets in `__init__` or a validator?** pydantic-settings validators run during model construction before the object is fully created; calling `load_secrets()` explicitly after `Settings()` gives full control and avoids validator ordering issues.

### 4.2 Logger (`src/utils/logger.py`)

loguru was chosen over the standard `logging` module because:
- Zero boilerplate — `from loguru import logger` and it works
- Built-in structured serialisation (`serialize=True` emits JSON)
- File rotation by size and retention by age in one line

**Production mode** (`environment="production"`):
```
{"time": "2026-04-29 13:00:00", "level": "INFO", "name": "src.api.main", "message": "..."}
```
JSON format is machine-parseable by log aggregators (Datadog, CloudWatch, ELK).

**Development mode** (`environment="development"`):
```
13:00:00 | DEBUG    | src.api.main:startup:42 - Server started
```
Colorised, human-readable, with exact file + line number.

File sink: rotates at 10 MB, keeps 7 days of history. The `logs/` directory is created if absent (`Path("logs").mkdir(exist_ok=True)`).

### 4.3 Response Builder (`src/utils/response.py`)

All API responses share a standard envelope:

```json
{
  "status": "success",
  "data": { ... },
  "message": "Success",
  "timestamp": "2026-04-29T13:00:00.000000+00:00",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Benefits:
- Clients always know where to find data vs metadata
- `request_id` (UUID4) ties API logs to specific requests for debugging
- `timestamp` is always UTC ISO 8601 — no timezone ambiguity
- Error responses add `code` and optional `details` field

### 4.4 Database Models (`src/db/models.py`)

Three ORM tables:

**`training_runs`** — one row per (state, model) training run:
- `run_id` — unique UUID for idempotent re-runs
- `fold_metrics` — JSON array of per-fold MAPE/RMSE/MAE; allows drill-down analysis
- `hyperparameters` — JSON; stores Optuna best params for reproducibility
- `is_champion` — boolean flag; only one row per state should have this true
- `model_path` — filesystem path to the serialised artifact

**`forecasts`** — one row per (state, date) prediction:
- `forecast_date` — the date being predicted (not the date the prediction was made)
- `lower_bound`, `upper_bound` — 95% confidence interval bounds
- `created_at` — when the prediction was stored

**`api_request_logs`** — one row per API call:
- Used by the Streamlit health page to show response time trends
- `response_time_ms` enables latency monitoring without external APM

All `DateTime` columns default to `lambda: datetime.now(timezone.utc)` — a lambda is used (not `datetime.now(timezone.utc)` directly) because SQLAlchemy evaluates the default at row insertion time, not at class definition time. Using a direct value would freeze the timestamp to when Python first imported the module.

### 4.5 Database Session (`src/db/session.py`)

```python
engine = create_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,       # 5 persistent connections
    max_overflow=settings.db_max_overflow, # 10 additional burst connections
    pool_pre_ping=True,                    # test connection before use
)
```

`pool_pre_ping=True` — before handing out a connection from the pool, SQLAlchemy sends a `SELECT 1`. If the connection is stale (e.g. PostgreSQL restarted), it is discarded and a fresh one is made. Without this, you get `OperationalError: server closed the connection unexpectedly` after periods of inactivity.

`get_db()` is a FastAPI dependency generator. FastAPI calls it for each request, yielding a session, then calling `session.close()` in the finally block when the request completes. The `contextmanager` version (`get_db_session()`) is used in the training pipeline where there is no FastAPI dependency injection.

`init_db()` calls `Base.metadata.create_all()` — creates tables that don't exist. Alembic handles schema migrations (column additions, renames) after initial creation.

### 4.6 Redis Client (`src/cache/redis_client.py`)

Cache key schema: `forecast:{state_lower}:{weeks}` — e.g. `forecast:alabama:8`

**`set_forecast` / `get_forecast`:** Serialise the forecast dict to JSON (`json.dumps`) before storing in Redis, deserialise on retrieval. Redis stores strings; JSON is the universal interchange format.

**`invalidate_state(state)`:** Uses `KEYS forecast:{state}:*` to find and delete all cached forecasts for a state. Called after retraining so stale predictions aren't served.

**`invalidate_all()`:** Clears the entire forecast cache. Called after a full retrain of all states.

**`increment(key, expire_seconds)`:** Atomic pipeline — `INCR key` + `EXPIRE key ttl` in one round-trip. Used by the rate limiter. Atomicity is critical: if `INCR` succeeds but `EXPIRE` fails (e.g. client disconnects), the key would never expire and the user would be permanently rate-limited. Redis pipelines send both commands together, and Redis processes them sequentially without interruption.

**`health_check()`:** `PING` command — returns `True` if Redis responds, `False` on any exception. Used by `GET /health`.

The module-level `redis_client = RedisClient()` creates a singleton. The connection is established lazily (on first command), so importing the module does not fail if Redis is unavailable.

---

## 5. Phase 2 — Data Preprocessing Pipeline

### Overview of cleaning steps

```
Raw CSV (8,084 rows)
  Step 1: load_raw()              → normalise columns, parse dates, parse Total
  Step 2: remove_duplicates()     → exact row deduplication
  Step 3: aggregate_duplicate_dates() → sum Total for same (state,date,category)
  Step 4: fill_missing_dates()    → complete daily range per (state,category)
  Step 5: impute_missing()        → interpolate gaps
  Step 6: handle_outliers()       → IQR flag + cap
  Step 7: aggregate_to_weekly()   → W-MON resample, sum Total
  Step 8: sort_data()             → state ASC, date ASC
Weekly DataFrame (11,008 rows)
```

### 5.1 Pandera Validation (`src/preprocessing/validator.py`)

Pandera enforces DataFrame schemas declaratively. Two schemas are defined:

**Raw schema** (applied after `load_raw()`, before cleaning):
```python
raw_schema = DataFrameSchema({
    "state":    Column(str,           nullable=False),
    "date":     Column(pa.DateTime,   nullable=False),
    "total":    Column(float,         checks=Check.ge(0), nullable=False),
    "category": Column(str,           nullable=False),
}, coerce=True, strict=False)
```

**Clean schema** (applied after all cleaning steps):
```python
clean_schema = DataFrameSchema({
    "state":    Column(str,   nullable=False, checks=Check.str_length(min_value=2)),
    "date":     Column(pa.DateTime, nullable=False),
    "total":    Column(float, nullable=False, checks=[Check.ge(0)]),
    "category": Column(str,   nullable=False),
}, coerce=True, strict=False)
```

`lazy=True` — collects ALL schema violations before raising, so you get a complete error report rather than stopping at the first failure.

`strict=False` — extra columns (like `is_outlier` added during cleaning) don't cause validation failure.

`coerce=True` — Pandera attempts to cast columns to the declared type before validating. This handles edge cases like date columns stored as object dtype.

**Why validate twice?** Validating the raw data catches upstream data quality issues early (before we waste time cleaning corrupt data). Validating the clean data confirms our cleaning logic produced the expected result.

### 5.2 Data Cleaner (`src/preprocessing/cleaner.py`)

#### Step 1: load_raw()

```python
df["Total"] = (
    df["Total"].astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
    .astype(float)
)
```
The `Total` column in the CSV contains strings like `"  109,574,036 "`. The commas are thousands separators, not decimal points. `regex=False` is passed to `str.replace` for a small performance gain (no regex compilation needed for a literal comma replacement).

```python
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)
```
`format="mixed"` was discovered to be necessary during smoke testing — position 75 in the CSV uses `DD-MM-YYYY` format while earlier rows use `M/D/YYYY`. `dayfirst=False` ensures `1/12/2019` is parsed as January 12 (US convention), not December 1.

#### Step 2: remove_duplicates()

`DataFrame.drop_duplicates()` removes rows where every column is identical. Called before aggregation to avoid double-counting.

#### Step 3: aggregate_duplicate_dates()

```python
df.groupby(["state", "date", "category"]).agg({"total": "sum"})
```
If the same state+date+category appears multiple times with different totals (partial shipments, corrections), we sum them. Summing is semantically correct for a sales total column — the actual sales for that day is the sum of all reported sub-totals.

#### Step 4: fill_missing_dates()

For each `(state, category)` group:
1. Find `min_date` and `max_date` in the group
2. Create a complete `pd.date_range(min_date, max_date, freq="D")`
3. Left-merge the group data onto the complete range

This exposes gaps as `NaN` rows. The result contained 68,757 missing entries — meaning the raw data records sales on specific dates only (e.g. delivery days) and silently skips others.

**Why fill per (state, category) and not globally?** Different states have different date ranges. Using a global date range would create spurious rows for states that didn't exist in the dataset until later dates.

#### Step 5: impute_missing()

```python
grp["total"] = grp["total"].interpolate(method="linear", limit_direction="both")
```
Linear interpolation draws a straight line between the two nearest known values and fills the gap along that line. This is appropriate for sales data which tends to change gradually.

`limit_direction="both"` handles edge cases where the gap is at the start or end of a group (one side has no known value). In those cases, it forward-fills or back-fills.

**Alternative methods** (`ffill`, `bfill`) are supported via the config — forward-fill repeats the last known value (good for sparse categorical-like data), back-fill repeats the next known value.

After interpolation, a safety check fills any remaining `NaN` with 0 and logs a warning — this should not occur for normal data but protects against all-null groups.

#### Step 6: handle_outliers() — IQR method

For each `(state, category)` group:
```
Q1  = 25th percentile of total
Q3  = 75th percentile of total
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
```
Values outside the fences are flagged as outliers. Instead of deleting them (which would lose information and create new gaps), they are **capped** at the fence values:
```python
grp_total.clip(lower=max(0, lower_fence), upper=upper_fence)
```
`max(0, lower_fence)` ensures sales are never capped to a negative value.

**Why not remove outliers?** Time series models need a continuous sequence. Removing rows would re-introduce gaps. Capping preserves the row while dampening extreme values that would otherwise distort model training.

In the actual dataset, 0 outliers were detected — the data is well-behaved after interpolation.

#### Step 7: aggregate_to_weekly()

```python
grp["total"].resample("W-MON", label="left", closed="left").sum()
```
`W-MON` — weeks start on Monday.  
`label="left"` — the week is labelled by its start date (Monday), not end date.  
`closed="left"` — the interval includes Monday, excludes the following Monday.

For example, the week 2019-01-07 (Monday) covers 2019-01-07 through 2019-01-13.

The daily data (with filled gaps) is summed into weekly totals. Summing is correct for sales: if Monday–Sunday combined revenue was £5M, the weekly total is £5M.

**Why weekly?** The forecast horizon is 8 weeks. Forecasting at daily granularity would require predicting 56 individual days — much harder and noisier. Weekly data smooths day-of-week effects and matches the business use case.

#### Step 8: sort_data()

Final sort by `["state", "date"]` ensures all downstream code (features, models) receives data in chronological order per state without needing to sort internally.

### 5.3 Preprocessing Pipeline Orchestrator (`src/preprocessing/pipeline.py`)

The `run()` function wires all steps together:

```python
load_raw(raw_path)
  → validate_raw(df)        # non-fatal: log warnings, continue
  → clean(df, ...)          # all 8 steps
  → validate_clean(df)      # fatal: raises if clean schema fails
  → save to CSV (optional)
  → log summary
  → return df
```

Raw schema failures are non-fatal (logged as warnings) because minor type coercions are expected and the cleaner handles them. Clean schema failures are fatal — if the output of our pipeline is invalid, something is seriously wrong and we should not proceed to feature engineering or training.

**Summary log output (actual run):**
```
[Pipeline] Summary | dates: 2019-01-07 → 2023-11-27 | states (43): ['Alabama', 'Arizona', ...]
```

---

## 6. Phase 3 — Feature Engineering

### Why feature engineering matters for this problem

SARIMA and Prophet use the time series itself (the sequence of weekly totals). XGBoost and LightGBM are regression trees — they don't inherently understand time. We must manually create features that encode the temporal structure:
- **What happened recently?** → Lag features
- **What's the recent trend?** → Rolling statistics
- **What time of year is it?** → Calendar features
- **Is a holiday coming up?** → Holiday features
- **Where are we in the overall time series?** → Linear trend

### 6.1 Data leakage prevention — the most critical constraint

**Data leakage** occurs when a model sees future information during training. It causes optimistically biased metrics that don't generalise to real future predictions.

Every feature in this system is designed to be computable at prediction time using ONLY data that existed before the point being predicted:

| Feature type | How leakage is prevented |
|---|---|
| Lag features | `total.shift(n)` — value at time t is `total[t-n]`, strictly past |
| Rolling mean/std | `total.rolling(w, min_periods=1).mean().shift(1)` — window ends at t-1 |
| Calendar features | Derived from the date itself — no future data needed |
| Holiday features | Holiday calendar is public knowledge — dates are known in advance |
| Linear trend | Integer index — purely positional, no future values |

The `.shift(1)` on rolling features is critical. Without it, `rolling(7).mean()` at time t would include the value at t itself — using the target to predict itself.

**NaN handling:** Lag features produce NaN for the first `n` rows of each state (before enough history exists). These are NOT dropped from the full feature matrix. They are only dropped from the **training split** using `drop_warmup_rows()`. The test split rows must include NaN-free features (computable from training history), which is guaranteed by the expanding window CV setup.

### 6.2 Lag features

```python
grp[f"lag_{n}"] = grp["total"].shift(n)
```

| Feature | Value at row t |
|---|---|
| `lag_1` | `total[t-1]` — last week's sales |
| `lag_7` | `total[t-7]` — sales 7 weeks ago (same quarter, prior year context) |
| `lag_14` | `total[t-14]` — 14 weeks ago |
| `lag_30` | `total[t-30]` — ~7 months ago (captures annual seasonality for tree models) |

Lag features are the most powerful signal for tree-based models — last week's sales is typically the strongest predictor of this week's sales.

### 6.3 Rolling statistics

```python
grp[f"rolling_mean_{w}"] = grp["total"].rolling(w, min_periods=1).mean().shift(1)
grp[f"rolling_std_{w}"]  = grp["total"].rolling(w, min_periods=1).std().shift(1)
```

Windows: 7, 14, 30 weeks.

| Feature | Encodes |
|---|---|
| `rolling_mean_7` | Short-term trend (last ~2 months) |
| `rolling_mean_14` | Medium-term trend (last ~3.5 months) |
| `rolling_mean_30` | Long-term trend (last ~7 months) |
| `rolling_std_7` | Recent volatility — high std = unstable sales |
| `rolling_std_14` | Medium volatility |
| `rolling_std_30` | Long-term volatility |

`min_periods=1` prevents NaN for early rows where fewer than `w` observations exist. The rolling window uses whatever data is available (e.g. 3 rows for a window of 7 at the start of a series). This is correct — it is still past data, just less of it.

### 6.4 Calendar features

```python
grp["week_of_year"]   = dt.dt.isocalendar().week.astype(int)   # ISO week 1–53
grp["month"]          = dt.dt.month                             # 1–12
grp["quarter"]        = dt.dt.quarter                           # 1–4
grp["year"]           = dt.dt.year                              # 2019–2023
grp["dayofweek"]      = dt.dt.dayofweek                         # 0=Mon, 6=Sun
grp["is_month_start"] = dt.dt.is_month_start.astype(int)        # 0 or 1
grp["is_month_end"]   = dt.dt.is_month_end.astype(int)          # 0 or 1
```

These encode seasonality patterns that repeat on calendar cycles:
- Sales often spike in Q4 (holiday season) — `quarter` and `month` capture this
- `week_of_year` captures annual patterns more granularly than `month`
- `is_month_start` / `is_month_end` capture billing cycle effects (many businesses report sales at month boundaries)

`.dt.isocalendar().week` is used for ISO week numbers (1–53) rather than `.dt.week` which was deprecated in pandas 2.x.

### 6.5 Holiday features

```python
us_holidays = hol.country_holidays("US", years=years)
holiday_dates = pd.to_datetime(sorted(us_holidays.keys()))
```

Three features:

| Feature | Meaning |
|---|---|
| `is_holiday` | 1 if this week contains a US public holiday, 0 otherwise |
| `days_to_next_holiday` | Calendar days until the next holiday |
| `days_from_last_holiday` | Calendar days since the last holiday |

`days_to_next_holiday` captures pre-holiday buying spikes (people stock up before Thanksgiving, Christmas). `days_from_last_holiday` captures post-holiday demand dips.

The `holidays` library covers all US federal holidays including floating dates (e.g. Thanksgiving = 4th Thursday in November).

### 6.6 Trend feature

```python
grp["linear_trend"] = np.arange(len(grp))
```

A simple 0, 1, 2, 3, ... index per state. Tree-based models cannot extrapolate trends on their own (they can only predict values they've seen in training). Providing an explicit trend feature allows XGBoost and LightGBM to learn the direction of long-term growth or decline.

### 6.7 Feature matrix summary

| Group | Features | Count |
|---|---|---|
| Lag | lag_1, lag_7, lag_14, lag_30 | 4 |
| Rolling mean | rolling_mean_7, rolling_mean_14, rolling_mean_30 | 3 |
| Rolling std | rolling_std_7, rolling_std_14, rolling_std_30 | 3 |
| Calendar | week_of_year, month, quarter, year, dayofweek, is_month_start, is_month_end | 7 |
| Holiday | is_holiday, days_to_next_holiday, days_from_last_holiday | 3 |
| Trend | linear_trend | 1 |
| **Total** | | **21** |

Full feature matrix shape: **(11,008 rows × 25 columns)** [state, date, total, category + 21 features]

---

## 13. Phase 10 — CI/CD Pipeline

CI runs on every push to `main` or `develop` and on every pull request targeting `main`.

### Pipeline stages

```
push / PR
    │
    ├─► lint job ─────────────────────────────────────────────────────────┐
    │     ruff check src/ tests/                                          │
    │     black --check src/ tests/                                       │
    │                                                                     │
    ├─► security-scan job (parallel with test, after lint) ───────────── ┤
    │     pip-audit --strict   ← checks for CVEs in dependencies         │
    │     safety check         ← advisory only, continue-on-error        │
    │                                                                     │
    ├─► test job (after lint) ────────────────────────────────────────── ┤
    │     services: postgres:16-alpine + redis:7-alpine                   │
    │     pytest tests/ -v --tb=short                                     │
    │     env: DATABASE_URL, REDIS_URL, ENVIRONMENT=test                  │
    │                                                                     │
    └─► docker-build job (after test) ────────────────────────────────── ┘
          docker-compose build api
          docker-compose build streamlit
          docker-compose up -d
          curl -f http://localhost:8000/health   ← smoke test
          docker-compose down
```

**Why test against real Postgres and Redis in CI?** Mocking the database would create false confidence. The integration tests run against actual services so schema issues, SQL errors, and Redis command failures are caught in CI — not in production.

### Pre-commit hooks

`.pre-commit-config.yaml` installs hooks that run on every `git commit`:
1. `ruff --fix` — auto-fixes import order, unused imports, style issues
2. `black` — auto-formats code to 88-character line length

This means linting failures are caught before code is committed, keeping the main branch always clean.

---

## 14. Design Decisions & Trade-offs

### Per-state models vs. global model

**Decision:** Train a separate model instance for each state.

**Rationale:** Sales patterns differ significantly by state (California has 4× the volume of Wyoming; states have different seasonal patterns). A single global model would average these differences and perform worse for all states. Per-state models allow each state to find its own best algorithm and hyperparameters.

**Trade-off:** 43 states × 5 models = 215 model training runs. This is computationally expensive but manageable (~41 hours estimated total). The training pipeline parallelises across states.

### MAPE as primary selection metric

**Decision:** Use MAPE (Mean Absolute Percentage Error) as the champion selection metric, with RMSE as tiebreaker.

**Rationale:** MAPE is scale-independent — it expresses error as a percentage of actual sales. This makes it comparable across states with very different sales volumes (California ~$400M vs. Maine ~$25M). A model that achieves 5% MAPE on a small state is directly comparable to one achieving 5% on a large state.

**Trade-off:** MAPE is undefined when actual values are 0, and inflates when values are near 0. Since our data is sales totals that are always positive (enforced by Pandera), this is not a concern here.

### Expanding window cross-validation

**Decision:** Use expanding window CV instead of k-fold or sliding window.

**Rationale:** In standard k-fold, test data can appear before training data in time — this is leakage. In sliding window CV, the training set size is fixed. In expanding window, each fold adds more training data (the model sees progressively more history), which mimics real deployment where more data accumulates over time.

```
Fold 1: train[0:38]  → test[38:46]   (8 weeks)
Fold 2: train[0:46]  → test[46:54]   (8 weeks)
Fold 3: train[0:54]  → test[54:62]   (8 weeks)
```

### Confidence intervals approach

Different models use different CI methods:
- **SARIMA:** `get_forecast(alpha=0.05)` — analytical 95% CI from the fitted ARIMA covariance
- **Prophet:** `yhat_lower` / `yhat_upper` — probabilistic forecast from Stan sampling
- **XGBoost/LightGBM:** Quantile regression — train three models (mean, q0.05, q0.95) using `objective='reg:quantileerror'` / `objective='quantile'`
- **LSTM:** Monte Carlo Dropout — keep dropout ON at inference, run 100 forward passes, take 2.5th–97.5th percentile. This approximates a Bayesian neural network without the computational overhead of full Bayesian inference.

### Synchronous vs. asynchronous SQLAlchemy

**Decision:** Use synchronous SQLAlchemy engine.

**Rationale:** FastAPI supports both sync and async. The training pipeline is CPU-bound (model fitting) and runs in a subprocess — async adds no benefit there. The API endpoints that query the DB are relatively infrequent (forecast results are cached in Redis, so DB is only hit on cache miss). Full async SQLAlchemy (`asyncpg`) adds significant complexity. The Redis cache is the performance layer; the DB is the persistence layer.

### Redis caching strategy

All forecast results are cached with a 24-hour TTL. Cache is invalidated per-state after retraining that state. The trade-off:
- **Hit:** Sub-millisecond response, no model loading, no DB query
- **Miss:** Load model from disk (~100ms), generate prediction, write to Redis and DB

For production, most requests will be cache hits because the same `(state, weeks)` pairs are queried repeatedly throughout the day.

---

*This document is updated as each phase is completed. Phases 4–9 sections will be filled in as implementation progresses.*
