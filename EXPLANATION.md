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
7. [Phase 4 — Model Implementations](#7-phase-4--model-implementations)
8. [Phase 5 — Training Pipeline & Model Selection](#8-phase-5--training-pipeline--model-selection)
9. [Phase 6 — REST API](#9-phase-6--rest-api)
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

## 7. Phase 4 — Model Implementations

Five forecasting algorithms are implemented, each extending `BaseForecaster`. Every model must implement: `fit(train_data, target_col)`, `predict(horizon)`, `save(path)`, `load(path)`.

### 7.0 Abstract Base Class (`src/models/base.py`)

```python
class BaseForecaster(ABC):
    def __init__(self, name: str, config: dict) -> None:
        self.name = name        # model identifier used in registry
        self.config = config    # full training_config.yaml dict
        self.model = None       # fitted model object (None until fit())
        self.is_fitted = False  # guard flag — predict() raises if False

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """Must return: date | predicted_value | lower_bound | upper_bound"""
```

The `predict()` contract guarantees every model returns identical column names and semantics. Routes and the registry can call `forecaster.predict(weeks)` without knowing which algorithm is underneath — classic Liskov Substitution.

The `is_fitted` guard means callers get a clear `RuntimeError("Model not fitted")` instead of a cryptic `AttributeError: 'NoneType' has no attribute 'predict'`.

### 7.1 SARIMA (`src/models/sarima_model.py`)

SARIMA (Seasonal AutoRegressive Integrated Moving Average) is a classical statistical time series model. It models the relationship between a value and its own past values (AR), its own past forecast errors (MA), and seasonal versions of those at period `m`.

**Why SARIMA for weekly sales?** Sales data exhibits strong yearly seasonality (holiday season spikes). With `m=52` (52 weeks = 1 year), SARIMA can learn this pattern from the data without needing engineered features.

**auto_arima:** Rather than manually specifying `(p,d,q)(P,D,Q,52)`, `pmdarima.auto_arima` performs a stepwise search over the order space, evaluating each candidate by AIC (Akaike Information Criterion). AIC penalises complexity — a model that fits marginally better but has more parameters gets a higher (worse) AIC. This prevents overfitting.

```python
self.model = auto_arima(
    series,
    seasonal=True,
    m=52,              # yearly weekly cycle
    stepwise=True,     # sequential search (faster than exhaustive grid)
    information_criterion="aic",
    suppress_warnings=True,
    trace=False,       # no per-iteration stdout
)
```

**Confidence intervals:** `model.predict(n_periods=h, return_conf_int=True, alpha=0.05)` returns the 95% CI analytically using the fitted ARIMA covariance matrix. This is the most statistically rigorous CI method available (no resampling needed).

**Save/load:** The fitted `pmdarima` model object (including all ARIMA parameters and covariance) is serialised with `joblib.dump`. The training series is also saved so `predict()` can reconstruct forecast dates accurately from the DatetimeIndex.

### 7.2 Prophet (`src/models/prophet_model.py`)

Prophet (Meta/Facebook) is an additive regression model that decomposes a time series into:
```
y(t) = trend(t) + seasonality(t) + holidays(t) + ε(t)
```

- **Trend:** Piecewise linear or logistic growth with automatic changepoint detection
- **Seasonality:** Fourier series approximation of yearly and weekly patterns
- **Holidays:** Additive effect around user-specified holiday dates

**Why Prophet?** It handles missing data natively, is robust to outliers in the training data, and requires no feature engineering — the date is the only input. It also provides uncertainty intervals via Stan (a probabilistic programming language) sampling.

**US holiday integration:**
```python
import holidays as hol
us_holidays = pd.DataFrame([
    {"holiday": name, "ds": pd.Timestamp(date)}
    for date, name in hol.US(years=range(2015, 2030)).items()
])
model = Prophet(holidays=us_holidays, ...)
```
The `holidays` library generates exact holiday dates for any year. This allows Prophet to learn that weeks containing Thanksgiving or Christmas have abnormal sales without the model needing to infer this from patterns alone.

**Logging suppression:** Prophet prints fitting progress to stdout via Stan. This pollutes logs and confuses log parsers. Suppression is done at two levels:
```python
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
```

**Multiplicative seasonality** (`seasonality_mode="multiplicative"`) means seasonal effects are proportional to the trend level — if sales double overall, the Christmas spike also roughly doubles. This is more realistic than additive seasonality for sales data where volume scales.

**Save/load:** Prophet objects are serialised with `joblib`. The `prophet` library itself supports `model_to_json` but joblib is simpler and consistent with all other models.

### 7.3 XGBoost (`src/models/xgboost_model.py`)

XGBoost is a gradient-boosted tree ensemble. Unlike SARIMA and Prophet which model time directly, XGBoost treats forecasting as a supervised regression problem: given features at time t, predict the target at t.

**Architecture decisions:**

**Three models trained per fit:**
1. `model` (mean prediction) — `objective="reg:squarederror"`
2. `model_lower` (lower CI) — `objective="reg:quantileerror"`, `quantile_alpha=0.05`
3. `model_upper` (upper CI) — `objective="reg:quantileerror"`, `quantile_alpha=0.95`

Quantile regression models the conditional quantile of the target distribution, not the mean. The 5th and 95th percentile models together form the 90% confidence interval. This is valid even when the error distribution is not Gaussian.

**Optuna hyperparameter search (50 trials):**
```python
study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50)
```
TPE (Tree-structured Parzen Estimator) is a Bayesian optimisation algorithm. After a few random trials, it builds a probabilistic model of which hyperparameter values lead to good performance and samples candidates from the promising region. It finds better hyperparameters than grid search in fewer evaluations.

The objective function runs 3-fold cross-validation on the training data, returning mean MAPE. This prevents selecting hyperparameters that overfit the training set.

**Recursive multi-step forecasting:**
To forecast `horizon` weeks ahead, the model predicts one week at a time. The predicted value at week t+1 is appended to the history, features are recomputed, and the model predicts t+2:
```
history → features → predict(t+1) → append t+1 to history
                                   → features → predict(t+2) → ...
```
This is necessary because lag features (like `lag_1`) at step t+2 require the value at t+1, which is not yet available. The downside is that errors compound — a poor prediction at t+1 degrades the prediction at t+2. This is an inherent limitation of autoregressive multi-step forecasting.

**Data normalisation for feature engineering:** XGBoost/LightGBM's `_ensure_feature_columns()` helper adds dummy `state` and `category` columns if missing. This allows the models to operate on nationally-aggregated data (without state breakdown) while still calling the shared `create_features()` function which expects those columns.

### 7.4 LightGBM (`src/models/lightgbm_model.py`)

LightGBM is architecturally similar to XGBoost but uses a different tree-growing strategy:

| | XGBoost | LightGBM |
|---|---|---|
| Tree growth | Depth-first (level-wise) | Leaf-wise (best-leaf) |
| Speed | Slower | Faster (especially on large datasets) |
| Overfitting risk | Lower | Higher on small datasets |
| Quantile CI | `objective="reg:quantileerror"` | `objective="quantile"`, `alpha=α` |

For this dataset (~11,000 rows), both perform similarly. LightGBM adds `num_leaves` to the Optuna search space — this controls tree complexity and is more direct than `max_depth` for leaf-wise growth.

**Why include both?** They are different algorithmic families that may find different optimal patterns for different states. The champion selection will choose whichever performs better for each state on the actual data.

### 7.5 LSTM (`src/models/lstm_model.py`)

LSTM (Long Short-Term Memory) is a recurrent neural network architecture designed to capture long-range temporal dependencies that tree models cannot express. Unlike XGBoost/LightGBM, LSTM learns the temporal structure directly from the sequence — no lag feature engineering needed.

**Architecture:**
```
Input sequence (seq_len × 1) → LSTM(hidden=64, layers=2, dropout=0.2) → Dropout(0.2) → Linear(64 → 1) → output scalar
```

**Data preparation:**
1. `MinMaxScaler` normalises the target to [0, 1] — LSTMs train more stably on normalised data
2. The scaler is fit on training data only; the same fitted scaler is used at inference (preventing leakage)
3. Sequences of length `seq_len=30` weeks are created: `X[i] = total[i-30:i]`, `y[i] = total[i]`

**Training loop:**
```python
optimizer = Adam(lr=0.001)
loss = MSELoss()
for epoch in range(50):
    for batch_X, batch_y in DataLoader(dataset, batch_size=32):
        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()
```

**Monte Carlo Dropout confidence intervals:**
Standard neural networks produce point predictions. To get uncertainty estimates, we use MC Dropout — a technique that approximates Bayesian inference:
1. Keep `Dropout` layers active at inference time (normally they are disabled)
2. Run 100 forward passes through the network
3. Each pass produces a slightly different prediction (because different neurons are dropped each time)
4. The mean of 100 predictions is the point forecast
5. The 2.5th and 97.5th percentiles are the 95% CI bounds

This works because a dropout network can be interpreted as an ensemble of exponentially many models sharing parameters. The variance across forward passes estimates epistemic uncertainty (model uncertainty).

**Save/load:** Two files are saved per model:
- `{path}.pt` — PyTorch state dict (`torch.save(net.state_dict(), ...)`)
- `{path}.meta` — joblib file containing scaler, last sequence, config, and last date

`weights_only=True` in `torch.load` prevents arbitrary code execution from untrusted checkpoint files (PyTorch security best practice since 2.x).

---

## 8. Phase 5 — Training Pipeline & Model Selection

### 8.1 Cross-Validation (`src/pipeline/evaluate.py`)

**Expanding window CV design:**
```
Total weeks: N
min_train_size: 52 (at least 1 year of training data for the first fold)
step = (N - 52 - horizon) // n_splits

Fold 0: train = data[0 : 52],        test = data[52 : 52+horizon]
Fold 1: train = data[0 : 52+step],   test = data[52+step : 52+step+horizon]
Fold 2: train = data[0 : 52+2*step], test = data[52+2*step : 52+2*step+horizon]
...
```

Each fold trains a fresh model instance (no state sharing between folds) and evaluates on `horizon` future weeks. This exactly mimics the production use case — train on historical data, predict N weeks ahead.

**`calculate_metrics()` implementation:**
```python
mask = actual != 0  # exclude zeros from MAPE denominator
mape = mean(|actual[mask] - predicted[mask]| / actual[mask]) × 100
rmse = sqrt(mean((actual - predicted)²))
mae  = mean(|actual - predicted|)
```

The zero-mask for MAPE prevents division by zero. Since Pandera enforces `total >= 0`, zero values represent weeks with no recorded sales (rare but possible). Including them in MAPE would produce `inf` or `nan`.

**Error handling:** If a model raises during a CV fold (e.g. SARIMA fails to converge for a particular state), the fold is logged as a warning and skipped. The model is not disqualified — it gets credit for folds it completes successfully. If all folds fail, the model receives `mape=inf` and will not be selected as champion.

### 8.2 Model Selection (`src/pipeline/select.py`)

```python
ranked = sorted(
    cv_results.items(),
    key=lambda kv: (kv[1]["mape"], kv[1]["rmse"])
)
champion = ranked[0][0]
```

Primary sort by `mape` ascending — the model with the lowest average MAPE across CV folds wins. RMSE is the tiebreaker (same sort tuple position). All rankings are returned by `rank_models()` for logging and dashboard display.

**Why MAPE over RMSE?** RMSE is scale-sensitive — a 1000-unit error on a state with $10M monthly sales is tiny, but the same error on a state with $100K is enormous. MAPE normalises by the actual value, making it directly comparable across states with different sales volumes.

### 8.3 Model Registry (`src/pipeline/registry.py`)

A lightweight JSON registry at `models/registry.json` tracks all trained model artifacts:
```json
{
  "models": [
    {
      "name": "xgboost",
      "version": "20260429_154821",
      "path": "models/xgboost_20260429_154821",
      "metrics": {"mape": 6.5, "rmse": 42.0, "mae": 33.0},
      "is_champion": true,
      "state": null
    }
  ],
  "champion": {"name": "xgboost", "version": "20260429_154821", "path": "..."}
}
```

**Why a JSON file instead of PostgreSQL for the registry?** The registry is read at API startup and on every `/forecast` cache miss (to find the champion model path). A JSON file is an order of magnitude faster to read than a DB query and requires no DB connection. PostgreSQL stores the full training metrics and history; the registry is the operational index.

**Version scheme:** `YYYYMMDD_HHMMSS` — this makes versions sort chronologically by string comparison. The latest version is always retrievable without parsing.

When `is_champion=True` is set for a new model, all other models' `is_champion` flags are cleared atomically (within the same registry write). This prevents two champions existing simultaneously.

### 8.4 Training Orchestrator (`src/pipeline/train.py`)

The orchestrator ties together all phases in a single CLI entry point:

```
1. Load config (YAML)
2. run_pipeline(data_path) → clean weekly DataFrame
3. Optional state filter (for per-state or single-state runs)
4. For each model:
   a. time_series_cv() → avg MAPE/RMSE/MAE
   b. Fit on full data (for production forecasting)
5. select_best_model() → champion name
6. save() all fitted models to disk
7. save_model() in registry (mark champion)
8. Log summary rankings
9. Return run metadata
```

**`--skip-cv` flag:** CV with 5 folds × 5 models × 50 Optuna trials each takes hours. During development, `--skip-cv` bypasses CV (assigns `mape=0` to all) and goes straight to fitting. This allows rapid iteration on API and dashboard code without waiting for full training.

**Background retraining via FastAPI:** The `/retrain` endpoint triggers `_run_retraining()` via `BackgroundTasks`. FastAPI continues serving requests while retraining runs in a thread. When complete, Redis cache is invalidated so the next forecast request loads the new champion.

---

## 9. Phase 6 — REST API

### 9.1 FastAPI Application (`src/api/main.py`)

**Lifespan context manager:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()           # create tables if they don't exist
    redis_client.health_check()  # verify Redis connection
    yield               # app runs
    # shutdown: connections close automatically
```

The `lifespan` pattern (replacing the deprecated `@app.on_event("startup")`) runs startup code before the first request and shutdown code after the last. DB and Redis availability are checked at startup — if they fail, a warning is logged but the server still starts (degraded mode allows `/health` to report the outage).

**Docs disabled in production:**
```python
_docs_url = None if settings.environment == "production" else "/docs"
app = FastAPI(docs_url=_docs_url, redoc_url=_redoc_url)
```
Swagger UI exposes your full API schema to anyone who visits `/docs`. In production, this is an unnecessary information disclosure. Disabling it forces clients to use the documented API rather than exploring via browser.

**Middleware order matters.** FastAPI processes middleware in reverse registration order (last-added = first-executed on request). The order:
1. `CORSMiddleware` — added last, runs first on ingress (must process preflight OPTIONS before auth)
2. `RequestLoggingMiddleware` — added second, runs second (logs the request including CORS-resolved headers)
3. `SecurityHeadersMiddleware` — added first, runs last (adds headers to the already-processed response)

### 9.2 API Key Authentication (`src/api/auth.py`)

```python
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key:
        raise HTTPException(401, "Missing API key")
    if api_key not in settings.api_keys:
        raise HTTPException(401, "Invalid API key")
    return api_key
```

`auto_error=False` means FastAPI does not automatically return 401 when the header is absent — the function handles it explicitly with a descriptive message. This is preferred over the auto error because it lets us return our standardised error envelope rather than FastAPI's default.

Applied as a route dependency: `@router.post("/forecast", dependencies=[Depends(verify_api_key)])`. This runs before the route handler and short-circuits with 401 if auth fails. The `/health` endpoint deliberately has no `verify_api_key` dependency — load balancers and monitoring systems need to hit health checks without credentials.

`settings.api_keys` is a `list[str]` loaded from the Docker secret at startup. Multiple API keys are supported — each client can have its own key (enabling per-client rate limiting and revocation).

### 9.3 Rate Limiting (`src/api/rate_limiter.py`)

The rate limiter uses a **Redis sorted set sliding window** — more accurate than simple counter-based approaches:

```python
# Each request is stored as a member with score = unix timestamp
pipe.zremrangebyscore(key, 0, now - window_seconds)  # remove old entries
pipe.zadd(key, {str(now): now})                       # add this request
pipe.zcard(key)                                        # count in window
pipe.expire(key, window_seconds)                       # auto-expire the key
```

This accurately counts requests in the last `window_seconds` regardless of when within the window they occurred. A simple counter approach (incr + expire) can allow up to `2 × max_requests` in a bad timing scenario (requests at the end of one window + start of the next).

**Per API key, not per IP.** The key includes the `X-API-Key` header value:
```python
client_id = request.headers.get("X-API-Key") or request.client.host
key = f"rate_limit:{client_id}"
```
IP-based limiting fails behind a NAT (all users of an office share one IP). API-key-based limiting is per-client.

**Rate limit response headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 73
X-RateLimit-Reset: 1745930400
```
Standard headers allow clients to implement their own throttling before hitting 429.

**Limits by endpoint:**
- `/forecast` — 100 req/min: frequent requests expected (dashboard polling)
- `/retrain` — 5 req/hr: expensive operation, should not be triggered carelessly
- `/models`, `/health` — 200 req/min: read-only, cheap

### 9.4 Middleware (`src/api/middleware.py`)

**`RequestLoggingMiddleware`:** Measures wall-clock latency with `time.perf_counter()` (higher resolution than `time.time()`). Generates a UUID4 per request and injects it as `X-Request-ID` response header. Log line example:
```
INFO | HTTP request | method=POST path=/forecast status=200 ms=47.3 request_id=550e8400-...
```

**`SecurityHeadersMiddleware`** — six headers added to every response:

| Header | Value | Protection |
|---|---|---|
| `X-Content-Type-Options` | `nosniff` | Prevents MIME-type sniffing (browser won't execute a file labelled as text/plain as JavaScript) |
| `X-Frame-Options` | `DENY` | Prevents clickjacking (this API cannot be embedded in an iframe) |
| `X-XSS-Protection` | `1; mode=block` | Legacy IE/Chrome XSS filter (belt-and-suspenders) |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` | Forces HTTPS for 1 year |
| `Content-Security-Policy` | `default-src 'self'` | Only this origin can load resources (no CDN injection, no script injection) |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Prevents full URL leaking in Referer header across origins |

### 9.5 Custom Exception Handlers (`src/api/exceptions.py`)

All exceptions are mapped to clean JSON responses. Raw Python tracebacks are never returned to clients — they expose internal structure and may reveal file paths, library versions, or sensitive data.

```python
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception", path=str(request.url))  # full traceback in logs
    return JSONResponse(500, content=error_response("Internal server error", 500))  # clean to client
```

`logger.exception()` logs the full traceback (including exception type, message, and stack frames) to the log file. The client only sees "Internal server error". Security through obscurity is a layer — not a strategy — but there is no reason to hand attackers stack traces.

Custom exception types and their HTTP codes:

| Exception | HTTP | When |
|---|---|---|
| `StateNotFoundException` | 404 | State not in whitelist or not trained |
| `ModelNotTrainedException` | 503 | No champion model found in registry |
| `ForecastGenerationError` | 500 | Model load or prediction fails |
| `RateLimitExceededError` | 429 | Sliding window limit exceeded |
| `UnauthorizedError` | 401 | Missing or invalid API key |
| Uncaught `Exception` | 500 | Any other runtime error |

### 9.6 Forecast Route (`src/api/routes/forecast.py`)

**Request flow for `POST /forecast`:**
```
1. verify_api_key() — 401 if invalid
2. rate_limiter.check() — 429 if exceeded
3. _validate_state() — 404 if state not in whitelist (title-cased for normalisation)
4. redis.get_forecast(state, weeks) — return cached response if hit (sub-ms)
5. get_champion() — load model path from registry
6. forecaster.load(model_path) — deserialise model from disk
7. forecaster.predict(weeks) — generate horizon weeks of forecasts
8. redis.set_forecast(state, weeks, data) — cache for 24h
9. db.add(Forecast(...)) — persist each forecast point to PostgreSQL
10. return success_response(data)
```

**State normalisation:** `state.strip().title()` converts `"california"`, `"CALIFORNIA"`, `"  California  "` all to `"California"`. This prevents cache fragmentation where the same state has different cache keys under different capitalisations.

**Redis before DB:** The cache is checked before loading the model — model loading involves disk I/O and potentially hundreds of milliseconds. A cache hit returns in under 1ms without touching the model or database. The 24-hour TTL means most daytime requests are served from cache.

### 9.7 Input Sanitisation

**State whitelist:** `VALID_STATES` is a module-level set populated at startup from the training data. A state string that isn't in this set gets a 404, not a 500. This is a whitelist approach — unknown inputs are rejected rather than processed.

**Pydantic field constraints:**
```python
class ForecastRequest(BaseModel):
    state: str = Field(..., min_length=2, max_length=100)
    weeks: int = Field(default=8, ge=1, le=52)
```
`weeks=0` or `weeks=100` are rejected at the schema level before reaching route logic. `min_length=2` prevents empty strings.

**SQLAlchemy ORM (no raw SQL):** All database writes use ORM models:
```python
db.add(Forecast(state=state, forecast_date=..., predicted_value=...))
```
SQLAlchemy generates parameterised SQL (`INSERT INTO forecasts (state, ...) VALUES ($1, ...)`). The `state` variable is never interpolated into a SQL string, making SQL injection impossible through the normal code path.

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

## 10. Phase 7 — Streamlit Dashboard

*(Implementation in progress — section will be completed after Phase 7 is merged.)*

The dashboard is a 4-page Streamlit application that calls the FastAPI backend via `httpx`. It never loads models directly — all data flows through the API.

**Pages:**
1. `01_forecast.py` — state dropdown, weeks slider, Plotly chart with historical + forecast + CI band, forecast table
2. `02_model_comparison.py` — grouped bar chart of MAPE across all 5 models, champion highlighted, state×model MAPE heatmap
3. `03_training_history.py` — table of training runs from PostgreSQL, per-fold metrics expandable, timeline chart
4. `04_api_health.py` — live status of API/DB/Redis, response time chart, recent request logs

---

*This document is updated as each phase is completed.*
