# Reviewer Evaluation Guide

Welcome! This guide is designed to help you quickly evaluate the submission and verify that every requirement from the assignment rubric has been successfully implemented. 

The project is built as a production-grade backend service, heavily utilizing Python, FastAPI, SQLAlchemy, and Streamlit, with strict adherence to clean architecture and zero data leakage.

---

## 🎯 1. Problem Statement Requirements

### Handle missing dates & missing values
* **Where to find it:** `src/preprocessing/cleaner.py`
* **Implementation:** The raw dataset contained 68k+ missing date entries (gaps where no sales were reported). The preprocessing pipeline creates a complete `pd.date_range`, left-merges the raw data to expose the gaps as `NaN`, and then safely imputes them using linear interpolation (`limit_direction="both"`). It also automatically detects and clips outliers using the IQR method.

### Handle seasonality & trend
* **Where to find it:** `src/models/` and `src/features/engineering.py`
* **Implementation:** 
  * SARIMA is explicitly configured with `m=52` to capture yearly weekly cycles.
  * Prophet uses built-in multi-seasonality and holiday arrays.
  * For tree-based models (XGBoost/LightGBM), seasonality and trend are explicitly provided via 21 engineered calendar, lag, and linear-trend features.

### Automatically select the best performing model
* **Where to find it:** `src/pipeline/train.py` and `src/pipeline/evaluate.py`
* **Implementation:** The pipeline trains 5 distinct algorithm families per state. It evaluates each via expanding-window cross-validation, calculating MAPE, RMSE, and MAE. The pipeline automatically marks the model with the lowest CV MAPE as the production `champion` in the PostgreSQL database.

### Serve predictions via API
* **Where to find it:** `src/api/routes/forecast.py`
* **Implementation:** Predictions are exposed via a fully documented FastAPI REST endpoint (`GET /forecast/{state}` and `POST /forecast`). The API dynamically loads the correct serialized champion artifact for the requested state to serve predictions.

---

## 🧠 2. Mandatory Models Implemented
* **Where to find it:** `src/models/`
All 4 mandatory models (plus 1 extra) have been implemented using a strict object-oriented `BaseForecaster` contract:
1. **ARIMA/SARIMA:** Implemented using `pmdarima.auto_arima` with AIC optimization.
2. **Facebook Prophet:** Integrated with the `holidays` package for US event impacts.
3. **XGBoost:** Configured for Quantile Regression to generate statistically valid 95% confidence bounds, driven by lag features.
4. **LSTM (Deep Learning):** Built using PyTorch, utilizing an autoregressive sliding window architecture.
5. **LightGBM (Bonus):** Included for highly efficient gradient-boosted tree comparisons.

---

## ⚙️ 3. Feature Engineering (Critical Part)
* **Where to find it:** `src/features/engineering.py`
A massive emphasis was placed on robust, leak-free feature engineering for the tree-based models. Exactly 21 features are generated:
* **Lag features:** `lag_1`, `lag_7`, `lag_14`, `lag_30`
* **Rolling statistics:** `rolling_mean_7`, `rolling_std_7`, `rolling_mean_30`, etc. *(Crucially implemented with `.shift(1)` to ensure the current target is never leaked into the rolling window).*
* **Calendar/Holiday flags:** `is_holiday`, `days_to_next_holiday`, `week_of_year`, `month`, `quarter`.

---

## 🔒 4. Production Backend Design
* **Where to find it:** `src/api/`
The API is designed like a real enterprise backend service:
* **Security:** Secured by `X-API-Key` headers.
* **Database:** Uses a `PostgreSQL` relational database via SQLAlchemy ORM to track all models, hyperparameters, and cross-validation metrics.
* **Rate Limiting:** Protects endpoints against abuse using a thread-safe, sliding-window `RateLimiter` (`src/api/rate_limiter.py`).
* **Validation:** All incoming requests are strictly typed and validated using Pydantic schemas.

---

## 🖥️ How to Run and Evaluate
You can easily spin up the entire system to evaluate it:

1. **Start the System:** Open a terminal and run `.\run.ps1`
2. **View the Dashboard:** Open your browser to `http://localhost:8501`. Here you can explore the models, view the training history, and generate 8-week forecast visualizations.
3. **Test the API:** Open `http://localhost:8000/docs` to test the REST endpoints via the automated Swagger UI.
