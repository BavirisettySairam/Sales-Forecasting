from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import declarative_base

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
    fold_metrics = Column(JSON)          # list[dict] — per-fold MAPE/RMSE/MAE
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
    forecast_date = Column(DateTime, nullable=False)
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
