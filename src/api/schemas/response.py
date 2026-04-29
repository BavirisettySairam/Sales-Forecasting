from typing import Any

from pydantic import BaseModel


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


class ModelInfo(BaseModel):
    state: str
    model_name: str
    is_champion: bool
    avg_mape: float
    avg_rmse: float
    avg_mae: float
    trained_at: str


class StandardResponse(BaseModel):
    status: str
    data: Any
    message: str
    timestamp: str
    request_id: str
