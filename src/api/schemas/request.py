from pydantic import BaseModel, Field
from typing import Optional


class ForecastRequest(BaseModel):
    state: str = Field(..., min_length=2, max_length=100, description="US state name")
    weeks: int = Field(default=8, ge=1, le=52, description="Forecast horizon in weeks")


class RetrainRequest(BaseModel):
    states: Optional[list[str]] = Field(default=None, description="States to retrain. None = all states.")
