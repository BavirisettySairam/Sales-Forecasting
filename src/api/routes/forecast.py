from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from src.api.auth import verify_api_key
from src.api.dependencies import get_db_dep
from src.api.exceptions import (
    ForecastGenerationError,
    ModelNotTrainedException,
    StateNotFoundException,
)
from src.api.rate_limiter import RateLimiter
from src.api.schemas.request import ForecastRequest
from src.api.schemas.response import ForecastData, ForecastPoint
from src.config.training import model_config
from src.db.models import Forecast
from src.pipeline.registry import get_champion
from src.utils.logger import logger
from src.utils.response import success_response

router = APIRouter(prefix="/forecast", tags=["forecast"])

_forecast_limiter = RateLimiter(max_requests=100, window_seconds=60)

VALID_STATES: set[str] = set()


def _validate_state(state: str) -> str:
    cleaned = state.strip().title()
    if VALID_STATES and cleaned not in VALID_STATES:
        raise StateNotFoundException(cleaned)
    return cleaned


async def _generate_forecast(state: str, weeks: int, db: Session) -> dict:

    champion = get_champion(state)
    if not champion:
        raise ModelNotTrainedException(state)

    model_name = champion["name"]
    model_path = champion["path"]
    mape = (champion.get("metrics") or {}).get("mape", 0.0)

    import yaml

    from src.models.lightgbm_model import LightGBMForecaster
    from src.models.lstm_model import LSTMForecaster
    from src.models.prophet_model import ProphetForecaster
    from src.models.sarima_model import SARIMAForecaster
    from src.models.xgboost_model import XGBoostForecaster

    model_cls_map = {
        "sarima": SARIMAForecaster,
        "prophet": ProphetForecaster,
        "xgboost": XGBoostForecaster,
        "lightgbm": LightGBMForecaster,
        "lstm": LSTMForecaster,
    }

    model_cls = model_cls_map.get(model_name)
    if not model_cls:
        raise ForecastGenerationError(f"Unknown model type: {model_name}")

    try:
        with open("config/training_config.yaml") as f:
            config = model_config(yaml.safe_load(f))
        forecaster = model_cls(config)
        forecaster.load(model_path)
        fc_df = forecaster.predict(weeks)
    except Exception as exc:
        logger.exception("Forecast failed", model=model_name, state=state)
        raise ForecastGenerationError(str(exc)) from exc

    points = [
        ForecastPoint(
            date=str(row["date"])[:10],
            predicted_value=round(float(row["predicted_value"]), 2),
            lower_bound=round(float(row["lower_bound"]), 2),
            upper_bound=round(float(row["upper_bound"]), 2),
        )
        for _, row in fc_df.iterrows()
    ]

    data = ForecastData(
        state=state,
        model_used=model_name,
        model_mape=round(mape, 4),
        forecast=points,
    ).model_dump()

    try:
        for pt in points:
            db.add(
                Forecast(
                    state=state,
                    model_name=model_name,
                    forecast_date=datetime.fromisoformat(pt.date).replace(
                        tzinfo=timezone.utc
                    ),
                    predicted_value=pt.predicted_value,
                    lower_bound=pt.lower_bound,
                    upper_bound=pt.upper_bound,
                )
            )
        db.commit()
    except Exception:
        logger.warning("Forecast DB save failed — continuing without persistence")

    return data


@router.post("", dependencies=[Depends(verify_api_key)])
async def post_forecast(
    body: ForecastRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db_dep),
):
    headers = await _forecast_limiter.check(request)
    for k, v in headers.items():
        response.headers[k] = v
    state = _validate_state(body.state)
    data = await _generate_forecast(state, body.weeks, db)
    return success_response(data=data, message="Forecast generated successfully")


@router.get("/{state}", dependencies=[Depends(verify_api_key)])
async def get_forecast_by_state(
    state: str,
    request: Request,
    response: Response,
    weeks: int = 8,
    db: Session = Depends(get_db_dep),
):
    headers = await _forecast_limiter.check(request)
    for k, v in headers.items():
        response.headers[k] = v
    state = _validate_state(state)
    data = await _generate_forecast(state, weeks, db)
    return success_response(data=data, message="Forecast generated successfully")
