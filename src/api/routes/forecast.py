from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from src.api.auth import verify_api_key
from src.api.dependencies import get_db_dep, get_redis
from src.api.exceptions import ForecastGenerationError, ModelNotTrainedException, StateNotFoundException
from src.api.rate_limiter import RateLimiter
from src.api.schemas.request import ForecastRequest
from src.api.schemas.response import ForecastData, ForecastPoint
from src.cache.redis_client import RedisClient
from src.db.models import Forecast
from src.pipeline.registry import get_champion, load_model
from src.utils.logger import logger
from src.utils.response import success_response

router = APIRouter(prefix="/forecast", tags=["forecast"])

_forecast_limiter = RateLimiter(redis_client=None, max_requests=100, window_seconds=60)

VALID_STATES: set[str] = set()


def _validate_state(state: str) -> str:
    cleaned = state.strip().title()
    if VALID_STATES and cleaned not in VALID_STATES:
        raise StateNotFoundException(cleaned)
    return cleaned


def _load_forecaster(state: str):
    champion = get_champion()
    if not champion:
        champion_all = get_champion()
        if not champion_all:
            raise ModelNotTrainedException(state)
    return champion


async def _generate_forecast(state: str, weeks: int, db: Session, redis_raw) -> dict:
    rc = RedisClient.__new__(RedisClient)
    rc.client = redis_raw
    rc.ttl = 86400
    rc.prefix = "forecast"

    cached = rc.get_forecast(state, weeks)
    if cached:
        logger.info("Cache hit", state=state, weeks=weeks)
        return cached

    champion = get_champion()
    if not champion:
        raise ModelNotTrainedException(state)

    model_name = champion["name"]
    model_path = champion["path"]
    mape = champion.get("metrics", {}).get("mape", 0.0)

    from src.models.sarima_model import SARIMAForecaster
    from src.models.prophet_model import ProphetForecaster
    from src.models.xgboost_model import XGBoostForecaster
    from src.models.lightgbm_model import LightGBMForecaster
    from src.models.lstm_model import LSTMForecaster
    import yaml

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
        config = yaml.safe_load(open("config/training_config.yaml"))
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

    rc.set_forecast(state, weeks, data)

    for pt in points:
        db.add(Forecast(
            state=state,
            model_name=model_name,
            forecast_date=datetime.fromisoformat(pt.date).replace(tzinfo=timezone.utc),
            predicted_value=pt.predicted_value,
            lower_bound=pt.lower_bound,
            upper_bound=pt.upper_bound,
        ))
    db.commit()

    return data


@router.post("", dependencies=[Depends(verify_api_key)])
async def post_forecast(
    body: ForecastRequest,
    request: Request,
    db: Session = Depends(get_db_dep),
    redis=Depends(get_redis),
):
    _forecast_limiter.redis = redis
    await _forecast_limiter.check(request)
    state = _validate_state(body.state)
    data = await _generate_forecast(state, body.weeks, db, redis)
    return success_response(data=data, message="Forecast generated successfully")


@router.get("/{state}", dependencies=[Depends(verify_api_key)])
async def get_forecast_by_state(
    state: str,
    request: Request,
    weeks: int = 8,
    db: Session = Depends(get_db_dep),
    redis=Depends(get_redis),
):
    _forecast_limiter.redis = redis
    await _forecast_limiter.check(request)
    state = _validate_state(state)
    data = await _generate_forecast(state, weeks, db, redis)
    return success_response(data=data, message="Forecast generated successfully")
