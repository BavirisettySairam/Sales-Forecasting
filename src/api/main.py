from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.exceptions import (
    ForecastGenerationError,
    ModelNotTrainedException,
    RateLimitExceededError,
    StateNotFoundException,
    UnauthorizedError,
    forecast_error_handler,
    generic_exception_handler,
    model_not_trained_handler,
    rate_limit_handler,
    state_not_found_handler,
    unauthorized_handler,
)
from src.api.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware
from src.api.routes import forecast, health, models, retrain
from src.config.settings import settings
from src.db.session import init_db
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up", environment=settings.environment)
    try:
        init_db()
        logger.info("Database initialised")
    except Exception as exc:
        logger.warning("DB init failed (continuing)", error=str(exc))

    try:
        from src.cache.redis_client import redis_client
        redis_client.health_check()
        logger.info("Redis connected")
    except Exception as exc:
        logger.warning("Redis unavailable (continuing)", error=str(exc))

    yield

    logger.info("API shutting down")


_docs_url = None if settings.environment == "production" else "/docs"
_redoc_url = None if settings.environment == "production" else "/redoc"

app = FastAPI(
    title="Forecasting System API",
    description="Production-ready time series forecasting with automatic model selection",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=_docs_url,
    redoc_url=_redoc_url,
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(StateNotFoundException, state_not_found_handler)
app.add_exception_handler(ModelNotTrainedException, model_not_trained_handler)
app.add_exception_handler(ForecastGenerationError, forecast_error_handler)
app.add_exception_handler(UnauthorizedError, unauthorized_handler)
app.add_exception_handler(RateLimitExceededError, rate_limit_handler)
app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(models.router)
app.include_router(retrain.router)
