from fastapi import APIRouter, BackgroundTasks, Depends, Request

from src.api.auth import verify_api_key
from src.api.rate_limiter import RateLimiter
from src.api.dependencies import get_redis
from src.api.schemas.request import RetrainRequest
from src.cache.redis_client import redis_client
from src.utils.logger import logger
from src.utils.response import success_response

router = APIRouter(prefix="/retrain", tags=["retrain"])

_retrain_limiter = RateLimiter(redis_client=None, max_requests=5, window_seconds=3600)


def _run_retraining(states: list[str] | None):
    import yaml
    from src.pipeline.train import run_training

    config = yaml.safe_load(open("config/training_config.yaml"))
    try:
        result = run_training(
            data_path="data/raw/dataset.csv",
            config_path="config/training_config.yaml",
            models_to_run=None,
            state_filter=states[0] if states and len(states) == 1 else None,
            skip_cv=False,
        )
        redis_client.invalidate_all()
        logger.info("Retraining complete", champion=result["champion"])
    except Exception as exc:
        logger.error("Retraining failed", error=str(exc))


@router.post("", dependencies=[Depends(verify_api_key)])
async def trigger_retrain(
    body: RetrainRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    redis=Depends(get_redis),
):
    _retrain_limiter.redis = redis
    await _retrain_limiter.check(request)

    background_tasks.add_task(_run_retraining, body.states)
    logger.info("Retraining scheduled", states=body.states)
    return success_response(
        data={"scheduled": True, "states": body.states or "all"},
        message="Retraining started in background. Cache will be invalidated on completion.",
    )
