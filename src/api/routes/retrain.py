from fastapi import APIRouter, BackgroundTasks, Depends, Request

from src.api.auth import verify_api_key
from src.api.rate_limiter import RateLimiter
from src.api.schemas.request import RetrainRequest
from src.utils.logger import logger
from src.utils.response import success_response

router = APIRouter(prefix="/retrain", tags=["retrain"])

_retrain_limiter = RateLimiter(max_requests=5, window_seconds=3600)


def _run_retraining(states: list[str] | None):
    import yaml

    from src.config.training import configured_data_path
    from src.pipeline.train import run_training_all_states

    try:
        config_path = "config/training_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        result = run_training_all_states(
            data_path=configured_data_path(config),
            config_path=config_path,
            models_to_run=None,
            states=states,
            skip_cv=False,
        )
        logger.info(
            "Retraining complete",
            states_succeeded=result["states_succeeded"],
            states_failed=result["states_failed"],
        )
    except Exception as exc:
        logger.error("Retraining failed", error=str(exc))


@router.post("", dependencies=[Depends(verify_api_key)])
async def trigger_retrain(
    body: RetrainRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    await _retrain_limiter.check(request)

    background_tasks.add_task(_run_retraining, body.states)
    logger.info("Retraining scheduled", states=body.states)
    return success_response(
        data={"scheduled": True, "states": body.states or "all"},
        message="Retraining started in background.",  # noqa: E501
    )
