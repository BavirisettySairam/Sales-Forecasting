from fastapi import APIRouter, Depends

from src.api.auth import verify_api_key
from src.api.dependencies import get_redis
from src.api.rate_limiter import RateLimiter
from src.pipeline.registry import list_models
from src.utils.response import success_response

router = APIRouter(prefix="/models", tags=["models"])

_models_limiter = RateLimiter(redis_client=None, max_requests=200, window_seconds=60)


@router.get("", dependencies=[Depends(verify_api_key)])
async def get_all_models(redis=Depends(get_redis)):
    models = list_models()
    return success_response(data=models, message=f"{len(models)} model(s) found")


@router.get("/{state}", dependencies=[Depends(verify_api_key)])
async def get_models_for_state(state: str, redis=Depends(get_redis)):
    state_clean = state.strip().title()
    filtered = list_models(state_clean)
    return success_response(
        data=filtered, message=f"{len(filtered)} model(s) for {state_clean}"
    )
