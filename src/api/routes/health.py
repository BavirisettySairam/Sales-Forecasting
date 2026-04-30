from fastapi import APIRouter, Depends

from src.api.dependencies import get_redis
from src.db.session import check_db_health
from src.utils.response import success_response

router = APIRouter(tags=["health"])


@router.get("/health")
def health(redis=Depends(get_redis)):
    db_ok = check_db_health()
    redis_ok = False
    try:
        redis_ok = bool(redis.ping())
    except Exception:
        redis_ok = False

    status = "healthy" if db_ok and redis_ok else "degraded"

    return success_response(
        data={
            "api": "ok",
            "database": "ok" if db_ok else "unavailable",
            "redis": "ok" if redis_ok else "unavailable",
        },
        message=status,
    )
