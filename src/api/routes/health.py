from fastapi import APIRouter

from src.db.session import check_db_health
from src.utils.response import success_response

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    db_ok = check_db_health()
    status = "healthy" if db_ok else "degraded"

    return success_response(
        data={
            "api": "ok",
            "database": "ok" if db_ok else "unavailable",
        },
        message=status,
    )
