import uuid
from datetime import datetime, timezone
from typing import Any, Optional


def success_response(data: Any, message: str = "Success") -> dict:
    return {
        "status": "success",
        "data": data,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4()),
    }


def error_response(message: str, code: int, details: Optional[Any] = None) -> dict:
    return {
        "status": "error",
        "message": message,
        "code": code,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4()),
    }
