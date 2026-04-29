from typing import Any

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    code: int
    details: Any = None
    timestamp: str
    request_id: str
