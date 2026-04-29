from pydantic import BaseModel
from typing import Any


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    code: int
    details: Any = None
    timestamp: str
    request_id: str
