from fastapi import Request
from fastapi.responses import JSONResponse

from src.utils.logger import logger
from src.utils.response import error_response


class StateNotFoundException(Exception):
    def __init__(self, state: str):
        self.state = state
        super().__init__(f"State '{state}' not found")


class ModelNotTrainedException(Exception):
    def __init__(self, state: str):
        self.state = state
        super().__init__(f"No trained model for state '{state}'")


class ForecastGenerationError(Exception):
    def __init__(self, detail: str):
        super().__init__(detail)


class UnauthorizedError(Exception):
    pass


class RateLimitExceededError(Exception):
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        super().__init__(f"Rate limit exceeded: {limit} req/{window}s")


async def state_not_found_handler(
    request: Request, exc: StateNotFoundException
) -> JSONResponse:
    return JSONResponse(status_code=404, content=error_response(str(exc), 404))


async def model_not_trained_handler(
    request: Request, exc: ModelNotTrainedException
) -> JSONResponse:
    return JSONResponse(status_code=503, content=error_response(str(exc), 503))


async def forecast_error_handler(
    request: Request, exc: ForecastGenerationError
) -> JSONResponse:
    logger.error("Forecast generation failed", detail=str(exc))
    return JSONResponse(
        status_code=500, content=error_response("Forecast generation failed", 500)
    )


async def unauthorized_handler(
    request: Request, exc: UnauthorizedError
) -> JSONResponse:
    return JSONResponse(status_code=401, content=error_response("Unauthorized", 401))


async def rate_limit_handler(
    request: Request, exc: RateLimitExceededError
) -> JSONResponse:
    resp = JSONResponse(
        status_code=429,
        content=error_response(str(exc), 429),
        headers={"Retry-After": str(exc.window)},
    )
    return resp


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception", path=str(request.url))
    return JSONResponse(
        status_code=500, content=error_response("Internal server error", 500)
    )
