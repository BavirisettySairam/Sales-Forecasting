import time

from fastapi import HTTPException, Request


class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._store: dict[str, list[float]] = {}

    async def check(self, request: Request) -> dict:
        client_id = request.headers.get("X-API-Key") or (
            request.client.host if request.client else "unknown"
        )

        now = time.time()
        window_start = now - self.window

        # Clean up old entries and count
        history = self._store.get(client_id, [])
        history = [t for t in history if t > window_start]

        count = len(history)
        remaining = max(self.max_requests - count, 0)
        reset_at = int(now + self.window)

        if count >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded. Max {self.max_requests} "
                    f"requests per {self.window}s."
                ),
                headers={
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "Retry-After": str(self.window),
                },
            )

        history.append(now)
        self._store[client_id] = history

        return {
            "X-RateLimit-Limit": str(self.max_requests),
            "X-RateLimit-Remaining": str(remaining - 1),
            "X-RateLimit-Reset": str(reset_at),
        }
