import time

from fastapi import Request, HTTPException
from redis import Redis


class RateLimiter:
    def __init__(self, redis_client: Redis, max_requests: int = 100, window_seconds: int = 60):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window_seconds

    async def check(self, request: Request) -> dict:
        client_id = request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")
        key = f"rate_limit:{client_id}"

        now = int(time.time())
        window_start = now - self.window

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, self.window)
        results = pipe.execute()

        count = results[2]
        remaining = max(self.max_requests - count, 0)
        reset_at = now + self.window

        if count > self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window}s.",
                headers={
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "Retry-After": str(self.window),
                },
            )

        return {
            "X-RateLimit-Limit": str(self.max_requests),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_at),
        }
