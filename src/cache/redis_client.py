import json
from typing import Optional

import redis

from src.config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(settings.environment)


class RedisClient:
    """Thin wrapper around redis.Redis with typed forecast cache helpers."""

    def __init__(self) -> None:
        self.client: redis.Redis = redis.from_url(
            settings.redis_url, decode_responses=True
        )
        self.ttl = settings.cache_ttl_seconds

    # ------------------------------------------------------------------
    # Forecast cache
    # ------------------------------------------------------------------

    def get_forecast(self, state: str, weeks: int) -> Optional[dict]:
        key = self._forecast_key(state, weeks)
        data = self.client.get(key)
        return json.loads(data) if data else None

    def set_forecast(self, state: str, weeks: int, data: dict) -> None:
        key = self._forecast_key(state, weeks)
        self.client.setex(key, self.ttl, json.dumps(data))
        logger.debug(f"Cache set: {key} (TTL {self.ttl}s)")

    def invalidate_state(self, state: str) -> int:
        """Delete all cached forecasts for a single state. Returns key count deleted."""
        pattern = f"forecast:{state.lower()}:*"
        keys = self.client.keys(pattern)
        if keys:
            deleted = self.client.delete(*keys)
            logger.info(f"Cache invalidated {deleted} keys for state '{state}'")
            return deleted
        return 0

    def invalidate_all(self) -> int:
        """Delete every forecast cache entry. Returns key count deleted."""
        keys = self.client.keys("forecast:*")
        if keys:
            deleted = self.client.delete(*keys)
            logger.info(f"Cache invalidated all {deleted} forecast keys")
            return deleted
        return 0

    # ------------------------------------------------------------------
    # Rate limiting helpers
    # ------------------------------------------------------------------

    def increment(self, key: str, expire_seconds: int) -> int:
        """Atomic incr + set TTL on first call. Returns current count."""
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, expire_seconds)
        results = pipe.execute()
        return results[0]

    def get_ttl(self, key: str) -> int:
        return self.client.ttl(key)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _forecast_key(state: str, weeks: int) -> str:
        return f"forecast:{state.lower()}:{weeks}"


redis_client = RedisClient()
