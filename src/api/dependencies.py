from typing import Generator

from sqlalchemy.orm import Session

from src.db.session import get_db
from src.cache.redis_client import redis_client


def get_db_dep() -> Generator[Session, None, None]:
    yield from get_db()


def get_redis():
    return redis_client.client
