from typing import Generator

from sqlalchemy.orm import Session

from src.cache.redis_client import redis_client
from src.db.session import get_db


def get_db_dep() -> Generator[Session, None, None]:
    yield from get_db()


def get_redis():
    return redis_client.client
