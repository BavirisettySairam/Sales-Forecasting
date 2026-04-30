from typing import Generator

from sqlalchemy.orm import Session

from src.db.session import get_db


def get_db_dep() -> Generator[Session, None, None]:
    yield from get_db()

