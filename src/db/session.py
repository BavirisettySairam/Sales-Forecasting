from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import settings
from src.db.models import Base
from src.utils.logger import setup_logger

logger = setup_logger(settings.environment)

_db_url = settings.database_url
if _db_url.startswith("sqlite"):
    engine = create_engine(
        _db_url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
else:
    engine = create_engine(
        _db_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables if they don't exist. Alembic handles schema migrations."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialised")
    except Exception as exc:
        logger.error(f"Database initialisation failed: {exc}")
        raise


def check_db_health() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager that yields a SQLAlchemy session and handles commit/rollback."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session per request."""
    with get_db_session() as session:
        yield session
