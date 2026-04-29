import sys
from pathlib import Path

from loguru import logger


def setup_logger(environment: str = "development") -> "logger":
    """Configure loguru with structured logging, file rotation, and JSON for production."""  # noqa: E501
    logger.remove()

    if environment == "production":
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",  # noqa: E501
            level="INFO",
            serialize=True,
        )
    else:
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",  # noqa: E501
            level="DEBUG",
            colorize=True,
        )

    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="DEBUG")

    return logger
