from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://app:password@localhost:5432/forecasting"
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"
    cors_origins: list[str] = ["http://localhost:8501"]

    # Authentication — loaded from Docker secret in production
    api_keys: list[str] = ["forecasting-api-key-2026"]

    # Paths
    model_artifacts_dir: str = "models"
    training_config_path: str = "config/training_config.yaml"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def load_secrets(self) -> None:
        """Replace placeholder credentials with Docker secret values when available."""
        db_secret = Path("/run/secrets/db_password")
        if db_secret.exists():
            password = db_secret.read_text().strip()
            self.database_url = self.database_url.replace("password", password)

        api_key_secret = Path("/run/secrets/api_key")
        if api_key_secret.exists():
            key = api_key_secret.read_text().strip()
            if key not in self.api_keys:
                self.api_keys.append(key)


settings = Settings()
settings.load_secrets()
