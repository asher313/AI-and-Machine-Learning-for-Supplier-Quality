# src/sqm_ai/settings.py
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sqm_env: str = "dev"
    log_level: str = "INFO"
    anthropic_api_key: SecretStr = SecretStr("")
    hana_host: str = "hana.northlake.internal"
    hana_port: int = 30015
    hana_user: str = "sqm_readonly"
    hana_password: SecretStr = SecretStr("")
    database_url: str = (
        "postgresql+psycopg://sqm@localhost"
        ":5432/sqm_analytics"
    )
    data_bucket: str = "northlake-sqm-data"
    model_bucket: str = "northlake-sqm-models"


@lru_cache
def get_settings() -> Settings:
    """Read .env and the environment once; reuse everywhere."""
    return Settings()
