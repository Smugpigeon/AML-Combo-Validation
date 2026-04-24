"""Environment-driven configuration.

All secrets come from env vars (loaded via docker-compose or `.env` file in
dev). We fail fast if a required secret is missing in production mode.
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path


def _require(key: str, default: str | None = None, allow_missing_in_dev: bool = False) -> str:
    val = os.environ.get(key, default)
    if val is None or val == "":
        env = os.environ.get("APP_ENV", "dev")
        if env == "dev" and allow_missing_in_dev:
            # Dev-only fallback
            val = secrets.token_urlsafe(32)
        else:
            raise RuntimeError(f"Missing required env var: {key}")
    return val


class Settings:
    APP_ENV: str = os.environ.get("APP_ENV", "dev")  # "dev" | "prod"
    DEBUG: bool = APP_ENV == "dev"

    # Domain config
    DOMAIN: str = os.environ.get("DOMAIN", "amlcombo.org")
    BASE_URL: str = os.environ.get("BASE_URL", f"https://{DOMAIN}")

    # Database
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://amlcombo:amlcombo@localhost:5432/amlcombo",
    )

    # Redis / Celery
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.environ.get("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND: str = os.environ.get("CELERY_RESULT_BACKEND", REDIS_URL)

    # Session / auth
    JWT_SECRET: str = _require("JWT_SECRET", allow_missing_in_dev=True)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24 * 7  # 7 days

    # Fernet symmetric key for encrypting user LLM API keys at rest.
    # MUST be a 32-byte urlsafe base64 string. Generate via:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    FERNET_KEY: str = _require("FERNET_KEY", allow_missing_in_dev=True)

    # File storage (per-user report artifacts)
    STORAGE_ROOT: Path = Path(os.environ.get("STORAGE_ROOT", "/data/storage"))

    # Kit assets (joblib bundle + reference stats) — readonly volume
    KIT_ASSETS_ROOT: Path = Path(
        os.environ.get("KIT_ASSETS_ROOT", "/data/canonical"),
    )

    # Rate limits (per user per day — MVP simple counter)
    DAILY_PATIENT_SUBMIT_LIMIT: int = int(
        os.environ.get("DAILY_PATIENT_SUBMIT_LIMIT", "10"),
    )

    # File upload caps (bytes) — reasonable for RNA-Seq counts
    MAX_UPLOAD_BYTES: int = int(
        os.environ.get("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)),  # 200 MB
    )

    # Password policy (light — researcher audience, not banking)
    MIN_PASSWORD_LEN: int = 8

    # CORS (only allow own domain in prod; permissive in dev)
    CORS_ORIGINS: list[str] = (
        ["*"] if APP_ENV == "dev"
        else [f"https://{DOMAIN}", f"https://www.{DOMAIN}"]
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
