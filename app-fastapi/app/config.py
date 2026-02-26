from __future__ import annotations

import logging
import os
import sys
from typing import Any

from pydantic import BaseModel, Field

loguru_logger: Any | None

try:
    from loguru import logger as _loguru_logger

    loguru_logger = _loguru_logger
except Exception:  # pragma: no cover
    loguru_logger = None


def _default_cors_origins() -> list[str]:
    return [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://localhost:3000",
        "https://localhost:8000",
    ]


def _load_cors_origins_from_env() -> list[str]:
    raw_origins = os.getenv("BACKEND_CORS_ORIGINS", "").strip()
    if not raw_origins:
        return _default_cors_origins()
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


class Settings(BaseModel):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "INSY-674 Employability Prediction API")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    BACKEND_CORS_ORIGINS: list[str] = Field(default_factory=_load_cors_origins_from_env)


def setup_logging(settings: Settings) -> None:
    log_level = settings.LOG_LEVEL.upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if loguru_logger is not None:
        loguru_logger.remove()
        loguru_logger.add(
            sys.stderr,
            level=log_level,
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )


def get_logger(name: str):
    if loguru_logger is not None:
        return loguru_logger.bind(module=name)
    return logging.getLogger(name)


settings = Settings()
