"""Structured logging setup with automatic secret redaction."""

import sys
from typing import Any

from loguru import logger

from doot.config import DootSettings


def configure_logging(settings: DootSettings, *, level: str = "INFO") -> None:
    """Configure loguru console logging and redact configured API secrets."""
    secrets = settings.secret_values()

    def _redact_patcher(record: Any) -> None:
        message = str(record["message"])
        for secret in secrets:
            if secret and secret in message:
                message = message.replace(secret, "[REDACTED_SECRET]")
        record["message"] = message

    logger.remove()
    logger.configure(patcher=_redact_patcher)
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level=level,
        format=log_format,
    )
