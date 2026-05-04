"""logging_config.py — Centralised logging via Loguru."""

from __future__ import annotations

import sys

from loguru import logger

from config import settings

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()

    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}:{function}:{line}</cyan> "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=False,
    )

    log_dir = settings.out_logs
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "pipeline_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="14 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    _CONFIGURED = True


def get_logger(name=None):
    _configure()
    return logger.bind(name=name) if name else logger


_configure()
