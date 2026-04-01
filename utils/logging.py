"""Logging configuration for the project."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    fmt: Optional[str] = None,
) -> logging.Logger:
    """Create and configure a logger instance.

    Args:
        name: Logger name, typically ``__name__``.
        level: Logging level string.
        fmt: Optional format string override.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt or "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger
