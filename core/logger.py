"""Structured logging configuration."""

import logging
import sys
from pathlib import Path
from typing import Optional, Any
import structlog


def setup_logging(
    level: str = "INFO",
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Setup structured logging with both stdout and optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Log format string (supports structlog placeholders)
        log_file: Optional path to log file
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Handlers
    handlers: list = [
        logging.StreamHandler(sys.stdout)
    ]

    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all levels
        handlers.append(file_handler)

    # Create formatters
    formatter = logging.Formatter(format_string)

    # Apply formatters to handlers
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def get_logger(
    name: str,
    extra: Optional[dict] = None,
    config: Optional[dict] = None,
) -> Any:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)
        extra: Additional fields to add to every log entry
        config: Optional logging configuration dict

    Returns:
        Structured logger instance
    """
    # If config provided, setup logging
    if config:
        setup_logging(
            level=config.get("level", "INFO"),
            format_string=config.get("format", "%(message)s"),
            log_file=Path(config.get("file")) if config.get("file") else None,
            max_bytes=config.get("max_bytes", 10 * 1024 * 1024),
            backup_count=config.get("backup_count", 5),
        )

    # Get logger
    logger = structlog.get_logger(name)

    if extra:
        return logger.bind(**extra)
    return logger


__all__ = ["setup_logging", "get_logger"]