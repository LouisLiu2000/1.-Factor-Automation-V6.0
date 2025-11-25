"""Centralised logging helpers used across CLI entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(*, log_file: Optional[Path] = None, level: str = "INFO") -> logging.Logger:
    """Configure root logger with console/file handlers.

    Parameters
    ----------
    log_file:
        Optional path to a log file. Parent directories are created on demand.
    level:
        Log level name (case insensitive). Defaults to ``INFO``.
    """

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(numeric_level)
        for handler in root.handlers:
            handler.setLevel(numeric_level)
        return root

    root.setLevel(numeric_level)
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(numeric_level)
    root.addHandler(console)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return child logger using the shared configuration."""

    return logging.getLogger(name)


def attach_file_handler(path: Path, *, level: Optional[str] = None) -> logging.Handler:
    """Attach a file handler to the root logger, returning the handler."""

    root = logging.getLogger()
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(formatter)
    if level is not None:
        handler.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    root.addHandler(handler)
    return handler
