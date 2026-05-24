"""
utils/logger.py
===============
Call setup_logging() once at the top of app.py.
Every other module just does:  logger = logging.getLogger(__name__)
"""

import logging
import sys
from src.config import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT


def setup_logging() -> None:
    """Configure root logger for the whole application."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Silence noisy third-party loggers
    for noisy in ("httpx", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
