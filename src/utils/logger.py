"""
Logging Utility
===============

This module provides a consistent logging configuration for the application.
It ensures logs are formatted correctly and output to the console for
debugging and monitoring purposes.
"""

import logging
import sys


def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
