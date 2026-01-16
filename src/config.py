"""
Configuration Management
========================

This module handles loading and parsing of the global project configuration
(YAML). It provides path resolution relative to the project root to ensure
scripts run correctly from any directory.
"""

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
