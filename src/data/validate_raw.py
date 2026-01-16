import os
import zipfile
from pathlib import Path
import logging

"""
Raw Data Validation Script
==========================

This script ensures the WESAD dataset is correctly downloaded and extracted.
It checks for the presence of specific subject folders and handles the
unzipping process if necessary.
"""

from src.config import load_config, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_and_unzip():
    config = load_config()
    raw_path = PROJECT_ROOT / config['data']['raw_path']
    zip_path = PROJECT_ROOT / config['data']['zip_path']

    # Ensure raw directory exists
    raw_path.mkdir(parents=True, exist_ok=True)

    # Check if subjects exist
    subjects = [f"S{i}" for i in range(2, 18)]
    missing_subjects = [s for s in subjects if not (raw_path / "WESAD" / s).exists()]

    if not missing_subjects:
        logger.info("All WESAD subjects found in raw directory.")
        return

    logger.info(f"Missing subjects: {missing_subjects}")
    
    if zip_path.exists():
        logger.info(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_path)
        logger.info("Unzip complete.")
        
        # Verify again
        missing_subjects = [s for s in subjects if not (raw_path / "WESAD" / s).exists()]
        if not missing_subjects:
            logger.info("Data validation successful.")
        else:
            logger.error(f"Still missing subjects after unzip: {missing_subjects}")
    else:
        logger.error(f"WESAD.zip not found at {zip_path} and raw data is incomplete.")
        raise FileNotFoundError(f"Please download WESAD.zip to {zip_path}")

if __name__ == "__main__":
    validate_and_unzip()
