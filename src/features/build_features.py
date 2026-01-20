"""
Feature Engineering Script
==========================

This script orchestrates the Feature Extraction process. It loads processed
windowed data (Parquet), applies the FeatureExtractor, and saves the
resulting feature matrix for model training.
"""

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.features.feature_extraction import FeatureExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    config = load_config()
    processed_path = PROJECT_ROOT / config["data"]["processed_path"]
    input_file = processed_path / "windows.parquet"
    output_file = processed_path / "features.parquet"

    if not input_file.exists():
        logger.error(f"{input_file} not found. Run 'make preprocess' first.")
        return

    logger.info(f"Loading windows from {input_file}...")
    df_windows = pd.read_parquet(input_file)

    logger.info("Extracting features (this may take a while)...")
    extractor = FeatureExtractor()
    df_features = extractor.process_windows(df_windows)

    logger.info(f"Saving {len(df_features)} feature vectors to {output_file}...")
    df_features.to_parquet(output_file, index=False)
    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    main()
