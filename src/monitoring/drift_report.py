"""
Drift Monitoring Module
=======================

This script detects distributional shifts in input data. It compares new
inference batches against the training reference using statistical tests
(KS-test) to alert MLOps teams of potential model degradation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_monitoring_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts scalar features from signal windows for monitoring.
    For array columns, calculates Mean and Std.
    """
    monitor_df = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col in ["subject_id", "label", "label_str"]:
            continue

        # Check if column contains arrays/lists
        if isinstance(df[col].iloc[0], (list, np.ndarray)):
            # Calculate simple stats for monitoring distribution
            # stack values to 2D array if possible for speed, else apply
            try:
                # Attempt fast vectorization
                matrix = np.stack(df[col].values)
                monitor_df[f"{col}_mean"] = matrix.mean(axis=1)
                monitor_df[f"{col}_std"] = matrix.std(axis=1)
            except Exception:
                # Fallback
                monitor_df[f"{col}_mean"] = df[col].apply(np.mean)
                monitor_df[f"{col}_std"] = df[col].apply(np.std)
        elif pd.api.types.is_numeric_dtype(df[col]):
            monitor_df[col] = df[col]

    return monitor_df


def compute_drift(reference_data, current_data):
    """
    Calculates KS-statistic for features between reference and current batches.
    Args:
        reference_data: DataFrame or Path to parquet
        current_data: DataFrame or Path to parquet
    """
    # Load Data if Paths
    if isinstance(reference_data, (str, Path)):
        df_ref = pd.read_parquet(reference_data)
    else:
        df_ref = reference_data.copy()

    if isinstance(current_data, (str, Path)):
        df_curr = pd.read_parquet(current_data)
    else:
        df_curr = current_data.copy()

    # Extract Scalar Features for Monitoring
    logger.info("Extracting monitoring features...")
    df_ref_scalar = extract_monitoring_features(df_ref)
    df_curr_scalar = extract_monitoring_features(df_curr)

    drift_report = {}
    alert_count = 0

    # Compare Distributions
    feature_cols = df_ref_scalar.columns

    for col in feature_cols:
        if col not in df_curr_scalar.columns:
            continue

        # Kolmogorov-Smirnov test
        # Null Hypothesis: distributions are the same
        stat, p_value = ks_2samp(df_ref_scalar[col], df_curr_scalar[col])

        drift_detected = p_value < 0.05
        if drift_detected:
            alert_count += 1

        drift_report[col] = {
            "ks_stat": float(stat),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "mean_ref": float(df_ref_scalar[col].mean()),
            "mean_curr": float(df_curr_scalar[col].mean()),
        }

    return drift_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to reference features parquet",
    )
    parser.add_argument(
        "--current", type=Path, required=True, help="Path to new batch features parquet"
    )
    parser.add_argument("--output", type=Path, default="reports/drift_report.json")

    args = parser.parse_args()
    compute_drift(args.reference, args.current, args.output)
