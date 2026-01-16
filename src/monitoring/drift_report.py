"""
Drift Monitoring Module
=======================

This script detects distributional shifts in input data. It compares new
inference batches against the training reference using statistical tests
(KS-test) to alert MLOps teams of potential model degradation.
"""

import pandas as pd
import argparse
from pathlib import Path
from scipy.stats import ks_2samp
import json
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_drift(reference_file: Path, current_file: Path, output_file: Path):
    logger.info(f"Loading reference: {reference_file}")
    df_ref = pd.read_parquet(reference_file)
    logger.info(f"Loading current: {current_file}")
    df_curr = pd.read_parquet(current_file)
    
    # Identify feature columns (numeric)
    feature_cols = [c for c in df_ref.columns if c not in ['subject_id', 'label'] and pd.api.types.is_numeric_dtype(df_ref[c])]
    
    drift_report = {}
    alert_count = 0
    
    for col in feature_cols:
        if col not in df_curr.columns:
            continue
            
        # Kolmogorov-Smirnov test
        stat, p_value = ks_2samp(df_ref[col], df_curr[col])
        
        drift_detected = p_value < 0.05
        if drift_detected:
            alert_count += 1
            
        drift_report[col] = {
            'ks_stat': float(stat),
            'p_value': float(p_value),
            'drift_detected': drift_detected,
            'mean_ref': float(df_ref[col].mean()),
            'mean_curr': float(df_curr[col].mean())
        }
        
    logger.info(f"Drift detection complete. {alert_count} features showed significant drift.")
    
    with open(output_file, "w") as f:
        json.dump(drift_report, f, indent=4)
    logger.info(f"Report saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True, help="Path to reference features parquet")
    parser.add_argument("--current", type=Path, required=True, help="Path to new batch features parquet")
    parser.add_argument("--output", type=Path, default="reports/drift_report.json")
    
    args = parser.parse_args()
    compute_drift(args.reference, args.current, args.output)
