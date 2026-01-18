import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
project_root = Path.cwd()
sys.path.append(str(project_root))

from src.monitoring.drift_report import compute_drift

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MonitoringTest")

def verify_monitoring():
    logger.info("=== Loading Data ===")
    data_path = project_root / "data" / "processed" / "windows.parquet"
    if not data_path.exists():
        logger.error("Dataset not found. Run make_dataset.py first.")
        return

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} windows.")

    # 1. Simulate "Drift": Split by Subjects
    # Different people have different physiology (HRV, EDA baseline), so this SHOULD trigger drift.
    subjects = df['subject_id'].unique()
    if len(subjects) < 2:
        logger.error("Not enough subjects to simulate drift.")
        return

    ref_subjects = subjects[:len(subjects)//2]
    curr_subjects = subjects[len(subjects)//2:]
    
    logger.info(f"Reference Group: {ref_subjects}")
    logger.info(f"Current Group (New Batch): {curr_subjects}")

    df_ref = df[df['subject_id'].isin(ref_subjects)].sample(100, replace=True) # Sample for speed
    df_curr = df[df['subject_id'].isin(curr_subjects)].sample(100, replace=True)

    # 2. Compute Drift
    logger.info("Running Drift Analysis...")
    report = compute_drift(df_ref, df_curr)

    # 3. Analyze Results
    drift_count = sum(1 for v in report.values() if v['drift_detected'])
    total_features = len(report)
    
    logger.info("-" * 30)
    logger.info(f"Drift Report Summary: {drift_count}/{total_features} features drifted.")
    
    # Check specific physiological marker (e.g., EDA Mean)
    if 'EDA_mean' in report:
        eda_drift = report['EDA_mean']
        logger.info(f"EDA Mean Drift: {eda_drift['drift_detected']} (p={eda_drift['p_value']:.4f})")
        logger.info(f"  Ref Mean: {eda_drift['mean_ref']:.2f}")
        logger.info(f"  Cur Mean: {eda_drift['mean_curr']:.2f}")
    
    if drift_count > 0:
        print("SUCCESS: Drift Detection logic works (Physiological differences detected).")
    else:
        print("WARNING: No drift detected. This is unusual for different subjects but possible if normalized.")

if __name__ == "__main__":
    verify_monitoring()