"""
Pipeline Consistency Tests
==========================

Integration tests to verify that the Training Logic and Inference Logic
remain consistent. It checks that a model saved by the trainer produces
identical predictions when loaded by the inference predictor.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Add project root
sys.path.append(str(Path.cwd()))

from src.config import PROJECT_ROOT  # noqa: E402
from src.models.deep import ResNet1D  # noqa: E402
from src.models.predict import StressPredictor  # noqa: E402


def test_pipeline_consistency():
    print("=== Pipeline Consistency Check (Train Logic vs Inference Logic) ===")

    # 1. Setup Temporary Artifact Directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Created temp directory: {temp_dir}")

    try:
        # 2. Simulate Training Environment
        # --------------------------------
        # Load Data
        data_path = PROJECT_ROOT / "data" / "processed" / "windows.parquet"
        if not data_path.exists():
            print("Skipping: Processed data not found.")
            return

        df = pd.read_parquet(data_path)
        # Filter (keep it small for speed)
        df_sample = df[df["label"].isin([1, 2])].sample(5, random_state=42).copy()

        # Prepare "Training" Data (N, C, T) - Logic from train.py
        signal_cols = ["ACC_x", "ACC_y", "ACC_z", "ECG", "EDA", "RESP", "TEMP"]
        X = []
        for _, row in df_sample.iterrows():
            X.append(np.stack([np.array(row[c]) for c in signal_cols]))
        X = np.stack(X)  # (5, 7, 2100)

        # --- TRAINING: PREPROCESSING LOGIC ---
        # "Instance Normalization" as defined in train.py (Validation loop)
        print("Applying Training Preprocessing (Instance Norm)...")
        mean_tr = X.mean(axis=2, keepdims=True)
        std_tr = X.std(axis=2, keepdims=True) + 1e-6
        X_norm = (X - mean_tr) / std_tr

        # --- TRAINING: MODEL EXECUTION ---
        # Initialize random model
        device = torch.device("cpu")
        # Check signature: currently accepts sequence_length (optional) or just ignores it?
        # Based on deep.py: __init__(self, num_channels, num_classes, sequence_length=None)
        model = ResNet1D(num_channels=7, num_classes=2).to(device)
        model.eval()

        # Run Forward Pass directly on Tensor
        print("Running Model directly (Training Path)...")
        with torch.no_grad():
            tensor_in = torch.tensor(X_norm, dtype=torch.float32).to(device)
            logits = model(tensor_in)
            probs_train = torch.softmax(logits, dim=1).numpy()

        # 3. Simulate Artifact Creation
        # -----------------------------
        # Save model
        torch.save(model.state_dict(), temp_dir / "model.pt")

        # Save Config (Required by StressPredictor)
        config = {
            "data": {
                "sensor_location": "chest",
                "labels": {1: "Baseline", 2: "Stress"},
            },
            "training": {"models": {"deep": {"confidence_threshold": 0.0}}},
        }
        with open(temp_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump(config, f)

        # 4. Simulate Inference Environment
        # ---------------------------------
        print("Initializing StressPredictor (Inference Path)...")
        predictor = StressPredictor(temp_dir)

        probs_inference = []

        print("Running Predictor on DataFrames...")
        for i in range(len(df_sample)):
            # Construct DataFrame input (One Window)
            row = df_sample.iloc[i]
            input_data = {}
            for c in signal_cols:
                input_data[c] = row[c]
            input_df = pd.DataFrame(input_data)

            # Predict
            result = predictor.predict(input_df)

            # Extract probability of Stress (Index 1)
            # predictor.class_names = ['Baseline', 'Stress']
            p_stress = result["probabilities"]["Stress"]
            p_baseline = result["probabilities"]["Baseline"]
            probs_inference.append([p_baseline, p_stress])

        probs_inference = np.array(probs_inference)

        # 5. Compare Results
        # ------------------
        print("\n=== Comparison Results ===")
        # Compare Prob(Stress) column
        diff = np.abs(probs_train - probs_inference)
        max_diff = np.max(diff)

        print(f"Training Probs (First 3):\n{probs_train[:3]}")
        print(f"Inference Probs (First 3):\n{probs_inference[:3]}")
        print(f"Maximum Difference: {max_diff:.8f}")

        if max_diff < 1e-6:
            print("\nSUCCESS: Inference Logic matches Training Logic perfectly.")
        else:
            print("\nFAIL: Discrepancy detected between pipelines.")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print("Cleaned up temp directory.")


if __name__ == "__main__":
    test_pipeline_consistency()
