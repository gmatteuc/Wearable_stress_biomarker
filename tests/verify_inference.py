import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path.cwd()))

from src.models.predict import StressPredictor
from src.config import load_config, PROJECT_ROOT

def verify_inference():
    print("=== Inference Pipeline Verification ===")
    
    # 1. Find latest Deep Learning Run
    reports_dir = PROJECT_ROOT / "reports"
    deep_runs = sorted(list(reports_dir.glob("deep_loso_*")))
    
    if not deep_runs:
        print("FAIL: No deep learning runs found in reports/")
        return
        
    latest_run = deep_runs[-1]
    print(f"Testing Artifact: {latest_run.name}")
    
    # 2. Initialize Predictor
    try:
        predictor = StressPredictor(latest_run)
        print("SUCCESS: Predictor Initialized.")
    except Exception as e:
        print(f"FAIL: Predictor Initialization error: {e}")
        return

    # 3. Load Sample Data (One Window from processed parquet)
    data_path = PROJECT_ROOT / "data" / "processed" / "windows.parquet"
    if not data_path.exists():
        print("FAIL: Data not found.")
        return
        
    df = pd.read_parquet(data_path)
    # Filter for S17 (usually held out in fold 1 or similar)
    s17 = df[df['subject_id'] == 'S17']
    
    if len(s17) == 0:
        print("WARNING: S17 not found, taking random subject.")
        window = df.sample(1)
    else:
        # Take a Stress window (Label 2) to see if it detects it
        stress_windows = s17[s17['label'] == 2]
        if len(stress_windows) > 0:
            window = stress_windows.iloc[0:1] # Keep as DataFrame
            print("Loaded a known STRESS window from Subject S17")
        else:
            window = s17.iloc[0:1]
            print(f"Loaded a window from Subject S17 (Label {window['label'].values[0]})")
            
    # 4. Run Prediction
    try:
        # Filter columns to only sensors
        sensor_cols = ['ACC_x', 'ACC_y', 'ACC_z', 'ECG', 'EDA', 'RESP', 'TEMP']
        
        # Convert the single row from windows.parquet into a Time-Series DataFrame
        # simulating a real-world CSV input file
        ts_data = {}
        for c in sensor_cols:
            ts_data[c] = window[c].iloc[0] # The array (2100,)
            
        input_df = pd.DataFrame(ts_data)
        print(f"Simulated Input Shape: {input_df.shape} (Time x Channels)")
        
        result = predictor.predict(input_df)
        
        print("\n--- Prediction Result ---")
        print(result)
        
        if result['status'] in ['Success', 'Abstained']:
            print("SUCCESS: Inference Pipeline end-to-end.")
            # Qualitative check
            if window['label'].iloc[0] == 2:
                print(f"Ground Truth: Stress. Prediction: {result['prediction']}")
            else:
                 print(f"Ground Truth: Baseline/Other. Prediction: {result['prediction']}")
        else:
            print("FAIL: Status unknown.")
            
    except Exception as e:
        print(f"FAIL: Inference Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_inference()
