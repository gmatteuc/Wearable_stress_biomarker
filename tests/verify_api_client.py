import pandas as pd
import numpy as np
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root
sys.path.append(str(Path.cwd()))

from src.api.app import app
from src.config import PROJECT_ROOT

def test_api_client():
    print("=== API Functional Test (FastAPI TestClient) ===")
    
    # 1. Initialize Client (Context Manager triggers startup events)
    with TestClient(app) as client:
        print("TestClient Initialized.")
        
        # 2. Check Health
        response = client.get("/health")
        print(f"Health Check: {response.status_code} - {response.json()}")
        if response.status_code != 200 or response.json().get("status") != "healthy":
            print("FAIL: Health check failed (Model not loaded?).")
            # Debug: Try to trigger load manually if exposed, or just fail
            return
            
        # 3. Load Sample Data
        data_path = PROJECT_ROOT / "data" / "processed" / "windows.parquet"
        if not data_path.exists():
            print("FAIL: No data found.")
            return
            
        df = pd.read_parquet(data_path)
        # Pick a random Stress window
        sample = df[df['label'] == 2].sample(1).iloc[0]
        
        # 4. Construct Payload
        payload = {
            "subject_id": str(sample['subject_id']),
            "ACC_x": sample['ACC_x'].tolist(),
            "ACC_y": sample['ACC_y'].tolist(),
            "ACC_z": sample['ACC_z'].tolist(),
            "ECG": sample['ECG'].tolist(),
            "EDA": sample['EDA'].tolist(),
            "RESP": sample['RESP'].tolist(),
            "TEMP": sample['TEMP'].tolist()
        }
        
        print("Payload constructed (Window size 2100). Sending Request...")
        
        # 5. Send Request
        response = client.post("/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n--- API Response ---")
            print(result)
            
            if result['prediction'] == 'Stress':
                print("SUCCESS: API correctly identified Stress.")
            else:
                print(f"WARNING: API predicted {result['prediction']} (Expected Stress). Check calibration.")
                
        else:
            print(f"FAIL: API Error {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    test_api_client()