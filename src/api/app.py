"""
FastAPI Deployment Service
==========================

This module implements the REST API for the stress detection system. It serves
predictions via a `/predict_window` endpoint, integrating SQI checks to
abstain from low-quality data and returning calibrated probabilities.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import joblib
import torch
import pandas as pd
from pathlib import Path

from src.models.deep import Simple1DCNN
from src.features.sqi import SignalQualityIndex
from src.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Wearable Stress Detection API", version="0.1.0")

# Load configuration and model on startup
class ModelWrapper:
    def __init__(self):
        self.config = load_config()
        self.sqi_checker = SignalQualityIndex(self.config)
        self.model = None
        self.model_type = "logistic" # Default
        self.preprocessor = None
        self.labels = None
        
    def load(self, model_type="logistic", split_type="loso"):
        artifacts_path = Path(f"reports/{model_type}_{split_type}")
        if not artifacts_path.exists():
            logger.warning(f"Artifacts not found at {artifacts_path}. API will fail to predict.")
            return

        if model_type == "logistic":
            self.model = joblib.load(artifacts_path / "model.joblib")
            self.preprocessor = None # Included in pipeline
            self.model_type = "logistic"
            # Infer labels (hardcoded for MVP or saved)
            self.labels = ["baseline", "stress", "amusement"] 
            
        elif model_type == "deep":
            # Load PyTorch model... (Simplified for MVP: stick to classical for API example)
            pass

model_wrapper = ModelWrapper()

@app.on_event("startup")
def startup_event():
    model_wrapper.load()

class WindowInput(BaseModel):
    subject_id: Optional[str] = "S_Test"
    EDA: List[float]
    ACC_x: List[float]
    ACC_y: List[float]
    ACC_z: List[float]
    TEMP: List[float]
    RESP: List[float]
    ECG: List[float]

class PredictionOutput(BaseModel):
    label: str
    probabilities: Dict[str, float]
    sqi_score: float
    abstain: bool
    message: Optional[str] = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict_window", response_model=PredictionOutput)
def predict(window: WindowInput):
    # 1. SQI Check
    data_dict = window.dict()
    sqi_out = model_wrapper.sqi_checker.compute_sqi(data_dict)
    sqi_score = sqi_out['overall_sqi']
    
    threshold = model_wrapper.config['features']['sqi_threshold']
    
    if sqi_score < threshold:
        return {
            "label": "unknown",
            "probabilities": {},
            "sqi_score": sqi_score,
            "abstain": True,
            "message": "Signal quality too low"
        }

    # 2. Featurization (for Classical)
    # We need to compute features on the fly
    from src.features.feature_extraction import FeatureExtractor
    extractor = FeatureExtractor()
    features = extractor.transform_window(data_dict)
    
    # Needs to match training feature order
    # Load expected feature names from artifact
    try:
        feature_names = joblib.load(Path("reports/logistic_loso/feature_names.joblib"))
    except:
         feature_names = list(features.keys()) # Fallback (risky)
         # Filter out non-numeric
         feature_names = [f for f in feature_names if f not in ['subject_id', 'label']]
    
    # Align features
    X_input = []
    vector = []
    for f in feature_names:
        vector.append(features.get(f, 0.0))
    X_input.append(vector)
    
    # 3. PredictCallback
    if model_wrapper.model:
        probs = model_wrapper.model.predict_proba(X_input)[0]
        pred_idx = np.argmax(probs)
        pred_label = model_wrapper.labels[pred_idx] if pred_idx < len(model_wrapper.labels) else str(pred_idx)
        
        prob_dict = {label: float(prob) for label, prob in zip(model_wrapper.labels, probs)}
        
        return {
            "label": pred_label,
            "probabilities": prob_dict,
            "sqi_score": sqi_score,
            "abstain": False
        }
    else:
        raise HTTPException(status_code=500, detail="Model not loaded")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
