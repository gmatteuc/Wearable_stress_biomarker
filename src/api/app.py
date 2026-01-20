"""
FastAPI Deployment Service
==========================

This module implements the REST API for the stress detection system.
It exposes a `/predict` endpoint that utilizes the Production `StressPredictor` class.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure src is in path
sys.path.append(str(Path.cwd()))

from src.config import PROJECT_ROOT  # noqa: E402
from src.models.predict import StressPredictor  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)
app = FastAPI(title="Wearable Stress Detection API", version="0.1.0")

# Global Predictor Instance
predictor: Optional[StressPredictor] = None


@app.on_event("startup")
def load_model():
    global predictor
    try:
        # Priority 1: Production Model Folder (Container/Deployment)
        prod_path = PROJECT_ROOT / "models" / "prod"

        # Check if prod folder has a config file (indicator of a valid run copy)
        if prod_path.exists() and (prod_path / "config_snapshot.yaml").exists():
            logger.info(f"Loading Production Model from: {prod_path}")
            predictor = StressPredictor(prod_path)

        else:
            # Priority 2: Latest Report (Development)
            reports_dir = PROJECT_ROOT / "reports"
            deep_runs = sorted(list(reports_dir.glob("deep_loso_CHEST_*")))

            if not deep_runs:
                logger.error("No trained models found in reports/ or models/prod!")
                return

            latest_run = deep_runs[-1]
            logger.info(f"Loading Latest Dev Artifact: {latest_run.name}")
            predictor = StressPredictor(latest_run)

        logger.info(
            f"StressPredictor initialized successfully ({predictor.model_type})."
        )

    except Exception as e:
        logger.error(f"Failed to load model: {e}")


class WindowInput(BaseModel):
    subject_id: Optional[str] = "Anonymous"
    # Sensor 1D Arrays (e.g. 2100 samples)
    ACC_x: List[float]
    ACC_y: List[float]
    ACC_z: List[float]
    ECG: List[float]
    EDA: List[float]
    RESP: List[float]
    TEMP: List[float]


class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    status: str
    probabilities: Dict[str, float]


@app.get("/health")
def health_check():
    status = "healthy" if predictor is not None else "degraded"
    return {"status": status, "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(window: WindowInput):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        # Convert Pydantic -> DataFrame expected by StressPredictor
        # StressPredictor expects a DataFrame where index=time, columns=channels (T x C)
        # This is inferred from src/models/predict.py which does:
        # x_np = np.stack([df[c].values for c in required_cols]) -> (C, T)
        # and then adds a batch dimension.

        data_dict = {
            "ACC_x": window.ACC_x,
            "ACC_y": window.ACC_y,
            "ACC_z": window.ACC_z,
            "ECG": window.ECG,
            "EDA": window.EDA,
            "RESP": window.RESP,
            "TEMP": window.TEMP,
        }

        input_df = pd.DataFrame(data_dict)

        result = predictor.predict(input_df)

        return PredictionOutput(
            prediction=result["prediction"],
            confidence=result["confidence"],
            status=result["status"],
            probabilities=result["probabilities"],
        )

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    # For local testing
    uvicorn.run(app, host="0.0.0.0", port=8000)
