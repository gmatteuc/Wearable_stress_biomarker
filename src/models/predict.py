"""
Inference Pipeline
==================

This module provides a Production-Ready inference class `StressPredictor`.
It loads a trained model artifact (Deep or Classical) from a specific report folder
and exposes a simple `predict()` method for new data.

Usage:
    predictor = StressPredictor("reports/deep_loso_CHEST_2023...")
    result = predictor.predict(new_window_data)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.features.feature_extraction import FeatureExtractor
from src.models.deep import ResNet1D

logger = logging.getLogger(__name__)


class StressPredictor:
    def __init__(self, run_dir: Union[str, Path]):
        """
        Initialize the predictor by loading artifacts from a run directory.

        Args:
            run_dir (str/Path): Path to the report folder containing config and model.
        """
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        # 1. Load Configuration
        self.config_path = self.run_dir / "config_snapshot.yaml"
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config snapshot not found in {self.run_dir}")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.sensor_loc = self.config["data"].get("sensor_location", "unknown").upper()
        self.labels = self.config["data"].get("labels", {1: "Baseline", 2: "Stress"})
        # Ensure we Map 0/1 back to Strings if needed.
        # Train mapping was normally 1->0, 2->1.
        self.class_names = [
            "Baseline",
            "Stress",
        ]  # Config might have more, but binary is this.

        # 2. Determine Model Type & Load Model
        self.model = None
        self.model_type = None
        self.device = torch.device(
            "cpu"
        )  # Inference usually on CPU unless batch is huge

        if (self.run_dir / "model.pt").exists():
            self.model_type = "deep"
            self._load_deep_model()
        elif (self.run_dir / "model.joblib").exists():
            self.model_type = "logistic"  # or classical
            self._load_classical_model()
        else:
            raise ValueError(
                "No valid model file (model.pt or model.joblib) found in run directory."
            )

        logger.info(f"Initialized {self.model_type} predictor from {self.run_dir}")

    def _load_deep_model(self):
        """Load PyTorch model."""
        # Note: Deep models use Instance Normalization (calculated at inference time),
        # so we do not need to load a global normalizer artifact.

        # Init Architecture
        # We need to infer dimensions from config or saved metdata.
        # Defaulting for WESAD: 7 channels.
        num_channels = 7
        num_classes = 2
        seq_len = 2100  # 60s * 35Hz

        self.model = ResNet1D(num_channels, num_classes, seq_len)

        # Load Weights
        state_dict = torch.load(self.run_dir / "model.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _load_classical_model(self):
        """Load Scikit-Learn Pipeline and feature names."""
        self.model = joblib.load(self.run_dir / "model.joblib")

        # Load expected feature order
        feat_path = self.run_dir / "feature_names.joblib"
        if feat_path.exists():
            self.expected_features = joblib.load(feat_path)
            # If loaded as array/list of strings
            if hasattr(self.expected_features, "tolist"):
                self.expected_features = self.expected_features.tolist()
        else:
            self.expected_features = None

        # Init Feature Extractor
        self.feature_extractor = FeatureExtractor(self.config)

    def predict(
        self, data: Union[pd.DataFrame, Dict[str, np.ndarray], str]
    ) -> Dict[str, Any]:
        """
        Run inference on new data.

        Args:
            data: Input can be:
                  - Path to CSV file (str).
                  - pandas DataFrame (Time x Channels).
                  - Dict with channel arrays.

        Returns:
            Dict containing 'prediction', 'probability', 'confidence', 'abstention'.
        """
        # 1. Ingest Data
        if isinstance(data, str) or isinstance(data, Path):
            df = pd.read_csv(data)
        elif isinstance(data, dict):
            # Assume dict of arrays -> Convert to DataFrame usually easier for feature extraction
            # Or Dict of arrays is exactly what we need for Deep (N, C, T) check?
            # Let's standardize on DataFrame for input of "Raw Signal"
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # 2. Process & Predict
        if self.model_type == "deep":
            return self._predict_deep(df)
        else:
            return self._predict_classical(df)

    def _predict_deep(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Preprocessing: DataFrame -> (1, C, T) tensor
        # Standardize Columns
        required_cols = ["ACC_x", "ACC_y", "ACC_z", "ECG", "EDA", "RESP", "TEMP"]

        # Check if missing
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            # Basic handling: fill with 0 or raise
            logger.warning(
                f"Missing columns for Deep Model: {missing}. Filling with 0."
            )
            for c in missing:
                df[c] = 0.0

        # Extract values
        # Assume input IS the window (e.g. 60s of data).
        # Resize/Resample logic is usually upstream.
        # Here we assume data is approx correct length or we truncate/pad.

        x_np = np.stack([df[c].values for c in required_cols])  # (C, T)

        # Resample logic check? If input is 700Hz and model needs 35Hz.
        # For this MVP Predictor, assume input is already resampled or close to strict.
        # Ideally FeatureExtractor or similar would handle resampling.

        # Add Batch Dim
        x_np = x_np[np.newaxis, ...]  # (1, C, T)

        # Normalization (Instance Norm as in training)
        # Train logic: (X - mean(time)) / std(time)
        # Note: We loaded 'normalizer.joblib' which contained global means?
        # Actually in train.py deep, we used *Instance Normalization* inside the loop:
        # mean_tr = X_train.mean(axis=2)
        # So we should apply Instance Norm here too. The loaded 'normalizer' might be irrelevant
        # if we purely used Instance Norm in the loop.
        # Checking train.py... yes: "Normalize: Instance Normalization (Per-Window)"

        mean = x_np.mean(axis=2, keepdims=True)
        std = x_np.std(axis=2, keepdims=True) + 1e-6
        x_norm = (x_np - mean) / std

        # Inference
        tensor_in = torch.tensor(x_norm, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = np.argmax(probs)
        confidence = np.max(probs)

        # Abstention Logic
        conf_threshold = (
            self.config["training"]
            .get("models", {})
            .get("deep", {})
            .get("confidence_threshold", 0.7)
        )
        abstained = confidence < conf_threshold

        return {
            "prediction": self.class_names[pred_idx] if not abstained else "Uncertain",
            "prediction_idx": int(pred_idx),
            "probabilities": {
                name: float(p) for name, p in zip(self.class_names, probs)
            },
            "confidence": float(confidence),
            "status": "Abstained" if abstained else "Success",
        }

    def _predict_classical(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Feature Extraction
        # FeatureExtractor expects dict or list of dicts(window)
        # We assume 'df' IS One Window.
        # FeatureExtractor.extract_features takes a dict of arrays.

        window_dict = {col: df[col].values for col in df.columns}
        features = self.feature_extractor.extract_features(window_dict)

        # Prepare for Model
        # Must align with self.expected_features
        if self.expected_features:
            # Create ordered array, filling missing with 0
            feat_vector = []
            for name in self.expected_features:
                val = features.get(name, 0.0)
                feat_vector.append(val)
            X = np.array([feat_vector])
        else:
            # Fallback (risky if order differs)
            X = pd.DataFrame([features]).values

        # Inference
        probs = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        confidence = np.max(probs)

        # Abstention Logic
        # Classical might have different threshold logic, usually simpler
        # abstained = False  # Default strict (Unused variable removed)

        return {
            "prediction": self.class_names[pred_idx],
            "prediction_idx": int(pred_idx),
            "probabilities": {
                name: float(p) for name, p in zip(self.class_names, probs)
            },
            "confidence": float(confidence),
            "status": "Success",
        }
