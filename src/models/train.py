import argparse
import pandas as pd
import numpy as np
import yaml
import joblib
import json
from pathlib import Path
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import logging

"""
Model Training Pipeline
=======================

This module helps train both classical (Logistic Regression) and deep learning
models. It handles data loading, splitting (including LOSO), training loops,
and artifact saving (models, scalers, metrics).
"""

from src.config import load_config, PROJECT_ROOT
from src.features.feature_extraction import FeatureExtractor
from src.models.deep import Simple1DCNN
from src.models.evaluate import evaluate_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Trainer:
    def __init__(self, model_type: str, split_type: str):
        self.config = load_config()
        self.model_type = model_type
        self.split_type = split_type
        self.processed_path = PROJECT_ROOT / self.config['data']['processed_path']
        self.run_dir = PROJECT_ROOT / "reports" / f"{model_type}_{split_type}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save run config
        with open(self.run_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump(self.config, f)

    def load_data(self):
        # Determine if we need raw windows (deep) or features (classical)
        if self.model_type == 'deep':
            file_path = self.processed_path / "windows.parquet"
            logger.info(f"Loading raw windows from {file_path}")
            df = pd.read_parquet(file_path)
            
            # Prepare tensors
            # Stack signal columns
            # Modalities: ['ACC_x', 'ACC_y', 'ACC_z', 'ECG', 'EDA', 'RESP', 'TEMP']
            # Sort columns to ensure consistent order
            signal_cols = ['ACC_x', 'ACC_y', 'ACC_z', 'ECG', 'EDA', 'RESP', 'TEMP']
            # Inspect first row to check available columns
            available_cols = [c for c in signal_cols if c in df.columns]
            
            X = []
            for _, row in df.iterrows():
                # stack: (Channels, Time)
                window_signals = np.stack([np.array(row[c]) for c in available_cols])
                X.append(window_signals)
            
            X = np.stack(X) # (N, C, T)
            y = df['label'].values
            groups = df['subject_id'].values
            
            return X, y, groups
            
        else:
            file_path = self.processed_path / "features.parquet"
            if not file_path.exists():
                raise FileNotFoundError("features.parquet not found. Run 'make features' first.")
            
            logger.info(f"Loading features from {file_path}")
            df = pd.read_parquet(file_path)
            
            feature_cols = [c for c in df.columns if c not in ['subject_id', 'label']]
            X = df[feature_cols].values
            y = df['label'].values
            groups = df['subject_id'].values
            
            return X, y, groups, feature_cols

    def get_split(self, groups):
        if self.split_type == 'loso':
            gkf = GroupKFold(n_splits=len(np.unique(groups)))
            # Logic: We just take one fold for 'Test'.
            # Ideally we do Cross Validation. For MVP we create one Train/Test split via first fold of GKF or manual.
            # Let's hold out Subject S17 (or last one) for test, or random subject.
            # But GroupKFold yields N splits.
            # We will grab all indices.
            train_idx, test_idx = next(gkf.split(X=np.zeros(len(groups)), groups=groups))
            return train_idx, test_idx
        else:
            # Random split
            indices = np.arange(len(groups))
            train_idx, test_idx = train_test_split(indices, test_size=self.config['training']['test_size'], random_state=42)
            return train_idx, test_idx

    def train_classical(self):
        X, y, groups, feature_names = self.load_data()
        train_idx, test_idx = self.get_split(groups)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Pipeline
        base_clf = LogisticRegression(max_iter=1000, random_state=42)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', base_clf)
        ])
        
        # Calibration (Classical needs it usually if not logistic, but logistic is well calibrated. 
        # XGBoost definitely needs it. Let's wrap in CalibratedClassifierCV just to demonstrate)
        # Using 'sigmoid' (Platt) or 'isotonic'.
        calibrated_clf = CalibratedClassifierCV(pipe, method='sigmoid', cv=3)
        
        logger.info("Training classical model...")
        calibrated_clf.fit(X_train, y_train)
        
        # Eval
        y_prob = calibrated_clf.predict_proba(X_test)
        
        # Save
        joblib.dump(calibrated_clf, self.run_dir / "model.joblib")
        joblib.dump(feature_names, self.run_dir / "feature_names.joblib")
        
        # Mapping labels
        labels = [self.config['data']['labels'][i] for i in sorted(np.unique(y))]
        
        evaluate_model(y_test, y_prob, labels, self.run_dir)
        logger.info(f"Results saved to {self.run_dir}")

    def train_deep(self):
        X, y, groups = self.load_data()
        
        # Remap labels to 0..K-1
        unique_labels = sorted(np.unique(y))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        y_mapped = np.array([label_map[l] for l in y])
        
        train_idx, test_idx = self.get_split(groups)
        
        # Standardize inputs (Channel-wise)
        # Compute mean/std on Train
        X_train = X[train_idx]
        X_test = X[test_idx]
        
        # (N, C, T) -> calculate scalar mean/std per channel over N, T
        means = X_train.mean(axis=(0, 2), keepdims=True)
        stds = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
        
        X_train = (X_train - means) / stds
        X_test = (X_test - means) / stds
        
        # Convert to Torch
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_mapped[train_idx], dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_mapped[test_idx], dtype=torch.long))
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Model
        num_channels = X.shape[1]
        num_classes = len(unique_labels)
        seq_len = X.shape[2]
        
        model = Simple1DCNN(num_channels, num_classes, seq_len)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        logger.info("Training Deep Model...")
        epochs = 10 # Short for MVP
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
        # Eval
        model.eval()
        probs = []
        trues = []
        with torch.no_grad():
            for bx, by in test_loader:
                out = model(bx) # Logits
                prob = torch.softmax(out, dim=1)
                probs.append(prob.numpy())
                trues.append(by.numpy())
                
        y_probs = np.concatenate(probs)
        y_trues = np.concatenate(trues)
        
        # Save artifacts
        torch.save(model.state_dict(), self.run_dir / "model.pt")
        normalizer = {'mean': means, 'std': stds}
        joblib.dump(normalizer, self.run_dir / "normalizer.joblib")
        
        # Need original labels for display
        display_labels = [self.config['data']['labels'][l] for l in unique_labels]
        evaluate_model(y_trues, y_probs, display_labels, self.run_dir)
        logger.info(f"Results saved to {self.run_dir}")

    def run(self):
        if self.model_type == 'logistic':
            self.train_classical()
        elif self.model_type == 'deep':
            self.train_deep()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['logistic', 'deep'], required=True)
    parser.add_argument("--split", type=str, choices=['random', 'loso'], default='loso')
    args = parser.parse_args()
    
    trainer = Trainer(args.model, args.split)
    trainer.run()
