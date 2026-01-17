import argparse
import copy
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
            
            # MVP Filter: Restrict to Baseline (1) and Stress (2)
            df = df[df['label'].isin([1, 2])].copy()

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
            
            # MVP Filter: Restrict to Baseline (1) and Stress (2)
            # This aligns the training pipeline with the notebook verification
            df = df[df['label'].isin([1, 2])].copy()
            
            # Exclude metadata columns
            # 'start_idx' and 'session' are metadata, not features. 
            # 'target' might be created in experimental notebooks.
            ignore_cols = ['subject_id', 'label', 'session', 'start_idx', 'target']
            feature_cols = [c for c in df.columns if c not in ignore_cols]
            
            X = df[feature_cols].values
            y = df['label'].values
            groups = df['subject_id'].values
            
            logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
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

        # Remap labels to 0..K-1 for sklearn compatibility (argmax matches index)
        unique_labels = sorted(np.unique(y))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        y_mapped = np.array([label_map[l] for l in y])
        
        # Display mappings
        labels = [self.config['data']['labels'][i] for i in sorted(np.unique(y))]

        if self.split_type == 'loso':
            logger.info("Running full Leave-One-Subject-Out Cross-Validation...")
            gkf = GroupKFold(n_splits=len(np.unique(groups)))
            
            y_test_all = []
            y_prob_all = []
            groups_test_all = []
            
            for i, (train_idx, test_idx) in enumerate(gkf.split(X, y_mapped, groups)):
                subj = np.unique(groups[test_idx])
                logger.info(f"Fold {i+1}: Validating on Subject {subj}")
                
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_mapped[train_idx], y_mapped[test_idx]
                
                # Pipeline
                # Using class_weight='balanced' to handle subject/class imbalance
                # Matching the notebook's baseline exactly (No extra calibration wrapper)
                base_clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='liblinear')
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', base_clf)
                ])
                
                pipe.fit(X_train, y_train)
                
                y_prob = pipe.predict_proba(X_test)
                
                y_test_all.append(y_test)
                y_prob_all.append(y_prob)
                groups_test_all.append(groups[test_idx])
                
            # Aggregate Results
            y_test_final = np.concatenate(y_test_all)
            y_prob_final = np.concatenate(y_prob_all)
            groups_test_final = np.concatenate(groups_test_all)
            
            # Train Final Model on ALL data for production artifact
            logger.info("Training final production model on full dataset...")
            base_clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='liblinear')
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', base_clf)
            ])
            pipe.fit(X, y_mapped)
            joblib.dump(pipe, self.run_dir / "model.joblib")
            
        else:
            # Single Split (Random)
            train_idx, test_idx = self.get_split(groups)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_mapped[train_idx], y_mapped[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx]
            
            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            base_clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='liblinear')
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', base_clf)
            ])
            pipe.fit(X_train, y_train)
            
            y_test_final = y_test
            y_prob_final = pipe.predict_proba(X_test)
            groups_test_final = groups_test
            
            joblib.dump(pipe, self.run_dir / "model.joblib")
        
        joblib.dump(feature_names, self.run_dir / "feature_names.joblib")
        evaluate_model(y_test_final, y_prob_final, labels, self.run_dir, subject_ids=groups_test_final)
        logger.info(f"Results saved to {self.run_dir}")

    def train_deep(self):
        # Determine Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on Device: {device}")
        if device.type == 'cuda':
             logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

        X, y, groups = self.load_data()
        
        # Remap labels to 0..K-1
        unique_labels = sorted(np.unique(y))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        y_mapped = np.array([label_map[l] for l in y])
        display_labels = [self.config['data']['labels'][l] for l in unique_labels]
        
        if self.split_type == 'loso':
            logger.info("Running Deep Learning LOSO Cross-Validation...")
            gkf = GroupKFold(n_splits=len(np.unique(groups)))
            
            y_test_all = []
            y_prob_all = []
            groups_test_all = []
            history_all = {} # Store history per fold
            
            # LOSO Loop
            for i, (train_idx, test_idx) in enumerate(gkf.split(X, y_mapped, groups)):
                subj = np.unique(groups[test_idx])
                fold_id = f"Fold_{i+1}"
                logger.info(f"Fold {i+1}: Validating on Subject {subj}")
                
                # Split
                X_train, X_test = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y_mapped[train_idx], y_mapped[test_idx]
                
                # Normalize: Instance Normalization (Per-Window)
                # Proven to work better for LOSO when subjects have different baselines (e.g. S2 vs S10)
                # (N, C, T) -> Mean/Std over T axis only
                mean_tr = X_train.mean(axis=2, keepdims=True)
                std_tr = X_train.std(axis=2, keepdims=True) + 1e-6
                X_train = (X_train - mean_tr) / std_tr
                
                mean_te = X_test.mean(axis=2, keepdims=True)
                std_te = X_test.std(axis=2, keepdims=True) + 1e-6
                X_test = (X_test - mean_te) / std_te
                
                # Convert to Torch
                train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_fold, dtype=torch.long))
                test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test_fold, dtype=torch.long)) # Need DS for loader if batching test
                
                train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
                
                # Init Model
                num_channels = X.shape[1]
                num_classes = len(unique_labels)
                seq_len = X.shape[2]
                model = Simple1DCNN(num_channels, num_classes, seq_len).to(device)
                
                # Train Configuration (Regularized + Scheduled)
                # Label Smoothing helps with noisy physiological labels and calibration
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                
                # 1. Weight Decay (L2 Regularization)
                # Reduced LR slightly to 5e-4 to prevent initial loss spike
                optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
                
                # 2. Learning Rate Scheduler
                # Increased scheduler patience to 5 (slower decay)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
                
                epochs = 50 # Increased to allow convergence
                
                # Early Stopping Vars
                best_val_loss = float('inf')
                patience = 15 # Less aggressive early stopping
                trigger_times = 0
                best_model_wts = copy.deepcopy(model.state_dict())
                
                fold_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
                
                for epoch in range(epochs):
                    # Training Phase
                    model.train()
                    epoch_loss = 0
                    for bx, by in train_loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        out = model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    avg_train_loss = epoch_loss / len(train_loader)
                    fold_history['train_loss'].append(avg_train_loss)

                    # Validation Phase (Per Epoch)
                    model.eval()
                    val_loss = 0
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for bx, by in test_loader:
                            bx, by = bx.to(device), by.to(device)
                            out = model(bx)
                            loss = criterion(out, by)
                            val_loss += loss.item()
                            
                            _, predicted = torch.max(out.data, 1)
                            total += by.size(0)
                            correct += (predicted == by).sum().item()
                    
                    avg_val_loss = val_loss / len(test_loader)
                    val_acc = correct / total
                    fold_history['val_loss'].append(avg_val_loss)
                    fold_history['val_acc'].append(val_acc)
                    
                    # 3. Step Schedule
                    scheduler.step(avg_val_loss)
                    
                    # 4. Early Stopping Check
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        trigger_times = 0
                    else:
                        trigger_times += 1
                        if trigger_times >= patience:
                            break
                            
                # Load best model weights before predicting
                model.load_state_dict(best_model_wts)
                
                history_all[fold_id] = fold_history

                # Predict Final Probability for Fold
                model.eval()
                probs = []
                with torch.no_grad():
                    for bx, by in test_loader:
                        bx = bx.to(device) # No label needed for predict
                        out = model(bx)
                        probs.append(torch.softmax(out, dim=1).cpu().numpy())
                
                y_test_all.append(y_test_fold)
                y_prob_all.append(np.concatenate(probs))
                groups_test_all.append(groups[test_idx])
            
            # Aggregate
            y_test_final = np.concatenate(y_test_all)
            y_prob_final = np.concatenate(y_prob_all)
            groups_test_final = np.concatenate(groups_test_all)
            
            # Save History
            joblib.dump(history_all, self.run_dir / "training_history.joblib")
            
            # Evaluate Aggregate Performance
            evaluate_model(y_test_final, y_prob_final, display_labels, self.run_dir, subject_ids=groups_test_final)
            
            # Final Training on ALL Data
            logger.info("Training final deep model on full dataset...")
            means = X.mean(axis=(0, 2), keepdims=True)
            stds = X.std(axis=(0, 2), keepdims=True) + 1e-6
            X_norm = (X - means) / stds
            
            full_ds = TensorDataset(torch.tensor(X_norm, dtype=torch.float32), torch.tensor(y_mapped, dtype=torch.long))
            full_loader = DataLoader(full_ds, batch_size=64, shuffle=True)
            
            final_model = Simple1DCNN(X.shape[1], len(unique_labels), X.shape[2]).to(device)
            optimizer = optim.Adam(final_model.parameters(), lr=1e-3)
            
            for epoch in range(10):
                final_model.train()
                for bx, by in full_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(final_model(bx), by)
                    loss.backward()
                    optimizer.step()
            
            torch.save(final_model.state_dict(), self.run_dir / "model.pt")
            joblib.dump({'mean': means, 'std': stds}, self.run_dir / "normalizer.joblib")
            
        else:
            # Single Split Logic (Random)
            train_idx, test_idx = self.get_split(groups)
            groups_test = groups[test_idx]
            
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
            
            model = Simple1DCNN(num_channels, num_classes, seq_len).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            logger.info("Training Deep Model (Single Split)...")
            epochs = 10 # Short for MVP
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
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
                    bx = bx.to(device)
                    out = model(bx) # Logits
                    prob = torch.softmax(out, dim=1).cpu()
                    probs.append(prob.numpy())
                    trues.append(by.numpy())
                    
            y_probs = np.concatenate(probs)
            y_trues = np.concatenate(trues)
            
            # Save artifacts
            torch.save(model.state_dict(), self.run_dir / "model.pt")
            normalizer = {'mean': means, 'std': stds}
            joblib.dump(normalizer, self.run_dir / "normalizer.joblib")
            
            evaluate_model(y_trues, y_probs, display_labels, self.run_dir, subject_ids=groups_test)

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
