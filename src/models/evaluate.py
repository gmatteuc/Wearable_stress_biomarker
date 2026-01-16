"""
Model Evaluation Module
=======================

This module provides tools for comprehensive model assessment. It generates
classification reports, confusion matrices, and reliability diagrams (calibration
curves) to ensure model trustworthiness and safety.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Any

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) for binary/multiclass.
    For multiclass, we essentially take the max prob and check if it matches accuracy.
    """
    pred_y = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(pred_y[in_bin] == y_true[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def evaluate_model(y_true, y_prob, labels, output_dir: Path):
    """
    Generate evaluation report: metrics, confusion matrix, calibration.
    """
    y_pred = np.argmax(y_prob, axis=1)
    
    # 1. Metrics
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    acc = accuracy_score(y_true, y_pred)
    ece = compute_ece(y_true, y_prob)
    
    metrics = {
        'accuracy': acc,
        'ece': ece,
        'report': report
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
    
    # 3. Reliability Diagram (Top-class calibration)
    # Simple binary-like plot for "Correctness"
    confidences = np.max(y_prob, axis=1)
    is_correct = (y_pred == y_true).astype(int)
    
    prob_true, prob_pred = calibration_curve(is_correct, confidences, n_bins=10, strategy='uniform')
    
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability (Confidence)')
    plt.ylabel('Fraction of Positives (Accuracy)')
    plt.title(f'Reliability Diagram (ECE={ece:.4f})')
    plt.legend()
    plt.savefig(output_dir / "calibration_plot.png")
    plt.close()
    
    return metrics
