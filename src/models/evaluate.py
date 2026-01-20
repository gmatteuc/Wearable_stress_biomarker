"""
Model Evaluation Module
=======================

This module provides tools for comprehensive model assessment. It generates
classification reports, confusion matrices, and reliability diagrams (calibration
curves) to ensure model trustworthiness and safety.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


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


def evaluate_model(y_true, y_prob, labels, output_dir: Path, subject_ids=None):
    """
    Generate evaluation report: metrics, confusion matrix, calibration.
    Returns a dictionary with metrics and granular results DataFrame.
    """
    y_pred = np.argmax(y_prob, axis=1)

    # 1. Metrics
    report = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True
    )
    acc = accuracy_score(y_true, y_pred)
    ece = compute_ece(y_true, y_prob)

    metrics = {"accuracy": acc, "ece": ece, "report": report}

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save granular results for diagnostics
    results = {"y_true": y_true, "y_pred": y_pred, "confidence": np.max(y_prob, axis=1)}
    # Add per-class probabilities
    for i, label in enumerate(labels):
        results[f"prob_{label}"] = y_prob[:, i]

    if subject_ids is not None:
        results["subject_id"] = subject_ids

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "predictions.csv", index=False)

    # 2. Confusion Matrix & Reliability Diagram
    # Removed redundant standalone plots here.
    # These are now generated as part of the unified diagnostic panels in src.models.train
    # via plot_model_diagnostics and plot_confidence_abstention_panel.

    return metrics, results_df
