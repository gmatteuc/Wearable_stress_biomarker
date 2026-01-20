"""
Evaluation Metrics Tests
========================

Unit tests for custom evaluation metrics, specifically Expected Calibration Error (ECE).
"""

import numpy as np
import pytest

from src.models.evaluate import compute_ece


def test_ece_perfect_calibration():
    """
    If a model is perfectly calibrated, ECE should be 0.
    """
    # 5 samples.
    # Prediction: Class 1 with 0.8 confidence -> True Class 1
    # Prediction: Class 0 with 0.6 confidence -> True Class 0
    # ...

    # Perfectly calibrated means:
    # Among all samples predicted with confidence 0.8, 80% are correct.

    # Let's construct a simple case:
    # 10 samples, all predicted with 0.8 confidence.
    # Exactly 8 are correct, 2 are wrong.
    # ECE = |0.8 - 0.8| * 1.0 = 0.0

    y_probs = np.zeros((10, 2))
    y_probs[:, 1] = 0.8  # Predict class 1 with 0.8 prob
    y_probs[:, 0] = 0.2

    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])  # 8 ones, 2 zeros

    ece = compute_ece(y_true, y_probs, n_bins=1)  # 1 bin for simplicity covers range

    assert ece == pytest.approx(0.0)


def test_ece_bad_calibration():
    """
    If model is overconfident (always 1.0) but accuracy is 0.5, ECE should be 0.5.
    """
    # 10 samples
    # Confidence 1.0
    # Accuracy 0.5 (random guess)

    y_probs = np.zeros((10, 2))
    y_probs[:, 1] = 1.0  # Absolute certainty

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # 50% split

    ece = compute_ece(y_true, y_probs)

    # Expected: |1.0 - 0.5| = 0.5
    assert ece == pytest.approx(0.5)
