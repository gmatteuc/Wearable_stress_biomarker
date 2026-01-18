"""
Signal Quality Index (SQI) Tests
================================

Unit tests for the SQI module. Verifies that flatlines and noise are correctly
flagged and that the composite quality score is robust.
"""

import pytest
import numpy as np
from src.features.sqi import SignalQualityIndex

def test_sqi_flatline():
    sqi = SignalQualityIndex()
    # Flatline EDA
    window = {
        'EDA': [1.0] * 100,
        'ACC_x': np.random.randn(100).tolist(),
        'ACC_y': np.random.randn(100).tolist(),
        'ACC_z': np.random.randn(100).tolist()
    }
    scores = sqi.compute_sqi(window)
    assert scores['eda_sqi'] == 0.0
    assert scores['overall_sqi'] < 1.0

def test_sqi_good_signal():
    sqi = SignalQualityIndex()
    # Good EDA (variance > 0)
    window = {
        'EDA': np.random.normal(1.0, 0.1, 100).tolist(),
        'ACC_x': np.random.normal(0, 0.1, 100).tolist(), 
        'ACC_y': np.random.normal(0, 0.1, 100).tolist(),
        'ACC_z': np.random.normal(1, 0.1, 100).tolist() # Gravity
    }
    scores = sqi.compute_sqi(window)
    assert scores['eda_sqi'] == 1.0
    assert scores['motion_score'] > 0.8 # Low motion
    assert scores['overall_sqi'] > 0.8
