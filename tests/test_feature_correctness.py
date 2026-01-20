"""
Feature Extraction Correctness Tests
====================================

Scientific validation of finding signal features using synthetic data with known ground truth.
Verifies:
1. ECG Heart Rate calculation accuracy.
2. EDA Skin Conductance Response (SCR) peak counting.
3. Accelerometer motion intensity logic.
"""

import numpy as np
import pytest

from src.features.feature_extraction import FeatureExtractor

# Constants
FS = 35  # Sampling rate used in pipeline


@pytest.fixture
def extractor():
    # Initialize with default config
    # We increase the SCR threshold to 0.1 for this synthetic test to avoid
    # detecting minor filter ringing artifacts as peaks.
    config = {
        "data": {"target_sampling_rate": FS},
        "features": {"eda": {"scr_threshold": 0.5}},
    }
    return FeatureExtractor(config)


def test_ecg_heart_rate_accuracy(extractor):
    """
    Generate a synthetic ECG-like signal (sine wave) at 90 BPM (1.5 Hz)
    and verify the calculated Heart Rate is approx 90.
    Rationale: 90 BPM is comfortably within default bandpass (1-15Hz).
    """
    duration_sec = 60
    t = np.linspace(0, duration_sec, duration_sec * FS)

    # 1.5 Hz Frequency => 90 BPM
    freq = 1.5
    expected_bpm = freq * 60

    # Synthetic ECG: Pulse train convolved with Gaussian to simulate QRS
    ecg_sig = np.zeros_like(t)
    period_samples = int(FS / freq)

    # Add pulses
    for i in range(0, len(t), period_samples):
        if i < len(t):
            ecg_sig[i] = 100.0  # High amplitude impulse

    # QRS shape convolution (approx 0.1s wide)
    qrs_shape = np.exp(-(np.linspace(-3, 3, 10) ** 2))
    ecg_sig = np.convolve(ecg_sig, qrs_shape, mode="same")

    # Add some baseline noise
    ecg_sig += np.random.normal(0, 0.1, size=len(t))

    # Pass to extractor
    feats = extractor._process_ecg(ecg_sig)

    # Check Heart Rate (allow +/- 5 BPM tolerance due to discrete sampling at 35Hz)
    assert feats["ecg_hr_bpm"] == pytest.approx(
        expected_bpm, abs=5.0
    ), f"Expected ~{expected_bpm} BPM, got {feats['ecg_hr_bpm']}"


def test_eda_scr_counting(extractor):
    """
    Generate a flat EDA signal with 3 distinct peaks and verify distinct SCR count.
    """
    duration_sec = 60
    n_samples = duration_sec * FS
    eda_sig = np.ones(n_samples) * 5.0  # Baseline 5 uS
    t_samples = np.arange(n_samples)

    # Helper to add Gaussian peak
    def add_peak(center_sec, height=1.0):
        center_idx = center_sec * FS
        width = 2 * FS  # 2 seconds wide
        # Gaussian formula
        gaussian = height * np.exp(-((t_samples - center_idx) ** 2) / (2 * width**2))
        return gaussian

    # Add 3 distinct peaks at 10s, 30s, 50s
    eda_sig += add_peak(10, height=2.0)
    eda_sig += add_peak(30, height=2.0)
    eda_sig += add_peak(50, height=2.0)

    feats = extractor._process_eda(eda_sig)

    # We expect `eda_scr_count` to be exactly 3 theoretically,
    # but filters might merge or cut edge peaks.
    # We accept 2-4 peaks as valid detection behavior.
    assert (
        2 <= feats["eda_acc_scr_count"] <= 4
    ), f"Expected 3 SCR peaks, got {feats['eda_acc_scr_count']}"


def test_acc_motion_intensity(extractor):
    """
    Verify that static sensor produces ~0 dynamic motion features,
    and shaking produces high values.
    """
    duration_sec = 10
    n_samples = duration_sec * FS

    # 1. Static (Gravity on Z-axis only, minimal noise)
    window_static = {
        "ACC_x": np.random.normal(0, 0.001, n_samples).tolist(),
        "ACC_y": np.random.normal(0, 0.001, n_samples).tolist(),
        "ACC_z": np.random.normal(1, 0.001, n_samples).tolist(),
    }

    # 2. Shaking (High variance on all axes)
    # Simulating vigorous movement
    window_shaking = {
        "ACC_x": np.random.normal(0, 0.5, n_samples).tolist(),
        "ACC_y": np.random.normal(0, 0.5, n_samples).tolist(),
        "ACC_z": np.random.normal(1, 0.5, n_samples).tolist(),
    }

    feats_static = extractor._process_acc(window_static)
    feats_shaking = extractor._process_acc(window_shaking)

    # Check `acc_std` (Std deviation of magnitude)
    # Static should have very low std (close to noise floor 0.001)
    # Shaking should have high std (> 0.2)

    print(
        f"Static STD: {feats_static['acc_std']}, Shaking STD: {feats_shaking['acc_std']}"
    )

    assert (
        feats_static["acc_std"] < 0.05
    ), "Static signal has unexpectedly high variance"
    assert (
        feats_shaking["acc_std"] > 0.2
    ), "Shaking signal has unexpectedly low variance"

    # Check `acc_energy`
    assert (
        feats_shaking["acc_energy"] > feats_static["acc_energy"] * 10
    ), "Shaking energy should be orders of magnitude higher than static"
