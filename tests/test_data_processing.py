"""
Data Processing & Splitting Tests
=================================

Tests the full data lifecycle: loading, windowing, and splitting.
Ensures:
1. Windows are created correctly (ETL).
2. Splits (LOSO) do not leak subjects (Training Prep).
"""

from unittest.mock import MagicMock, patch

import numpy as np

from src.data.make_dataset import WESADPreprocessor
from src.models.train import Trainer


class MockWESADPreprocessor(WESADPreprocessor):
    """
    Subclass to bypass __init__ and heavy config loading.
    """

    def __init__(self):
        # Setup minimal config for testing
        self.config = {
            "data": {
                "sensor_location": "chest",
                "target_sampling_rate": 10,  # Low Hz for easy math
                "window_size_sec": 10,
                "window_overlap_sec": 5,
                "target_labels": [1, 2],
            }
        }
        self.target_fs = 10
        self.window_size = 10
        self.overlap = 5
        self.target_labels = [1, 2]
        self.subjects = ["S_TEST"]

    def load_subject(self, subject_id):
        # Return dummy dummy dictionary mimicking WESAD pickle structure
        # Source 700Hz signal, 20 seconds long => 14000 samples
        n_samples = 14000

        # Create dictionary structure
        return {
            "label": np.ones(n_samples),  # All label 1
            "signal": {
                "chest": {
                    "ACC": np.zeros((n_samples, 3)),
                    "ECG": np.zeros((n_samples, 1)),
                    "EDA": np.zeros((n_samples, 1)),
                    "Resp": np.zeros((n_samples, 1)),
                    "Temp": np.zeros((n_samples, 1)),
                }
            },
        }


def test_windowing_overlap_logic():
    """
    Verify that 20 seconds of data with 10s window and 5s overlap
    produces exactly 3 windows (0-10, 5-15, 10-20).
    """
    processor = MockWESADPreprocessor()

    # Run processing on dummy subject
    windows = processor.process_subject("S_TEST")

    # Expected behavior:
    # Duration: 20s
    # Window: 10s
    # Stride: 10 - 5 = 5s
    # Windows:
    # 1. Start 0, End 10
    # 2. Start 5, End 15
    # 3. Start 10, End 20
    # Next start 15+10 = 25 > 20 -> Stop.

    assert len(windows) == 3, f"Expected 3 windows, got {len(windows)}"

    # Check start indices (in native target_fs=10Hz units)
    # Window 1: start index 0
    # Window 2: start index 50 (5s * 10Hz)
    # Window 3: start index 100 (10s * 10Hz)

    assert windows[0]["start_idx"] == 0
    assert windows[1]["start_idx"] == 50
    assert windows[2]["start_idx"] == 100


def test_label_filtering():
    """
    Verify that windows with ignored labels (e.g. 0 or 4) are dropped.
    """
    processor = MockWESADPreprocessor()

    # Override load_subject to return ignored labels
    def mock_load(sid):
        n_samples = 14000
        labels = np.zeros(n_samples)  # Label 0 is ignored
        return {
            "label": labels,
            "signal": {
                "chest": {
                    "ACC": np.zeros((n_samples, 3)),
                    "ECG": np.zeros((n_samples, 1)),
                    "EDA": np.zeros((n_samples, 1)),
                    "Resp": np.zeros((n_samples, 1)),
                    "Temp": np.zeros((n_samples, 1)),
                }
            },
        }

    processor.load_subject = mock_load

    windows = processor.process_subject("S_TEST")
    assert len(windows) == 0, "Should drop windows with non-target labels"


def test_loso_split_no_leakage():
    """
    Verify that Leave-One-Subject-Out (LOSO) splitting prevents data leakage.
    Train and Test sets must have disjoint subject IDs.
    """
    # Mock data
    groups = np.array(["S1"] * 10 + ["S2"] * 10 + ["S3"] * 10)

    # Instantiate trainer (mock config)
    # We patch load_config inside Trainer to avoid filesystem issues
    with patch("src.models.train.load_config") as mock_conf:
        # Need to mock enough config so Trainer __init__ doesn't fail
        mock_conf.return_value = {
            "data": {"processed_path": "data/processed", "sensor_location": "chest"}
        }
        # Also need to patch PROJECT_ROOT because Trainer uses it to make dir
        with patch("src.models.train.PROJECT_ROOT", MagicMock()) as mock_root:
            mock_root.__truediv__.return_value.mkdir.return_value = None

            trainer = Trainer(model_type="logistic", split_type="loso")

            # The Trainer might need a .run_dir to exist for logging, but we mocked mkdir
            trainer.run_dir = MagicMock()
            # Mock the dump config call
            with patch("yaml.dump"):
                train_idx, test_idx = trainer.get_split(groups)

                train_subjects = np.unique(groups[train_idx])
                test_subjects = np.unique(groups[test_idx])

                # Intersection must be empty
                intersection = np.intersect1d(train_subjects, test_subjects)
                assert (
                    len(intersection) == 0
                ), f"Subject leakage detected: {intersection}"

                # Verify specific behavior: S3 is usually the last one, or random depends on implementation
                # Trainer's get_split typically yields one fold. Let's ensure we got valid indices.
                assert len(train_idx) > 0
                assert len(test_idx) > 0
