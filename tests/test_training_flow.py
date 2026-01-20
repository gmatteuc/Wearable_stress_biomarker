"""
Training Flow Smoke Tests
=========================

Integration tests that run the full training loop (for a tiny number of steps)
to ensure that data loading, model forward/backward pass, loss calculation,
and artifact saving are all working together without crashing.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.train import Trainer


@pytest.fixture
def mock_trainer_setup():
    """
    Creates a Trainer instance with:
    1. A temporary run directory.
    2. Mocked data loading to return a tiny dataset.
    3. Overridden config for 1 epoch and small batch size.
    """
    # Create temp dir
    temp_dir = Path(tempfile.mkdtemp())

    # Instantiate trainer (Deep Learning LOPO)
    trainer = Trainer(model_type="deep", split_type="loso")

    # Override run directory to temp
    trainer.run_dir = temp_dir

    # Mock Config for speed
    trainer.config["training"] = {
        "batch_size": 2,
        "epochs": 1,  # Only 1 epoch
        "learning_rate": 0.001,
    }

    # Create Dummy Data
    # 6 samples: 3 subjects (S1, S2, S3), 2 samples each.
    # Shape: (6, 7, 210)  (Time reduced for speed)
    n_samples = 6
    n_channels = 7
    seq_len = 210

    X_dummy = np.random.randn(n_samples, n_channels, seq_len).astype(np.float32)
    y_dummy = np.array([1, 1, 1, 2, 2, 2])  # Classes 1 and 2
    groups_dummy = np.array(["S1", "S1", "S2", "S2", "S3", "S3"])

    # Mock load_data
    trainer.load_data = MagicMock(return_value=(X_dummy, y_dummy, groups_dummy))

    yield trainer

    # Cleanup
    shutil.rmtree(temp_dir)


def test_deep_learning_training_loop(mock_trainer_setup):
    """
    Run the `train()` method. If it finishes without exception and produces
    artifacts, the test passes.
    """
    trainer = mock_trainer_setup

    # Mock the evaluate_model to prevent it from trying to plot complex things
    # or write to disk, which might slow us down or error on missing fonts.
    # However, Trainer.train() calls `evaluate_model` at the END.
    # Let's mock it to return dummy metrics.
    with patch("src.models.train.evaluate_model") as mock_eval:
        mock_eval.return_value = ({"accuracy": 0.5}, pd.DataFrame())

        # Overwrite the model creation to handle the reduced sequence length if necessary.
        # ResNet1D uses AdaptiveAvgPool so sequence length shouldn't matter too much,
        # provided it's larger than the receptive field of first layers.
        # 210 samples @ 35Hz is 6 seconds. Might be tight for convolutions.
        # Let's retry with standard length in fixture if this fails, but Adaptive should handle it.

        # RUN
        try:
            trainer.run()
        except Exception as e:
            pytest.fail(f"Training loop crashed: {e}")

        # CHECK ARTIFACTS
        # The trainer should save `model.pt` at the end
        assert (
            trainer.run_dir / "model.pt"
        ).exists(), "Model artifact model.pt not found after training"

        # We mocked evaluate_model, so predictions.csv won't be created.
        # pass
