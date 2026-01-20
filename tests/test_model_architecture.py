"""
Model Architecture Tests
========================

Unit tests for the Deep Learning models to ensure:
1. Input tensor shapes are handled correctly.
2. Output shapes match (batch_size, num_classes).
3. Forward pass runs without errors on CPU.
"""

import torch

from src.models.deep import ResNet1D


def test_resnet1d_output_shape():
    """
    Verify that ResNet1D processes a standard batch of windows
    and produces logits of shape [batch, num_classes].
    """
    batch_size = 4
    channels = 8  # e.g., ACCx, ACCy, ACCz, EDA, TEMP, ...
    seq_len = 2100  # 60s @ 35Hz
    num_classes = 2

    model = ResNet1D(num_channels=channels, num_classes=num_classes)

    # Create dummy input: [Batch, Channels, Time]
    dummy_input = torch.randn(batch_size, channels, seq_len)

    # Forward pass
    output = model(dummy_input)

    # Check shape
    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Expected output shape {(batch_size, num_classes)}, got {output.shape}"


def test_resnet1d_variable_length():
    """
    Verify the model can handle slightly different lengths (due to Global Average Pooling).
    """
    model = ResNet1D(num_channels=4, num_classes=3)

    # Length 1000
    out1 = model(torch.randn(2, 4, 1000))
    assert out1.shape == (2, 3)

    # Length 2000
    out2 = model(torch.randn(2, 4, 2000))
    assert out2.shape == (2, 3)
