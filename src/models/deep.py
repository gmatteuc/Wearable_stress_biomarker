"""
Deep Learning Models Module
===========================

This module defines PyTorch neural network architectures for stress detection.
It includes a lightweight 1D-CNN optimized for raw time-series sensor data,
designed to be efficient enough for near-real-time inference.
"""

import torch
import torch.nn as nn

class Simple1DCNN(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, sequence_length: int):
        super(Simple1DCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # /2
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # /4
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Pool -> (B, 128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        x = self.features(x)
        x = self.classifier(x)
        return x
