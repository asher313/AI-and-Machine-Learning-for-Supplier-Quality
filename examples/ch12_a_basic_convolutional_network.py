# Chapter 12 — A basic convolutional network
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    """3 conv blocks + a linear head, for 3x32x32 images."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)     # halves H and W
        self.dropout = nn.Dropout(0.3)
        # after 3 pools: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))   # (B, 32,16,16)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))   # (B,128, 4, 4)
        x = x.view(x.size(0), -1)              # (B, 2048)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
