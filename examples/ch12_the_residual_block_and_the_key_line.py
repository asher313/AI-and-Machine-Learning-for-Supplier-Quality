import torch
import torch.nn as nn
import torch.nn.functional as F
# Chapter 12 — The residual block, and the key line
class ResidualBlock(nn.Module):
    """Two convolutions plus a skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride,
            padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, 1,
            padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # match shapes on the skip path when they differ
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1,
                    stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity        # THE KEY LINE
        return F.relu(out)
