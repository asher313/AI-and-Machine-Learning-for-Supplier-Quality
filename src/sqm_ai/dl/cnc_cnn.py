# sqm_ai/dl/cnc_cnn.py
import torch
import torch.nn as nn


class CNCWindowNet(nn.Module):
    """1-D CNN over one machining cycle of sensor data."""

    def __init__(
        self, in_channels: int = 5, width: int = 32
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, width, 7, padding=3),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.MaxPool1d(2),               # 512 -> 256
            nn.Conv1d(width, width * 2, 5, padding=2),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),               # 256 -> 128
            nn.Conv1d(width * 2, width * 4, 3, padding=1),
            nn.BatchNorm1d(width * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),       # -> (B, 128, 1)
        )
        self.head = nn.Linear(width * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, 512)
        h = self.features(x).squeeze(-1)   # (batch, 128)
        return self.head(h).squeeze(-1)    # (batch,)


if __name__ == "__main__":
    net = CNCWindowNet()
    print(sum(p.numel() for p in net.parameters()))
    batch = torch.randn(8, 5, 512)
    print(net(batch).shape)
