# Chapter 12 — 12.1 The Multi-Layer Perceptron
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Feedforward network: features in, logits out."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = MLP(20, [128, 64, 32], output_dim=1)
x = torch.randn(16, 20)
print(model(x).shape)          # torch.Size([16, 1])
