# Chapter 12 — 12.3 Recurrent Networks: RNN and LSTM
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Sequence in, one prediction per sequence out."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_classes: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,     # input is (B, T, F)
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        output, (h_n, c_n) = self.lstm(x)
        # output: (B, T, hidden_dim)  — every timestep
        # h_n, c_n: (num_layers, B, hidden_dim) — final
        last = output[:, -1, :]        # (B, hidden_dim)
        return self.classifier(last).squeeze(-1)
