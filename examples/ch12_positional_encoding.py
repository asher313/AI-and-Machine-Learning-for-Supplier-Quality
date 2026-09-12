# Chapter 12 — Positional encoding
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Add fixed sinusoidal position signals to embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model < 1 or max_len < 1:
            raise ValueError("positive position dimensions required")
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        step = torch.arange(0, d_model, 2).float()
        div = torch.exp(
            step * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        if x.size(1) > self.pe.size(1):
            raise ValueError("sequence exceeds positional capacity")
        return x + self.pe[:, : x.size(1), :].to(x.dtype)
