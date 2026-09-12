from sqm_ai.dl.architectures import scaled_dot_product_attention
# Chapter 12 — Multi-head attention
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """h attention heads in parallel, then a projection."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("positive width divisible by heads required")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, h, T, d_k)
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def merge(self, x: torch.Tensor) -> torch.Tensor:
        # (B, h, T, d_k) -> (B, T, d_model)
        B, _, T, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.split(self.W_q(query))
        K = self.split(self.W_k(key))
        V = self.split(self.W_v(value))
        out, weights = scaled_dot_product_attention(
            Q, K, V, mask
        )
        out = self.W_o(self.merge(out))
        return self.dropout(out), weights
