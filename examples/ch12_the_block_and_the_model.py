import torch
import torch.nn as nn
from sqm_ai.dl.architectures import MultiHeadAttention, FeedForward
# Chapter 12 — The block, and the model
class TransformerBlock(nn.Module):
    """Pre-norm attention + feed-forward, both residual."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model, num_heads, dropout
        )
        self.feed_forward = FeedForward(
            d_model, d_ff, dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.attention(
            normed, normed, normed, mask
        )
        x = x + attn_out  # attention already applies output dropout

        normed = self.norm2(x)
        x = x + self.dropout(self.feed_forward(normed))
        return x
