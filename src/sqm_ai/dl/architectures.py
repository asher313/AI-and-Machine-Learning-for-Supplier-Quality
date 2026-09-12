# src/sqm_ai/dl/architectures.py

"""Chapter 12 educational architectures; no import-time execution."""

import math

import torch
import torch.nn.functional as F
from torch import nn


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
        output, (_h_n, _c_n) = self.lstm(x)
        # output: (B, T, hidden_dim)  — every timestep
        # h_n, c_n: (num_layers, B, hidden_dim) — final
        last = output[:, -1, :]        # (B, hidden_dim)
        return self.classifier(last).squeeze(-1)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(dk)) V

    Q:    (B, h, Tq, dk)   queries
    K:    (B, h, Tk, dk)   keys
    V:    (B, h, Tk, dv)   values
    mask: (B, 1, Tq, Tk) or None; 0 = block

    Returns (B, h, Tq, dv) and (B, h, Tq, Tk).
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1)      # (B,h,Tq,Tk)
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        allowed = torch.broadcast_to(mask.to(Q.device).bool(), scores.shape)
        if not allowed.any(dim=-1).all():
            raise ValueError("every query needs at least one allowed key")
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output = weights @ V                  # (B,h,Tq,dv)
    return output, weights

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

class FeedForward(nn.Module):
    """Expand to d_ff, activate, project back to d_model."""

    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)   # usually 4x
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

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

class Transformer(nn.Module):
    """Token ids in, logits over the vocabulary out."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )
        self.positional_encoding = PositionalEncoding(
            d_model, max_len
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask=None):
        # x: (B, T), unpadded or right-padded token IDs
        causal = causal_mask(x.size(1), device=x.device)
        mask = causal if mask is None else causal & mask.to(x.device).bool()
        h = self.token_embedding(x)
        h = h * math.sqrt(self.d_model)
        h = self.dropout(self.positional_encoding(h))
        for block in self.blocks:
            h = block(h, mask)
        h = self.norm(h)
        return self.output(h)      # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        start_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample tokens one at a time, autoregressively."""
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if max_new_tokens < 0 or start_ids.ndim != 2 or start_ids.size(1) == 0:
            raise ValueError("nonempty (B, T) prompt and nonnegative length required")
        if start_ids.size(1) + max_new_tokens > self.positional_encoding.pe.size(1):
            raise ValueError("requested sequence exceeds context capacity")
        self.eval()
        ids = start_ids
        for _ in range(max_new_tokens):
            logits = self(ids)[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids

def causal_mask(size: int, device=None) -> torch.Tensor:
    """(1, 1, size, size); position i sees 0..i only."""
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)
