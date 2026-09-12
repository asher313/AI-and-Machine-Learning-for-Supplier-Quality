import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sqm_ai.dl.architectures import PositionalEncoding, TransformerBlock, causal_mask
# Chapter 12 — The block, and the model
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
