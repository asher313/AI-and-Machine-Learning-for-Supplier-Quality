# Chapter 12 — Position-wise feed-forward
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
