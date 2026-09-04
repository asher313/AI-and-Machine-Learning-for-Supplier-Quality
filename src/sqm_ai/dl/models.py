# src/sqm_ai/dl/models.py
# Chapter 11 — 11.4 nn.Module
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupplierRiskNet(nn.Module):
    """MLP: supplier-month features -> logit of 90-day risk."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()          # required, always first
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # no sigmoid here: BCEWithLogitsLoss wants the logit
        return x.squeeze(-1)        # (batch,)



# The book's demonstration lines follow. They are kept
# verbatim but commented, so importing the module does not
# build a model as a side effect.
# model = SupplierRiskNet(input_dim=20)
# print(sum(p.numel() for p in model.parameters()))
# batch = torch.randn(32, 20)
# logits = model(batch)               # (32,)
# probs = torch.sigmoid(logits)       # probabilities
