# src/sqm_ai/dl/data.py
# Chapter 11 — 11.6 Datasets and DataLoaders
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SupplierDataset(Dataset):
    """Rows of a feature matrix plus a 0/1 label per row."""

    def __init__(
        self, features: np.ndarray, labels: np.ndarray
    ):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]



# The book's demonstration lines follow. They are kept
# verbatim but commented, because X_train/y_train are
# supplied by the caller, not by this module.
# # X_train, X_val: float feature matrices (NumPy)
# # y_train, y_val: 0/1 label arrays, one per row
# train_ds = SupplierDataset(X_train, y_train)
# val_ds = SupplierDataset(X_val, y_val)
#
# train_loader = DataLoader(
#     train_ds,
#     batch_size=32,
#     shuffle=True,        # reshuffle every epoch; critical
#     num_workers=4,       # parallel loading processes
#     pin_memory=True,     # faster CPU -> GPU copies
#     drop_last=True,      # skip the ragged final batch
# )
# val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
