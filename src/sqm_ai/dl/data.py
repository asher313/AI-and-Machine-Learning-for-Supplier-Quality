# src/sqm_ai/dl/data.py
import numpy as np
import torch
from torch.utils.data import Dataset


class SupplierDataset(Dataset):
    """Finite feature matrix and one binary label per row."""

    def __init__(
        self, features: np.ndarray, labels: np.ndarray
    ):
        features, labels = (
            np.asarray(features),
            np.asarray(labels),
        )
        if features.ndim != 2 or labels.shape != (
            len(features),
        ):
            raise ValueError(
                "features (n, d) and labels (n,) required"
            )
        if (
            not np.isfinite(features).all()
            or not np.isin(labels, [0, 1]).all()
        ):
            raise ValueError(
                "impute features and supply finite binary labels"
            )
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
