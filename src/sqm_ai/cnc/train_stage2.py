# sqm_ai/cnc/train_stage2.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sqm_ai.dl.cnc_cnn import CNCWindowNet
from sqm_ai.dl.train import train


def stage2_loaders(
    windows: np.ndarray,      # (n, 5, 512), flagged only
    labels: np.ndarray,
    split: int,
) -> tuple[DataLoader, DataLoader]:
    x = torch.tensor(windows, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    tr = TensorDataset(x[:split], y[:split])
    va = TensorDataset(x[split:], y[split:])
    return (
        DataLoader(tr, batch_size=64, shuffle=True,
                   drop_last=True),
        DataLoader(va, batch_size=256, shuffle=False),
    )


def fit_stage2(windows, labels, split, device="cpu"):
    pos = float(labels[:split].sum())
    neg = float(split - pos)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / pos], device=device)
    )
    tr, va = stage2_loaders(windows, labels, split)
    model = CNCWindowNet(in_channels=5)
    train(
        model, tr, va,
        epochs=30, lr=1e-3, device=device,
        criterion=criterion,
        ckpt_path="artifacts/cnc_stage2.pt",
    )
    return model
