# src/sqm_ai/cnc/train_stage2.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sqm_ai.cnc.preprocessing import (
    channel_statistics,
    prepare_window,
)
from sqm_ai.dl.cnc_cnn import CNCWindowNet
from sqm_ai.dl.train import train


def fit_stage2(
    windows,
    labels,
    split,
    valid_lengths,
    *,
    device="cpu",
    epochs=30,
    ckpt_path="artifacts/cnc_stage2.pt",
):
    """Chronologically ordered, out-of-sample gated windows only.

    Caller enforces gate fit times and inspection availability. The tail
    selects the checkpoint; final tests must be provided separately.
    """
    labels, valid_lengths = (
        np.asarray(labels),
        np.asarray(valid_lengths),
    )
    if (
        not 2 <= split < len(labels)
        or len(windows) != len(labels)
        or valid_lengths.shape != labels.shape
    ):
        raise ValueError(
            "nonempty aligned chronological train/validation split required"
        )
    if (
        not np.isin(labels, [0, 1]).all()
        or np.unique(labels[:split]).size != 2
    ):
        raise ValueError(
            "binary labels and both training classes required"
        )
    if not np.isin(valid_lengths, np.arange(2, 513)).all():
        raise ValueError(
            "valid lengths must be integers between 2 and 512"
        )
    mean, scale = channel_statistics(
        windows[:split], valid_lengths[:split]
    )
    x = np.stack(
        [
            prepare_window(w[:, : int(n)], mean, scale)
            for w, n in zip(
                windows, valid_lengths, strict=True
            )
        ]
    )
    x, y = (
        torch.from_numpy(x),
        torch.tensor(labels, dtype=torch.float32),
    )
    tr = DataLoader(
        TensorDataset(x[:split], y[:split]),
        batch_size=min(64, split),
        shuffle=True,
    )
    va = DataLoader(
        TensorDataset(x[split:], y[split:]), batch_size=256
    )
    pos = labels[:split].sum()
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [(split - pos) / pos], dtype=torch.float32
        )
    )
    torch.manual_seed(42)
    model = CNCWindowNet()
    train(
        model,
        tr,
        va,
        epochs=epochs,
        device=device,
        criterion=criterion,
        ckpt_path=ckpt_path,
    )
    return model, mean, scale
