"""Binary-loop regressions: accounting, checkpoints and learnability."""

import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sqm_ai.dl.data import SupplierDataset
from sqm_ai.dl.models import SupplierRiskNet
from sqm_ai.dl.train import train


def test_parameter_count_and_dataset_contract():
    assert (
        sum(
            p.numel()
            for p in SupplierRiskNet(20).parameters()
        )
        == 5825
    )
    with pytest.raises(ValueError, match="binary"):
        SupplierDataset(np.zeros((3, 2)), np.array([0, 1, 2]))


def test_dropped_rows_do_not_dilute_loss(tmp_path):
    model = nn.Sequential(nn.Linear(2, 1), nn.Flatten(0))
    nn.init.zeros_(model[0].weight)
    nn.init.zeros_(model[0].bias)
    ds = TensorDataset(
        torch.zeros(5, 2),
        torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
    )
    loader = DataLoader(ds, batch_size=2, drop_last=True)
    history = train(
        model,
        loader,
        loader,
        epochs=1,
        ckpt_path=str(tmp_path / "nested/best.pt"),
    )
    assert history["train_loss"] == pytest.approx(
        [math.log(2)], abs=1e-6
    )
    assert history["val_acc"] == [0.5]
    assert not model.training


def test_best_epoch_restored(tmp_path, monkeypatch):
    model = nn.Linear(1, 1)
    model.forward = lambda x: nn.functional.linear(
        x, model.weight, model.bias
    ).squeeze(-1)
    ds = TensorDataset(torch.ones(4, 1), torch.zeros(4))
    loader = DataLoader(ds, batch_size=4)

    # Force progressively worse predictions after every update.
    def deteriorate(self, closure=None):
        with torch.no_grad():
            model.bias.add_(1)

    monkeypatch.setattr(
        torch.optim.AdamW, "step", deteriorate
    )
    path = tmp_path / "best.pt"
    history = train(
        model, loader, loader, epochs=3, ckpt_path=str(path)
    )
    assert history["val_loss"][0] < history["val_loss"][-1]
    saved = torch.load(path, weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name])


def test_overfits_one_batch(tmp_path):
    torch.manual_seed(12)
    x = torch.cat((-torch.ones(16, 2), torch.ones(16, 2)))
    y = torch.cat((torch.zeros(16), torch.ones(16)))
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Flatten(0),
    )
    history = train(
        model,
        loader,
        loader,
        epochs=80,
        lr=0.05,
        ckpt_path=str(tmp_path / "best.pt"),
    )
    assert min(history["val_loss"]) < 0.02
    assert history["val_acc"][-1] == 1.0


def test_rejects_empty_loader_and_wrong_loss(tmp_path):
    model = SupplierRiskNet(2)
    ds = SupplierDataset(np.zeros((1, 2)), np.zeros(1))
    loader = DataLoader(ds, batch_size=2, drop_last=True)
    with pytest.raises(ValueError, match="no examples"):
        train(
            model,
            loader,
            loader,
            ckpt_path=str(tmp_path / "empty.pt"),
        )
    with pytest.raises(ValueError, match="binary loop"):
        train(
            model,
            loader,
            loader,
            criterion=nn.CrossEntropyLoss(),
        )
