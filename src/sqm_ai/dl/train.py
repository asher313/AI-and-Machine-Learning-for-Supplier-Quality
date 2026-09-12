# src/sqm_ai/dl/train.py
"""Binary mini-batch training; restore the best validation checkpoint."""

import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    criterion: nn.Module | None = None,
    ckpt_path: str = "best_model.pt",
) -> dict[str, list[float]]:
    if epochs < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError(
            "positive epochs and learning rate required"
        )
    criterion = (
        criterion
        if criterion is not None
        else nn.BCEWithLogitsLoss()
    )
    if (
        not isinstance(criterion, nn.BCEWithLogitsLoss)
        or criterion.reduction != "mean"
    ):
        raise ValueError(
            "this binary loop requires mean BCEWithLogitsLoss"
        )
    model = model.to(device)
    criterion = criterion.to(
        device
    )  # includes pos_weight buffers
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    checkpoint = Path(ckpt_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        losses, accuracy = {}, 0.0
        for phase, loader in (
            ("train", train_loader),
            ("val", val_loader),
        ):
            training = phase == "train"
            model.train(training)
            loss_sum, seen, correct = 0.0, 0, 0
            with torch.set_grad_enabled(training):
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device).float()
                    if batch_y.ndim != 1 or not torch.all(
                        (batch_y == 0) | (batch_y == 1)
                    ):
                        raise ValueError(
                            "one 0/1 target per example required"
                        )
                    if training:
                        optimizer.zero_grad()  # 1. clear
                    logits = model(batch_x)  # 2. forward
                    if logits.shape != batch_y.shape:
                        raise ValueError(
                            "logits and target shapes differ"
                        )
                    loss = criterion(
                        logits, batch_y
                    )  # 3. loss
                    if not torch.isfinite(loss):
                        raise ValueError(
                            "non-finite loss; inspect data and weights"
                        )
                    if training:
                        loss.backward()  # 4. backward
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=1.0,
                            error_if_nonfinite=True,
                        )
                        optimizer.step()  # 5. update
                    seen += len(batch_y)
                    loss_sum += loss.item() * len(batch_y)
                    correct += (
                        ((logits >= 0) == batch_y.bool())
                        .sum()
                        .item()
                    )
            if seen == 0:
                raise ValueError(
                    f"{phase} loader produced no examples"
                )
            losses[phase] = (
                loss_sum / seen
            )  # honors drop_last and samplers
            if not training:
                accuracy = correct / seen

        scheduler.step()
        history["train_loss"].append(losses["train"])
        history["val_loss"].append(losses["val"])
        history["val_acc"].append(accuracy)
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={losses['train']:.4f} "
            f"val_loss={losses['val']:.4f} val_acc={accuracy:.4f}"
        )
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(
        torch.load(
            checkpoint, map_location=device, weights_only=True
        )
    )
    model.eval()
    return history
