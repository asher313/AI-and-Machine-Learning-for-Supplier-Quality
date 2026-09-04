# Appendix D — D.1 The Canonical Training Loop (Chapter 11.5)
import torch
import torch.nn as nn
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
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    criterion = criterion or nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ---- training ----
        model.train()            # dropout on, batchnorm learning
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()              # 1. clear grads
            logits = model(batch_x)            # 2. forward
            loss = criterion(logits, batch_y)  # 3. loss
            loss.backward()                    # 4. backward
            torch.nn.utils.clip_grad_norm_(    # stop explosions
                model.parameters(), max_norm=1.0
            )
            optimizer.step()                   # 5. update
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- validation ----
        model.eval()             # dropout off, batchnorm frozen
        val_loss, correct = 0.0, 0
        with torch.no_grad():    # no graph, less memory
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == batch_y).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_loss < best_val_loss:   # keep the best weights
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    return history
