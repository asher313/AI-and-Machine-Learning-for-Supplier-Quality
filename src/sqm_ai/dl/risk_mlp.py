# src/sqm_ai/dl/risk_mlp.py
"""Leakage-safe chronological comparison; no import-time training."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from sqm_ai.build1.features import load_modelling_frame
from sqm_ai.build1.train import split_train_tail
from sqm_ai.dl.data import SupplierDataset
from sqm_ai.dl.models import SupplierRiskNet
from sqm_ai.dl.train import train
from sqm_ai.features import month_folds, prep


def run(feat, output, epochs=20, device="cpu"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    feat = feat.sort_values(
        ["month", "supplier_id"]
    ).reset_index(drop=True)
    X, y, _ = load_modelling_frame(
        feat
    )  # named predictors only
    results = []
    for fold, (tr, te) in enumerate(month_folds(feat), 1):
        fit_idx, val_idx = split_train_tail(feat, tr)
        transform = clone(prep).fit(X.iloc[fit_idx])
        x_fit, x_val, x_test = [
            np.asarray(
                transform.transform(X.iloc[idx]),
                dtype=np.float32,
            )
            for idx in (fit_idx, val_idx, te)
        ]
        torch.manual_seed(42 + fold)
        train_loader = DataLoader(
            SupplierDataset(
                x_fit, y.iloc[fit_idx].to_numpy()
            ),
            batch_size=256,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            SupplierDataset(
                x_val, y.iloc[val_idx].to_numpy()
            ),
            batch_size=1024,
            shuffle=False,
        )
        model = SupplierRiskNet(input_dim=x_fit.shape[1])
        train(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            device=device,
            ckpt_path=str(output / f"fold-{fold}.pt"),
        )  # restores best INNER-validation checkpoint
        with torch.no_grad():
            logits = model(
                torch.from_numpy(x_test).to(device)
            )
            probs = torch.sigmoid(logits).cpu().numpy()
        results.append(
            {
                "fold": fold,
                "roc_auc": float(
                    roc_auc_score(y.iloc[te], probs)
                ),
                "average_precision": float(
                    average_precision_score(y.iloc[te], probs)
                ),
            }
        )
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/supplier_month_features.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/risk_mlp"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    metrics = run(
        pd.read_parquet(args.input),
        args.output,
        args.epochs,
        args.device,
    )
    print(metrics.round(3).to_string(index=False))
    print("mean ROC-AUC:", round(metrics.roc_auc.mean(), 3))


if __name__ == "__main__":
    main()
