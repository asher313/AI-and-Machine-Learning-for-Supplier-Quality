# sqm_ai/dl/risk_mlp.py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from sqm_ai.dl.data import SupplierDataset
from sqm_ai.dl.models import SupplierRiskNet
from sqm_ai.dl.train import train
from sqm_ai.features import month_folds   # Chapter 7.4

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

feat = pd.read_parquet("data/supplier_month_features.parquet")
feat = feat.sort_values("month")
label = "sev3_next_90d"
X = feat.drop(columns=["supplier_id", "month", label])
y = feat[label]

aucs = []
for tr, va in month_folds(feat):        # same folds as Ch 10
    scaler = StandardScaler().fit(X.loc[tr])    # fit on train
    X_tr = scaler.transform(X.loc[tr]).astype(np.float32)
    X_va = scaler.transform(X.loc[va]).astype(np.float32)
    train_loader = DataLoader(
        SupplierDataset(X_tr, y.loc[tr].to_numpy()),
        batch_size=256, shuffle=True,
    )
    val_loader = DataLoader(
        SupplierDataset(X_va, y.loc[va].to_numpy()),
        batch_size=1024,
    )
    model = SupplierRiskNet(input_dim=X.shape[1])
    train(
        model, train_loader, val_loader,
        epochs=20, device=device,
    )
    model.load_state_dict(
        torch.load("best_model.pt", weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_va).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    aucs.append(roc_auc_score(y.loc[va], probs))

print("MLP ROC-AUC per fold:", np.round(aucs, 3))
print("mean:", round(float(np.mean(aucs)), 3))
