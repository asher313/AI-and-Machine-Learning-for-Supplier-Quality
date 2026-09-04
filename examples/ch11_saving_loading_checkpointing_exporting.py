# Chapter 11 — 11.8 Saving, Loading, Checkpointing, Exporting
import torch

torch.save(model.state_dict(), "model.pt")

model = SupplierRiskNet(input_dim=20)    # same architecture
model.load_state_dict(
    torch.load("model.pt", weights_only=True)
)
model.eval()                             # before inference
