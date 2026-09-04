# Chapter 11 — 11.7 Optimizers, Schedulers, Losses
import torch

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9,
    weight_decay=1e-4, nesterov=True,
)
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8
)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-2
)

# different learning rates per part of the model
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},
    {"params": model.head.parameters(), "lr": 1e-3},
])
