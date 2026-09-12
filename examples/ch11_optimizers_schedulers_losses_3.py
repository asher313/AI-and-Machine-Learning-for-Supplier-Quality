# Chapter 11 — 11.7 Optimizers, Schedulers, Losses
import torch
import torch.nn as nn

# illustrative training weight, tuned on development data;
# not automatically the operating false-negative cost ratio
pos_weight = torch.tensor([10.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# multi-class: one weight per class
class_weights = torch.tensor([1.0, 2.0, 5.0])
criterion = nn.CrossEntropyLoss(weight=class_weights)
