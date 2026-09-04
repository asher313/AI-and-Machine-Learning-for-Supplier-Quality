# Chapter 11 — 11.7 Optimizers, Schedulers, Losses
import torch
import torch.nn as nn

# count positives 10x: a missed failure costs more than a
# false alarm (Chapter 9 sets the number from the cost table)
pos_weight = torch.tensor([10.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# multi-class: one weight per class
class_weights = torch.tensor([1.0, 2.0, 5.0])
criterion = nn.CrossEntropyLoss(weight=class_weights)
