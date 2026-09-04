# Chapter 13 — 13.1 Transfer Learning
import torch
import torch.nn as nn
from torchvision import models

num_classes = 4
model = models.resnet50(weights="IMAGENET1K_V2")

# freeze everything the pretrained model learned
for param in model.parameters():
    param.requires_grad = False

# replace the classifier head; it starts trainable
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# pass ONLY trainable parameters to the optimizer
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
)
