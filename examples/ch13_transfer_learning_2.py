# Chapter 13 — 13.1 Transfer Learning
model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(model.fc.in_features, num_classes)

optimizer = torch.optim.AdamW(
    [
        {"params": model.conv1.parameters(), "lr": 1e-5},
        {"params": model.bn1.parameters(), "lr": 1e-5},
        {"params": model.layer1.parameters(), "lr": 1e-5},
        {"params": model.layer2.parameters(), "lr": 1e-4},
        {"params": model.layer3.parameters(), "lr": 1e-4},
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ]
)
