# Chapter 11 — 11.2 Devices
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

t = torch.randn(3, 3).to(device)   # move a tensor
t = t.cpu()                        # bring it back
