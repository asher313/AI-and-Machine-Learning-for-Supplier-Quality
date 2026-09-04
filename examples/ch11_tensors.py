# Chapter 11 — 11.1 Tensors
import numpy as np
import torch

t = torch.tensor([1, 2, 3])                 # int64, inferred
t = torch.tensor([1.0, 2.0, 3.0])           # float32
t = torch.tensor([1, 2, 3], dtype=torch.float32)

arr = np.array([1, 2, 3])
t = torch.from_numpy(arr)                   # shares memory
t = torch.tensor(arr)                       # copies

torch.zeros(3, 4)                           # shape (3, 4)
torch.ones(2, 3)
torch.eye(4)                                # identity
torch.arange(0, 10, 2)                      # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 11)

torch.manual_seed(42)                       # always seed
torch.rand(3, 3)                            # uniform [0, 1)
torch.randn(3, 3)                           # normal(0, 1)
torch.randint(0, 10, (3, 3))
