# Chapter 11 — 11.3 Autograd
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x + 1
y.backward()
print(x.grad)          # dy/dx = 2x + 3 = 7.0 at x = 2
