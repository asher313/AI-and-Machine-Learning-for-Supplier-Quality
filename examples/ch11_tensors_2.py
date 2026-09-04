# Chapter 11 — 11.1 Tensors
t = torch.arange(12)              # shape (12,)

t.view(3, 4)                      # (3, 4), no copy
t.reshape(3, 4)                   # (3, 4), copies if it must
t.view(-1, 4)                     # -1 means "work it out": (3, 4)

m = torch.randn(2, 3, 4)
m.transpose(0, 1)                 # swap two dims: (3, 2, 4)
m.permute(2, 0, 1)                # reorder all dims: (4, 2, 3)

t = torch.randn(1, 3, 1, 4)
t.squeeze()                       # drop every size-1 dim: (3, 4)
t.squeeze(0)                      # drop dim 0 only: (3, 1, 4)
t.unsqueeze(0)                  # add dim at 0: (1, 1, 3, 1, 4)

a = torch.randn(3, 4)
b = torch.randn(3, 4)
torch.cat([a, b], dim=0)          # (6, 4): extends a dim
torch.cat([a, b], dim=1)          # (3, 8)
torch.stack([a, b], dim=0)        # (2, 3, 4): creates a dim
