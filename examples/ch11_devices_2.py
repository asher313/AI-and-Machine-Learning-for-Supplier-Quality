# Chapter 11 — 11.2 Devices
a = torch.randn(3, 3, device=device)
b = torch.randn(3, 3)              # on the CPU
# a + b                            # RuntimeError if device != cpu
c = a + b.to(a.device)             # works everywhere
