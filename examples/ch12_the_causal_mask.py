# Chapter 12 — The causal mask
def causal_mask(size: int) -> torch.Tensor:
    """(1, 1, size, size); position i sees 0..i only."""
    mask = torch.tril(torch.ones(size, size))
    return mask.unsqueeze(0).unsqueeze(0)


print(causal_mask(4)[0, 0].int())
