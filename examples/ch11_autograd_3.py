# Chapter 11 — 11.3 Autograd
with torch.no_grad():
    preds = model(batch)           # no graph; outputs still allocate memory

loss_value = loss.detach().item()  # plain Python float
