# Chapter 11 — 11.7 Optimizers, Schedulers, Losses
from torch.optim import lr_scheduler as ls
import math

# halve the rate every 5 epochs
sched = ls.StepLR(optimizer, step_size=5, gamma=0.5)

# smooth decay to zero over the run (the loop's default)
sched = ls.CosineAnnealingLR(optimizer, T_max=epochs)

# react: cut the rate when validation loss stalls
sched = ls.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)
# call sched.step(val_loss) for this one, with the metric

# warmup then cosine: required for transformers
def warmup_cosine(step, warmup=1000, total=100_000):
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / (total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))

sched = ls.LambdaLR(optimizer, lr_lambda=warmup_cosine)
