# Chapter 11 — 11.8 Saving, Loading, Checkpointing, Exporting
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_val_loss": best_val_loss,
    "config": {"input_dim": 20, "hidden_dim": 64},
}
torch.save(checkpoint, "checkpoint.pt")

ck = torch.load("checkpoint.pt", weights_only=True)
model.load_state_dict(ck["model_state_dict"])
optimizer.load_state_dict(ck["optimizer_state_dict"])
scheduler.load_state_dict(ck["scheduler_state_dict"])
start_epoch = ck["epoch"] + 1
