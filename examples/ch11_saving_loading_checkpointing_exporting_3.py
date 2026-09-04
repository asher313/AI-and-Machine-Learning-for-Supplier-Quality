# Chapter 11 — 11.8 Saving, Loading, Checkpointing, Exporting
import torch

model.eval()
dummy = torch.randn(1, 20)         # one representative batch

# TorchScript: trace the graph with a sample input
traced = torch.jit.trace(model, dummy)
traced.save("model_traced.pt")

# ONNX: portable, runs outside PyTorch
torch.onnx.export(
    model,
    (dummy,),
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"},
    },
)
