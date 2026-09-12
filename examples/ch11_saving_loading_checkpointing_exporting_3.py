# Chapter 11 — 11.8 Saving, Loading, Checkpointing, Exporting
import torch

model = model.cpu().eval()
dummy = torch.randn(2, 20)         # batch > 1 for dynamic export
batch_dim = torch.export.Dim("batch", min=1)

# Current PyTorch graph capture; deployment needs a runtime
exported = torch.export.export(
    model, (dummy,), dynamic_shapes=({0: batch_dim},),
)
torch.export.save(exported, "model.pt2")

# ONNX: portable, runs outside PyTorch
torch.onnx.export(
    model,
    (dummy,),
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamo=True,
    dynamic_shapes=({0: batch_dim},),
)
