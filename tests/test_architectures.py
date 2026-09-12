"""Architectural invariants that shape-only smoke tests miss."""
import pytest
import torch
import torch.nn.functional as F

from sqm_ai.dl.architectures import (
    MLP,
    BasicCNN,
    LSTMClassifier,
    PositionalEncoding,
    Transformer,
    causal_mask,
    scaled_dot_product_attention,
)
from sqm_ai.dl.cnc_cnn import CNCWindowNet


def test_attention_matches_torch_and_rejects_empty_rows():
    torch.manual_seed(42)
    q, k, v = [torch.randn(2, 3, 5, 4) for _ in range(3)]
    mask = causal_mask(5)
    out, weights = scaled_dot_product_attention(q, k, v, mask)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3, 5))
    assert weights[..., 0, 1:].eq(0).all()
    with pytest.raises(ValueError, match="allowed key"):
        scaled_dot_product_attention(q, k, v, torch.zeros_like(mask))


def test_decoder_cannot_read_future_and_greedy_is_defined():
    torch.manual_seed(3)
    model = Transformer(17, d_model=12, num_heads=3, num_layers=2, d_ff=24, max_len=10, dropout=0).eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 8, 9]])
    torch.testing.assert_close(model(a)[:, :2], model(b)[:, :2])
    expected = model(a)[:, -1].argmax(-1)
    generated = model.generate(a, 1, temperature=0)
    torch.testing.assert_close(generated[:, -1], expected)
    with pytest.raises(ValueError, match="temperature"):
        model.generate(a, 1, temperature=-1)
    with pytest.raises(ValueError, match="capacity"):
        model.generate(a, 10)
    F.cross_entropy(model(a[:, :-1]).reshape(-1, 17), a[:, 1:].reshape(-1)).backward()
    assert model.token_embedding.weight.grad is not None


def test_odd_positional_width_and_model_shapes():
    assert PositionalEncoding(5)(torch.zeros(2, 7, 5)).shape == (2, 7, 5)
    assert MLP(20, [8])(torch.zeros(3, 20)).shape == (3, 1)
    assert BasicCNN()(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    assert LSTMClassifier(5, 8)(torch.zeros(2, 7, 5)).shape == (2,)
    cnn = CNCWindowNet().eval()
    assert sum(p.numel() for p in cnn.parameters()) == 36737
    assert cnn(torch.zeros(2, 5, 512)).shape == (2,)
