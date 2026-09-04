# Chapter 12 — Scaled dot-product attention
import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(dk)) V

    Q:    (B, h, Tq, dk)   queries
    K:    (B, h, Tk, dk)   keys
    V:    (B, h, Tk, dv)   values
    mask: (B, 1, Tq, Tk) or None; 0 = block

    Returns (B, h, Tq, dv) and (B, h, Tq, Tk).
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1)      # (B,h,Tq,Tk)
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(
            mask == 0, float("-inf")
        )
    weights = F.softmax(scores, dim=-1)
    output = weights @ V                  # (B,h,Tq,dv)
    return output, weights
