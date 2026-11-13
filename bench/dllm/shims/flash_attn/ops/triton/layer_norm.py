"""The eager RMSNorm the reference code carries commented out under its fused call."""

import torch


def rms_norm_fn(x, weight, bias=None, eps=1e-6, **_):
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    out = weight * h.to(dtype)
    return out if bias is None else out + bias
