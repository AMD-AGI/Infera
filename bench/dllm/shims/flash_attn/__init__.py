"""flash_attn stand-in for HF reference code on ROCm without a flash-attn build.

Attention runs through SDPA (aiter's fmha_fwd on ROCm), (batch, seq, heads, dim)
layout like flash_attn_func; the varlen entry points are not provided.
"""

import torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=causal,
        scale=softmax_scale,
        enable_gqa=q.shape[2] != k.shape[2],
    )
    return out.transpose(1, 2)


def flash_attn_varlen_func(*_, **__):
    raise NotImplementedError("varlen attention is not shimmed")
