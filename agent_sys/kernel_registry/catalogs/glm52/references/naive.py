"""PyTorch semantic references; no SGLang, aiter, Triton or GPU dependency.

These references were authored for the catalog import. Historical campaign tests
used the preserved task drivers, not this module. FP32 reduction and top-k tie
ordering may differ from production; GPU differential validation is outstanding.
"""

from __future__ import annotations

import math

import torch


def sparse_mla(q_nope, q_rope, kv, indices, sm_scale, d_v=512, lengths=None):
    """Return FP32 attention and log2 LSE over selected physical KV slots.

    The DCP form passes per-row owned lengths. Empty rows have zero output and
    the production sentinel -2**30 for LSE. Non-DCP callers may ignore LSE.
    """
    slots = indices.reshape(indices.shape[0], -1).long()
    cache = kv.reshape(-1, kv.shape[-1]).float()
    output = torch.zeros(q_nope.shape[:2] + (d_v,), device=kv.device)
    lse = torch.full(q_nope.shape[:2], float(-(2**30)), device=kv.device)
    for row in range(slots.shape[0]):
        count = slots.shape[1] if lengths is None else int(lengths[row])
        chosen = slots[row, :count]
        chosen = chosen[chosen >= 0]
        if chosen.numel() == 0:
            continue
        key = cache[chosen]
        query = torch.cat((q_nope[row].float(), q_rope[row].float()), dim=-1)
        scores = (query @ key.T) * sm_scale
        output[row] = scores.softmax(-1) @ key[:, :d_v]
        lse[row] = scores.logsumexp(-1) / math.log(2)
    return output, lse


def unpack_index_cache(cache, page_size=64, head_dim=128, preshuffle=True):
    """Decode page bytes: FP8 keys followed by one FP32 scale per token."""
    raw = cache.contiguous().view(torch.uint8).reshape(-1, page_size * (head_dim + 4))
    values = raw[:, : page_size * head_dim].contiguous().view(torch.float8_e4m3fn)
    if preshuffle:
        values = values.reshape(-1, page_size // 16, head_dim // 16, 16, 16)
        values = values.permute(0, 1, 3, 2, 4).reshape(-1, page_size, head_dim)
    else:
        values = values.reshape(-1, page_size, head_dim)
    scales = raw[:, page_size * head_dim :].contiguous().view(torch.float32)
    return values.float() * scales.reshape(-1, page_size, 1)


def _scores(q, keys, weights):
    return (torch.relu(q.float() @ keys.float().T) * weights.float()[:, None]).sum(0)


def dsa_indexer(q, cache, weights, lengths, page_table, topk=2048, page_size=64):
    """Weighted ReLU dot scores, top-k, and logical-to-physical slot mapping."""
    keys = unpack_index_cache(cache, page_size, q.shape[-1])
    result = torch.full((q.shape[0], topk), -1, dtype=torch.int32, device=q.device)
    for row in range(q.shape[0]):
        length = int(lengths[row])
        if length == 0:
            continue
        pages = page_table[row, : math.ceil(length / page_size)].long()
        row_keys = keys[pages].reshape(-1, q.shape[-1])[:length]
        positions = _scores(q[row], row_keys, weights[row]).topk(min(topk, length)).indices
        slots = pages[positions // page_size] * page_size + positions % page_size
        result[row, : positions.numel()] = slots.to(torch.int32)
    return result


def dcp_indexer(
    q,
    cache,
    weights,
    local_table,
    local_lengths,
    real_table,
    other_packs,
    topk=2048,
    rank=0,
    world_size=4,
    page_size=64,
):
    """One DCP rank's local logits/top-k, supplied all-gather packs, and merge.

    other_packs has shape [world_size-1, 2, rows, topk]. Each pack stores FP32
    score bits in its first int32 plane and global logical positions in its second.
    The function performs no communication, matching the campaign operator boundary.
    """
    keys = unpack_index_cache(cache, page_size, q.shape[-1])
    rows = q.shape[0]
    values = torch.full((rows, topk), -float("inf"), device=q.device)
    positions = torch.full((rows, topk), -1, dtype=torch.int32, device=q.device)
    for row in range(rows):
        length = int(local_lengths[row])
        if length == 0:
            continue
        pages = local_table[row, : math.ceil(length / page_size)].long()
        row_keys = keys[pages].reshape(-1, q.shape[-1])[:length]
        score, local = _scores(q[row], row_keys, weights[row]).topk(min(length, topk))
        values[row, : score.numel()] = score
        global_pos = ((local // page_size) * world_size + rank) * page_size + local % page_size
        positions[row, : score.numel()] = global_pos.to(torch.int32)
    local_pack = torch.stack((values.contiguous().view(torch.int32), positions))
    others = other_packs.reshape(world_size - 1, 2, rows, topk)
    packs = torch.cat((local_pack[None], others), dim=0)
    scores = packs[:, 0].contiguous().view(torch.float32).permute(1, 0, 2).reshape(rows, -1)
    logical = packs[:, 1].permute(1, 0, 2).reshape(rows, -1)
    best, pick = scores.topk(topk, dim=1)
    chosen = logical.gather(1, pick).long()
    valid = torch.isfinite(best) & (chosen >= 0)
    safe = chosen.clamp(min=0)
    slots = real_table.long().gather(1, safe // page_size) * page_size + safe % page_size
    return torch.where(valid, slots, -1).to(torch.int32)


FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant_mxfp4(packed, scales):
    positive = torch.tensor(FP4_GRID, dtype=torch.float32, device=packed.device)
    lookup = torch.cat((positive, -positive))
    bits = packed.view(torch.uint8).long()
    values = torch.stack((lookup[bits & 15], lookup[bits >> 4]), dim=-1).flatten(-2)
    scale = torch.pow(2.0, scales.view(torch.uint8).float() - 127)
    return values * scale.repeat_interleave(32, dim=-1)


def quant_dequant_mxfp4(value):
    """Power-of-two scale covering amax/6, E2M1 round-to-nearest-even."""
    groups = value.float().reshape(-1, 32)
    maximum = groups.abs().amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2(maximum / 6)).clamp(-127, 127)
    exponent = torch.where(maximum > 0, exponent, -127)
    scale = torch.pow(2.0, exponent)
    normalized = (groups / scale).clamp(-6, 6)
    grid = torch.tensor(FP4_GRID, device=value.device)
    distance = (normalized.abs()[..., None] - grid).abs()
    matches = distance == distance.amin(-1, keepdim=True)
    even = torch.tensor([True, False] * 4, device=value.device)
    matches = torch.where((matches.sum(-1) > 1)[..., None], matches & even, matches)
    rounded = grid[matches.float().argmax(-1)] * normalized.sign()
    return (rounded * scale).reshape(value.shape)


def moe_mxfp4(x, ids, weights, w13, w13_scales, w2, w2_scales):
    """Routed SwiGLU MoE with quant-dequant at both GEMM inputs."""
    xq = quant_dequant_mxfp4(x)
    output = torch.zeros_like(x, dtype=torch.float32)
    intermediate = w13.shape[1] // 2
    for expert in torch.unique(ids.long()).tolist():
        rows, slots = torch.nonzero(ids == expert, as_tuple=True)
        first = xq[rows] @ dequant_mxfp4(w13[expert], w13_scales[expert]).T
        activated = torch.nn.functional.silu(first[:, :intermediate]) * first[:, intermediate:]
        second = quant_dequant_mxfp4(activated) @ dequant_mxfp4(w2[expert], w2_scales[expert]).T
        output.index_add_(0, rows, second * weights[rows, slots, None])
    return output


def _hadamard(value):
    size = value.shape[-1]
    result = value.float()
    stride = 1
    while stride < size:
        blocks = result.reshape(*value.shape[:-1], -1, 2, stride)
        left, right = blocks[..., 0, :], blocks[..., 1, :]
        result = torch.stack((left + right, left - right), -2).reshape(value.shape)
        stride *= 2
    return result / math.sqrt(size)


def _rope(value, positions, cos, sin):
    result = value.float().clone()
    position = positions.long().clamp(0, cos.shape[0] - 1)
    cosine, sine = cos[position].float(), sin[position].float()
    while cosine.ndim < value.ndim:
        cosine, sine = cosine.unsqueeze(1), sine.unsqueeze(1)
    width = cosine.shape[-1] * 2
    pairs = value[..., :width].float().reshape(*value.shape[:-1], -1, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * cosine - odd * sine, odd * cosine + even * sine), -1)
    result[..., :width] = rotated.flatten(-2).to(torch.bfloat16).float()
    return result


def _fp8(value, floor):
    scale = torch.pow(
        2.0, torch.ceil(torch.log2(value.float().abs().amax(-1).clamp(min=floor) / 448))
    )
    return (value.float() / scale[..., None]).to(torch.float8_e4m3fn), scale


def fused_indexer(
    *,
    q_lora,
    x_bf16,
    w_q_b,
    w_kw,
    norm_w,
    norm_b,
    cos2d,
    sin2d,
    positions,
    out_cache_loc,
    kv_cache,
    seqlens,
    page_table_64,
    eps,
    weights_scale,
    topk=2048,
):
    """Fused-indexer semantics, including in-place preshuffled cache writes.

    FP32 matrix products/reductions express the operator but are not a bit-exact
    emulator of HIP MFMA/reduction order. Production comparison remains the oracle
    for strict byte equality; this function is an independent readable reference.
    """
    rows, heads, dimension, page_size = q_lora.shape[0], 32, 128, 64
    projected = (
        (q_lora.float() @ w_q_b.float().T).to(torch.bfloat16).reshape(rows, heads, dimension)
    )
    kw = (x_bf16.float() @ w_kw.float().T).to(torch.bfloat16)
    query = _hadamard(_rope(projected, positions, cos2d, sin2d)).to(torch.bfloat16)
    q_fp8, q_scale = _fp8(query, 1e-10)
    head_scale = 1 / math.sqrt(heads)
    gate = (kw[:, dimension:].float() * head_scale).to(torch.bfloat16).float()
    gate = gate * q_scale * (weights_scale / head_scale)
    key = kw[:, :dimension].float()
    centered = key - key.mean(-1, keepdim=True)
    key = centered * torch.rsqrt((centered * centered).mean(-1, keepdim=True) + eps)
    key = (key * norm_w.float() + norm_b.float()).to(torch.bfloat16)
    key = _hadamard(_rope(key, positions, cos2d, sin2d)).to(torch.bfloat16)
    k_fp8, k_scale = _fp8(key, 1e-4)
    raw = kv_cache.view(torch.uint8).reshape(-1, page_size * (dimension + 4))
    dims = torch.arange(dimension, device=kv_cache.device)
    for row in range(rows):
        slot = int(out_cache_loc[row])
        if slot < 0:
            continue
        page, token = divmod(slot, page_size)
        offset = (
            (token // 16) * (16 * dimension) + (dims // 16) * 256 + (token % 16) * 16 + dims % 16
        )
        raw[page, offset] = k_fp8[row].view(torch.uint8)
        begin = page_size * dimension + token * 4
        raw[page, begin : begin + 4] = k_scale[row : row + 1].contiguous().view(torch.uint8)
    return dsa_indexer(q_fp8, kv_cache, gate, seqlens, page_table_64, topk, page_size)
