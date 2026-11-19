###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Which attention and MLP each layer runs, and which pipeline stage holds it.

The shape reports count kernels per rank. Under pipeline parallelism a rank
holds one stage, so its counts are those of the layers in that stage rather
than the whole model's, and a model that mixes layer types (MoE after dense
layers, sliding-window or linear layers among full-attention ones, DeepSeek-V4's
compression schedule) needs each layer's type to split them.

Where a config gives only a count rather than positions, the layers are spread
evenly. That only decides which stage gets them; whole-model totals are exact.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

FULL, WINDOW, LINEAR = "full", "window", "linear"
V4_SWA, V4_CSA, V4_HCA = "v4_swa", "v4_csa", "v4_hca"
_V4_KINDS = {0: V4_SWA, 4: V4_CSA, 128: V4_HCA}
V4_RATIO = {V4_SWA: 0, V4_CSA: 4, V4_HCA: 128}


@dataclass(frozen=True)
class Layer:
    attn: str
    moe: bool
    indexed: bool = False


@dataclass(frozen=True)
class Counts:
    """Layers of each type on one rank."""

    layers: int
    dense: int
    moe: int
    attn: Counter
    indexed: int
    first: bool = True
    last: bool = True

    def n(self, *kinds: str) -> int:
        return sum(self.attn.get(k, 0) for k in kinds)

    @property
    def softmax(self) -> int:
        """Layers that run softmax attention over a KV cache."""
        return self.layers - self.n(LINEAR)


def _spread(n: int, total: int) -> set[int]:
    """``n`` of ``total`` positions, evenly spaced."""
    if n <= 0 or total <= 0:
        return set()
    n = min(n, total)
    return {min(total - 1, (i * total) // n + (total // n) - 1) for i in range(n)}


def window_of(cfg) -> tuple[int, float]:
    """``(window, share of attention layers windowed)``; ``(0, 0)`` for none."""
    mc, req = cfg.model_config, getattr(cfg, "request_config", None)
    model_w = int(getattr(mc, "sink_sliding_window", 0) or 0)
    w = req.resolved_sliding_window(model_w) if hasattr(req, "resolved_sliding_window") else model_w
    if not w:
        return 0, 0.0
    even = bool(getattr(mc, "sink_window_even_layers_only", False))
    frac = float(getattr(mc, "sink_window_layer_fraction", 0.0) or 0.0)
    if hasattr(req, "resolved_sliding_window_fraction"):
        return w, float(req.resolved_sliding_window_fraction(even, frac))
    return w, frac or (0.5 if even else 1.0)


def _v4_schedule(mc) -> list[int] | None:
    if not getattr(mc, "hybrid_attention_enabled", False):
        return None
    from infera.projection.core.projection.training_config import parse_compress_ratios

    sched = parse_compress_ratios(mc) or []
    if not sched or any(int(c) not in _V4_KINDS for c in sched):
        return None
    return [int(c) for c in sched]


def plan(cfg) -> list[Layer]:
    """One entry per layer of the model, in order."""
    mc = cfg.model_config
    n = int(mc.num_layers or 0)
    pattern = list(mc.moe_pattern or [0] * n)
    moe = [
        bool(getattr(mc, "num_experts", 0)) and bool(pattern[i] if i < len(pattern) else 0)
        for i in range(n)
    ]

    sched = _v4_schedule(mc)
    if sched:
        attn = [_V4_KINDS[sched[i]] if i < len(sched) else V4_SWA for i in range(n)]
    else:
        n_lin = (
            mc.linear_attention_layer_count() if hasattr(mc, "linear_attention_layer_count") else 0
        )
        freq = int(getattr(mc, "linear_attention_freq", 0) or 0)
        if n_lin and freq > 1 and not int(getattr(mc, "linear_attention_layers", 0) or 0):
            # Qwen-style interleave: every ``freq``-th layer is full attention.
            linear = {i for i in range(n) if (i + 1) % freq}
        else:
            linear = set(range(n)) - _spread(n - n_lin, n)
        attn = [LINEAR if i in linear else FULL for i in range(n)]
        w, frac = window_of(cfg)
        if w:
            softmax = [i for i in range(n) if attn[i] == FULL]
            n_win = min(len(softmax), round(len(softmax) * frac))
            if getattr(mc, "sink_window_even_layers_only", False) and abs(frac - 0.5) < 1e-9:
                picked = set(softmax[0::2][:n_win])
            else:
                picked = {softmax[j] for j in _spread(n_win, len(softmax))}
            attn = [WINDOW if i in picked else a for i, a in enumerate(attn)]

    idx_layers = int(getattr(mc, "sparse_index_layers", 0) or 0)
    if int(getattr(mc, "sparse_index_n_heads", 0) or 0) and int(
        getattr(mc, "sparse_index_head_dim", 0) or 0
    ):
        softmax = [i for i in range(n) if attn[i] != LINEAR]
        k = min(len(softmax), idx_layers or len(softmax))
        indexed = {softmax[j] for j in _spread(k, len(softmax))}
    else:
        indexed = {i for i in range(n) if attn[i] == V4_CSA}
    return [Layer(attn[i], moe[i], i in indexed) for i in range(n)]


def stages(cfg) -> list[range]:
    """The layers each pipeline stage holds, split as the language-model profiler splits them."""
    from infera.projection.core.projection.module_profilers.language_model import (
        _get_balanced_layer_distribution,
        _get_explicit_layer_distribution,
        _parse_layout_stage_layer_counts,
    )

    mc, mp = cfg.model_config, cfg.model_parallel_config
    n = int(mc.num_layers or 0)
    pp = max(1, int(getattr(mp, "pipeline_model_parallel_size", 1) or 1))
    if pp <= 1 or n <= 0:
        return [range(n)]
    layout = getattr(mp, "pipeline_model_parallel_layout", None)
    first = getattr(mp, "decoder_first_pipeline_num_layers", None)
    last = getattr(mp, "decoder_last_pipeline_num_layers", None)
    if layout:
        sizes = _parse_layout_stage_layer_counts(layout, pp, n)
    elif first is not None or last is not None:
        sizes = _get_explicit_layer_distribution(n, pp, first, last)
    else:
        sizes = _get_balanced_layer_distribution(n, pp)
    out, start = [], 0
    for size in sizes:
        out.append(range(start, start + int(size)))
        start += int(size)
    return out


def counts(cfg, layers: range | None = None, *, first: bool = True, last: bool = True) -> Counts:
    """Layer counts over ``layers`` (default: the whole model)."""
    p = plan(cfg)
    sel = [p[i] for i in (layers if layers is not None else range(len(p)))]
    n_moe = sum(1 for x in sel if x.moe)
    return Counts(
        layers=len(sel),
        dense=len(sel) - n_moe,
        moe=n_moe,
        attn=Counter(x.attn for x in sel),
        indexed=sum(1 for x in sel if x.indexed),
        first=first,
        last=last,
    )


def stage_counts(cfg) -> list[Counts]:
    st = stages(cfg)
    return [counts(cfg, r, first=i == 0, last=i == len(st) - 1) for i, r in enumerate(st)]
