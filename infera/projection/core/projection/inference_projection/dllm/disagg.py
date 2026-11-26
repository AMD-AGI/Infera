"""Prefill/decode disaggregation (xPyD) for canvas and block dLLMs.

A dLLM has no autoregressive decode, so the split differs from an AR model's: the prefill pool
runs the prompt forward plus ``prefill_steps`` denoise steps (the step that also builds the
prompt KV a serving proxy asks for), ships the prompt KV, and the decode pool runs every
remaining denoise step on it. A request's latency is P + KV transfer + D; ``x`` prefill and
``y`` decode instances share ``concurrency`` requests in flight, each pool at its own batch.

The P and D step costs come from the step model (``simulate_static``'s schedule). A measured
colocated latency curve can anchor them: the colocated simulation is scaled to the measurement
at each concurrency and the same factor applies to both pools; the KV transfer stays modeled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .hardware import dtype_bytes
from .model import DLLMModelSpec, Parallel
from .step_model import DLLMStepModel
from .workload import DLLMWorkload


@dataclass
class PDConfig:
    prefill_steps: int = 1
    # KV link per engine rank (e.g. a MoRI-IO RDMA read over one NIC) and its per-transfer cost.
    link_gbps: float = 40.0
    link_latency_us: float = 50.0
    # Measured colocated request latency (ms) by per-instance concurrency, to anchor P and D;
    # empty = none.
    anchor_ms: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prefill_steps < 0 or self.link_gbps <= 0 or self.link_latency_us < 0:
            raise ValueError("[dllm-sim] invalid PDConfig")


@dataclass
class PDResult:
    concurrency: int
    x: int
    y: int
    latency_ms: float
    prefill_ms: float
    transfer_ms: float
    decode_ms: float
    tokens_per_s: float
    tokens_per_s_per_gpu: float
    colocated_ms: float
    anchor_scale: float

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def prompt_kv_bytes(spec: DLLMModelSpec, par: Parallel, prompt_len: int) -> float:
    """KV of a ``prompt_len`` prompt on one engine rank, with sliding-window layers holding at
    most the window."""
    kv_bpe = dtype_bytes(spec.kv_cache_dtype)
    kv_heads = max(1, spec.num_query_groups // par.tp)
    L = spec.num_layers
    if spec.sliding_window and spec.global_layer_every:
        n_glob = L // spec.global_layer_every
        g_hd = spec.global_head_dim or spec.head_dim
        g_kvh = max(1, spec.global_kv_heads // par.tp) if spec.global_kv_heads else kv_heads
        local = (L - n_glob) * min(prompt_len, spec.sliding_window) * kv_heads * spec.head_dim
        total = local + n_glob * prompt_len * g_kvh * g_hd
    else:
        total = L * prompt_len * kv_heads * spec.head_dim
    return 2.0 * total * kv_bpe / par.pp


def _split(wl: DLLMWorkload, prefill_steps: int) -> tuple[list, list]:
    """Schedule steps on the prefill and decode pools."""
    sched = wl.schedule()
    n_prompt = 1 if (wl.cache == "block" and wl.prompt_len > 0) else 0
    cut = min(len(sched), n_prompt + prefill_steps)
    return sched[:cut], sched[cut:]


def _pool_ms(model: DLLMStepModel, steps: list, wl: DLLMWorkload, batch: int) -> float:
    return sum(model.step([s.shape] * (wl.seqs_per_request * batch)).total_ms for s in steps)


def _batch(requests: float, instances: int, par: Parallel) -> int:
    return max(1, math.ceil(requests / (instances * par.dp)))


def project_pd(
    model: DLLMStepModel,
    wl: DLLMWorkload,
    concurrency: int,
    x: int = 1,
    y: int = 1,
    cfg: PDConfig | None = None,
) -> PDResult:
    """Closed-loop xPyD projection at ``concurrency`` requests in flight (all instances).

    The decode pool holds nearly every in-flight request; the prefill pool holds the share of time
    a request spends there, so its batch is resized once from a first pass."""
    if concurrency < 1 or x < 1 or y < 1:
        raise ValueError("[dllm-sim] concurrency, x and y must be >= 1")
    cfg = cfg or PDConfig()
    par = model.par
    p_steps, d_steps = _split(wl, cfg.prefill_steps)
    # Colocated reference: one instance at the decode pool's per-instance load.
    per_inst = concurrency / y
    colo = _pool_ms(model, p_steps + d_steps, wl, _batch(per_inst, 1, par))
    scale = 1.0
    if cfg.anchor_ms:
        near = min(cfg.anchor_ms, key=lambda c: abs(math.log(c / per_inst)))
        scale = cfg.anchor_ms[near] / _pool_ms(model, p_steps + d_steps, wl, _batch(near, 1, par))
    kv = prompt_kv_bytes(model.spec, par, wl.prompt_len) * wl.seqs_per_request
    xfer = cfg.link_latency_us / 1e3 + kv / (cfg.link_gbps * 1e6)
    d_ms = scale * _pool_ms(model, d_steps, wl, _batch(concurrency, y, par))
    p_ms = scale * _pool_ms(model, p_steps, wl, _batch(concurrency, x, par))
    share = p_ms / (p_ms + xfer + d_ms)
    p_ms = scale * _pool_ms(model, p_steps, wl, _batch(concurrency * share, x, par))
    lat = p_ms + xfer + d_ms
    tps = concurrency * wl.gen_len / (lat / 1e3)
    gpus = (x + y) * par.gpus_per_replica * par.dp
    return PDResult(
        concurrency=concurrency,
        x=x,
        y=y,
        latency_ms=lat,
        prefill_ms=p_ms,
        transfer_ms=xfer,
        decode_ms=d_ms,
        tokens_per_s=tps,
        tokens_per_s_per_gpu=tps / gpus,
        colocated_ms=colo * scale,
        anchor_scale=scale,
    )
