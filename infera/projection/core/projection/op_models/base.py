###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Pluggable analytical op models for inference projection.

The built-in profilers cover the architectures InferaSim already knows. A new
model often ships one op they do not -- a new attention, a new gate, a new
mixer -- and without a model the projection prices it as whatever block it
resembles.

An :class:`OpModel` is the unit a new op is added as. It declares which config
keys select and parameterise it (so the coverage check can tell modeled from
unmodeled), which slot of the layer it fills, and returns an :class:`OpCost`
for one forward step of one layer on one rank, built from the same pricing
primitives the built-in profilers use:

* projections and other dense matmuls on the GEMM backend (Origami);
* softmax-attention cores on the FAv3 SDPA simulator, so a new attention keeps
  the tile-level, 1-CU wave-quantised cost model;
* any other fused kernel through :meth:`OpContext.fused_tiles`: a
  per-workgroup tile program priced on one CU over the waves, the construction
  the SDPA simulator uses for FAv3;
* weight reads, KV / state traffic and elementwise work as HBM bytes at the
  sustained fraction the built-in profilers use.

A step is a prefill (``q_len`` new tokens per sequence over ``kv_len`` of
context) or a decode (``q_len`` = 1, or the speculative width). Decode is
usually weight- and cache-bound, so count bytes as carefully as FLOPs.

Ops live in :mod:`infera.projection.core.projection.op_models.generated` (written
by the tuning agent's op bring-up stage) or on ``INFERASIM_OP_MODEL_DIRS``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

SLOT_ATTENTION = "attention"
SLOT_ATTENTION_EXTRA = "attention_extra"
SLOT_MLP = "mlp"
SLOT_LAYER_EXTRA = "layer_extra"
SLOTS = (SLOT_ATTENTION, SLOT_ATTENTION_EXTRA, SLOT_MLP, SLOT_LAYER_EXTRA)

# The activation / weight-stream fraction of peak HBM the MoE, linear-attention
# and sampling profilers use (``moe_mlp._ACTIVATION_BW_FRACTION``).
DEFAULT_STREAM_EFFICIENCY = 0.566
_FALLBACK_HBM_GBPS = 5300.0
_FALLBACK_PEAK_TFLOPS = {"bf16": 1307.0, "fp16": 1307.0, "fp8": 2614.0, "mxfp4": 2614.0}
_FALLBACK_N_CU = 304

_INACTIVE = (None, False, 0, 0.0, "", "none", "None", "null")


def cfg_value(model_config, key: str, default: Any = None) -> Any:
    """Read *key* from the model config, then from the args it was built from.

    ``ModelConfig`` is a fixed dataclass, so keys a new op introduces are not
    fields on it; :func:`convert_config_to_projection_config` keeps them in
    ``model_config.extra_args`` instead of dropping them.
    """
    val = getattr(model_config, key, None)
    if val is not None:
        return val
    extra = getattr(model_config, "extra_args", None) or {}
    val = extra.get(key)
    return default if val is None else val


def is_active(value: Any) -> bool:
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    try:
        return value not in _INACTIVE
    except TypeError:
        return True


def full_num_layers(model_config) -> int:
    n = cfg_value(model_config, "projection_full_num_layers", None)
    if n:
        return int(n)
    return int(getattr(model_config, "num_layers", 0) or 0)


def interleave_fraction(model_config, every: int) -> float:
    """Share of layers that are *not* the every-``every``-th layer (Qwen3-Next convention)."""
    n = full_num_layers(model_config)
    every = int(every or 0)
    if n <= 0 or every <= 1:
        return 1.0 if every == 1 else 0.0
    return (n - max(1, n // every)) / n


@dataclass
class OpCost:
    """One forward step of one layer on one rank."""

    ms: float
    flops: float
    bytes: float
    # name -> ms; reported, and checked to sum to ``ms``.
    breakdown: dict[str, float] = field(default_factory=dict)
    # Memory-model terms, per layer the op runs on. ``None`` keeps the
    # built-in estimate. ``kv_bytes_per_token`` replaces the layer's KV cache
    # entry; ``state_bytes_per_seq`` is fixed-size per-sequence state (a
    # recurrent state, a conv window) the op keeps instead of or besides it.
    kv_bytes_per_token: float | None = None
    state_bytes_per_seq: float | None = None
    num_params: int | None = None


@dataclass
class OpContext:
    """Shapes, parallelism and pricing primitives for one forward step."""

    config: Any  # TrainingConfig view of the InferenceConfig
    batch: int
    q_len: int
    kv_len: int
    phase: str = "prefill"  # "prefill" | "decode" | "refine"
    gemm_backend: Any = None
    sdpa_backend: Any = None
    hbm_gbps: float | None = None

    @property
    def mc(self):
        return self.config.model_config

    @property
    def mp(self):
        return self.config.model_parallel_config

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def dp(self) -> int:
        """Attention data parallelism: splits TP and the requests, as the attention profiler does."""
        return max(1, int(getattr(self.mp, "attention_data_parallel_size", 1) or 1))

    @property
    def tp(self) -> int:
        tp = max(1, int(getattr(self.mp, "tensor_model_parallel_size", 1) or 1))
        return max(1, tp // self.dp) if self.dp > 1 else tp

    @property
    def cp(self) -> int:
        return max(1, int(getattr(self.mp, "context_model_parallel_size", 1) or 1))

    @property
    def batch_per_rank(self) -> int:
        return max(1, -(-int(self.batch) // self.dp))

    @property
    def tokens(self) -> int:
        """New tokens this rank processes in the step (GEMM ``M``)."""
        return max(1, self.batch_per_rank * int(self.q_len) // self.cp)

    def per_rank(self, width: int) -> int:
        """A TP-sharded width (heads x dim, channels) on one rank."""
        return max(1, int(width) // self.tp)

    def value(self, key: str, default: Any = None) -> Any:
        return cfg_value(self.mc, key, default)

    @property
    def weight_dtype(self) -> str:
        w = getattr(self.mc, "linear_weight_dtype", None)
        if w:
            return str(w)
        return "fp8" if getattr(self.mc, "fp8", None) else "bf16"

    @staticmethod
    def dtype_bytes(dtype: str) -> float:
        return {"fp8": 1.0, "mxfp4": 0.5, "fp4": 0.5, "int4": 0.5, "fp32": 4.0}.get(
            str(dtype).lower(), 2.0
        )

    # ---- hardware ----
    @property
    def _hw(self):
        return getattr(self.sdpa_backend, "_hw", None)

    @property
    def hbm_bytes_per_ms(self) -> float:
        gbps = self.hbm_gbps or getattr(self.gemm_backend, "hbm_bandwidth_gbps", None)
        if not gbps and self._hw is not None:
            gbps = getattr(self._hw, "hbm_bandwidth_gbps", None)
        return float(gbps or _FALLBACK_HBM_GBPS) * 1e6

    def peak_flops(self, dtype: str | None = None) -> float:
        dt = (dtype or self.weight_dtype).lower()
        key = "fp8" if dt in ("fp8", "mxfp4", "fp4") else ("fp16" if dt == "fp16" else "bf16")
        tf = getattr(self._hw, f"peak_tflops_{key}", None) if self._hw is not None else None
        return float(tf or _FALLBACK_PEAK_TFLOPS.get(key, 1307.0)) * 1e12

    @property
    def n_cu(self) -> int:
        return int(getattr(self._hw, "n_cu", 0) or _FALLBACK_N_CU)

    def roofline_ms(self, flops: float, nbytes: float, dtype: str | None = None) -> float:
        """Lower bound: neither peak compute nor peak HBM can be beaten."""
        return max(flops / self.peak_flops(dtype) * 1e3, nbytes / self.hbm_bytes_per_ms)

    # ---- pricing primitives ----
    def gemm(self, m: int, n: int, k: int, dtype: str | None = None, batch: int = 1) -> float:
        """One (batched) GEMM ``[m,k] x [k,n]``, in ms."""
        m, n, k, batch = max(1, int(m)), max(1, int(n)), max(1, int(k)), max(1, int(batch))
        dt = dtype or self.weight_dtype
        if self.gemm_backend is not None:
            return self.gemm_backend.simulate_gemm(m, n, k, dt, batch=batch).forward_time_ms
        bpe = self.dtype_bytes(dt)
        flops = 2.0 * m * n * k * batch
        nbytes = (m * k * 2 + k * n * bpe + m * n * 2) * batch
        return self.roofline_ms(flops, nbytes, dt)

    def linear(
        self, n_full: int, k_full: int, *, shard: str = "column", dtype: str | None = None
    ) -> float:
        """A projection on this rank's tokens. ``column`` shards N over TP, ``row`` shards K."""
        n = self.per_rank(n_full) if shard == "column" else int(n_full)
        k = self.per_rank(k_full) if shard == "row" else int(k_full)
        return self.gemm(self.tokens, n, k, dtype)

    def linear_bytes(
        self, n_full: int, k_full: int, *, shard: str = "column", dtype: str | None = None
    ) -> float:
        """HBM bytes of :meth:`linear`: weights once plus activations in and out."""
        n = self.per_rank(n_full) if shard == "column" else int(n_full)
        k = self.per_rank(k_full) if shard == "row" else int(k_full)
        return n * k * self.dtype_bytes(dtype or self.weight_dtype) + self.tokens * (n + k) * 2.0

    def sdpa(
        self,
        heads: int,
        head_dim: int,
        *,
        q_len: int | None = None,
        kv_len: int | None = None,
        head_dim_v: int | None = None,
        num_heads_kv: int | None = None,
        causal: bool | None = None,
    ) -> float:
        """Softmax-attention core for this rank's requests on the FAv3 simulator, ms."""
        q = int(q_len or self.q_len)
        kv = int(kv_len or self.kv_len)
        causal = (not self.is_decode) if causal is None else causal
        if self.sdpa_backend is not None:
            r = self.sdpa_backend.simulate_sdpa(
                batch_size=self.batch_per_rank,
                num_heads=int(heads),
                seq_len=q,
                head_dim=int(head_dim),
                causal=causal,
                dtype="bf16",
                seq_len_kv=kv,
                num_heads_kv=num_heads_kv,
                head_dim_v=head_dim_v,
            )
            return r.forward_time_ms
        d_v = head_dim_v or head_dim
        cf = 0.5 if causal and q == kv else 1.0
        flops = 2.0 * self.batch_per_rank * heads * q * kv * (head_dim + d_v) * cf
        nbytes = self.batch_per_rank * kv * (num_heads_kv or heads) * (head_dim + d_v) * 2.0
        return self.roofline_ms(flops, nbytes, "bf16")

    def fused_tiles(
        self,
        per_workgroup_gemms: Sequence[tuple[int, int, int]],
        n_workgroups: int,
        dtype: str = "bf16",
    ) -> float:
        """A fused kernel as a per-workgroup GEMM program, priced on one CU.

        ``per_workgroup_gemms`` lists the ``(m, n, k)`` matmuls one workgroup
        runs back to back; the kernel time is their 1-CU sum over
        ``ceil(n_workgroups / N_CU)`` waves -- the FAv3 construction, for any
        fused kernel. A loop a workgroup runs over the sequence is one matmul
        with the loop folded into a dimension (``(n_chunks * C, n, k)``).
        """
        tiles = [
            (max(1, int(m)), max(1, int(n)), max(1, int(k))) for m, n, k in per_workgroup_gemms
        ]
        n_wg = max(1, int(n_workgroups))
        sim = getattr(self.sdpa_backend, "simulate_fused_tiles", None)
        if callable(sim):
            return sim(tiles, n_wg, dtype)
        per_wg_flops = sum(2.0 * m * n * k for m, n, k in tiles)
        waves = math.ceil(n_wg / self.n_cu)
        return self.roofline_ms(per_wg_flops * self.n_cu * waves, 0.0, dtype)

    def stream_ms(self, nbytes: float, efficiency: float = DEFAULT_STREAM_EFFICIENCY) -> float:
        """Memory-bound traffic of ``nbytes`` at a sustained fraction of peak HBM."""
        return float(nbytes) / (efficiency * self.hbm_bytes_per_ms)


class OpModel:
    """Analytical model of one op the built-in profilers do not cover.

    Subclasses set the class attributes, implement :meth:`cost`, and are
    registered with :func:`register_op_model`. :meth:`cost` must handle both
    phases: a prefill of ``q_len`` tokens over ``kv_len`` context and a decode
    step of ``q_len`` (usually 1) token per sequence.
    """

    name: ClassVar[str] = ""
    slot: ClassVar[str] = SLOT_ATTENTION
    claims: ClassVar[dict[str, tuple[Any, ...]]] = {}
    consumes: ClassVar[tuple[str, ...]] = ()
    ignores: ClassVar[dict[str, str]] = {}
    references: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def matches(cls, model_config) -> bool:
        for key, values in cls.claims.items():
            val = cfg_value(model_config, key)
            if values == ("*",):
                if is_active(val):
                    return True
                continue
            norm = {str(v).lower() for v in values}
            if isinstance(val, (list, tuple)):
                if any(str(v).lower() in norm for v in val):
                    return True
            elif val is not None and str(val).lower() in norm:
                return True
        return False

    def layer_fraction(self, model_config) -> float:
        return 1.0

    def cost(self, ctx: OpContext) -> OpCost:
        raise NotImplementedError

    def reference_flops(self, ctx: OpContext) -> float | None:
        """Closed-form forward FLOPs from the paper's equations, independent of :meth:`cost`."""
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name}, slot={self.slot})"
