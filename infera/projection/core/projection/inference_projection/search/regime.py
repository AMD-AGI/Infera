###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Regime signature + config keying — the canonical parameter split.

Every inference recipe parameter is one of two kinds (see the design note §2):

* **regime-defining** — swaps the kernel/execution regime (dtype → different
  GEMM/attention kernels, backend, cudagraph/AITER path).  Two recipes that
  differ on any of these are *not* transportable from one another; each regime
  needs its own measured anchor.
* **transportable** — moves cost analytically from an existing measurement
  (layers, TP/EP/PP, batch, sequence length, concurrency, arrival rate).

This module is the single source of truth for that split.  It is deliberately
**stdlib-only** so the dependency-light ``benchmark_vllm.py`` (which runs inside
a bare vLLM container) can import it to key its result cache with the exact same
scheme the anchor store uses.

Public API
----------
``regime_signature(recipe)``  -> 16-hex hash over the regime axes only.
``config_key(recipe, extra)`` -> 16-hex hash over regime + transport (+extra);
                                 an exact-run identity (used by the bench cache).
``regime_distance(a, b)``     -> Hamming distance over the regime axes.
``recipe_from_meta(meta)``    -> canonical recipe dict from an artifact ``meta``.
``recipe_from_bench_args(args, env)`` -> canonical recipe from benchmark CLI args.
``recipe_from_inference_config(cfg)`` -> canonical recipe from an InferenceConfig.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Optional

# Regime-defining axes: differ on any of these → a different measured anchor.
#
# ``speculative`` is regime-defining rather than transportable because a
# speculative step emits a variable number of tokens: per-output-token latency
# becomes ``step_cost(batch * (k + 1)) / (1 + a + ... + a**k)``, and the
# acceptance rate ``a`` depends on the model, the draft head and the data. There
# is no way to derive ``a`` analytically from a non-speculative anchor, so a
# target that speculates needs its own measurement.
REGIME_AXES = (
    "model",
    "weight_dtype",
    "kv_cache_dtype",
    "moe_expert_dtype",
    "attention_backend",
    "cudagraph",
    "aiter",
    "aiter_ops",
    "speculative",
)

# Canonical value of the ``speculative`` axis when speculation is off. Distinct
# from ``None``, which means "this artifact predates speculative tracking".
SPECULATIVE_OFF = "off"

# Canonical value of the ``aiter_ops`` axis when no per-op override is set.
# Distinct from ``None``, which means "this artifact predates the tracking".
AITER_OPS_DEFAULT = "default"

# ``VLLM_ROCM_USE_AITER`` is the master switch; each kernel family is gated
# behind its own ``VLLM_ROCM_USE_AITER_<OP>``. The trailing underscore keeps the
# master switch itself out of the per-op axis, where ``aiter`` already carries it.
_AITER_OP_PREFIX = "VLLM_ROCM_USE_AITER_"

# Transportable axes: reconstructed analytically from an anchor in the same
# regime (the projector's restore + interpolation already implement these).
TRANSPORT_AXES = (
    "tp",
    "pp",
    "ep",
    "num_layers",
    "batch",
    "input_len",
    "output_len",
    "concurrency",
    "request_rate",
)


def _canon(v: Any) -> str:
    """Canonical, comparison-stable string for a single axis value."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        # Trim trailing zeros so 8.0 and 8 compare equal.
        if v == int(v):
            return str(int(v))
        return repr(round(v, 6))
    if isinstance(v, (int,)):
        return str(v)
    return str(v).strip().lower()


def normalise_model_id(name: Any) -> str:
    """Reduce a model name to a form comparable across its spellings.

    The same model reaches us as a preset ("gpt_oss_120B"), a HuggingFace id
    ("openai/gpt-oss-120b") and a checkout path ("/models/gpt-oss-120b"), so
    separators and case carry no information and only get in the way.
    """
    return "".join(c for c in str(name or "").lower() if c.isalnum())


def models_match(a: Any, b: Any) -> bool:
    """Whether two model names plausibly name the same model.

    Compared on the last path segment, because everything before it is
    provenance rather than identity: "/models/DeepSeek-R1" and
    "deepseek-ai/DeepSeek-R1" are the same checkpoint, but their full strings
    have no containment relation in either direction -- the local mount point
    and the org prefix each break it.

    Containment rather than equality on that segment, since a preset drops
    decoration the id keeps. Deliberately generous: the caller's alternative is
    refusing to reuse a warmup that is genuinely for this model, and the regime
    axes still have to agree before it is used for anything.
    """
    def leaf(v: Any) -> str:
        return normalise_model_id(str(v or "").rstrip("/").rsplit("/", 1)[-1])

    na, nb = leaf(a), leaf(b)
    if not na or not nb:
        return False
    return na in nb or nb in na


def _sig(recipe: Dict[str, Any], axes: Iterable[str]) -> str:
    payload = {k: _canon(recipe.get(k)) for k in axes}
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def regime_signature(recipe: Dict[str, Any]) -> str:
    """Hash over the regime-defining axes only.  Two recipes with the same
    signature are mutually transportable (same kernels/regime)."""
    return _sig(recipe, REGIME_AXES)


def config_key(recipe: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> str:
    """Exact-run identity: hash over regime + transport axes, plus any ``extra``
    (measurement knobs that change the number but not the regime, e.g.
    decode-steps).  Used by the benchmark result cache."""
    payload = {k: _canon(recipe.get(k)) for k in (*REGIME_AXES, *TRANSPORT_AXES)}
    if extra:
        for k, v in extra.items():
            payload[f"x_{k}"] = _canon(v)
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def regime_distance(
    a: Dict[str, Any], b: Dict[str, Any], *, ignore_missing: bool = True
) -> int:
    """Hamming distance over the regime axes.  0 => same regime (fully
    transportable).  When ``ignore_missing`` (default), an axis absent/None on
    *either* side is not counted — so a partially-specified target still matches
    an anchor on the axes both actually pin (e.g. an anchor with no
    ``attention_backend`` is not penalised against a target that sets one)."""
    d = 0
    for k in REGIME_AXES:
        av, bv = a.get(k), b.get(k)
        missing = av is None or bv is None or av == "" or bv == ""
        if k == "speculative" and missing:
            # Asymmetric on purpose. An artifact with no recorded speculative
            # setting predates the tracking, and everything measured then ran
            # without speculation — so treating unknown as "off" is right when
            # the other side is also off. The reverse is the dangerous case:
            # silently reusing a non-speculative anchor for a speculating target
            # under-predicts throughput by the whole acceptance factor, which is
            # a large error that looks like a plausible number. Count it.
            other = bv if (av is None or av == "") else av
            if other and _canon(other) != SPECULATIVE_OFF:
                d += 1
            continue
        if k == "aiter_ops" and missing:
            # Asymmetric for the same reason as ``speculative``. An artifact
            # with no recorded per-op state predates the tracking, and those
            # runs were measured on the default kernel set -- so unknown may
            # match a target that is also on defaults. It may not match one that
            # overrides an op: that anchor was taken on different kernels, and
            # reusing it returns a confident number for a stack never run.
            other = bv if (av is None or av == "") else av
            if other and _canon(other) != AITER_OPS_DEFAULT:
                d += 1
            continue
        if ignore_missing and missing:
            continue
        if _canon(av) != _canon(bv):
            d += 1
    return d


def speculative_axis(
    method: Optional[str], num_tokens: Optional[int] = None
) -> Optional[str]:
    """Canonical ``speculative`` axis value: ``None``, ``"off"`` or ``"spec:k"``.

    ``None`` propagates as "unknown" (artifact predates the tracking); an
    explicit absence of a method canonicalizes to :data:`SPECULATIVE_OFF`.

    The *method* deliberately does not appear in the value. A structural
    ``InferenceConfig`` has no notion of which draft head is used — it models
    speculation as ``k`` plus an acceptance rate — so hashing the method name
    would make every config-side target mismatch every benchmark-side anchor.
    ``k`` is included because it changes the tokens emitted per step. The method
    is still recorded in the artifact ``meta`` for provenance.
    """
    if method is None:
        return None
    m = str(method).strip().lower()
    if not m or m in ("none", "off", "0", "false"):
        return SPECULATIVE_OFF
    return f"spec:{int(num_tokens)}" if num_tokens else "spec"


def aiter_ops_axis(env: Optional[Dict[str, str]]) -> Optional[str]:
    """Canonical ``aiter_ops`` value: ``None``, ``"default"`` or ``"op=v,..."``.

    A recipe can turn off one AITER kernel family while the master switch stays
    on, so the ``aiter`` boolean alone does not identify which kernels ran: every
    such recipe hashes equal to base, and an anchor measured on one is handed
    back for the other.

    Every ``VLLM_ROCM_USE_AITER_*`` variable present is folded in, rather than a
    fixed list, so a switch vLLM adds later is tracked without a change here.

    ``None`` propagates as "unknown" (no environment was recorded); an
    environment with no per-op override canonicalizes to
    :data:`AITER_OPS_DEFAULT`.
    """
    if env is None:
        return None
    ops = sorted(
        (k[len(_AITER_OP_PREFIX):].lower(), _canon(v))
        for k, v in env.items()
        if str(k).upper().startswith(_AITER_OP_PREFIX)
    )
    return ",".join(f"{op}={val}" for op, val in ops) if ops else AITER_OPS_DEFAULT


# --------------------------------------------------------------------------
# Adapters: build a canonical recipe from the three sources that produce one.
# --------------------------------------------------------------------------

def recipe_from_meta(meta: Dict[str, Any], *, model: Optional[str] = None) -> Dict[str, Any]:
    """Canonical recipe from a benchmark artifact's ``meta`` block.  The
    *benchmark* parallelism (what it actually ran at) is recorded on the
    transport axes so the anchor's coverage is described in benchmark space;
    the restore to the target happens at reconstruction time."""
    # Weight dtype is what the run actually executed in, which is not the same
    # as what was asked for: plenty of checkpoints ship already quantized
    # (gpt-oss in mxfp4, DeepSeek-R1 in fp8) and vLLM resolves the dtype from the
    # checkpoint when ``--quantization`` is unset. Reading an unset flag as bf16
    # therefore mislabels those runs, and an anchor labelled bf16 can never match
    # the mxfp4 target it was actually measured for -- the warmup becomes
    # unusable for its own deployment.
    #
    # Prefer the resolved dtype the benchmark recorded; fall back to the
    # requested quantization; otherwise leave it unknown rather than assert a
    # value, since an unknown axis is skipped by ``regime_distance`` while a
    # wrong one forces a mismatch.
    quant = meta.get("weight_dtype") or meta.get("quantization")
    return {
        "model": model or meta.get("model"),
        "weight_dtype": quant if quant else None,
        "kv_cache_dtype": meta.get("kv_cache_dtype") or "bf16",
        "moe_expert_dtype": meta.get("moe_expert_dtype"),
        "attention_backend": meta.get("attention_backend"),
        "cudagraph": "eager" if meta.get("enforce_eager") else "graph",
        "aiter": bool(meta.get("use_aiter")),
        # Absent key => unknown (pre-tracking artifact), not "default".
        "aiter_ops": meta.get("aiter_ops"),
        # Absent key => unknown (pre-tracking artifact), not "off".
        "speculative": (
            speculative_axis(
                meta.get("speculative_method") or "",
                meta.get("speculative_num_tokens"),
            )
            if "speculative_method" in meta
            else None
        ),
        # transport (benchmark space)
        "tp": meta.get("benchmark_tp") or meta.get("tp"),
        "pp": meta.get("benchmark_pp") or meta.get("pp"),
        "ep": meta.get("benchmark_ep") or meta.get("ep"),
        "num_layers": meta.get("num_hidden_layers"),
        "batch": meta.get("batch"),
        "input_len": meta.get("input_len"),
        "output_len": meta.get("output_len"),
    }


def recipe_from_bench_args(args: Any, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Canonical recipe from ``benchmark_vllm.py`` CLI args (stdlib-only path).

    Also returns nothing extra here — measurement-only knobs are passed as the
    ``extra`` dict to :func:`config_key` by the caller so the cache stays exact
    without polluting the regime/transport axes."""
    env = env or {}
    aiter_on = env.get("VLLM_ROCM_USE_AITER", "0") == "1" and not getattr(args, "no_aiter", False)
    ep = int(getattr(args, "tp", 1) or 1) if getattr(args, "enable_expert_parallel", False) else 1
    return {
        "model": getattr(args, "model", None),
        "weight_dtype": getattr(args, "quantization", None) or "bf16",
        "kv_cache_dtype": getattr(args, "kv_cache_dtype", None) or "bf16",
        "moe_expert_dtype": None,
        # A flag since vLLM 0.25; the env var it replaced is read as a fallback
        # so anchors measured under the older name still match.
        "attention_backend": (
            getattr(args, "attention_backend", None) or env.get("VLLM_ATTENTION_BACKEND")
        ),
        "cudagraph": "eager" if getattr(args, "enforce_eager", False) else "graph",
        "aiter": aiter_on,
        "aiter_ops": aiter_ops_axis(env),
        "speculative": speculative_axis(
            getattr(args, "speculative_method", None) or "",
            getattr(args, "speculative_num_tokens", None),
        ),
        "tp": getattr(args, "tp", 1),
        "pp": getattr(args, "pp", 1),
        "ep": ep,
        "num_layers": getattr(args, "num_hidden_layers", None),
        "batch": getattr(args, "batch", None),
        "input_len": getattr(args, "input_len", None),
        "output_len": getattr(args, "output_len", None),
    }


def recipe_from_inference_config(cfg: Any) -> Dict[str, Any]:
    """Canonical recipe from an ``InferenceConfig`` (the reconstruction target).

    Structural configs carry no HF model *name*, so ``model`` is left ``None``
    and matching is expected within a per-model anchor store (the common case:
    you harvest anchors for the model you are tuning).  ``num_layers`` is the
    full target depth (restore extrapolates the anchor's reduced depth to it)."""
    req = getattr(cfg, "request_config", None)
    mc = getattr(cfg, "model_config", None)
    mp = getattr(cfg, "model_parallel_config", None)

    def g(o, name, default=None):
        return getattr(o, name, default) if o is not None else default

    tp = int(g(mp, "tensor_model_parallel_size", 1) or 1)
    ep = int(g(mp, "expert_model_parallel_size", 1) or 1)
    return {
        "model": None,
        "weight_dtype": g(req, "weight_dtype", "bf16"),
        "kv_cache_dtype": g(req, "kv_cache_dtype", "bf16"),
        "moe_expert_dtype": g(req, "moe_expert_dtype"),
        "attention_backend": g(req, "attention_backend"),
        "cudagraph": _cudagraph_from_mode(g(req, "cudagraph_mode")),
        # Neither is represented on the config side; both are ignored in distance.
        "aiter": None,
        "aiter_ops": None,
        "speculative": speculative_axis(
            "spec" if g(req, "speculative_num_tokens") else "",
            g(req, "speculative_num_tokens"),
        ),
        "tp": tp,
        "pp": int(g(mp, "pipeline_model_parallel_size", 1) or 1),
        "ep": ep,
        "num_layers": g(mc, "num_layers"),
        "batch": g(req, "batch_size"),
        "input_len": g(req, "input_seq_len"),
        "output_len": g(req, "output_seq_len"),
        "concurrency": g(req, "max_concurrency"),
        "request_rate": g(req, "request_rate"),
    }


def _cudagraph_from_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    return "eager" if str(mode).lower() in ("none", "off", "eager") else "graph"
