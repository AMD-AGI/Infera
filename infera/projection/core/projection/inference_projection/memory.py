###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Inference memory projection.

Per-rank HBM for serving is dominated by two terms that training does *not*
share:

  * **Weights only** — no gradients, no fp32 master copy, no Adam moments.
  * **KV cache** — grows with resident concurrency × context length.

plus a (comparatively small) **forward activation working set** for the
in-flight batch.  We also report how many concurrent sequences fit in the
remaining HBM, which is the headline capacity number for a serving config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from infera.projection.core.projection.module_profilers.language_model import (
    build_profiler,
    get_language_model_profiler_spec,
)
from infera.projection.core.projection.training_config import InferenceConfig, dtype_num_bytes

from .kv_cache import (
    KVCacheBreakdown,
    attention_dp_size,
    estimate_kv_cache,
    max_concurrent_sequences,
)


@dataclass
class InferenceMemoryResult:
    rank: int
    num_params: int
    weight_bytes: int
    kv_cache_bytes: int
    activation_bytes: int
    total_bytes: int
    kv: KVCacheBreakdown
    layers_on_rank: int
    hbm_capacity_bytes: int | None = None
    max_concurrent_sequences: int | None = None
    fits: bool | None = None


_GB = 1024.0**3


def _layers_on_rank(inference_config: InferenceConfig) -> int:
    mc = inference_config.model_config
    pp = max(1, inference_config.model_parallel_config.pipeline_model_parallel_size)
    return max(1, (mc.num_layers + pp - 1) // pp)


def _forward_activation_bytes(inference_config: InferenceConfig, profiler) -> int:
    """A bounded estimate of the transient forward activation working set.

    Inference keeps no activations for backward, so only a few in-flight
    layers' worth of hidden states exist at once.  We proxy this with one
    transformer layer's activation footprint at the prefill token count,
    scaled by a small number of in-flight layers.
    """
    req = inference_config.request_config
    batch = max(1, req.batch_size)
    # The working set is one scheduler step's worth of tokens. The engine's
    # prefill budget (vLLM --max-num-batched-tokens / SGLang
    # --chunked-prefill-size) caps that step across the WHOLE batch, not per
    # sequence, so it bounds the product rather than one sequence's prompt.
    tokens = batch * req.input_seq_len
    budget = int(req.max_num_batched_tokens or 0) or int(req.chunked_prefill_size or 0)
    if budget > 0:
        tokens = min(tokens, budget)

    layer = profiler.sub_profilers.get("dense_transformer_layer")
    moe_layer = profiler.sub_profilers.get("moe_transformer_layer")
    chosen = layer
    # Prefer whichever layer type actually exists in the model.
    if (inference_config.model_config.num_experts or 0) and moe_layer is not None:
        chosen = moe_layer
    if chosen is None:
        return 0
    per_layer = chosen.estimated_activation_memory(1, tokens)
    inflight_layers = min(_layers_on_rank(inference_config), 2)
    return int(per_layer * inflight_layers)


def project_inference_memory(
    inference_config: InferenceConfig,
    *,
    rank: int | None = None,
    hbm_capacity_gb: float | None = None,
    verbose: bool = True,
) -> InferenceMemoryResult:
    eff_rank = int(os.getenv("RANK", "0")) if rank is None else int(rank)

    # A disaggregated deployment's memory ceiling is the decode pool's -- that is
    # where the KV cache lives and what caps concurrency. Everything below reads
    # parallelism off ``model_parallel_config``, so rebinding it once here points
    # the whole projection at the pool that actually holds the cache. Reporting
    # the global parallelism instead described a worker nobody runs, and once
    # per-pool TP existed without a matching global TP it reported a 744B model
    # as fitting zero sequences.
    #
    # The resident batch is split the same way the decode projector splits it:
    # each decode replica holds ``C / decode_replicas`` sequences, not the
    # system-wide concurrency. Sizing KV at the full batch on every replica
    # double-counted the cache by the replica count and rejected the 2-decode
    # MiniMax GB300 winner at 4096 in-flight -- 7 GB over 288 -- while silicon
    # ran it. Activations follow the same local batch.
    disagg = getattr(inference_config, "disaggregation_config", None)
    if disagg is not None and disagg.enabled:
        reps = max(1, int(disagg.decode_replicas or 1))
        local = max(1, int(inference_config.request_config.resolved_max_concurrency()) // reps)
        inference_config = replace(
            inference_config,
            model_parallel_config=disagg.decode_parallel(inference_config.model_parallel_config),
            request_config=replace(
                inference_config.request_config,
                batch_size=local,
                max_concurrency=local,
            ),
        )

    view = inference_config.as_training_config(
        batch_size=inference_config.request_config.batch_size,
        seq_len=inference_config.request_config.input_seq_len,
    )
    # ``rank`` selects this rank's pipeline stage AND applies the profiler's
    # expert-parallel split, which is right for training, where EP is a separate
    # axis of GPUs from TP. A serving engine places experts on the same GPUs as
    # the tensor shards (vLLM sets EP = TP), so a rank holds 1/TP of every
    # weight class -- experts by expert assignment, everything else by tensor
    # sharding. Counting the expert split here as well would shard experts
    # twice and report a fraction of the weights a rank really loads.
    view.model_parallel_config = replace(view.model_parallel_config, expert_model_parallel_size=1)
    profiler = build_profiler(get_language_model_profiler_spec(view))

    # With experts left whole above, ``estimated_num_params`` counts this rank's
    # pipeline stage at full width and the TP divide below is the only split
    # applied. Norms and the router are replicated rather than sharded, and are
    # well under a percent of the total.
    tp = max(1, inference_config.model_parallel_config.tensor_model_parallel_size)
    num_params = profiler.estimated_num_params(rank=eff_rank) // tp
    weight_bytes = _weight_bytes(inference_config, profiler, num_params, tp, eff_rank)

    layers_on_rank = _layers_on_rank(inference_config)
    kv = estimate_kv_cache(inference_config, layers_on_rank)
    activation_bytes = _forward_activation_bytes(inference_config, profiler)

    total = weight_bytes + int(kv.bytes_total) + activation_bytes

    hbm_bytes = int(hbm_capacity_gb * _GB) if hbm_capacity_gb else None
    max_conc = None
    fits = None
    if hbm_bytes is not None:
        # Serving engines only hand a fraction of HBM to the runtime (vLLM
        # gpu_memory_utilization / SGLang mem_fraction_static); the rest is
        # reserved for the driver/CUDA context and fragmentation headroom.
        fraction = inference_config.request_config.kv_cache_memory_fraction
        usable_bytes = int(hbm_bytes * float(fraction)) if fraction else hbm_bytes
        free_for_kv = usable_bytes - weight_bytes - activation_bytes
        # Blocks spilled to the host tier still hold a live session's context, so
        # they raise how many sessions a replica can keep resident even though
        # the actively-decoding batch stays in HBM. This is why the measured
        # agentic configs all run "dram" offload: agent sessions are idle most of
        # the time, and idle KV does not need HBM bandwidth.
        free_for_kv += inference_config.request_config.kv_offload_gb_per_gpu * _GB
        max_conc = max_concurrent_sequences(inference_config, layers_on_rank, free_for_kv)
        fits = total <= usable_bytes

    result = InferenceMemoryResult(
        rank=eff_rank,
        num_params=int(num_params),
        weight_bytes=weight_bytes,
        kv_cache_bytes=int(kv.bytes_total),
        activation_bytes=activation_bytes,
        total_bytes=total,
        kv=kv,
        layers_on_rank=layers_on_rank,
        hbm_capacity_bytes=hbm_bytes,
        max_concurrent_sequences=max_conc,
        fits=fits,
    )

    if verbose:
        _print_memory(inference_config, result)
    return result


def _routed_expert_params(model_config, layers) -> tuple[int, int]:
    """Routed-expert parameters per MoE layer, and the MoE layers among ``layers``."""
    num_experts = int(getattr(model_config, "num_experts", 0) or 0)
    pattern = getattr(model_config, "moe_pattern", None) or []
    if num_experts <= 1 or not pattern:
        return 0, 0
    ffn = model_config.moe_ffn_hidden_size or model_config.ffn_hidden_size
    proj = 3 if model_config.swiglu else 2
    n_moe = sum(1 for layer in layers if layer < len(pattern) and pattern[layer])
    return num_experts * proj * model_config.hidden_size * ffn, n_moe


def _attention_params(inference_config: InferenceConfig, profiler, rank: int) -> int:
    """Attention parameters across this rank's layers, unsharded."""
    pattern = inference_config.model_config.moe_pattern or []
    total = 0
    for layer in profiler.layers:
        kind = (
            "moe_transformer_layer"
            if layer < len(pattern) and pattern[layer]
            else "dense_transformer_layer"
        )
        attn = getattr(profiler.sub_profilers.get(kind), "sub_profilers", {}).get("self_attention")
        if attn is not None:
            total += attn.estimated_num_params(rank)
    return total


_V4_TP_REPLICATED = frozenset({"attn_q_a", "attn_kv"})


def _v4_attention_params(inference_config: InferenceConfig, layers) -> tuple[int, int] | None:
    """DeepSeek-V4 attention over ``layers`` as ``(TP-sharded, TP-replicated)`` parameters.

    The profiler's attention count has no V4 branch and prices V4's one 512-wide
    KV head and grouped output LoRA as dense 128-head Q/K/V/O -- 0.95B a layer
    against the checkpoint's ~0.31B. The GEMM shapes the performance model runs
    are V4's real ones, so the weights are read off them: the Q down-projection,
    the shared KV head, compressors, indexer and mHC mixers stay whole on every
    TP rank; the Q up-projection and both output LoRA halves are split.
    """
    from . import gemm_shapes as gs
    from . import layer_plan as lp

    mc = inference_config.model_config
    layers = list(layers)
    if not gs.is_v4(mc) or not layers:
        return None
    mp1 = replace(inference_config.model_parallel_config, tensor_model_parallel_size=1)
    c = lp.counts(
        inference_config,
        range(min(layers), max(layers) + 1),
        first=0 in layers,
        last=int(mc.num_layers) - 1 in layers,
    )
    sharded = replicated = 0
    for g in gs._attention(mc, mp1, 1, "bf16"):
        n = g.groups * g.n * g.k * c.softmax
        if g.op in _V4_TP_REPLICATED:
            replicated += n
        else:
            sharded += n
    for g, calls in gs._v4_extras(mc, mp1, 1, "bf16", c):
        replicated += g.groups * g.n * g.k * calls
    return sharded, replicated


def _attention_dp(inference_config: InferenceConfig, tp: int) -> int:
    mp = getattr(inference_config, "model_parallel_config", None)
    return min(tp, max(1, int(getattr(mp, "attention_data_parallel_size", 1) or 1)))


def _non_expert_params(
    inference_config: InferenceConfig, profiler, rest: int, tp: int, rank: int
) -> int:
    """This rank's non-routed-expert parameters, given ``rest`` = their 1/tp share.

    Two corrections to an even 1/tp split. V4 attention is re-counted from its
    GEMM shapes, with its TP-replicated part held whole (see
    :func:`_v4_attention_params`). And under attention-DP of size ``dp`` a rank
    shards what it would otherwise split ``tp`` ways only ``tp / dp`` ways. What
    that covers depends on the engine: a vLLM (or ATOM) DP rank is a whole
    engine, so everything but the routed experts -- attention, embedding, LM
    head, dense and shared-expert MLPs -- goes to ``tp / dp``; SGLang's
    ``--enable-dp-attention`` replicates the attention block and keeps the rest
    tensor-parallel across all ``tp`` ranks. Measured on DeepSeek-V4-Pro / vLLM /
    MI325X: tp4 -> tp4ep4dp4 adds 18.5 GiB a rank and tp8ep8 -> tp8ep8dp8 adds
    21.6, both 1 - 1/dp of the same ~25 GiB non-expert block.
    """
    dp = _attention_dp(inference_config, tp)
    v4 = _v4_attention_params(inference_config, profiler.layers)
    if v4 is None and dp <= 1:
        return rest
    attn_sharded = _attention_params(inference_config, profiler, rank)
    tp_replicated = 0
    if v4 is not None:
        rest = max(0, rest - attn_sharded // tp)
        attn_sharded, tp_replicated = v4
        rest += attn_sharded // tp
    if dp > 1:
        engine = str(getattr(inference_config.request_config, "serving_engine", "") or "").lower()
        rest += (attn_sharded // tp if "sglang" in engine else rest) * (dp - 1)
    return rest + tp_replicated


def _weight_bytes(
    inference_config: InferenceConfig, profiler, num_params: int, tp: int, rank: int = 0
) -> int:
    """This rank's weight bytes, with experts and the rest at their own widths.

    ``moe_expert_dtype`` and ``linear_weight_dtype`` fall back to
    ``weight_dtype``, so a single-precision checkpoint costs ``num_params``
    times one width. A mixed one (mxfp4 experts, bf16 attention) does not.
    """
    req = inference_config.request_config
    mc = inference_config.model_config
    base = req.weight_dtype
    expert_dt = getattr(req, "moe_expert_dtype", None) or base
    linear_dt = getattr(req, "linear_weight_dtype", None) or base
    per_layer, n_moe = _routed_expert_params(mc, profiler.layers)
    if per_layer == 0 or n_moe == 0:
        params = _non_expert_params(inference_config, profiler, num_params, tp, rank)
        return int(params * dtype_num_bytes(base))
    n_unq = min(n_moe, int(getattr(mc, "moe_unquantized_expert_layers", 0) or 0))
    quantized = per_layer * (n_moe - n_unq) // tp
    unquantized = per_layer * n_unq // tp
    rest = _non_expert_params(
        inference_config, profiler, max(0, num_params - quantized - unquantized), tp, rank
    )
    return int(
        quantized * dtype_num_bytes(expert_dt) + (unquantized + rest) * dtype_num_bytes(linear_dt)
    )


def _print_memory(inference_config: InferenceConfig, r: InferenceMemoryResult) -> None:
    req = inference_config.request_config
    print("\n" + "=" * 100)
    print(f"[inferasim:Inference] Memory Projection (Rank {r.rank})")
    print("=" * 100)
    print(f"  Params (this rank):       {r.num_params / 1e9:.4f} B")
    print(f"  Weights ({req.weight_dtype}):           {r.weight_bytes / _GB:.4f} GB")
    print(
        f"  KV cache ({req.kv_cache_dtype}):         {r.kv_cache_bytes / _GB:.4f} GB "
        f"(concurrency={r.kv.concurrency}, ctx={r.kv.max_context_len}, "
        f"layers/rank={r.layers_on_rank})"
    )
    print(f"    KV per sequence:        {r.kv.bytes_per_sequence / _GB:.4f} GB")
    attn_dp = attention_dp_size(inference_config)
    if attn_dp > 1:
        print(
            f"    Attention DP:           {attn_dp} "
            f"({r.kv.sequences_on_rank} sequences on this rank)"
        )
    print(f"  Activation working set:   {r.activation_bytes / _GB:.4f} GB")
    print(f"  Projected Total Memory:   {r.total_bytes / _GB:.4f} GB")
    if r.hbm_capacity_bytes is not None:
        print(f"  HBM capacity:             {r.hbm_capacity_bytes / _GB:.4f} GB")
        if req.kv_cache_memory_fraction:
            usable = r.hbm_capacity_bytes * float(req.kv_cache_memory_fraction)
            print(f"  Usable HBM (frac={req.kv_cache_memory_fraction:.2f}): {usable / _GB:.4f} GB")
        print(f"  Fits:                     {r.fits}")
        if req.kv_offload_gb_per_gpu:
            print(
                f"  Host KV offload:          {req.kv_offload_gb_per_gpu:.1f} GB/GPU "
                f"@ {req.kv_offload_bw_gbps:.0f} GB/s"
            )
        print(f"  Max concurrent sequences: {r.max_concurrent_sequences}")
    print("=" * 100)
