###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""CLI launcher for ``inferasim projection inference``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from infera.projection.core.launcher.parser import load_config
from infera.projection.core.projection.training_config import (
    convert_config_to_inference_config,
)

from .memory import project_inference_memory
from .performance import InferencePerformanceProjector, project_inference_performance

# Map CLI arg attribute names → InferenceRequestConfig field names.
_ARG_TO_FIELD = {
    "input_len": "input_seq_len",
    "output_len": "output_seq_len",
    "inference_batch_size": "batch_size",
    "max_concurrency": "max_concurrency",
    "max_context_len": "max_context_len",
    "weight_dtype": "weight_dtype",
    "kv_cache_dtype": "kv_cache_dtype",
    "chunked_prefill_size": "chunked_prefill_size",
    "prefix_cache_hit_rate": "prefix_cache_hit_rate",
    "speculative_num_tokens": "speculative_num_tokens",
    "speculative_acceptance_rate": "speculative_acceptance_rate",
    "serving_model": "serving_model",
    "decode_step_overhead_us": "decode_step_overhead_us",
    "detokenize_overhead_us": "detokenize_overhead_us",
    "tokenize_overhead_us": "tokenize_overhead_us",
    "stream_interval": "stream_interval",
    "decode_admission_steps": "decode_admission_steps",
    "mixed_batch_penalty": "mixed_batch_penalty",
    "cudagraph_mode": "cudagraph_mode",
    "kv_cache_memory_fraction": "kv_cache_memory_fraction",
    "kv_block_size": "kv_block_size",
    "kv_offload_gb_per_gpu": "kv_offload_gb_per_gpu",
    "kv_offload_bw_gbps": "kv_offload_bw_gbps",
    "max_num_batched_tokens": "max_num_batched_tokens",
    "ep_load_balance": "ep_load_balance",
    "redundant_experts": "redundant_experts",
    "request_rate": "request_rate",
    "arrival_model": "arrival_model",
    "attention_backend": "attention_backend",
    "sparse_attention_topk": "sparse_attention_topk",
    "sliding_window": "sliding_window",
    "sliding_window_layer_fraction": "sliding_window_layer_fraction",
    "moe_expert_dtype": "moe_expert_dtype",
    "speculative_draft_cost_factor": "speculative_draft_cost_factor",
    "sampling_top_k": "sampling_top_k",
    "sampling_top_p": "sampling_top_p",
    "sampling_temperature": "sampling_temperature",
    "act_quant_dtype": "act_quant_dtype",
    "kernel_launch_latency_us": "kernel_launch_latency_us",
    "kernels_per_layer": "kernels_per_layer",
    "decode_kernel_occupancy_us": "decode_kernel_occupancy_us",
}

# Feature B (custom collective ops): CLI arg → ``collective_<field>`` key.
_COLL_ARG_TO_FIELD = {
    "tp_allreduce_algo": "collective_tp_allreduce_algo",
    "ep_a2a_algo": "collective_ep_a2a_algo",
    "prefill_comm_overlap": "collective_prefill_overlap",
    "decode_comm_overlap": "collective_decode_overlap",
    "tp_allreduce_efficiency": "collective_tp_allreduce_efficiency",
    "ep_a2a_efficiency": "collective_ep_a2a_efficiency",
}

# Feature A (prefill/decode disaggregation): CLI arg → ``disagg_<field>`` key.
_DISAGG_ARG_TO_FIELD = {
    "prefill_tp": "disagg_prefill_tp",
    "prefill_pp": "disagg_prefill_pp",
    "prefill_ep": "disagg_prefill_ep",
    "decode_tp": "disagg_decode_tp",
    "decode_pp": "disagg_decode_pp",
    "decode_ep": "disagg_decode_ep",
    "prefill_attention_dp": "disagg_prefill_attention_dp",
    "decode_attention_dp": "disagg_decode_attention_dp",
    "prefill_replicas": "disagg_prefill_replicas",
    "decode_replicas": "disagg_decode_replicas",
    "kv_transfer_bw_gbps": "disagg_kv_transfer_bw_gbps",
    "kv_transfer_latency_us": "disagg_kv_transfer_latency_us",
    "transfer_backend": "disagg_transfer_backend",
}

_GB = 1024.0 ** 3


def _collect_inference_overrides(args) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for mapping in (_ARG_TO_FIELD, _COLL_ARG_TO_FIELD, _DISAGG_ARG_TO_FIELD):
        for arg_name, field_name in mapping.items():
            if hasattr(args, arg_name):
                val = getattr(args, arg_name)
                if val is not None:
                    overrides[field_name] = val

    # Attention DP is a parallel axis, but a serving-only one, so it arrives on
    # an inference flag rather than in the trainer's parallelism block.
    attn_dp = getattr(args, "attention_dp_size", None)
    if attn_dp is not None:
        overrides["attention_data_parallel_size"] = int(attn_dp)
    # --comm-model {explicit,builtin} → collective_enabled.
    comm_model = getattr(args, "comm_model", None)
    if comm_model is not None:
        overrides["collective_enabled"] = comm_model != "builtin"
    # --disaggregate flag → disagg_enabled.
    if getattr(args, "disaggregate", False):
        overrides["disagg_enabled"] = True
    # store_true serving flags: only override when actually set, so they never
    # clobber a value carried by a YAML ``inference:`` block.
    if getattr(args, "fused_kernels", False):
        overrides["fused_kernels"] = True
    if getattr(args, "no_sampling", False):
        overrides["sampling_enabled"] = False
    if getattr(args, "quick_reduce", False):
        overrides["collective_quick_reduce"] = True
    if getattr(args, "fuse_rmsnorm_allreduce", False):
        overrides["collective_fuse_rmsnorm_allreduce"] = True
    return overrides


def _print_performance(inference_config, perf, gpu_cost_per_hour=None) -> None:
    req = inference_config.request_config
    mc = inference_config.model_config
    print("\n" + "=" * 100)
    print("[inferasim:Inference] Performance Projection")
    print("=" * 100)
    print(
        f"  Workload: input={req.input_seq_len} tok, output={req.output_seq_len} tok, "
        f"batch={req.batch_size}"
    )
    feats = []
    if req.chunked_prefill_size:
        feats.append(f"chunked_prefill={req.chunked_prefill_size}")
    if req.resolved_prefix_cache_hit_rate() > 0.0:
        feats.append(f"prefix_cache_hit={req.resolved_prefix_cache_hit_rate():.2f}")
    if req.speculative_num_tokens:
        feats.append(
            f"speculative(k={req.speculative_num_tokens}, "
            f"accept={req.speculative_acceptance_rate})"
        )
    if req.kv_cache_dtype != "bf16":
        feats.append(f"kv_dtype={req.kv_cache_dtype}")
    _win = req.resolved_sliding_window(getattr(mc, "sink_sliding_window", 0))
    if _win > 0:
        _frac = req.resolved_sliding_window_fraction(
            getattr(mc, "sink_window_even_layers_only", False)
        )
        feats.append(
            f"sliding_window={_win}" + (f"(frac={_frac:g})" if _frac < 1.0 else "")
        )
    if not getattr(req, "sampling_enabled", True):
        feats.append("sampling=off")
    elif getattr(req, "sampling_top_k", 0) or 0.0 < getattr(req, "sampling_top_p", 1.0) < 1.0:
        feats.append(
            "sampling("
            + (f"top_k={req.sampling_top_k}" if req.sampling_top_k else f"top_p={req.sampling_top_p}")
            + ")"
        )
    _adq = req.resolved_act_quant_dtype(getattr(mc, "fp8", None))
    if _adq:
        feats.append(f"act_quant={_adq}")
    if getattr(req, "kernel_launch_latency_us", 0.0):
        feats.append(f"launch_floor={req.kernel_launch_latency_us:g}us/kernel")
    if getattr(mc, "use_turbo_deepep", False):
        sync_free = getattr(mc, "turbo_sync_free_moe_stage", 0) or 0
        feats.append("deepep" + (f"(sync_free={sync_free})" if sync_free else ""))
    if req.cudagraph_mode:
        feats.append(f"cudagraph={req.cudagraph_mode}")
    if req.kv_cache_memory_fraction:
        feats.append(f"kv_mem_frac={req.kv_cache_memory_fraction:.2f}")
    if req.kv_block_size:
        feats.append(f"kv_block={req.kv_block_size}")
    if req.kv_offload_gb_per_gpu:
        feats.append(
            f"kv_offload={req.kv_offload_gb_per_gpu:g}GB@{req.kv_offload_bw_gbps:g}GB/s"
        )
    if req.max_num_batched_tokens:
        feats.append(f"max_batched_tokens={req.max_num_batched_tokens}")
    if req.ep_load_balance and req.ep_load_balance != 1.0:
        feats.append(f"ep_load_balance={req.ep_load_balance:.2f}")
    if req.redundant_experts:
        feats.append(f"redundant_experts={req.redundant_experts}")
    if getattr(req, "attention_backend", None):
        feats.append(f"attn_backend={req.attention_backend}")
    if getattr(req, "sparse_attention_topk", 0):
        feats.append(f"sparse_attn_topk={req.sparse_attention_topk}")
    if getattr(req, "moe_expert_dtype", None):
        feats.append(f"moe_expert_dtype={req.moe_expert_dtype}")
    if getattr(req, "fused_kernels", False):
        feats.append("fused_kernels")
    if getattr(req, "speculative_draft_cost_factor", 0.0):
        feats.append(f"draft_cost={req.speculative_draft_cost_factor:.2f}")
    if getattr(req, "request_rate", 0.0) and (req.arrival_model or "closed") != "closed":
        feats.append(f"request_rate={req.request_rate:g}/s({req.arrival_model})")
    cc = inference_config.collective_config
    if getattr(cc, "quick_reduce", False):
        feats.append("quick_reduce")
    if getattr(cc, "fuse_rmsnorm_allreduce", False):
        feats.append("fused_rmsnorm_ar")
    if feats:
        print(f"  Features: {', '.join(feats)}")
    if perf.is_disaggregated:
        print("  Mode: prefill/decode DISAGGREGATED")
    if perf.extras.get("serving_continuous_batching"):
        print(
            f"  Serving model: CONTINUOUS BATCHING "
            f"(concurrency={int(perf.extras.get('concurrency', req.batch_size))})"
        )
    elif not perf.is_disaggregated:
        print("  Serving model: STATIC (pure-decode batch)")
    src = "BENCHMARK (GPU-calibrated)" if perf.extras.get("benchmark_calibrated") else "SIMULATION"
    print(f"  Profiling source: {src}")
    if "sustainable_concurrency" in perf.extras:
        sc = int(perf.extras.get("sustainable_concurrency", 0) or 0)
        used = int(perf.extras.get("concurrency_used", 0) or 0)
        hbm = float(perf.extras.get("hbm_capacity_gb", 0.0) or 0.0)
        hbm_src = perf.extras.get("hbm_capacity_source", "")
        capped = bool(perf.extras.get("concurrency_capped", 0.0))
        sc_s = str(sc) if sc > 0 else "n/a (HBM/KV sizing unavailable)"
        print(f"  Max sustainable concurrency: {sc_s}  (HBM={hbm:.0f} GB via {hbm_src})")
        cap_note = "  [capped to sustainable max]" if capped else ""
        print(f"  Concurrency used: {used}{cap_note}")
    print("-" * 100)
    print(f"  TTFT (time to first token):      {perf.ttft_ms:.2f} ms")
    if perf.is_disaggregated:
        print(f"    prefill compute:               {perf.extras.get('prefill_compute_ttft_ms', 0.0):.2f} ms")
        print(f"    KV-cache transfer:             {perf.kv_transfer_ms:.2f} ms")
    print(f"  ITL / TPOT (per token):          {perf.itl_ms:.2f} ms")
    print(f"  Interactivity (per user):        {perf.per_request_decode_tps:.1f} tok/s/user")
    if perf.extras.get("serving_continuous_batching"):
        print(
            f"  Decode step latency (pure):      {perf.decode_step_latency_ms:.2f} ms"
            f"  | mixed: {perf.extras.get('mixed_step_latency_ms', 0.0):.2f} ms"
        )
        print(
            f"    Mixed-step fraction:           {perf.extras.get('mixed_step_fraction', 0.0) * 100:.2f}%"
            f"  → TPOT pollution: {perf.extras.get('tpot_pollution_pct', 0.0):.1f}%"
        )
    else:
        print(f"  Decode step latency (batch):     {perf.decode_step_latency_ms:.2f} ms")
    print(f"  End-to-end request latency:      {perf.request_latency_ms:.2f} ms")
    if "offered_request_rate" in perf.extras:
        sat = " [SATURATED]" if perf.extras.get("saturated") else ""
        print(
            f"  Offered load:                    {perf.extras['offered_request_rate']:g} req/s "
            f"(max sustainable {perf.extras.get('max_sustainable_request_rate', 0.0):.2f} req/s, "
            f"utilization {perf.extras.get('utilization', 0.0) * 100:.0f}%){sat}"
        )
        print(
            f"  Queue wait (excl. from TTFT):    {perf.extras.get('queue_wait_ms', 0.0):.2f} ms"
            f"   (TTFT+queue {perf.extras.get('ttft_with_queue_ms', 0.0):.2f} ms)"
        )
    print("-" * 100)
    print(f"  Per-request decode throughput:   {perf.per_request_decode_tps:.1f} tok/s")
    print(f"  Aggregate decode throughput:     {perf.decode_throughput_tps:.1f} tok/s")
    print(f"  Decode throughput / GPU:         {perf.decode_throughput_tps_per_gpu:.1f} tok/s/gpu")
    if perf.is_disaggregated:
        fleet_gpus = (perf.prefill_replica_gpus * int(perf.extras.get("prefill_replicas", 1))
                      + perf.decode_replica_gpus * int(perf.extras.get("decode_replicas", 1)))
    else:
        fleet_gpus = perf.replica_gpus
    # Billed tokens, not computed ones: a prompt served from the prefix cache is
    # still charged, so this is what a $/token figure divides into. Prefill and
    # decode GPUs are both in the denominator, which is what makes it comparable
    # across P:D ratios that a decode-only figure would rank identically.
    total_tps = perf.decode_throughput_tps * (
        req.input_seq_len + req.output_seq_len
    ) / max(1, req.output_seq_len)
    print(
        f"  Total throughput / GPU:          {total_tps / max(1, fleet_gpus):.1f} tok/s/gpu"
        f"   (in+out over {fleet_gpus} GPU)"
    )
    print(f"  Prefill throughput:              {perf.prefill_throughput_tps:.1f} tok/s")
    if perf.is_disaggregated:
        print(
            f"  Prefill pool:                    {int(perf.extras.get('prefill_replicas', 1))} "
            f"replica(s) x {perf.prefill_replica_gpus} GPU"
        )
        print(
            f"  Decode pool:                     {int(perf.extras.get('decode_replicas', 1))} "
            f"replica(s) x {perf.decode_replica_gpus} GPU"
        )
    else:
        print(f"  Replica GPUs (TP×PP):            {perf.replica_gpus}")
    if perf.extras.get("speculative_tokens_per_step", 1.0) > 1.0:
        print(
            f"  Speculative tokens / step:       "
            f"{perf.extras['speculative_tokens_per_step']:.2f}"
        )
    # Throughput priced in the unit a serving budget is quoted in. Prefill and
    # decode share the GPUs under continuous batching, so a token's cost is the
    # whole replica's cost divided by what the replica emits -- which is why a
    # recipe is charged for GPUs it needs but does not keep busy.
    if gpu_cost_per_hour:
        gpus = fleet_gpus
        out_tps = perf.decode_throughput_tps
        # The same requests carry their prompts, so the blended rate follows
        # from the workload's own input/output mix.
        all_tps = total_tps
        print("-" * 100)
        print(f"  {'Cost basis:':<33}${gpu_cost_per_hour:g}/GPU-h x {gpus} GPU")
        for label, tps in (("output", out_tps), ("in+out", all_tps)):
            cost = (gpu_cost_per_hour * gpus * 1e6 / (tps * 3600.0)) if tps > 0 else float("inf")
            # Three decimals: a competitive recipe lands in cents per million,
            # where two would print every one of them as the same number.
            print(f"  {f'Cost / 1M {label} tokens:':<33}${cost:,.3f}")

    # Feature B: explicit communication breakdown (exposed, post-overlap).
    if "comm_prefill_total_ms" in perf.extras:
        print("-" * 100)
        print("  Communication breakdown (exposed ms/forward):")
        print(
            f"    prefill:  TP-AR {perf.extras['comm_prefill_tp_allreduce_ms']:.2f} | "
            f"EP-A2A {perf.extras['comm_prefill_ep_a2a_ms']:.2f} | "
            f"PP-P2P {perf.extras['comm_prefill_pp_p2p_ms']:.2f} | "
            f"total {perf.extras['comm_prefill_total_ms']:.2f}"
        )
        print(
            f"    decode:   TP-AR {perf.extras['comm_decode_tp_allreduce_ms']:.2f} | "
            f"EP-A2A {perf.extras['comm_decode_ep_a2a_ms']:.2f} | "
            f"PP-P2P {perf.extras['comm_decode_pp_p2p_ms']:.2f} | "
            f"total {perf.extras['comm_decode_total_ms']:.2f}"
        )
    print("=" * 100)


def _scaling_bench_paths(args) -> list:
    """Extra benchmark artifacts for the TP-scaling fit (repeatable or comma-sep)."""
    raw = getattr(args, "load_benchmark_scaling", None) or []
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for item in raw:
        out.extend(p for p in str(item).split(",") if p)
    return out


def _emit_restore_confidence(anchor_paths, target_gpus: int) -> None:
    """Advisory: how far past its anchor is this target being extrapolated?

    Anchors are measured at ``min(tp, 4)`` and everything above is projected, so
    the only question left is reach: a flat anchor carries one doubling, and a
    target beyond that is extrapolated rather than restored. Climbing to another
    rung is not the fix -- scored against the measured ladder, a second anchor
    tied or lost against a single four-GPU anchor on every target.
    Best-effort — never raises into the projection.
    """
    try:
        import json as _json

        gpus = []
        for p in anchor_paths:
            if not p:
                continue
            try:
                meta = (_json.load(open(p)) or {}).get("meta", {})
            except Exception:
                continue
            gpus.append(int(meta.get("tp", 1) or 1) * int(meta.get("pp", 1) or 1))
        if not gpus or target_gpus <= 1:
            return
        best = max(gpus)
        rungs = ",".join(str(g) for g in sorted(set(gpus)))
        cap_on = os.getenv("INFERASIM_DECODE_ETP_CAP", "0").strip().lower() in ("1", "true", "yes")
        if best * 2 >= target_gpus:
            # The raw origami restore is trustworthy here, and the ETP cap would
            # over-correct an anchor this close to the target.
            note = (" (ETP cap is ON — consider INFERASIM_DECODE_ETP_CAP=0; it can "
                    "over-correct a near-target anchor)") if cap_on else ""
            print(
                f"[inferasim:Inference] restore confidence: HIGH — target {target_gpus} GPUs "
                f"is within one doubling of the {best}-GPU anchor (anchors: {rungs}).{note}"
            )
        else:
            stopgap = (
                " The ETP cap is currently masking this with a blunt, "
                "model-dependent correction; prefer reporting the extrapolation "
                "honestly over relying on it."
                if cap_on else
                " (ETP cap OFF: expect the raw decode over-projection.)"
            )
            print(
                f"[inferasim:Inference] restore confidence: LOW — target {target_gpus} GPUs is "
                f"more than one doubling above the {best}-GPU anchor (anchors: {rungs}). "
                f"Treat the number as extrapolated, not measured.{stopgap}"
            )
    except Exception:
        pass


def _print_des(des: Dict[str, object]) -> None:
    point = des["point"]
    print("\n" + "=" * 100)
    print("[inferasim:Inference] Discrete-Event Simulation (arrival-driven)")
    print("=" * 100)
    sat = " [SATURATED]" if point.saturated else ""
    print(
        f"  Arrivals: {point.arrival_model} @ {point.offered_rate:g} req/s offered  "
        f"(achieved {point.achieved_rate:.2f} req/s, "
        f"utilization {point.utilization * 100:.0f}%){sat}"
    )
    print(
        f"  Simulated: {point.num_requests} requests over {point.makespan_ms / 1000.0:.2f} s  "
        f"→ system throughput {point.system_throughput_tps:.0f} tok/s"
    )
    print("-" * 100)
    print(f"  {'metric':<22}{'mean':>12}{'p50':>12}{'p90':>12}{'p99':>12}")

    def _row(label: str, d: Dict[str, float], unit: str = "ms") -> None:
        print(
            f"  {label:<22}"
            f"{d.get('mean', 0.0):>10.2f} {unit:<1}"
            f"{d.get('p50', 0.0):>10.2f} {unit:<1}"
            f"{d.get('p90', 0.0):>10.2f} {unit:<1}"
            f"{d.get('p99', 0.0):>10.2f} {unit:<1}"
        )

    _row("TTFT (from admit)", point.ttft)
    if getattr(point, "queue_wait", None):
        _row("  queue wait", point.queue_wait)
        _row("  TTFT (from arrival)", point.ttft_arrival)
    _row("TPOT (per token)", point.tpot)
    _row("ITL (inter-token)", point.itl)
    _row("End-to-end latency", point.e2e)

    pfx = getattr(point, "prefix", None) or {}
    if pfx and pfx.get("num_instances", 0):
        _routes = ("round_robin", "random", "prefix_aware", "kv")
        ri = int(pfx.get("routing", -1))
        rname = _routes[ri] if 0 <= ri < len(_routes) else "?"
        trace_driven = bool(pfx.get("trace_driven", 0.0))
        source = (
            "mooncake trace"
            if trace_driven
            else f"prefix pool: {int(pfx.get('num_prefixes', 0))} prefixes"
        )
        print("-" * 100)
        print(
            f"  Fleet: {int(pfx['num_instances'])} instance(s), routing={rname} | {source}"
        )
        bs = int(pfx.get("block_size", 0))
        cap = int(pfx.get("cache_blocks", 0))
        cap_str = f"{cap} blocks" if cap > 0 else "unbounded"
        print(
            f"  KV block cache: block_size={bs} tok, capacity/instance={cap_str}, "
            f"evictions={int(pfx.get('evictions', 0))}, "
            f"block-reuse={pfx.get('block_hit_rate', 0.0) * 100:.1f}%"
        )
        print(
            f"  Prefix-cache hit rate: {pfx.get('hit_rate', 0.0) * 100:.1f}% of requests "
            f"(avg {pfx.get('avg_cached_tokens', 0.0):.0f} cached tok/req; "
            f"per-instance {pfx.get('min_inst_hit_rate', 0.0) * 100:.0f}–"
            f"{pfx.get('max_inst_hit_rate', 0.0) * 100:.0f}%)"
        )

    pk = getattr(point, "packing", None) or {}
    if pk:
        print("-" * 100)
        print(
            f"  Batch packing: avg batch {pk.get('avg_batch_size', 0.0):.1f} "
            f"(max {int(pk.get('max_batch_size', 0))}) | "
            f"avg prefill/decode reqs {pk.get('avg_prefill_reqs', 0.0):.1f}/"
            f"{pk.get('avg_decode_reqs', 0.0):.1f} | "
            f"avg query tokens/step {pk.get('avg_query_tokens', 0.0):.0f} | "
            f"mixed-step frac {pk.get('prefill_step_fraction', 0.0) * 100:.0f}%"
        )
        if pk.get("kv_peak_tokens", 0):
            util = pk.get("kv_utilization", 0.0)
            util_s = f" ({util * 100:.0f}% of pool)" if util else ""
            print(f"  KV peak: {int(pk['kv_peak_tokens'])} tokens{util_s}")

    curve = des.get("curve")
    if curve:
        mu = des.get("max_sustainable_rate", 0.0)
        print("-" * 100)
        print(f"  Throughput–latency curve (max sustainable ≈ {mu:.2f} req/s):")
        print(
            f"  {'load(req/s)':>12}{'util%':>8}{'tok/s':>10}"
            f"{'TTFT p50':>11}{'TTFT p99':>11}{'TPOT p50':>11}{'TPOT p99':>11}"
        )
        for r in curve:
            flag = "*" if r.saturated else " "
            print(
                f"  {r.offered_rate:>11.2f}{flag}{r.utilization * 100:>7.0f}"
                f"{r.system_throughput_tps:>10.0f}"
                f"{r.ttft.get('p50', 0.0):>11.1f}{r.ttft.get('p99', 0.0):>11.1f}"
                f"{r.tpot.get('p50', 0.0):>11.2f}{r.tpot.get('p99', 0.0):>11.2f}"
            )
    print("=" * 100)


def _anchor_model_filter(store):
    """Which model's anchors this projection may use, or None for any.

    A structural config carries no HF model name, so a store holding one model's
    anchors needs no filter. A store holding several does: without one, the
    nearest-anchor search would happily calibrate gpt-oss against DeepSeek's
    warmup, which is the kind of mistake that produces a confident wrong number.
    Resolved against INFERASIM_MODEL, whose preset spelling ("gpt_oss_120B") is
    matched loosely against the artifact's id ("openai/gpt-oss-120b").
    """
    from .search.regime import models_match

    models = {e.get("model") for e in store.entries() if e.get("model")}
    if len(models) <= 1:
        return None
    preset = os.environ.get("INFERASIM_MODEL")
    if preset:
        hits = [m for m in models if models_match(preset, m)]
        if len(hits) == 1:
            return hits[0]
    raise ValueError(
        f"store holds anchors for {len(models)} models {sorted(models)} and the "
        "target model could not be identified; set INFERASIM_MODEL or use a "
        "per-model store"
    )


def _anchor_from_store(args, inference_config):
    """Find a warmup measurement this projection can be calibrated against.

    The point of a warmup store is that a deployment is measured once and then
    reused, rather than re-measured for every config a search wants to score.
    What makes that safe is the regime: anchors are only interchangeable with
    targets that run the same kernels, so an anchor at a different dtype,
    attention backend, cudagraph mode or speculation setting is reported and
    refused rather than silently applied.

    Returns the anchor path to load, or None to project analytically.
    """
    root = getattr(args, "anchor_store", None) or os.environ.get(
        "INFERASIM_ANCHOR_STORE"
    )
    if not root:
        return None
    if not os.path.isdir(root):
        print(f"[inferasim:Inference] anchor store '{root}' does not exist — "
              "projecting analytically.")
        return None

    try:
        from .search.anchor_store import AnchorStore
        from .search.regime import recipe_from_inference_config

        store = AnchorStore(root)
        recipe = recipe_from_inference_config(inference_config)
        model = _anchor_model_filter(store)
        entry, distance = store.nearest(recipe, model=model)
    except Exception as exc:  # noqa: BLE001 - a broken store must not fail a projection
        print(f"[inferasim:Inference] anchor store unusable ({exc}) — "
              "projecting analytically.")
        return None

    if entry is None:
        print(f"[inferasim:Inference] no anchor in {root} for this model — "
              "projecting analytically.")
        return None
    if distance:
        print(f"[inferasim:Inference] nearest anchor differs on {distance} regime "
              f"axis(es) ({entry.get('path')}); it describes different kernels, so "
              "projecting analytically instead. Warm up in this regime to calibrate.")
        return None

    path = entry.get("path")
    tr = entry.get("transport") or {}
    print(f"[inferasim:Inference] calibrating from warmup anchor {path} "
          f"(same regime, measured at TP={tr.get('tp')} EP={tr.get('ep')} "
          f"PP={tr.get('pp')})")
    return path


def launch_projection_from_cli(args, overrides):
    """Entry point for ``inferasim projection inference``."""
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[inferasim:Inference] Config file '{cfg_path}' not found.")

    config, _unknown = load_config(args, overrides or [])

    inf_overrides = _collect_inference_overrides(args)
    inference_config = convert_config_to_inference_config(
        config, inference_overrides=inf_overrides
    )

    # Serving weight precision drives the *compute* GEMM dtype, not just the
    # memory report. The layer profilers pick the GEMM dtype from
    # ``model_config.fp8``, so an explicit ``--weight-dtype`` must be reflected
    # there; otherwise projecting an FP8-trained model in BF16 (or forcing FP8
    # on a BF16 model) silently kept the training precision for compute. Only
    # applied when the flag is explicitly set (default ``None``) so omitting it
    # preserves the model's native precision.
    # How unevenly the router spreads tokens over experts. It belongs to the
    # model's trained router, but stays overridable so a measured imbalance can
    # be supplied per run.
    _skew = getattr(args, "moe_routing_skew", None)
    if _skew is not None:
        inference_config.model_config.moe_routing_skew = float(_skew)

    _cov = getattr(args, "moe_router_coverage", None)
    if _cov:
        import json as _json

        with open(_cov) as _f:
            _blob = _json.load(_f)
        _curve = _blob.get("coverage_vs_uniform", _blob)
        inference_config.model_config.moe_router_coverage = {
            int(k): float(v) for k, v in _curve.items()
        }
        print(f"[inferasim:Inference] measured router coverage from {_cov} "
              f"({len(_curve)} batch points, "
              f"min {min(float(v) for v in _curve.values()):.2f}x independent)")

    explicit_wdt = getattr(args, "weight_dtype", None)
    if explicit_wdt is not None:
        wdt = str(explicit_wdt).lower()
        if wdt.startswith("fp8") or wdt in ("e4m3", "e5m2"):
            inference_config.model_config.fp8 = "hybrid"
        elif wdt in ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32"):
            inference_config.model_config.fp8 = None

    # Origami (simulate) calibration defaults. The analytical GEMM/attention
    # model over-predicts small-batch MoE decode by several x when it is left at
    # its bf16 + Triton defaults, because real serving runs low-precision expert
    # kernels (mxfp4/fp8) on the AITER backend. Reflect the serving stack so the
    # simulate ratio lands closer to measured, WITHOUT overriding an explicit
    # user choice (both remain no-ops when already set).
    _rc = inference_config.request_config
    if getattr(_rc, "moe_expert_dtype", None) is None:
        # The expert grouped-GEMM runs at the weight precision. mxfp4-weighted
        # models (e.g. gpt-oss) are served as mxfp4; fp8 weights → fp8 experts.
        _wdt = str(explicit_wdt or "").lower()
        if _wdt in ("mxfp4", "fp4"):
            _rc.moe_expert_dtype = "mxfp4"
        elif _wdt.startswith("fp8") or _wdt in ("e4m3", "e5m2"):
            _rc.moe_expert_dtype = "fp8"
    if getattr(_rc, "attention_backend", None) is None:
        _arch = str(getattr(args, "gpu_arch", "") or "").lower()
        if _arch.startswith("mi") or _arch.startswith("gfx") or "rocm" in _arch:
            _rc.attention_backend = "aiter"

    # MoE simulate kernel: vLLM serving uses a fused-MoE decode kernel (single
    # batched, weight-bandwidth-bound op), NOT the Megatron per-expert grouped
    # GEMM. Default the inference simulate path to ``vllm_fused`` so the origami
    # MoE cost model matches the serving stack (no-op if explicitly set).
    if getattr(inference_config.model_config, "moe_sim_kernel", None) is None:
        inference_config.model_config.moe_sim_kernel = "vllm_fused"

    # DeepEP / SyncFree (shared perf flags) — enable async EP All-to-All overlap
    # for the serving projection, mirroring the training projection override.
    if getattr(args, "enable_deepep", False):
        inference_config.model_config.use_turbo_deepep = True
    sync_free_stage = getattr(args, "sync_free_stage", 0) or 0
    if sync_free_stage > 0:
        inference_config.model_config.turbo_sync_free_moe_stage = sync_free_stage
        inference_config.model_config.use_turbo_deepep = True

    mode = getattr(args, "inference_mode", "both") or "both"
    hbm_gb = getattr(args, "hbm_capacity_gb", None)
    profiling_mode = getattr(args, "profiling_mode", "simulate") or "simulate"

    # Benchmark mode: measure this recipe on real GPUs, then calibrate the
    # analytical projection to what was measured.
    benchmark_layer_times = None
    scaling_benchmarks: list = []
    load_bench = getattr(args, "load_benchmark", None)
    if not load_bench and mode in ("performance", "both"):
        load_bench = _anchor_from_store(args, inference_config)
    if load_bench and mode in ("performance", "both"):
        # Reuse a previously-saved GPU layer benchmark (skips the spawn). Lets a
        # concurrency sweep calibrate against one bench run.
        import json as _json

        with open(load_bench) as _f:
            benchmark_layer_times = _json.load(_f)
        print(f"[inferasim:Inference] loaded GPU benchmark from {load_bench}")
        for _p in _scaling_bench_paths(args):
            with open(_p) as _f:
                scaling_benchmarks.append(_json.load(_f))
            print(f"[inferasim:Inference] loaded TP-scaling benchmark from {_p}")

    # Decode latency floor: a sharded probe's measured decode curve caps the
    # restored decode step from below (decode = max(restored, floor)). Above the
    # roofline knee the step is fixed by per-step launch/dispatch overhead and is
    # ~parallelism-invariant, so one probe supplies the floor for all more-
    # sharded targets. Parsed into {batch: decode_ms}.
    decode_floor = None
    floor_bench = getattr(args, "decode_floor_benchmark", None)
    if floor_bench and mode in ("performance", "both"):
        import json as _json

        with open(floor_bench) as _f:
            _blob = _json.load(_f)
        decode_floor = {
            int(e["batch"]): float(e["decode_ms"])
            for e in _blob.get("sweep", [])
            if e.get("decode_ms")
        }
        if decode_floor:
            print(
                f"[inferasim:Inference] loaded decode latency floor from {floor_bench} "
                f"({len(decode_floor)} batch points, "
                f"min {min(decode_floor.values()):.2f} ms)"
            )
        else:
            decode_floor = None
    elif profiling_mode == "benchmark" and mode in ("performance", "both"):
        # Measure this recipe on GPUs now. Asking for a benchmark and quietly
        # receiving a simulation would be the worst of both, so failures here
        # are raised rather than absorbed; --profiling-mode simulate is how you
        # ask for the projection that needs no GPU.
        from .benchmark import spawn_inference_benchmark

        benchmark_layer_times = spawn_inference_benchmark(args, inference_config)

    # Default multi-anchor policy ("TP=1 + TP=2 scaling"): when several
    # whole-model benchmarks are loaded, use the one measured AT the target TP
    # as the primary (its step latency is then reproduced exactly) and keep the
    # others only to fit the TP-scaling law for *unmeasured* target TPs. Without
    # this, projecting to a TP we actually measured would needlessly restore it
    # from a different TP through the analytical comm model and lose accuracy.
    if benchmark_layer_times and scaling_benchmarks and mode in ("performance", "both"):
        tgt_tp = int(inference_config.model_parallel_config.tensor_model_parallel_size)

        def _btp(blob):
            m = blob.get("meta", {}) if isinstance(blob, dict) else {}
            return int(m.get("benchmark_tp") or m.get("tp") or 1)

        if _btp(benchmark_layer_times) != tgt_tp:
            for _i, _sb in enumerate(scaling_benchmarks):
                if _btp(_sb) == tgt_tp:
                    scaling_benchmarks[_i] = benchmark_layer_times
                    benchmark_layer_times = _sb
                    print(
                        f"[inferasim:Inference] using the TP={tgt_tp} benchmark as the "
                        f"primary anchor (exact); other anchors fit the TP-scaling law."
                    )
                    break

    # Advisory: flag when the target is being restored from an out-of-regime
    # anchor (per-GPU decode throughput not yet flat within one doubling of the
    # target) and recommend the GPU count to benchmark next.
    if mode in ("performance", "both"):
        _tgt_pp = int(getattr(inference_config.model_parallel_config, "pipeline_model_parallel_size", 1) or 1)
        _tgt_tp = int(inference_config.model_parallel_config.tensor_model_parallel_size)
        _anchor_paths = ([load_bench] if load_bench else []) + _scaling_bench_paths(args)
        if getattr(args, "decode_floor_benchmark", None):
            _anchor_paths.append(args.decode_floor_benchmark)
        _emit_restore_confidence(_anchor_paths, _tgt_tp * _tgt_pp)

    # The fully-merged config, so programmatic callers can see what the model
    # preset, module defaults and CLI overrides actually resolved to rather
    # than re-deriving it.
    results = {"config": inference_config}
    if mode in ("memory", "both"):
        results["memory"] = project_inference_memory(
            inference_config, hbm_capacity_gb=hbm_gb, verbose=True
        )
    if mode in ("performance", "both"):
        projector = InferencePerformanceProjector(
            inference_config, args=args, benchmark_layer_times=benchmark_layer_times,
            scaling_benchmarks=scaling_benchmarks, decode_floor=decode_floor,
        )
        perf = projector.project()
        _print_performance(inference_config, perf,
                           getattr(args, "gpu_cost_per_hour", None))
        results["performance"] = perf

        # Phase 3: opt-in discrete-event simulation for arrival-driven
        # percentiles. Reuses ``projector`` as the (possibly benchmark-
        # calibrated) cost kernel for each step's duration.
        req = inference_config.request_config
        arrival_model = (getattr(req, "arrival_model", "closed") or "closed").lower()
        workload_file = getattr(args, "des_workload_file", None)
        dump_steps = getattr(args, "des_dump_steps", None)
        mooncake_trace = getattr(args, "des_mooncake_trace", None)
        run_des_enabled = (
            arrival_model in ("poisson", "deterministic") and (req.request_rate or 0) > 0
        ) or bool(workload_file) or bool(mooncake_trace)
        if run_des_enabled:
            from .des import run_des

            des = run_des(
                inference_config,
                projector,
                arrival_model=arrival_model if arrival_model in ("poisson", "deterministic") else "poisson",
                rate_per_s=float(req.request_rate or 0.0),
                num_requests=int(getattr(args, "des_num_requests", 400) or 400),
                seed=int(getattr(args, "des_seed", 0) or 0),
                sweep=bool(getattr(args, "des_sweep", False)),
                burstiness=float(getattr(args, "des_burstiness", 1.0) or 1.0),
                range_ratio=float(getattr(args, "des_range_ratio", 1.0) or 1.0),
                kv_cache_tokens=int(getattr(args, "des_kv_cache_tokens", 0) or 0),
                workload_file=workload_file,
                record_steps=bool(dump_steps),
                num_instances=int(getattr(args, "des_instances", 1) or 1),
                routing=(getattr(args, "des_routing", "round_robin") or "round_robin"),
                overlap_weight=float(getattr(args, "des_overlap_weight", None) or 1.0),
                num_prefixes=int(getattr(args, "des_num_prefixes", 0) or 0),
                prefix_len=int(getattr(args, "des_prefix_len", 0) or 0),
                prefix_zipf=float(getattr(args, "des_prefix_zipf", 0.0) or 0.0),
                cache_slots=int(getattr(args, "des_cache_slots", 0) or 0),
                block_size=int(getattr(args, "des_block_size", 0) or 0),
                cache_blocks=int(getattr(args, "des_kv_blocks", 0) or 0),
                mooncake_trace=mooncake_trace,
            )
            _print_des(des)
            results["des"] = des

            if dump_steps and des["point"].steps is not None:
                import json as _json

                payload = {
                    "config": {
                        "input_len": req.input_seq_len,
                        "output_len": req.output_seq_len,
                        "max_concurrency": req.resolved_max_concurrency(),
                        "max_num_batched_tokens": req.max_num_batched_tokens,
                        "chunked_prefill_size": req.chunked_prefill_size,
                        "request_rate": req.request_rate,
                        "arrival_model": arrival_model,
                        "burstiness": float(getattr(args, "des_burstiness", 1.0) or 1.0),
                        "range_ratio": float(getattr(args, "des_range_ratio", 1.0) or 1.0),
                        "kv_cache_tokens": int(getattr(args, "des_kv_cache_tokens", 0) or 0),
                    },
                    "packing": des["point"].packing,
                    "steps": des["point"].steps,
                }
                with open(dump_steps, "w") as _f:
                    _json.dump(payload, _f)
                print(f"[inferasim:Inference] wrote {len(des['point'].steps)} DES step records to {dump_steps}")

    return results
