###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Inference performance projection: prefill + autoregressive decode.

This reuses the existing analytical profiler tree (in *simulation* mode, so
no GPU is required) to estimate **forward-only** per-component latency, then
composes those into the two serving phases:

  * **Prefill** — process the whole prompt (optionally in chunks) to produce
    the first token.  Drives **TTFT** (time-to-first-token).
  * **Decode** — generate ``output_seq_len`` tokens autoregressively, each
    step attending to a growing KV cache.  Drives **ITL / TPOT** and decode
    throughput.

Serving features modelled here: chunked prefill, KV-cache quantization
(via the SDPA/KV dtype), batching / concurrency, and speculative decoding.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace

from infera.projection.core.projection.module_profilers.language_model import (
    build_profiler,
    get_language_model_profiler_spec,
)
from infera.projection.core.projection.module_profilers.quantization import QuantCastProfiler
from infera.projection.core.projection.module_profilers.sampling import SamplingProfiler
from infera.projection.core.projection.module_profilers.transformer_layer import (
    _dense_tp_allreduce_count,
    _estimate_moe_a2a_time_ms,
    _estimate_tp_allreduce_time_ms,
    _moe_tp_allreduce_count,
)
from infera.projection.core.projection.simulation_backends.factory import (
    get_gemm_simulation_backend,
    get_sdpa_simulation_backend,
)
from infera.projection.core.projection.training_config import (
    INTERCONNECT_PROFILES,
    InferenceConfig,
    resolve_decode_occupancy_us,
)

from .collectives import (
    CommBreakdown,
    InferenceCollectiveModel,
    deepep_overlap_efficiency,
)


def _usable_packed_probe(packed, seq_rate_ms_per_tok=0.0, seq_cost_at=None):
    """A packed-prefill probe block, or None if it is not a measurement.

    Checked on read rather than trusted, because artifacts already on disk
    carry blocks written before the harvester validated them. Three vLLM
    anchors harvested with the probe came back with the widest wave *faster*
    than the one before it -- 481.6 ms at 2048 tokens, 532.9 at 4096, then
    334.8 at 8192 -- and least squares still fitted a tidy positive rate
    through them. A wider step cannot be cheaper, so those points are the
    scheduler's admission tail rather than one step's cost, and the rate
    derived from them is not a measurement of anything.

    Rejecting returns the projector to the single-sequence curve and the
    warning that names the packed term as unmeasured: a worse prediction,
    honestly labelled, instead of a confident wrong one.
    """
    if not packed or not packed.get("ms_per_token"):
        return None
    # Diagnostic only: drop the packed term and fall back to the single-
    # sequence curve, to separate "the probe is wrong" from "the step the
    # probe measured is not the step the scheduler builds under load".
    if os.environ.get("INFERASIM_IGNORE_PACKED_PROBE"):
        return None
    # Each point's p99 is taken over exactly as many requests as the wave is
    # wide, so a single pass is the max of a handful of samples and one slow
    # request sets the point. An early harvest taken that way produced a
    # monotonic-looking ladder -- 122, 300, 339, 600 ms -- that still put the
    # S=2 point almost level with S=4, and reading a rate off it moved a
    # 16384-token step from 827 ms to 1089 in the wrong direction. Monotonicity
    # alone does not catch that, so the repeat count is checked as well.
    repeats = int(packed.get("repeats") or 1)
    if repeats < 2:
        print(
            "[inferasim:Inference] WARNING: this anchor's packed-prefill "
            "probe recorded one pass per point, so each is the p99 of a "
            "handful of requests rather than a repeatable measurement. "
            "Ignoring it and falling back to the single-sequence curve; "
            "re-harvest to measure the packing term."
        )
        return None
    pts = sorted(
        (int(p.get("step_tokens") or 0), float(p.get("last_ttft_ms") or 0.0))
        for p in (packed.get("points") or [])
    )
    if len(pts) < 2:
        return None
    if any(pts[i + 1][1] <= pts[i][1] for i in range(len(pts) - 1)):
        print(
            f"[inferasim:Inference] WARNING: this anchor's packed-prefill "
            f"probe is not monotonic in step width "
            f"({[round(v, 1) for _, v in pts]} ms across "
            f"{[n for n, _ in pts]} tokens), so its "
            f"{packed['ms_per_token'] * 1000:.1f} us/token is a fit through "
            f"scheduler tail rather than a step cost. Ignoring it and falling "
            f"back to the single-sequence curve. Re-harvest to measure the "
            f"packing term."
        )
        return None
    # Monotonic and repeated is still not measured. Going from one sequence to
    # two adds a step's worth of latency on top of the tokens, so the first
    # rung is inflated on every probe on file and the rungs above it are the
    # marginal cost; least squares lets that first rung set the rate whenever
    # it is large enough to, leaving the packed term describing the scheduler
    # rather than the step.
    #
    # Rejected only when both tests fail, because either alone takes good
    # probes with it: a rate several times the single-sequence one (packing
    # cannot do that -- same attention per token, no less efficient a GEMM),
    # and a first rung that dwarfs the rest. MiniMax-M2.7 at 256 tokens fails
    # both -- 27.4 us/token against the curve's 10.1 -- and scored +55.9% TTFT.
    # DeepSeek-R1-0528 at 1024 reads 2.67x but its implied 8192-token step
    # matches Atom's scheduler logs to 10%; DeepSeek-V4-Flash-0731 has the
    # steep rung but a rate 1.06x the curve's.
    pr = float(packed["ms_per_token"])
    slopes = [(y2 - y1) / (n2 - n1) for (n1, y1), (n2, y2) in zip(pts, pts[1:])]
    later = sorted(slopes[1:])
    first_dominates = bool(later and slopes[0] > 5.0 * later[len(later) // 2] > 0.0)
    # The widest rung is the one the probe exists to measure, and it can be
    # checked against something independent: a step holding N tokens does about
    # the work of one sequence of N tokens. Packing changes the GEMM shape and
    # removes the attention a long context would have cost, so the packed step
    # should come in a little under the curve, never far over it.
    #
    # Across every probe on file the ratio sits between 0.63 and 1.39 when the
    # measurement is real. The failures are not near that band: a DeepSeek-V4-Pro
    # probe reads 320.7 ms for an 8192-token step whose single-sequence curve
    # says 89.3, and one probed at 2048 reads 7.06x. Both are the requests
    # behind the first one waiting their turn -- serialisation recorded as
    # packing -- and both pass the rate-and-first-rung test below, because
    # their rungs are uniformly inflated rather than front-loaded.
    if seq_cost_at is not None:
        n_wide, t_wide = pts[-1]
        expect = float(seq_cost_at(n_wide) or 0.0)
        if expect > 0.0 and not (0.5 <= t_wide / expect <= 1.5):
            print(
                f"[inferasim:Inference] WARNING: this anchor's packed-prefill "
                f"probe times a {n_wide}-token step at {t_wide:.1f} ms, where "
                f"the single-sequence curve puts {n_wide} tokens at "
                f"{expect:.1f} ms ({t_wide / expect:.2f}x). A step does not "
                f"change what its tokens cost by that much, so the probe is "
                f"timing queued sequences rather than one step. Ignoring it "
                f"and falling back to the single-sequence curve; re-harvest "
                f"with a token budget that admits the whole wave at once."
            )
            return None
    # Last, the combination that will actually be used has to reproduce the
    # measurement it came from. The projector prices a step as the curve's
    # intercept plus the packed rate over the step's tokens, and those two
    # numbers come from different fits: if the probe's intercept is far from
    # the curve's, the pair prices a step at something neither fit ever saw.
    #
    # Qwen3.6-35B-A3B is the case. Its probe times an 8192-token step at 79.0
    # ms and solves 66.6 ms fixed + 1.5 us/token; the length sweep solves 17.5
    # ms fixed for a single sequence, because the probe's p99-over-a-wave
    # method carries about 36 ms the sweep does not. Pairing the sweep's
    # intercept with the probe's slope prices that same step at 29.6 ms, and
    # scored -76.9% on TTFT. Taking the probe's intercept instead is not the
    # fix -- that 36 ms is the measurement, not the step, and substituting it
    # cost DeepSeek-V4-Flash 42 points. The pair simply cannot be used.
    if seq_cost_at is not None:
        n_wide, t_wide = pts[-1]
        modelled = float(seq_cost_at(0) or 0.0) + pr * n_wide
        if t_wide > 0.0 and not (0.6 <= modelled / t_wide <= 1.4):
            print(
                f"[inferasim:Inference] WARNING: this anchor's packed rate of "
                f"{pr * 1000:.1f} us/token, on the single-sequence curve's "
                f"fixed cost, prices a {n_wide}-token step at "
                f"{modelled:.1f} ms, against the {t_wide:.1f} ms the probe "
                f"timed for that step ({modelled / t_wide:.2f}x). The slope "
                f"and the intercept were fitted against different baselines "
                f"and do not describe one step together. Ignoring the packed "
                f"term and falling back to the single-sequence curve."
            )
            return None
    if seq_rate_ms_per_tok > 0.0 and pr > 2.0 * seq_rate_ms_per_tok and first_dominates:
        print(
            f"[inferasim:Inference] WARNING: this anchor's packed-prefill "
            f"probe reads {pr * 1000:.1f} us/token against the "
            f"single-sequence curve's {seq_rate_ms_per_tok * 1000:.1f} at the "
            f"same sequence length, and its 1->2 rung "
            f"({slopes[0] * 1000:.1f} us/token) dwarfs the rungs above it "
            f"({[round(s * 1000, 1) for s in slopes[1:]]}). That is the one-off "
            f"latency of widening the step, not the cost of the tokens in it. "
            f"Ignoring it and falling back to the single-sequence curve; "
            f"re-harvest at a sequence length where the packing signal clears "
            f"the step's fixed cost."
        )
        return None
    return packed


# There is deliberately no tuning knob here for how ``implied_fixed_ms``
# behaves across parallelism.
#
# An earlier revision carried one: a fitted exponent applied as
# ``fixed * shard ** exp`` when restoring a chord-fitted anchor at a width it
# was not harvested at. Swept against measured TP8 runs anchored from TP4, the
# per-model optima came out at 0.4, 1.0 and 0.5 -- spread across the whole
# range, so no single value described anything and the number was only ever a
# curve-fit to three models. It has been removed rather than retuned.
#
# What replaced it is a measurement. ``_fit_prefill_curve`` fits
# ``F + a*n + b*n^2`` over four or more probe lengths, which separates the
# per-step fixed cost from per-token work instead of leaving them mixed in a
# two-point chord's intercept. ``F`` is then held across parallelism and the
# per-token terms are sharded, because that is what each one does. The
# decomposition checks out against hardware: solving ``T(n,TP) = F + S(n)/ratio``
# from measured DeepSeek-V4-Pro TP4 and TP8 prefill gives F = 146.90 ms at 4096
# tokens and 145.86 ms at 8192, two independent lengths agreeing to 1%.
#
# When an anchor predates that and carries only a chord, the fallback holds the
# intercept whole -- exponent zero in the old parameterisation. That is the
# physical reading of the quantity's name and it needs no calibration; it is
# also, measurably, not as accurate as the curve, which is why the fallback
# warns and asks for a re-harvest rather than quietly standing in for one.


def _safe_forward(profiler, batch: int, seq_len: int) -> float:
    """Forward time of a sub-profiler, or 0 if it does not implement timing.

    Some element-wise profilers (LayerNorm, residual) only model memory and
    raise ``NotImplementedError`` for timing; those contributions are
    negligible for serving latency.
    """
    if profiler is None:
        return 0.0
    try:
        return float(profiler.measured_forward_time(batch, seq_len))
    except NotImplementedError:
        return 0.0


def _layers_on_rank(inference_config: InferenceConfig) -> int:
    mc = inference_config.model_config
    pp = max(1, inference_config.model_parallel_config.pipeline_model_parallel_size)
    return max(1, (mc.num_layers + pp - 1) // pp)


def _replica_gpus(inference_config: InferenceConfig) -> int:
    """GPUs in one model replica that serves a request.

    Latency-wise a request traverses TP×PP GPUs; for MoE the EP ranks live
    within that mesh, so we lower-bound the replica by EP.
    """
    mp = inference_config.model_parallel_config
    tp = max(1, mp.tensor_model_parallel_size)
    pp = max(1, mp.pipeline_model_parallel_size)
    ep = max(1, mp.expert_model_parallel_size)
    return max(tp * pp, ep)


def _split_replica_loads(total: int, replicas: int) -> list[int]:
    """Split an integer population evenly without dropping the remainder.

    Empty replicas are omitted from the returned work list; they are still
    counted separately when deployment cost is divided by the configured GPU
    fleet.
    """
    replicas = max(1, int(replicas))
    q, r = divmod(max(1, int(total)), replicas)
    return [n for n in ([q + 1] * r + [q] * (replicas - r)) if n > 0]


@dataclass
class PhaseForwardTimes:
    """Forward latency (ms) of each component for one forward pass."""

    layers_ms: float
    embedding_ms: float
    final_norm_ms: float
    output_ms: float
    dense_layer_ms: float
    moe_layer_ms: float
    # Token sampling / logits post-processing (memory-bound vocab reduction).
    sampling_ms: float = 0.0
    # Runtime activation quantization / cast (fp8 / mxfp4) over all layers.
    quant_ms: float = 0.0
    # Explicit communication (exposed, i.e. after overlap) for this forward.
    comm: CommBreakdown = field(default_factory=CommBreakdown)

    @property
    def total_ms(self) -> float:
        return (
            self.layers_ms
            + self.embedding_ms
            + self.final_norm_ms
            + self.output_ms
            + self.sampling_ms
            + self.quant_ms
            + self.comm.pp_p2p_ms
        )


@dataclass
class InferencePerfResult:
    ttft_ms: float
    decode_total_ms: float
    itl_ms: float  # inter-token latency per sequence (= TPOT)
    request_latency_ms: float  # TTFT + full decode for one sequence
    per_request_decode_tps: float
    decode_throughput_tps: float  # aggregate, whole batch
    decode_throughput_tps_per_gpu: float
    prefill_throughput_tps: float
    decode_step_latency_ms: float  # one decode forward (whole batch)
    replica_gpus: int
    # Disaggregation (feature A). ``is_disaggregated`` toggles the extra report.
    is_disaggregated: bool = False
    kv_transfer_ms: float = 0.0
    prefill_replica_gpus: int = 0
    decode_replica_gpus: int = 0
    extras: dict[str, float] = field(default_factory=dict)


def _prefix_caching_from_server_args(server_args) -> bool | None:
    """Whether the anchor's server ran with prefix caching, per its own flags.

    Returns None when the flags do not say, which keeps the caller's
    "untrusted unless proven otherwise" default: a prefill measured against a
    warm prefix cache is a block lookup, not prompt processing, and using one
    as if it were the latter under-prices TTFT badly.

    Every engine's spelling has to be listed, because an unrecognised flag
    reads as "not stated" and therefore as untrusted, and the consequence is
    not a warning but a discarded measurement: a prefill harvested with
    caching genuinely off gets thrown away and TTFT falls back to the
    analytical model. That cost 70% of corpus TTFT until SGLang's spelling was
    added here -- it calls the feature a radix cache, so
    --disable-radix-cache is the same statement vLLM makes with
    --no-enable-prefix-caching.
    """
    if not server_args:
        return None
    text = server_args if isinstance(server_args, str) else " ".join(server_args)
    flat = text.replace("_", "-")
    off = (
        "--no-enable-prefix-caching",
        "--disable-prefix-caching",
        # SGLang. --disable-radix-cache is the documented switch;
        # --enable-radix-cache does not exist, the cache being on by default.
        "--disable-radix-cache",
    )
    if any(flag in flat for flag in off):
        return False
    if "--enable-prefix-caching" in flat:
        return True
    return None


def _attention_dp_from_server_args(server_args, *, tp: int) -> int | None:
    """The attention-DP degree the anchor's server actually ran at.

    Unlike prefix caching, this is safe to infer from silence: data-parallel
    attention is opt-in on every engine, so server flags that do not ask for it
    describe a run without it. ``None`` is therefore reserved for having no flag
    string at all -- an artifact that predates this tracking, which may have run
    either way and must not be assumed to match.

    Each engine spells the same layout differently. SGLang and ATOM gate it
    behind ``--enable-dp-attention`` and size it with ``--dp-size``, defaulting
    to the whole tensor-parallel group; vLLM drives it from
    ``--data-parallel-size`` alongside expert parallelism.
    """
    if not server_args:
        return None
    text = server_args if isinstance(server_args, str) else " ".join(server_args)
    flat = text.replace("_", "-")
    tokens = flat.split()

    def value(flag: str) -> str | None:
        for i, tok in enumerate(tokens):
            if tok == flag and i + 1 < len(tokens):
                return tokens[i + 1]
            if tok.startswith(flag + "="):
                return tok.split("=", 1)[1]
        return None

    def as_degree(flag: str) -> int | None:
        got = value(flag)
        try:
            return max(1, int(got)) if got else None
        except ValueError:
            return None

    if "--disable-dp-attention" in flat:
        return 1
    if "--enable-dp-attention" in flat or "--enable-dp-attn" in flat:
        for flag in ("--dp-size", "--attention-dp-size", "--data-parallel-size"):
            got = as_degree(flag)
            if got:
                return got
        # Sized to the tensor-parallel group when left implicit, which is what
        # both engines do and how the MLA recipes are actually served.
        return max(1, int(tp))
    return as_degree("--data-parallel-size") or 1


class InferencePerformanceProjector:
    """Builds the profiler once and answers prefill / decode timing queries."""

    def __init__(
        self,
        inference_config: InferenceConfig,
        args=None,
        benchmark_layer_times=None,
        scaling_benchmarks=None,
        decode_floor=None,
        pool_benchmarks=None,
    ):
        self.cfg = inference_config
        self._args_ref = args
        # Optional per-pool anchors {"prefill": artifact, "decode": artifact}
        # for a disaggregated projection. Each pool prefers its own measurement
        # and falls back to the shared anchor -- see ``_project_disaggregated``.
        self._pool_benchmarks = dict(pool_benchmarks or {})
        # Optional measured decode latency floor {batch: ms} from a sharded
        # probe. Applied as decode = max(restored, floor(batch)) — see
        # ``_decode_floor_ms``.
        self._decode_floor = {int(b): float(v) for b, v in (decode_floor or {}).items() if v}
        # The loaded anchor's own invariant decode floor, filled in from its
        # batch sweep once one is loaded. Drives the decode restore; 0.0 means
        # "no sweep, no floor", which falls through to the other scaling laws.
        self._anchor_decode_floor_ms = 0.0
        gpu_arch = getattr(args, "gpu_arch", None) if args else None
        gpu_clock = getattr(args, "gpu_clock_mhz", None) if args else None
        gemm_name = getattr(args, "gemm_backend", None) if args else None
        # Kept for the origami-ratio restore, which builds its own guaranteed-
        # simulating profilers at the bench/target views (see _setup_restoration).
        self._gpu_arch, self._gpu_clock, self._gemm_name = gpu_arch, gpu_clock, gemm_name
        # TP/EP-restore scaling: how a measured anchor is extrapolated to another
        # TP. "origami" (default) scales the measured step by the simulator's
        # (vLLM-fused MoE) TP-scaling ratio — validated to beat a 2-point measured
        # fit at high TP; "fit" forces the measured shardable/invariant fit;
        # "blind" the naive TP^-1. Env override for A/B testing.
        self._scaling_mode = (os.getenv("INFERASIM_RESTORE_SCALING") or "origami").strip().lower()
        self._lm_ratio_bench = None

        # In benchmark mode the projection is driven *entirely* by measured
        # layer times, so the analytical GEMM/SDPA simulators (origami) are not
        # exercised for a dense model.  Don't hard-require origami there — fall
        # back to a metadata-only backend and skip SDPA if unavailable.
        benchmark_mode = benchmark_layer_times is not None
        self._gemm = get_gemm_simulation_backend(
            backend_name=gemm_name,
            gpu_arch=gpu_arch,
            gpu_clock_mhz=gpu_clock,
            require_simulation=not benchmark_mode,
        )
        try:
            self._sdpa = get_sdpa_simulation_backend(gpu_arch=gpu_arch, gpu_clock_mhz=gpu_clock)
        except RuntimeError:
            if not benchmark_mode:
                raise
            self._sdpa = None

        # Serving engines (vLLM/SGLang with AITER) run the MoE experts through a
        # *batched* grouped GEMM, not the training-time legacy sequential kernel.
        # The MoE profiler selects the batched Origami model via
        # ``use_turbo_grouped_gemm``, but the ModelConfig dataclass only carries
        # ``use_turbo_grouped_mlp`` — so the profiler-facing flag is never set on
        # the inference path and decode is otherwise mis-modelled as N sequential
        # per-expert GEMMs (grossly inflating small-batch decode). Set it here for
        # MoE serving, respecting an explicit legacy request.
        _mc = inference_config.model_config
        if getattr(_mc, "num_experts", 0) and not getattr(
            _mc, "moe_use_legacy_grouped_gemm", False
        ):
            _mc.use_turbo_grouped_gemm = True

        # Build profiler tree against a representative TrainingConfig view.
        view = inference_config.as_training_config(
            batch_size=inference_config.request_config.batch_size,
            seq_len=inference_config.request_config.input_seq_len,
        )
        self._view = view
        self._lm = build_profiler(get_language_model_profiler_spec(view))
        self._lm.set_simulation_backends(self._gemm, self._sdpa)

        # Token-sampling / logits post-processing model (memory-bound reduction
        # over the vocab, forward-only). Priced off the target GPU's HBM BW.
        req0 = inference_config.request_config
        self._sampling_enabled = bool(getattr(req0, "sampling_enabled", True))
        _hbm = getattr(self._gemm, "hbm_bandwidth_gbps", None)
        self._sampler = SamplingProfiler(
            inference_config.model_config.padded_vocab_size,
            hbm_bandwidth_gbps=_hbm,
            top_k=int(getattr(req0, "sampling_top_k", 0) or 0),
            top_p=float(getattr(req0, "sampling_top_p", 1.0) or 1.0),
            temperature=float(getattr(req0, "sampling_temperature", 1.0) or 1.0),
        )

        # Runtime activation quantization / cast (fp8 / mxfp4). Auto-detected
        # from weight_dtype / model fp8 unless explicitly set; ``None`` (bf16
        # serving) disables the term.
        self._act_quant_dtype = req0.resolved_act_quant_dtype(
            getattr(inference_config.model_config, "fp8", None)
        )
        self._quant = QuantCastProfiler(
            view, hbm_bandwidth_gbps=_hbm, dtype=self._act_quant_dtype or "fp8"
        )

        mc = inference_config.model_config
        self._moe_pattern = mc.moe_pattern or [0] * mc.num_layers
        self._n_moe = sum(1 for x in self._moe_pattern if x)
        self._n_dense = mc.num_layers - self._n_moe

        # DeepEP / SyncFree EP-A2A compute-overlap fraction (0 = disabled).
        # Applied to the *builtin* comm path here; the explicit comm model
        # (``InferenceCollectiveModel``) applies the same factor internally.
        self._deepep_overlap = deepep_overlap_efficiency(mc)

        # MoE expert-routing imbalance multiplier (>= 1.0).  Real routing is
        # skewed, so the MoE step is gated by the busiest EP rank rather than
        # the perfectly-balanced average.  Only meaningful for an EP-sharded MoE
        # model; a no-op (1.0) otherwise.
        self._moe_imbalance = self._moe_imbalance_factor()
        # Per-view imbalance for the origami ratio (populated in
        # _setup_restoration when a bench<->target restore is active).
        self._imb_tgt = self._moe_imbalance
        self._imb_bench = self._moe_imbalance
        # Mirror the routing-imbalance knobs onto the (shared) model_config so the
        # expert-GEMM simulator can apply imbalance *inside* the roofline, per
        # view (EP lives on each view's parallel config). Default ("roofline")
        # mode; INFERASIM_MOE_IMB_ROOFLINE=0 falls back to the outer multiplier.
        self._imb_roofline = os.getenv("INFERASIM_MOE_IMB_ROOFLINE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        try:
            mc.ep_load_balance = float(self.cfg.request_config.ep_load_balance or 1.0)
            mc.redundant_experts = int(self.cfg.request_config.redundant_experts or 0)
            # Expert weight precision, read by the MoE profiler so the expert
            # grouped-GEMM roofline streams the right number of weight bytes.
            mc.moe_expert_dtype = self.cfg.request_config.moe_expert_dtype
            # The same, for the linears that are not experts: attention's
            # projections and the dense MLP. Left unset those GEMMs stream fp8
            # weights whatever the checkpoint says, so a 4-bit model was sized
            # at 4-bit by the memory model and then read at twice that width by
            # the roofline. Decode is weight-bound, so the two have to agree.
            mc.linear_weight_dtype = self.cfg.request_config.linear_weight_dtype
        except Exception:
            pass

        # Kernel-backend (AITER/Triton/CK/HIP) attention multiplier, native
        # sparse-attention selection, and MoE expert-dtype (mxfp4/fp8/bf16)
        # compute speedup.  All affect the *simulation* path only (the measured
        # path bundles these into the whole-model step).  Defaults are no-ops.
        self._attn_backend_mult = (
            inference_config.request_config.resolved_attention_backend_multiplier()
        )
        # The expert dtype is now priced inside the expert GEMM roofline (real
        # operand bytes + matrix throughput), so there is no outer multiplier to
        # apply. Kept at 1.0 rather than removed so the restore/ratio paths that
        # reference it stay arithmetically identical.
        self._moe_expert_speedup = 1.0

        # Feature B: explicit, knob-driven communication model. When enabled we
        # replace the layer profiler's *implicit* TP-AllReduce / EP-AllToAll
        # cost with this model (delta applied per layer), enabling algorithm
        # selection, comm/compute overlap, fused-op speedups and a reportable
        # per-phase breakdown.
        # Per-kernel decode occupancy is a property of the silicon, so an
        # unset one resolves from the architecture rather than from a single
        # global default that can only be right for the part it was solved on.
        if inference_config.request_config.decode_kernel_occupancy_us is None:
            inference_config.request_config.decode_kernel_occupancy_us = (
                resolve_decode_occupancy_us(gpu_arch)
            )

        self._cc = inference_config.collective_config
        if self._cc and self._cc.interconnect is None and gpu_arch:
            self._cc.interconnect = INTERCONNECT_PROFILES.get(str(gpu_arch).lower())
        self._comm = (
            InferenceCollectiveModel(mc, inference_config.model_parallel_config, self._cc)
            if (self._cc and self._cc.enabled)
            else None
        )

        # Benchmark mode (BENCHMARK-BASED PROJECTION — no calibration factors).
        # We use the *measured* silicon times directly as the projection. Two
        # measurement schemas are supported:
        #
        #   * whole-model (vLLM/SGLang): measured prefill / decode *step* latency
        #     (ms) for the full model, optionally swept over batch. Stored as
        #     per-phase (batch -> ms) curves; interpolated by concurrency.
        #   * per-layer (Megatron worker): measured forward time of one dense and
        #     one MoE layer per phase. Composed directly by layer counts.
        #
        # Empty => pure simulation.
        self._meas_whole: dict[str, list] = {}  # {"prefill": [(batch, ms)], "decode": [...]}
        # When the benchmark swept the decode curve at the engine's CUDA-graph
        # capture sizes, runtime pads the decode batch UP to the nearest captured
        # size — so decode latency is a staircase and we look it up by bucket
        # rather than interpolating. Set from meta in set_benchmark_calibration.
        self._decode_pad_to_capture: bool = False
        # Attention KV term for the FULL decode step: the step grows with context
        # by an ~batch-independent additive per-token amount (measured: the rise
        # over context is nearly the same at low and high batch, so it is NOT
        # proportional to batch). Fit from the benchmark's decode-vs-context grid;
        # 0 => flat (no grid), preserving prior behaviour.
        self._decode_kv_slope_ms: float = 0.0  # ms per KV token (batch-independent)
        self._decode_ctx_ref: float = 0.0  # context the batch curve was measured at
        self._decode_ctx_max: float = 0.0  # largest measured context (guard)
        self._decode_kv_slope_by_batch: list = []  # (batch, ms per KV token)
        self._meas_prefill_rate_ms_per_tok: float = 0.0  # for sub-prompt prefill pieces
        # Fixed per-prefill-step cost measured alongside that rate.
        self._meas_prefill_fixed_ms: float = 0.0
        # tp -> the prefill length curve measured at that width.
        self._bench_prefill_curves: dict = {}
        # True when the benchmark deliberately repeated prompts with prefix
        # caching enabled. Such a curve is a cache-hit lookup curve and is only
        # usable for a target configured as a full prefix hit.
        self._meas_prefill_cache_hit: bool = False
        self._meas_layer: dict[tuple, float] = {}  # {(phase, ltype): ms}
        self._meas_ref_input: int = 0
        self._bench_backend: str = "megatron"
        self._bench_measured = benchmark_layer_times
        # Training-style TP/EP restoration state (populated for the per-layer
        # schema in set_benchmark_calibration). Off unless the benchmark ran at
        # a reduced parallelism vs the target.
        self._restore = False
        self._bench_tp = 1
        self._bench_ep = 1
        self._bench_pp = 1
        # Attention layout the anchor was harvested at. ``None`` means the
        # artifact never recorded it, which is not the same as 1: it may have
        # run either way, so it is reported as unverifiable rather than assumed
        # to match. See ``_setup_restoration``.
        self._bench_attn_dp: int | None = None
        # Set by ``_setup_restoration``: the layout the bench view is actually
        # built at, and whether it differs from the target's.
        self._bench_attn_dp_eff = 1
        self._restore_layout_moved = False
        # Draft depth the anchor itself was harvested at. This decides what a
        # measured decode number *means*: harvested with speculation it is a
        # per-output-token time with acceptance already folded in, harvested
        # without it is a single-token step. 0 = no speculation on the anchor.
        self._bench_spec_k = 0
        # (batch, context, width) -> verify/single step ratio, or None where the
        # analytical path could not be evaluated. Memoised because the replay
        # prices a step per simulated event and the ratio moves only with the
        # step shape.
        self._verify_ratio_cache: dict = {}
        # (batch, ref_batch, context, phase) -> modelled step ratio, for giving
        # a single-point anchor a slope. Same memoisation reason.
        self._batch_shape_cache: dict = {}
        # phase -> batch -> tp -> (ms, ep, pp), and the split fitted from it.
        self._bench_scaling_raw: dict = {}
        self._bench_scaling_fit: dict = {}
        for _blob in scaling_benchmarks or []:
            self.add_scaling_benchmark(_blob)
        if benchmark_layer_times:
            self.set_benchmark_calibration(benchmark_layer_times)

    @property
    def is_benchmark_calibrated(self) -> bool:
        return bool(self._meas_whole or self._meas_layer)

    @property
    def _measured_mode(self) -> bool:
        return bool(self._meas_whole or self._meas_layer)

    @staticmethod
    def _interp(batch: int, pts: list) -> float:
        """Piecewise-linear interpolation of a sorted (batch, value) curve.

        Clamps to the endpoints outside the measured range so concurrencies
        below/above the swept batches reuse the nearest measured anchor.
        """
        if not pts:
            return 0.0
        if batch <= pts[0][0]:
            return pts[0][1]
        if batch >= pts[-1][0]:
            return pts[-1][1]
        for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
            if b0 <= batch <= b1:
                w = (batch - b0) / (b1 - b0) if b1 > b0 else 0.0
                return v0 + w * (v1 - v0)
        return pts[-1][1]

    # -- batch transport (measured curve only) --------------------------------

    @staticmethod
    def _nearest_anchor(batch: int, pts: list):
        """Nearest measured point to ``batch`` in log-space (batch grids are
        geometric, e.g. 1/4/16/64)."""
        lb = math.log(max(1, batch))
        return min(pts, key=lambda p: abs(lb - math.log(max(1, p[0]))))

    def _fit_decode_kv_slope(self, decode_ctx: list, ref_ctx: float) -> None:
        """Fit the FULL-decode attention KV term from the benchmark's
        decode-vs-context grid: ``step(b, c) = bucket(b) + slope * (c - ref)``.
        ``slope`` is the median per-token increment (``(step - bucket) / (c-ref)``)
        over the measured points — batch-independent, because the measured rise
        with context is ~the same at low and high batch. 0 when no grid (→ decode
        stays flat in context, prior behaviour). Skipped under parallelism restore
        (the grid is only emitted un-restored)."""
        self._decode_ctx_ref = ref_ctx
        self._decode_ctx_max = ref_ctx
        # Differenced against the sweep as measured, not as restored: the grid
        # is recorded at the anchor's own parallelism, so the KV term is fitted
        # there and carried to the target width below.
        dec = getattr(self, "_meas_decode_bench", None) or self._meas_whole.get("decode")
        if not decode_ctx or not dec or ref_ctx <= 0:
            return
        slopes, per_batch = [], {}
        for e in decode_ctx:
            try:
                b, c, ms = int(e["batch"]), float(e["context"]), float(e["decode_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            self._decode_ctx_max = max(self._decode_ctx_max, c)
            # Points on either side of the reference are equally informative:
            # below it the step is measured cheaper and the numerator turns
            # negative along with the denominator, so the slope comes out
            # positive from the same expression. Only points AT the reference
            # say nothing.
            if b > 0 and c != ref_ctx:
                s = (ms - self._bucket_up(b, dec)) / (c - ref_ctx)
                if s > 0:
                    slopes.append(s)
                    per_batch.setdefault(b, []).append(s)
        # Reading the KV of one sequence is work, and work shards: the
        # attention heads that do the reading divide across ranks like
        # everything else in the step. So the term moves to the target width by
        # the same ratio the restore applies to the part of the step above the
        # floor, which is where this term lives.
        shard = self._bench_tp / self._tgt_tp if (self._restore and self._tgt_tp > 0) else 1.0
        # Held per batch rather than as one median over all of them. The step
        # carries the KV of every resident sequence, so the cost of context
        # grows with how many are in flight, and measured it does: carrying
        # DeepSeek-V4-Flash at TP4 from 1024 tokens of context to 8192 costs
        # 0.15 us per token of context at batch 1, 1.07 at 4, 1.26 at 16 and
        # 1.92 at 64. One median over that span reproduces none of them, and
        # collapsing it cost DeepSeek-V4-Pro's ISL-1024 rows 6.0% -> 12.5% on
        # TPOT against reading their own matched-length sweep directly.
        if per_batch:
            self._decode_kv_slope_by_batch = sorted(
                (b, sorted(v)[len(v) // 2] * shard) for b, v in per_batch.items()
            )
        if slopes:
            slopes.sort()
            self._decode_kv_slope_ms = slopes[len(slopes) // 2] * shard

    @staticmethod
    def _bucket_up(batch: int, pts: list) -> float:
        """Decode value with the runtime CUDA-graph padding applied: the batch is
        padded UP to the nearest captured size, so return the measured value at
        the smallest measured (capture-aligned) batch >= ``batch``. Clamp to the
        largest measured point above the top capture size. ``pts`` sorted asc."""
        for b0, v0 in pts:
            if b0 >= batch:
                return v0
        return pts[-1][1]

    @staticmethod
    def _loglog_transport(batch: int, pts: list) -> float:
        """Piecewise power-law (log-log linear) interpolation of a measured
        ``(batch -> ms)`` curve; extrapolate with the nearest end segment's
        slope. ``pts`` must be sorted with >= 2 points.

        Interpolating the *measured* curve directly is more accurate than
        modulating the analytical simulator, which can carry a spurious knee
        (e.g. an MoE expert-coverage bump) that the real silicon does not show.
        Latency-vs-batch is close to a local power law between adjacent measured
        points, so a straight line in (log batch, log ms) tracks it tightly and
        extrapolates monotonically."""
        lb = math.log(max(1, batch))
        xs = [(math.log(max(1, b)), math.log(max(1e-9, v))) for b, v in pts]
        if lb <= xs[0][0]:
            (x0, y0), (x1, y1) = xs[0], xs[1]
        elif lb >= xs[-1][0]:
            (x0, y0), (x1, y1) = xs[-2], xs[-1]
        else:
            (x0, y0), (x1, y1) = xs[0], xs[1]
            for i in range(len(xs) - 1):
                if xs[i][0] <= lb <= xs[i + 1][0]:
                    (x0, y0), (x1, y1) = xs[i], xs[i + 1]
                    break
        slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
        return math.exp(y0 + slope * (lb - x0))

    def _transport_batch(self, batch: int, pts: list, phase: str = "decode") -> float:
        """Transport a measured ``(batch -> ms)`` curve to an arbitrary
        ``batch`` — MEASUREMENT-ONLY.

        The benchmark protocol always sweeps batch within a single run, so
        ``pts`` normally carries >= 2 measured points and we interpolate/
        extrapolate the real curve in log-log space
        (:meth:`_loglog_transport`) — the analytical (origami) simulator is
        never consulted for the batch shape.

        A lone point is a degenerate, non-swept artifact, and holding it flat
        is not the neutral choice it looks like. Step latency grows with batch
        on every measured curve here, so a flat hold asserts a zero slope --
        the one shape the hardware never has -- and it asserts it over the
        whole ladder a sweep is ranked on, which collapses the ordering the
        projection exists to produce. The measured point still sets the level;
        the shape around it comes from the model, which is the same split
        :meth:`_restore_whole` uses to carry an anchor across widths. Falls
        back to the flat hold only where that ratio cannot be evaluated.

        Returns the exact measured value when ``batch`` is itself measured."""
        if not pts:
            return 0.0
        P = sorted(pts)
        for b0, v0 in P:
            if b0 == batch:
                return v0
        if len(P) >= 2:
            return self._loglog_transport(batch, P)
        b0, v0 = P[0]
        shape = self._batch_shape_ratio(batch, b0, phase)
        return v0 if shape is None else v0 * shape

    def _batch_shape_ratio(
        self, batch: int, ref_batch: int, phase: str
    ) -> float | None:
        """How a step at ``batch`` compares with one at ``ref_batch``, modelled.

        Used only to give a single-point anchor a slope. ``None`` when the
        analytical path cannot be evaluated, which leaves the caller holding
        the measured value flat.
        """
        b, b0 = max(1, int(batch)), max(1, int(ref_batch))
        if b == b0:
            return 1.0
        kv = max(
            1, int(self._meas_ref_input or self.cfg.request_config.input_seq_len or 1024)
        )
        key = (b, b0, kv, phase)
        if key in self._batch_shape_cache:
            return self._batch_shape_cache[key]
        try:
            if phase == "prefill":
                # Prefill work is per token, so the comparable pair is the same
                # per-sequence prompt carried at each batch.
                here = self._forward_times(b, kv, "prefill", kv).total_ms
                there = self._forward_times(b0, kv, "prefill", kv).total_ms
            else:
                here = self._forward_times(b, 1, "decode", kv).total_ms
                there = self._forward_times(b0, 1, "decode", kv).total_ms
        except Exception:  # noqa: BLE001 - simulator availability is arch-dependent
            self._batch_shape_cache[key] = None
            return None
        if here <= 0.0 or there <= 0.0:
            self._batch_shape_cache[key] = None
            return None
        ratio = here / there
        self._batch_shape_cache[key] = ratio
        return ratio

    # -- measured-time accessors (benchmark-based projection) ------------------

    def _decode_kv_slope_at(self, batch: int) -> float:
        """Per-token cost of resident context at ``batch``.

        Interpolated across the batches the grid was measured at, the same way
        the decode sweep itself is, and held flat outside them: the term grows
        with how many sequences are resident, so one number for every batch
        fits neither end."""
        pts = getattr(self, "_decode_kv_slope_by_batch", None)
        if not pts:
            return self._decode_kv_slope_ms
        return self._loglog_transport(batch, pts) if len(pts) >= 2 else pts[0][1]

    def _measured_decode_step_ms(self, batch: int, context: float | None = None) -> float:
        """Measured whole-model / composed decode *step* latency at ``batch``.

        ``context`` (the resident KV length) adds the fitted attention KV term on
        top of the batch-bucket value; omitting it (or a zero slope / no grid)
        reproduces the flat-in-context behaviour."""
        if self._meas_whole.get("decode"):
            pts = self._meas_whole["decode"]
            if self._decode_pad_to_capture:
                base = self._bucket_up(batch, pts)
            else:
                base = self._transport_batch(batch, pts, "decode")
            if (
                context is not None
                and self._decode_kv_slope_ms > 0.0
                and self._decode_ctx_ref > 0.0
            ):
                # Signed, not one-sided. The batch sweep is measured at one
                # context and the attention in it reads exactly that much KV,
                # so billing it at a shorter context charges for KV that is
                # not resident -- which is the larger of the two errors here,
                # because anchors get harvested long and read short. On
                # MI355X, DeepSeek-V4-Flash at TP4 batch 16 steps in 22.12 ms
                # at 8192 tokens of context and 15.72 at 1024; holding the
                # first for ISL-1024 rows is what puts their TPOT +27% out.
                #
                # Held above zero because the line is only a local statement
                # about the KV term: extrapolated far enough below the
                # reference it eventually crosses the context-free cost of the
                # step, which is weights and compute and does not go away.
                slope = self._decode_kv_slope_at(batch)
                base = max(
                    base + slope * (float(context) - self._decode_ctx_ref),
                    self._decode_floor_ms(batch),
                )
            return base
        # Per-layer schema: restore each layer to the target TP/EP, then sum by
        # layer count. Decode processes 1 token/step.
        d = self._restore_per_layer(
            "dense", self._meas_layer.get(("decode", "dense"), 0.0), batch, 1
        )
        m = self._restore_per_layer("moe", self._meas_layer.get(("decode", "moe"), 0.0), batch, 1)
        return self._n_dense * d + self._n_moe * m + self._restore_pp_ms(batch, 1)

    def _measured_full_prefill_ms(self, batch: int) -> float:
        """Measured whole-model / composed prefill latency for the full prompt."""
        if self._meas_whole.get("prefill"):
            return self._transport_batch(batch, self._meas_whole["prefill"], "prefill")
        tok = self._meas_ref_input or 1
        d = self._restore_per_layer(
            "dense", self._meas_layer.get(("prefill", "dense"), 0.0), batch, tok
        )
        m = self._restore_per_layer(
            "moe", self._meas_layer.get(("prefill", "moe"), 0.0), batch, tok
        )
        return self._n_dense * d + self._n_moe * m + self._restore_pp_ms(batch, tok)

    def _measured_prefill_tokens_ms(self, total_tokens: int) -> float:
        """Measured prefill time for an arbitrary token count (chunk pieces).

        Prefill is compute-bound and ~linear in total processed tokens, so we
        scale by a measured per-token rate rather than re-simulating.
        """
        rate = self._meas_prefill_rate_ms_per_tok
        if rate <= 0:
            # Decode-only artifact (or an untrusted prefill, see
            # set_benchmark_calibration): simulate the chunk rather than bill it
            # as free.
            return self._forward_times(
                1, max(1, total_tokens), "prefill", max(1, total_tokens)
            ).total_ms
        # Charged once per step, not per token and not per request: a step
        # packing eight 1024-token prompts pays it once, which is what makes
        # short prompts clear in ceil(C*ISL/budget) steps rather than C.
        return self._meas_prefill_fixed_ms + rate * max(1, total_tokens)

    # -- benchmark ingestion ---------------------------------------------------

    def _builtin_comm_ms(self, ltype: str, batch: int, q_len: int) -> float:
        """Implicit comm baked into the layer profiler's forward time."""
        tp_ar_one = _estimate_tp_allreduce_time_ms(self._view, batch, q_len)
        if ltype == "moe":
            # EP>1 (expert_tp==1) drops the post-expert TP-AR (combined by A2A).
            n_ar = _moe_tp_allreduce_count(self._view)
            return n_ar * tp_ar_one + _estimate_moe_a2a_time_ms(
                self._view, batch, q_len, self._gemm
            )
        return _dense_tp_allreduce_count(self._view) * tp_ar_one

    def set_benchmark_calibration(self, benchmark_layer_times: dict) -> None:
        """Ingest measured silicon times for a **benchmark-based** projection.

        No calibration factors are applied to the analytical model: the measured
        times are used *directly* as the projection. Two schemas are accepted:

        * whole-model (vLLM/SGLang)::

              {"backend": "vllm",
               "measured": {"model": {"prefill_ms", "decode_ms"}},
               "sweep": [{"batch", "prefill_ms", "decode_ms"}, ...],
               "meta": {"batch", "input_len", ...}}

          ``prefill_ms``/``decode_ms`` are full-model step latencies; ``sweep``
          gives the per-concurrency curve (preferred), interpolated by batch.

        * per-layer (Megatron worker)::

              {"measured": {"dense"|"moe": {"prefill_ms", "decode_ms"}},
               "meta": {"batch", "input_len"}}

          composed by layer counts.
        """
        if not benchmark_layer_times:
            return
        measured = benchmark_layer_times.get("measured", benchmark_layer_times)
        meta = benchmark_layer_times.get("meta", {})
        self._bench_backend = str(benchmark_layer_times.get("backend", "megatron"))
        ref_batch = int(meta.get("batch") or self.cfg.request_config.batch_size or 1)
        ref_input = int(meta.get("input_len") or self.cfg.request_config.input_seq_len or 1)
        # Parallelism the benchmark ran at (for training-style restoration of a
        # reduced-parallelism per-layer bench to the target TP/EP).
        self._bench_tp = int(meta.get("benchmark_tp") or meta.get("tp") or 1)
        self._bench_ep = int(meta.get("benchmark_ep") or meta.get("ep") or 1)
        self._bench_pp = int(meta.get("benchmark_pp") or meta.get("pp") or 1)
        _attn_dp = meta.get("attention_data_parallel_size")
        self._bench_attn_dp = int(_attn_dp) if _attn_dp else None
        self._bench_spec_k = int(meta.get("speculative_num_tokens") or 0)
        self._decode_pad_to_capture = bool(meta.get("decode_pad_to_capture"))

        self._meas_ref_input = ref_input

        # Whole-model schema (vLLM/SGLang): measured step latencies, used
        # DIRECTLY (no factor, no simulator). ``sweep`` gives the per-batch
        # curve; fall back to the single ``model`` anchor at ``ref_batch``.
        model_step = measured.get("model")
        if model_step:
            # Restore a reduced-parallelism (benchmark) whole-model measurement to
            # the target TP/EP/PP, in the same pp -> ep -> tp order as the Megatron
            # per-layer path. Builds the bench/target collective models; a no-op
            # when the benchmark already ran at the target parallelism.
            self._setup_restoration()
            # The primary artifact is itself a point on the scaling curve.
            self.add_scaling_benchmark(benchmark_layer_times)
            if self._restore:
                for _phase in ("decode", "prefill"):
                    self._fit_tp_scaling(_phase)
                # Resolved before the report so it can name the law it will use,
                # and before any restore call consumes it.
                self._anchor_decode_floor_ms = self._anchor_floor_from(benchmark_layer_times)
                self._report_tp_scaling()
            sweep = benchmark_layer_times.get("sweep") or []
            pre_pts, dec_pts = [], []
            for e in sweep:
                try:
                    b = int(e["batch"])
                except (KeyError, TypeError, ValueError):
                    continue
                if e.get("prefill_ms"):
                    pre_pts.append((b, float(e["prefill_ms"])))
                if e.get("decode_ms"):
                    dec_pts.append((b, float(e["decode_ms"])))
            if not pre_pts and model_step.get("prefill_ms"):
                pre_pts = [(ref_batch, float(model_step["prefill_ms"]))]
            if not dec_pts and model_step.get("decode_ms"):
                dec_pts = [(ref_batch, float(model_step["decode_ms"]))]
            # A prefill timed against a warm prefix cache measures a block
            # lookup rather than prompt processing, and vLLM caches by default.
            # Artifacts predating the harness forcing it off carry no marker, so
            # a missing key counts as untrusted and prefill falls back to
            # simulation. Decode is unaffected either way: it is differenced
            # between two runs that both hit the cache.
            #
            # The size of what is being refused: harvesting gpt-oss-120b at TP1
            # both ways -- once as the old artifacts were collected and once with
            # the cache forced off -- put the cached prefill 3.3x under the real
            # one at the median, and 9x under it at batch 64. So this is loud
            # rather than incidental. It is also easy to miss that the fallback
            # silently moves TTFT onto the simulator, which is usually the number
            # the projection was run for.
            cache_mode = meta.get("prefix_caching")
            if cache_mode is None:
                # Older harnesses recorded the launch flags but not the resolved
                # cache state. The answer is still in the artifact -- an
                # explicit --no-enable-prefix-caching on the server it measured
                # settles it -- and reading it there is the difference between
                # using a measured prefill and silently simulating TTFT, which
                # is usually the number the projection was run for. Only an
                # explicit flag counts; a server_args string that never mentions
                # caching leaves this None and stays untrusted.
                cache_mode = _prefix_caching_from_server_args(meta.get("server_args"))
                if cache_mode is not None:
                    print(
                        f"[inferasim:Inference] anchor records no prefix_caching "
                        f"flag; its server_args say prefix caching was "
                        f"{'ON' if cache_mode else 'OFF'}, using that."
                    )
            target_hit = self.cfg.request_config.resolved_prefix_cache_hit_rate()
            if pre_pts and cache_mode is True and target_hit >= 0.999:
                self._meas_prefill_cache_hit = True
                print(
                    "[inferasim:Inference] using measured PREFIX-CACHE-HIT prefill "
                    "curve (target prefix hit fraction is 1.0)."
                )
            elif pre_pts and cache_mode is not False:
                print(
                    "[inferasim:Inference] WARNING: PREFILL IS NOT CALIBRATED. This "
                    "artifact's prefill is a prefix-cache-hit lookup curve, but the "
                    f"target prefix hit fraction is {target_hit:.2f}, not 1.0. Decode "
                    "is calibrated as usual; prefill and TTFT remain simulated. Use "
                    "a cache-off anchor for partial/miss traffic or set the target "
                    "hit fraction to 1.0 for repeated-prefix traffic."
                )
                pre_pts = []
            # A prefill curve also has to be transportable to the prompt length
            # being asked about, and only the length probe makes it so. Four or
            # more probed lengths fit a quadratic, which is read at the target
            # (held at the last probed length past the ladder). Two lengths fit
            # only a chord, and what survives is the sweep's flat per-token rate
            # at the anchor's own ``input_len`` -- a number that is then sharded
            # across TP width and billed unchanged at any prompt length.
            #
            # For a 16x reach that is not a small error, because prefill
            # per-token cost is not context-free: attention grows with context.
            # DeepSeek-V4-Pro's TP8 MI355X anchor probes 4096 and 8192 only, so
            # at ISL 130000 it charges the 8192-token rate, which prices the
            # prompt as very nearly free -- TTFT 531 ms against a measured
            # 15570, and 108005 tok/s/gpu against a best measured 49370, a 2.2x
            # over-read on the headline metric. Simulated prefill lands within
            # 0.3% of the same measurement, so the fallback is strictly better
            # than the flat rate here.
            #
            # Refused rather than scaled: transporting it would mean inventing
            # the context term from the analytical model and calling the result
            # measured, which is the fallback's job and is reported as such.
            # Decode is unaffected and stays calibrated.
            _prefill_probed = sorted(
                int(p.get("input_len") or 0)
                for p in ((meta.get("prefill_anchor") or {}).get("points") or [])
            )
            _has_curve = bool(
                ((meta.get("prefill_anchor") or {}).get("curve_fit") or {}).get("ms_per_token")
            )
            _target_n = int(self.cfg.request_config.input_seq_len or ref_input or 0)
            if pre_pts and not _has_curve and _prefill_probed and _target_n > 1.5 * max(
                _prefill_probed
            ):
                print(
                    f"[inferasim:Inference] WARNING: PREFILL IS NOT CALIBRATED. This "
                    f"anchor probed prefill at {min(_prefill_probed)}..{max(_prefill_probed)} "
                    f"tokens, too few lengths to fit a context curve, and the target "
                    f"runs {_target_n}. Its flat per-token rate carries no context "
                    f"growth, so billing it here would price a "
                    f"{_target_n / max(_prefill_probed):.0f}x longer prompt at the same "
                    f"cost per token. Decode is calibrated as usual; prefill and TTFT "
                    f"are simulated. Re-harvest with --prefill-anchor-points 4 or more "
                    f"over a ladder that brackets this prompt length."
                )
                pre_pts = []
            pre_pts_bench = list(pre_pts)
            dec_pts_bench = list(dec_pts)
            if self._restore:
                # Prefill processes ``ref_input`` tokens/seq; decode 1 token/step.
                pre_pts = [
                    (b, self._restore_whole(ms, b, ref_input, "prefill")) for b, ms in pre_pts
                ]
                dec_pts = [(b, self._restore_whole(ms, b, 1, "decode")) for b, ms in dec_pts]
            # Kept at the width it was measured at. The decode-vs-context grid
            # is recorded un-restored, so the KV term has to be differenced
            # against the un-restored sweep and carried across parallelism
            # afterwards; differencing a raw grid against a restored sweep
            # would read the width transport as context dependence.
            self._meas_decode_bench = sorted(dec_pts_bench)
            self._meas_whole = {
                k: v for k, v in (("prefill", sorted(pre_pts)), ("decode", sorted(dec_pts))) if v
            }
            # Which decode observable the sweep holds changes what the
            # scheduler is allowed to add on top of it. An anchor recording
            # median inter-token latency holds the unblocked decode step, which
            # is what the simulator wants: it schedules the prefill stalls
            # itself. One recording mean TPOT already contains those stalls,
            # so simulating them again bills them twice -- measured at 1.14x
            # TPOT at 16 concurrent, 1.23x at 32 and 1.41x at 64 on the
            # DeepSeek-V4-Pro TP8 ladder. Artifacts harvested before the key
            # existed are the mean-TPOT kind; they are used as-is rather than
            # corrected by a guess, and the warning says what to re-harvest.
            observable = meta.get("decode_observable")
            # "median_itl" without the single-wave qualifier was harvested with
            # a refilling loop, so on a chunked co-scheduled engine its median
            # step carries prefill work -- 51% of TPOT on a vLLM harvest, while
            # the same protocol on an exclusive-prefill engine was within 3%.
            if observable not in ("median_itl_single_wave", "median_itl"):
                print(
                    f"[inferasim:Inference] WARNING: this anchor's decode "
                    f"sweep is {observable or 'mean TPOT'}, which already "
                    f"includes the prefill work the scheduler also models. "
                    f"TPOT will be over-predicted, increasingly so with "
                    f"concurrency. Re-harvest to record median inter-token "
                    f"latency instead."
                )
            elif observable == "median_itl" and str(self._bench_backend).lower() == "vllm":
                # Right observable, refilling loop. Harmless where prefill is
                # exclusive, because a handful of huge stalls do not move a
                # median; wrong where it is chunked and co-scheduled, because
                # then the typical step carries prefill and the median carries
                # it too. Named by engine rather than guessed at, since that is
                # what decides which of the two applies.
                print(
                    "[inferasim:Inference] WARNING: this anchor's decode sweep "
                    "is median inter-token latency over a refilling loop, and "
                    "it was harvested on a chunked co-scheduled engine. Every "
                    "step in such a loop carries part of some prefill, so the "
                    "median is a mixed step rather than a decode step and the "
                    "sweep also climbs too steeply with batch. Measured on "
                    "DeepSeek-V4-Flash-0731 that read 19.71 ms at batch 4 "
                    "against 13.44 measured, and TPOT 51% high across the "
                    "corpus. Re-harvest for a single-wave sweep."
                )
            # A decode step reads the KV of every sequence in it, so what the
            # sweep measures depends on the prompt length it was measured at --
            # but only on engines whose attention actually reads all of it.
            # Same model, same width, same anchor prompt length of 8192,
            # applied at 1024: SGLang lands within 3% of the measured step
            # (10.81 ms against 11.19 at four sequences) because its DeepSeek-V4
            # backend reads a fixed top-k of the cache and is therefore flat in
            # context, while vLLM's dense MLA path reads the whole thing and
            # comes out 41% high (18.96 against 13.44). Across the corpus that
            # one difference was worth 51% on TPOT.
            #
            # Nothing here converts one into the other -- how a step scales with
            # context is a property of the attention kernel, and a fitted ratio
            # would be exactly the kind of factor this model does not carry. So
            # the mismatch is reported and the harvest is named as the fix.
            bench_isl = meta.get("input_len")
            target_isl = self.cfg.request_config.input_seq_len
            if (
                bench_isl
                and target_isl
                and max(bench_isl, target_isl) >= 2 * min(bench_isl, target_isl)
            ):
                print(
                    f"[inferasim:Inference] WARNING: decode sweep was measured "
                    f"at {int(bench_isl)}-token prompts and is being applied at "
                    f"{int(target_isl)}. A decode step reads the KV of every "
                    f"sequence in it, so on an engine whose attention is dense "
                    f"in context this sweep is the wrong step: the same "
                    f"{int(bench_isl)}-token anchor read a 1024-token workload's "
                    f"decode 41% high on vLLM, and within 3% on SGLang, whose "
                    f"backend reads a fixed top-k instead. Harvest at this "
                    f"prompt length to remove the question."
                )

            # Decode's width scaling is restored, not measured, and for some
            # models the restore assumes a speedup the hardware does not
            # deliver. GLM-5.2-MXFP4's measured TP8 decode step comes out at
            # 0.89 to 1.00 of its own TP4 anchor's across concurrency 16 to 64
            # -- that is, essentially no gain from doubling the width -- while
            # DeepSeek-V4-Flash-0731, scored against a width-matched anchor,
            # sits within 3% on TPOT at every concurrency from 2 to 256. A
            # width-matched anchor is the fix; short of one, the size of this
            # term is unknown rather than small, so it is said out loud.
            if self._bench_tp != max(1, self.cfg.model_parallel_config.tensor_model_parallel_size):
                print(
                    f"[inferasim:Inference] WARNING: decode sweep was measured "
                    f"at TP{self._bench_tp} and is being restored to TP"
                    f"{self.cfg.model_parallel_config.tensor_model_parallel_size}. "
                    f"How a decode step scales with width is modelled here, "
                    f"not measured, and it is not the same across models: one "
                    f"MoE in this corpus gains nothing at all from TP4 to TP8. "
                    f"Harvest an anchor at the target width before trusting "
                    f"TPOT from this run."
                )
            # Batch transport interpolates the MEASURED curve and never falls
            # back to the simulator. That requires a batch sweep (>= 2 points);
            # a single-batch artifact would force a flat hold. Warn so the
            # benchmark is (re)run with a sweep rather than silently degrading.
            for _ph in ("prefill", "decode"):
                if len(self._meas_whole.get(_ph, [])) < 2:
                    print(
                        f"[inferasim:Inference] WARNING: {_ph} benchmark has a single "
                        f"batch point — batch transport will hold it flat. Re-run the "
                        f"benchmark with a batch sweep for an accurate {_ph} batch curve."
                    )
            # And having a sweep is not the same as having one that reaches the
            # batch being asked about. DeepSeek-V4-Pro's MI355X anchor sweeps
            # batch 1..64 and the agentic recipe runs concurrency 256, so the
            # curve is read 4x past its widest measured rung.
            #
            # The sign of that error is not stated here because it is not a
            # property of the anchor alone: against the same measured runs, the
            # closed-form path comes out 2.57x high at concurrency 256 while
            # trace replay of the same candidate comes out 0.74x, and both use
            # this curve. What is common to them is that the step cost past
            # batch 64 is modelled rather than measured. Which is the thing to
            # say, and the reason the fix is a harvest rather than a factor.
            _tgt_b = max(
                int(self.cfg.request_config.batch_size or 0),
                int(getattr(self.cfg.request_config, "max_concurrency", 0) or 0),
            )
            for _ph in ("prefill", "decode"):
                _pts = self._meas_whole.get(_ph, [])
                if len(_pts) < 2 or _tgt_b <= 0:
                    continue
                _widest = max(int(b) for b, _ in _pts)
                if _tgt_b > _widest:
                    print(
                        f"[inferasim:Inference] WARNING: {_ph} was swept to batch "
                        f"{_widest} and this config runs {_tgt_b}. The curve is being "
                        f"read {_tgt_b / _widest:.1f}x past its widest measured rung, "
                        f"so the step cost out here is modelled rather than measured. "
                        f"Harvest with --inference-batch-size {_tgt_b} (or "
                        f"--benchmark-batches up to {_tgt_b}) before trusting "
                        f"throughput or TPOT from this run."
                    )
            # Per-token prefill rate (for sub-prompt chunk pieces): full-prompt
            # prefill of ``b`` seqs processes ``b * ref_input`` tokens.
            if pre_pts and ref_input > 0:
                # Sharded by the parallelism ratio rather than by the
                # simulator's prefill ratio. Prefill splits into per-token work
                # that shards and a per-step intercept that does not, and
                # harvesting this model at both TP4 and TP8 pins each half:
                # the rate went 0.034373 -> 0.017100 ms/token, a ratio of
                # 0.4975 against the 0.5 ideal sharding predicts, while the
                # intercept moved only 204.8 -> 176.4 ms. One multiplicative
                # ratio applied to the sum cannot honour both, and applying the
                # simulator's lands between them -- right for a model whose
                # prefill step is mostly per-token work, badly wrong for one
                # where the intercept dominates.
                rates = [ms / (b * ref_input) for b, ms in pre_pts_bench if b > 0]
                rate_bench = sum(rates) / len(rates) if rates else 0.0
                _diag = meta.get("prefill_anchor") or {}
                _probed = sorted(int(p.get("input_len") or 0) for p in (_diag.get("points") or []))
                # Recorded before the scale is asked for, not after it is used.
                # A second anchor registers its curve during setup, so leaving
                # this one until the rate is being written left exactly one
                # width on file at the moment the ratio between two widths was
                # wanted, and the floor silently never applied.
                _curve = _diag.get("curve_fit") or {}
                if _curve.get("ms_per_token"):
                    self._bench_prefill_curves[self._bench_tp] = _curve
                shard, fixed_shard = self._prefill_width_scale(_probed)
                self._meas_prefill_rate_ms_per_tok = rate_bench * shard
                # The per-step fixed cost the same probe measured. Its TTFT
                # difference across two prompt lengths yields a slope AND an
                # intercept; the sweep's prefill_ms carries only slope*tokens,
                # so billing a prefill step at rate*tokens drops the intercept
                # entirely. That is not a rounding error: on the MI355X anchor
                # it is 204.8 ms against a 281.6 ms marginal cost for an
                # 8192-token prompt, i.e. 42% of the step, and dropping it
                # under-read TTFT by a near-constant 0.65x at every measured
                # concurrency. Transported like the rest of the prefill curve,
                # since it is prefill work rather than the parallelism-invariant
                # launch overhead the decode floor describes.
                # Carried across parallelism unscaled, being the half that does
                # not shard. 204.8 -> 176.4 ms across TP4 and TP8 is a 14% drop
                # against a 50% one for the rate, so treating it as fixed costs
                # a little and treating it as shardable costs a lot. An anchor
                # harvested at the target TP needs none of this, and
                # --load-benchmark-scaling with a second parallelism measures
                # the split outright rather than assuming it.
                anchor_diag = meta.get("prefill_anchor") or {}
                curve = anchor_diag.get("curve_fit") or {}
                if curve:
                    # Harvested at four or more prompt lengths, so the split
                    # between fixed cost and per-token work is measured rather
                    # than guessed.
                    #
                    # Both per-token terms shard: they are compute, and the
                    # quadratic one is attention, whose heads divide across
                    # ranks like everything else. Only the intercept is held.
                    #
                    # Collapsed to one effective rate because the simulator
                    # bills a prefill step as rate*tokens, and (a + b*n)*n
                    # reproduces a*n + b*n^2 exactly -- but only at the n it is
                    # evaluated at, so that has to be the length being asked
                    # about rather than the length the anchor was harvested at.
                    # Evaluating at the anchor's 8192 and then billing 1024-token
                    # prompts charges them the marginal cost of a token arriving
                    # at position 8192, which for a convex curve is far too
                    # much: DeepSeek-V4-Flash-0731 came out +61.4% on TTFT at
                    # ISL 1024 against +2.5% at 8192, from one anchor that
                    # probed both lengths.
                    #
                    # Taken wholly from the probe, not blended with the sweep's
                    # rate, so the intercept and the slope come from the same
                    # measurement and cannot double-count each other.
                    a = float(curve.get("ms_per_token") or 0.0)
                    b = float(curve.get("ms_per_token_sq") or 0.0)
                    self._bench_prefill_curves[self._bench_tp] = curve
                    at_n = float(self.cfg.request_config.input_seq_len or ref_input or 1)
                    # The curvature is read no further out than it was measured.
                    # ``b`` carries GEMM efficiency improving with step width,
                    # so it is routinely negative -- DeepSeek-V4-Pro's TP4
                    # anchor fits -1.05e-6 over 1024..8192 at R2=0.98, which is
                    # a good fit and a fine local approximation. Continued as a
                    # quadratic it crosses zero at about 53.5k tokens, and at
                    # ISL 130000 the same curve prices prefill at -80.7 us/token:
                    # negative per-token work, which then reads out as a 2450
                    # tok/s prefill and a 480-second disaggregated TTFT, and
                    # ranked that config 31st on a metric where the trace puts
                    # it first. Past the last probed length the marginal rate is
                    # therefore held where the measurement left it rather than
                    # allowed to keep improving, which is the conservative half
                    # of the choice: it stops crediting efficiency nobody timed,
                    # and it leaves the fixed cost and the linear term exactly
                    # as fitted. An anchor that brackets the target length needs
                    # none of this and the warning below names that re-harvest.
                    _probed_n = [
                        int(p.get("input_len") or 0) for p in (anchor_diag.get("points") or [])
                    ]
                    _curv_n = at_n
                    if _probed_n and at_n > max(_probed_n):
                        _curv_n = float(max(_probed_n))
                    rate_at_n = a + b * _curv_n
                    # The length sweep runs at concurrency 1, so every point it
                    # holds has one sequence in the step and the step's token
                    # count is that sequence's attention context. GEMM
                    # efficiency, which improves as the step widens, and
                    # attention, which grows with context, are therefore the
                    # same variable in those points and no fit over them can
                    # separate the two -- ``b`` absorbs whatever narrow-step
                    # inefficiency is there, and ``a`` settles near the rate at
                    # the narrow end. Reading the result back for a step that
                    # packs many short sequences then charges them attention
                    # they do not have and a GEMM rate they beat.
                    #
                    # Which is the corpus error exactly: one anchor, scored at
                    # ISL 8192 where a 16384-token budget packs two sequences,
                    # is within 5% on TTFT at every concurrency from 2 to 256;
                    # at ISL 1024, where the same budget packs sixteen, it
                    # over-reads by 39% at 8 concurrent and 157% at 256, while
                    # its TPOT and throughput stay right.
                    _pk_l0 = float((anchor_diag.get("packed") or {}).get("seq_len") or at_n or 0.0)
                    packed = _usable_packed_probe(
                        anchor_diag.get("packed"),
                        a + b * _pk_l0,
                        seq_cost_at=lambda n: (
                            float(curve.get("fixed_ms") or 0.0) + a * n + b * n * n
                        ),
                    )
                    budget = float(
                        getattr(self.cfg.request_config, "max_num_batched_tokens", 0) or 0
                    )
                    seqs = (budget / at_n) if (budget > 0 and at_n > 0) else 1.0
                    # A packed rate describes a step holding several sequences.
                    # When the budget admits one -- ISL 8192 against an
                    # 8192-token budget, which is most of the corpus's long
                    # prompts -- the step the engine builds is the step the
                    # length sweep measured, and the packed rate is an answer
                    # to a question nobody asked. It was being applied anyway:
                    # GLM-5.2-MXFP4's 39 single-sequence configs were billed
                    # 290 ms for a step its own curve prices at 446 and the
                    # server takes about 509.
                    if seqs < 1.5:
                        packed = None
                    # And it describes a step built from sequences of the
                    # length it probed. Re-centring by the curve's ``b`` below
                    # adds back the attention a longer context would cost, but
                    # that is a correction, not a transport: across an 8x
                    # change in context it does not hold. GLM-5.2-MXFP4's probe
                    # is taken at 1024 and 32 of its configs run at 8192, where
                    # the re-centred rate prices a 16384-token step at 696 ms
                    # against the 1017 the server takes, while simply reading
                    # two sequences off the curve lands at 892. Every model the
                    # packed term measurably helps -- MiniMax-M2.7,
                    # DeepSeek-V4-Flash, Qwen3-14B-FP8, DeepSeek-R1-0528 --
                    # runs at the length its probe was taken at.
                    # It also describes steps no wider than the widest it
                    # timed. The probe fits a rate *because* the rate moves
                    # with step width, so reading it past the last rung is
                    # extrapolating the one thing it exists to measure.
                    if packed and budget > 0:
                        _wide = max(
                            (int(q.get("step_tokens") or 0) for q in (packed.get("points") or [])),
                            default=0,
                        )
                        if _wide > 0 and budget > 1.5 * _wide:
                            print(
                                f"[inferasim:Inference] WARNING: this config's "
                                f"token budget builds steps of {int(budget)} "
                                f"tokens and the packed probe timed nothing "
                                f"wider than {_wide}. Using the single-"
                                f"sequence curve rather than reading the "
                                f"packed rate past its last measured step."
                            )
                            packed = None
                    if packed:
                        _l0 = float(packed.get("seq_len") or 0.0)
                        if _l0 > 0 and not (0.5 <= at_n / _l0 <= 2.0):
                            print(
                                f"[inferasim:Inference] WARNING: the packed "
                                f"prefill probe holds {int(_l0)}-token "
                                f"sequences and this config runs at "
                                f"{int(at_n)}. Attention per token differs by "
                                f"{at_n / _l0:.1f}x between them, which is too "
                                f"far to carry a packed rate across. Using the "
                                f"single-sequence curve; re-harvest the packed "
                                f"probe at this prompt length."
                            )
                            packed = None
                    if packed and packed.get("ms_per_token"):
                        # Measured in the regime being billed. Re-centred on
                        # this prompt length by the curve's own ``b``, which is
                        # the only measured statement about context dependence
                        # available; at the probe's own length the correction
                        # vanishes and the rate is used as measured.
                        l0 = float(packed.get("seq_len") or at_n)
                        p = float(packed["ms_per_token"])
                        rate_at_n = (p - b * l0) + b * at_n
                        print(
                            f"[inferasim:Inference] packed prefill rate "
                            f"{p * 1000:.1f} us/token measured at {int(l0)}-token "
                            f"sequences, re-centred to {int(at_n)} as "
                            f"{rate_at_n * 1000:.1f} us/token (the "
                            f"single-sequence curve reads "
                            f"{(a + b * at_n) * 1000:.1f} here)."
                        )
                    elif seqs > 2.0:
                        print(
                            f"[inferasim:Inference] WARNING: a step of "
                            f"{int(budget)} tokens packs about "
                            f"{seqs:.0f} sequences at ISL {int(at_n)}, but this "
                            f"anchor only probed prefill one sequence at a "
                            f"time. Its per-token rate therefore carries "
                            f"attention for a {int(at_n)}-token context and "
                            f"GEMM efficiency for a {int(at_n)}-token step, and "
                            f"the second of those is wrong by however much "
                            f"wider the real step is -- TTFT will be "
                            f"over-predicted, increasingly so with "
                            f"concurrency. Re-harvest with "
                            f"--prefill-packed-points 4 to measure it."
                        )
                    # Per-token prefill compute cannot be negative or zero
                    # however the fit was shaped, so this is a floor on the
                    # arithmetic rather than a modelling choice. Placed after
                    # the packed branch because that one re-centres by ``b`` too
                    # and would otherwise escape it. Held at the rate of the
                    # shortest length probed: the slowest per-token cost the
                    # anchor actually measured.
                    if rate_at_n <= 0.0:
                        _floor = a + b * float(min(_probed_n)) if _probed_n else a
                        print(
                            f"[inferasim:Inference] WARNING: the prefill curve "
                            f"prices per-token work at {rate_at_n * 1000:.1f} "
                            f"us/token, which is not physical. Holding it at "
                            f"{max(1e-9, _floor) * 1000:.1f} us/token, the "
                            f"slowest rate this anchor measured. Re-harvest a "
                            f"probe that brackets ISL {int(at_n)}."
                        )
                        rate_at_n = max(1e-9, _floor)
                    self._meas_prefill_rate_ms_per_tok = rate_at_n * shard
                    # The intercept stays the curve's even when the rate comes
                    # from the packed probe, which looks like mixing two fits
                    # and is worth saying why it is not. The probe's own
                    # intercept carries what it costs to ask: it times p99 TTFT
                    # over a client-side wave, and reads 68.1 ms for the single
                    # 1024-token sequence that the length sweep times at 31.8.
                    # That 36 ms is the measurement, not the step. Substituting
                    # it moved DeepSeek-V4-Flash from +1.1% to +43.2% on TTFT
                    # and Qwen3-14B-FP8 from -4.4% to -49.3%.
                    self._meas_prefill_fixed_ms = float(curve.get("fixed_ms") or 0.0) * fixed_shard
                    probed = _probed_n
                    # Said out loud when the curve is being read outside the
                    # span it was fitted over, where a quadratic stops being a
                    # local approximation and starts being an extrapolation.
                    if probed and not (min(probed) <= at_n <= max(probed)):
                        print(
                            f"[inferasim:Inference] WARNING: prefill curve was "
                            f"fitted over {min(probed)}..{max(probed)} tokens "
                            f"and is being evaluated at {int(at_n)}. Outside "
                            f"that span the quadratic is an extrapolation; "
                            f"re-harvest with a probe that brackets this "
                            f"prompt length."
                        )
                    if _curv_n != at_n:
                        print(
                            f"[inferasim:Inference] prefill curvature held at "
                            f"{int(_curv_n)} tokens, the longest prompt the "
                            f"curve was fitted over, rather than continued to "
                            f"{int(at_n)}: the quadratic term would read "
                            f"{(a + b * at_n) * 1000:.1f} us/token there."
                        )
                    print(
                        f"[inferasim:Inference] prefill curve fit: "
                        f"{self._meas_prefill_fixed_ms:.1f} ms fixed (x{fixed_shard:.3f}) + "
                        f"{rate_at_n * 1000:.1f} us/token at "
                        f"{int(_curv_n)} tokens, sharded by {shard:.3f}; "
                        f"R2={curve.get('r2')}"
                    )
                    self._fit_decode_kv_slope(
                        benchmark_layer_times.get("decode_ctx") or [],
                        float(ref_input),
                    )
                    return
                fixed = float(anchor_diag.get("implied_fixed_ms") or 0.0)
                if fixed > 0.0:
                    # Chord-fitted anchor: this "fixed" is part genuine fixed
                    # cost and part prefill's curvature in prompt length, which
                    # a two-point fit cannot tell apart and folds into the
                    # intercept. It is held whole across parallelism, which is
                    # what the name claims and what needs no calibration, but
                    # only the genuine half deserves that -- so the restore is
                    # biased by however much curvature got swallowed.
                    # --prefill-anchor-points replaces the whole business with
                    # a measurement; a same-width restore avoids it too, since
                    # shard is then 1 and the term is a no-op.
                    self._meas_prefill_fixed_ms = fixed * fixed_shard
                    print(
                        f"[inferasim:Inference] measured prefill: "
                        f"{self._meas_prefill_rate_ms_per_tok * 1000:.1f} us/token "
                        f"+ {self._meas_prefill_fixed_ms:.1f} ms fixed per step."
                    )
                    if abs(shard - 1.0) > 1e-9:
                        print(
                            f"[inferasim:Inference] WARNING: restoring a "
                            f"chord-fitted anchor across parallelism "
                            f"(shard={shard:.3f}). Its {fixed:.1f} ms "
                            f"intercept mixes fixed cost with prompt-length "
                            f"curvature and is being held whole, which is "
                            f"right for the first and wrong for the second. "
                            f"Re-harvest with --prefill-anchor-points 5 to fit "
                            f"the curve and measure the split."
                        )
            self._fit_decode_kv_slope(
                benchmark_layer_times.get("decode_ctx") or [], float(ref_input)
            )
            return

        # Per-layer schema (Megatron worker): measured forward time of one dense
        # and one MoE layer per phase. Used directly, composed by layer counts.
        layer: dict[tuple, float] = {}
        for ltype in ("dense", "moe"):
            entry = measured.get(ltype)
            if not entry:
                continue
            if entry.get("prefill_ms"):
                layer[("prefill", ltype)] = float(entry["prefill_ms"])
            if entry.get("decode_ms"):
                layer[("decode", ltype)] = float(entry["decode_ms"])
        self._meas_layer = layer
        self._setup_restoration()
        # Prefill rate from the dominant (per-layer * count) prefill total.
        if ref_input > 0:
            full_pre = self._measured_full_prefill_ms(ref_batch)
            self._meas_prefill_rate_ms_per_tok = full_pre / (ref_batch * ref_input)

    def _setup_restoration(self) -> None:
        """Prepare restoration when a per-layer benchmark was captured at a
        different parallelism than the target (mirrors training's
        benchmark-at-fewer-GPUs → target extrapolation). Builds analytical
        collective models at the benchmark and target layout so
        ``_restore_per_layer`` can strip the benchmark's comm, scale the sharded
        compute, and add the target comm back.

        Attention-DP is one of the axes here, and the only one the fallback laws
        cannot express -- see the refusal at the end."""
        mp = self.cfg.model_parallel_config
        self._tgt_tp = max(1, mp.tensor_model_parallel_size)
        tgt_ep = max(1, getattr(mp, "expert_model_parallel_size", 1) or 1)
        tgt_pp = max(1, mp.pipeline_model_parallel_size)
        tgt_attn_dp = max(1, getattr(mp, "attention_data_parallel_size", 1) or 1)

        # An anchor that never recorded its attention layout cannot be checked
        # against a data-parallel target, and neither available guess is free:
        # calling it non-DP applies a ratio the measurement may already contain,
        # while calling it a match is precisely how an attn_dp=1 anchor gets
        # reused at attn_dp=8 unchanged. So it is reported instead, with the
        # re-harvest named, and the number is left as measured.
        bench_attn_dp = self._bench_attn_dp
        if bench_attn_dp is None:
            if tgt_attn_dp > 1:
                print(
                    f"[inferasim:Inference] anchor does not record the attention "
                    f"layout it was harvested at and the target runs "
                    f"attention-DP={tgt_attn_dp}; the measured step is being used "
                    f"as-is and is NOT transported across the layout change. "
                    f"Re-harvest to record it."
                )
            bench_attn_dp = tgt_attn_dp
        # Attention-DP subdivides the tensor-parallel group, so it cannot exceed
        # it; a recorded degree wider than the bench TP is a malformed artifact.
        bench_attn_dp = max(1, min(int(bench_attn_dp), self._bench_tp))

        layout_moved = bench_attn_dp != tgt_attn_dp
        self._restore = (
            self._bench_tp != self._tgt_tp
            or self._bench_ep != tgt_ep
            or self._bench_pp != tgt_pp
            or layout_moved
        )
        # Recorded because the fallback laws below cannot express it: they are
        # all functions of the GPU count, which attention-DP does not change.
        self._restore_layout_moved = layout_moved
        self._bench_attn_dp_eff = bench_attn_dp
        if not self._restore:
            return
        mc = self.cfg.model_config
        bench_mp = replace(
            mp,
            tensor_model_parallel_size=self._bench_tp,
            expert_model_parallel_size=self._bench_ep,
            pipeline_model_parallel_size=self._bench_pp,
            attention_data_parallel_size=bench_attn_dp,
        )
        self._comm_bench = InferenceCollectiveModel(mc, bench_mp, self._cc)
        self._comm_tgt = InferenceCollectiveModel(mc, mp, self._cc)

        # Per-view MoE routing imbalance for the origami ratio.  The target view
        # already uses ``self._moe_imbalance`` (constructor).  The bench view is
        # evaluated at the *bench* EP so that, e.g., an EP=1 bench (experts
        # TP-sharded, balanced -> 1.0) restored to an EP=8 target (all-to-all,
        # busiest-rank gated -> ep_load_balance) keeps the EP sharding penalty
        # in ``sim(target)/sim(bench)`` instead of cancelling it.
        self._imb_tgt = self._moe_imbalance
        self._imb_bench = self._moe_imbalance_for_ep(self._bench_ep)
        # Diagnostic escape hatch: INFERASIM_ORIGAMI_IMB_PERVIEW=0 restores the old
        # behaviour (single target imbalance on both views, which cancels in the
        # ratio) for before/after validation of the per-view fix.
        if os.getenv("INFERASIM_ORIGAMI_IMB_PERVIEW", "1").strip().lower() in ("0", "false", "no"):
            self._imb_bench = self._moe_imbalance

        # Origami-ratio setup: build guaranteed-simulating profiler trees at the
        # bench and target views so ``_restore_whole`` can scale the measured
        # anchor by the simulator's TP-scaling ratio (the absolute origami bias
        # cancels in the ratio). Benchmark mode may hold a metadata-only GEMM
        # backend, so build dedicated simulating backends here. On any failure
        # (e.g. no SDPA simulator for the arch) origami is disabled and the
        # restore falls back to the measured fit / blind TP^-1.
        self._view_tgt = self._view
        self._lm_ratio_bench = None
        if self._scaling_mode == "origami":
            try:
                self._gemm_sim = get_gemm_simulation_backend(
                    backend_name=self._gemm_name,
                    gpu_arch=self._gpu_arch,
                    gpu_clock_mhz=self._gpu_clock,
                    require_simulation=True,
                )
                self._sdpa_sim = get_sdpa_simulation_backend(
                    gpu_arch=self._gpu_arch,
                    gpu_clock_mhz=self._gpu_clock,
                )
                rc = self.cfg.request_config
                saved_mp = self.cfg.model_parallel_config
                self.cfg.model_parallel_config = bench_mp
                self._view_bench = self.cfg.as_training_config(
                    batch_size=rc.batch_size,
                    seq_len=rc.input_seq_len,
                )
                self.cfg.model_parallel_config = saved_mp
                self._lm_ratio_tgt = build_profiler(
                    get_language_model_profiler_spec(self._view_tgt)
                )
                self._lm_ratio_tgt.set_simulation_backends(self._gemm_sim, self._sdpa_sim)
                self._lm_ratio_bench = build_profiler(
                    get_language_model_profiler_spec(self._view_bench)
                )
                self._lm_ratio_bench.set_simulation_backends(self._gemm_sim, self._sdpa_sim)
            except Exception as e:  # pragma: no cover - arch-dependent
                self._lm_ratio_bench = None
                print(
                    f"[inferasim:Inference] origami-ratio unavailable ({e}); "
                    "falling back to measured fit / blind TP^-1."
                )

        # The simulator ratio is the only mechanism here that describes a change
        # of attention layout. Every other law is a function of the GPU count --
        # the measured TP fit, the ideal TP^-1 sharding -- and attention-DP does
        # not change the GPU count, so with the ratio gone they leave the layout
        # change entirely unpriced and report the result as transported anyway.
        # That holds whether or not TP also moved, so the refusal is on the
        # layout alone rather than on it being the only axis.
        if self._restore_layout_moved and self._lm_ratio_bench is None:
            if not os.getenv("INFERASIM_ALLOW_LAYOUT_MISMATCH"):
                raise ValueError(
                    f"The measured anchor was harvested at attention-DP="
                    f"{self._bench_attn_dp_eff} but the target runs attention-DP="
                    f"{tgt_attn_dp}, and the analytical ratio that would transport "
                    "that change is unavailable here (no simulating GEMM/SDPA "
                    "backend for this architecture, or INFERASIM_RESTORE_SCALING "
                    "set away from 'origami'). Data-parallel attention changes the "
                    "batch a rank holds and removes one all-reduce per layer, and "
                    "no remaining scaling law describes either. Harvest an anchor "
                    "at the target attention layout, project this point "
                    "analytically instead of from the anchor, or set "
                    "INFERASIM_ALLOW_LAYOUT_MISMATCH=1 to reuse it unchanged and "
                    "accept the error."
                )
            print(
                f"[inferasim:Inference] attention-DP {self._bench_attn_dp_eff} -> "
                f"{tgt_attn_dp} is NOT being transported "
                "(INFERASIM_ALLOW_LAYOUT_MISMATCH set); the measured step is "
                "reused unchanged and the decode number is wrong by whatever the "
                "layout change is worth."
            )

    def _comm_model_at_tp(self, tp: int, ep: int, pp: int) -> InferenceCollectiveModel:
        """Collective model at an arbitrary parallelism, for the scaling fit."""
        mp = self.cfg.model_parallel_config
        return InferenceCollectiveModel(
            self.cfg.model_config,
            replace(
                mp,
                tensor_model_parallel_size=max(1, tp),
                expert_model_parallel_size=max(1, ep),
                pipeline_model_parallel_size=max(1, pp),
            ),
            self._cc,
        )

    def _step_tokens(self, at_n):
        """How many tokens a prefill step actually carries.

        Not the prompt length. The scheduler fills a step to its token budget,
        so a 16384-token budget puts sixteen 1024-token prompts in one step and
        only two 8192-token ones -- and a ratio between two measured widths is
        a statement about steps, so it has to be read at the width of the step
        the scheduler will build rather than the length of one request in it.

        Getting this wrong is not a detail. Read at the request length instead,
        the floor lifts GLM-5.2-MXFP4's ISL-1024 TTFT from -34.0% to +0.8% at 8
        concurrent and then to +56.0% at 128, because at 128 the steps are
        sixteen prompts wide and dominated by per-token work that does shard,
        while a single 1024-token prompt looks like almost pure fixed cost.
        Concurrency caps it at the low end: two requests in flight cannot fill
        a step past two prompts however large the budget is.
        """
        budget = float(getattr(self.cfg.request_config, "max_num_batched_tokens", 0) or 0)
        # Falling back to batch size the way the rest of the codebase does:
        # ``max_concurrency`` is optional and means the batch when unset.
        rc = self.cfg.request_config
        conc = float(getattr(rc, "max_concurrency", 0) or getattr(rc, "batch_size", 0) or 0)
        if at_n <= 0:
            return at_n
        packed = at_n
        if conc > 0:
            packed = conc * at_n
        if budget > 0:
            packed = min(packed, budget) if conc > 0 else budget
        return max(at_n, packed)

    def _measured_ratio_at(self, tokens):
        """What two measured widths carried a ``tokens``-long prefill step by,
        or ``None``.

        Only returned when the two anchor widths are the same factor apart as
        bench is from target, because the two are compared and a ratio over a
        wider jump is not comparable to one over a narrower jump: four-fold
        sharding is not two applications of two-fold. Equal factors make it
        apples to apples -- TP2 -> TP4 against TP4 -> TP8 -- and anything else
        is declined rather than approximated.
        """
        curves = self._bench_prefill_curves
        if len(curves) < 2 or tokens <= 0:
            return None
        lo, hi = min(curves), max(curves)
        if lo <= 0 or hi <= lo:
            return None
        if abs((hi / lo) - (self._tgt_tp / self._bench_tp)) > 1e-9:
            return None

        def step(c):
            return (
                float(c.get("fixed_ms") or 0.0)
                + float(c.get("ms_per_token") or 0.0) * tokens
                + float(c.get("ms_per_token_sq") or 0.0) * tokens * tokens
            )

        at_lo, at_hi = step(curves[lo]), step(curves[hi])
        if at_lo <= 0.0 or at_hi <= 0.0:
            return None
        return at_hi / at_lo, lo, hi

    def _measured_width_ratio(self):
        """The two-anchor fit's answer for ``bench -> target``, or ``None``.

        Averaged over batch exactly as ``rate_bench`` is, so the ratio being
        applied describes the same set of points the rate was taken from.
        """
        fits = self._bench_scaling_fit.get("prefill") or {}
        ratios = []
        for _batch, (shardable, invariant) in sorted(fits.items()):
            at_bench = shardable / self._bench_tp + invariant
            at_tgt = shardable / self._tgt_tp + invariant
            if at_bench > 0.0 and at_tgt > 0.0:
                ratios.append(at_tgt / at_bench)
        return sum(ratios) / len(ratios) if ratios else None

    def _prefill_width_scale(self, probed=None) -> tuple:
        """``(rate_scale, fixed_scale)`` carrying prefill from the anchor's width
        to the target's.

        Three laws, in the order they are preferred:

        **The simulator's own ratio** (``origami``, the default). Evaluates a
        prefill step at both widths analytically and moves the measured anchor
        by ``sim(target) / sim(bench)``. The absolute analytical cost is not
        trusted anywhere here -- it is several-fold off -- but it cancels in the
        ratio, leaving a shape that knows what the other two laws cannot: how
        wide each rank's GEMMs end up and what the collectives cost there.

        **A fit through two measured widths**, if a second anchor was supplied.

        **Ideal sharding**, ``bench_tp / tgt_tp``, with a warning.

        Ideal is preferred by neither and is only a floor, because whether it
        holds is a property of the model that one anchor width cannot reveal.
        Measured at TP4 and TP8, DeepSeek's whole prefill step carries at 0.651
        and GLM-5.2-MXFP4's at 0.825 -- GLM barely gains from the extra width at
        all, because its per-token work is MXFP4 expert GEMMs already too narrow
        at TP4 to fill the matrix cores. Against those two:

            law              DeepSeek   GLM-5.2
            ideal sharding     +9.2%     -32.6%
            two-width fit         --     -25.8%
            simulator ratio    -4.4%      -6.9%

        The two-width fit losing to the simulator is not an accident of GLM. Its
        anchors are at TP2 and TP4, and GLM shards at 0.563 between those before
        falling off a cliff to 0.825 above them, so every point the fit can see
        says scaling is healthy. Extrapolating from below the target cannot find
        a breakdown that only happens above it; evaluating the target width
        directly can.

        ``fixed_scale`` is 1.0 for the latter two laws, which shard the
        per-token half and hold the per-step intercept -- right for them,
        because ideal sharding is a statement about per-token compute.

        The simulator's ratio instead applies to **both** halves, and that is
        deliberate and hard-won. The prefill probe separates a step cleanly:
        its slope times the token count reproduces the engine's own prefill
        time to four figures (DeepSeek-V4-Pro at TP4, 46.68 us/tok x 8192 =
        382.4 ms against 382.43 measured), so the slope is engine work and the
        intercept is host and queueing cost outside it. It is tempting to carry
        those two separately, and the simulator will offer a ratio for each.
        Do not: it gets the whole step right and the split wrong.

            DeepSeek-V4-Pro, TP4 -> TP8      simulator   measured
            per-token (engine) work            x0.665     x0.365
            per-step (host) cost               x0.520     x1.630
            whole step                         x0.622     x0.648

        Its per-token work shards *better* than ideal while its host cost rises
        by nearly two thirds, and the simulator has both backwards. The whole
        step survives because the two errors are opposed and cancel. Carrying
        only the per-token half and holding the intercept reads +14.7% here,
        and +20.1% on the earlier whole-prefill observable, against -4.4% for
        the blended ratio.

        Llama-3.1-8B does not contradict this, it merely cannot see it: its
        host cost is 8.06 +/- 2.49 ms across TP1..TP8, about 8% of a step, so
        any split is nearly the same number. Splitting reads +1.6% there
        against +4.9% blended -- a real but small gain, bought at the price of
        +14.7% on a model where the intercept is 22% of the step. And on GLM
        the split costs more still: scored through the scheduler against
        measured TP8 at ISL 1024, the median error goes 18.2% -> 24.4%.

        A split is the right shape and will be worth having once each half can
        be calibrated against something. Neither can today: the engine slope is
        measured only at one sequence per step while the scheduler packs
        sixteen at ISL 1024, and every anchor's batch sweep is derived by
        multiplication rather than measured, so nothing on disk constrains the
        packed regime. That is what ``--prefill-packed-points`` is for.
        """
        if not self._restore:
            return 1.0, 1.0
        ideal = self._bench_tp / self._tgt_tp
        at_n = int(self.cfg.request_config.input_seq_len or self._meas_ref_input or 1)
        measured = self._measured_width_ratio()
        if self._scaling_mode == "origami":
            # The ISL probe measures a prefill step as a per-step cost plus a
            # per-token rate. Those two halves answer to width completely
            # differently, so they are carried differently: only the per-token
            # half is scaled, and the simulator is asked for that ratio alone.
            #
            # Llama-3.1-8B, measured at every width from one GPU to eight,
            # shows why. Its per-step cost is 7.3, 5.5, 7.9, 11.5 ms at TP1,
            # 2, 4 and 8 -- it never shards, and drifts slightly upward as
            # collectives are added. Its per-token rate is 17.0, 19.3, 11.2
            # and 7.0 us/tok. Transporting its TP4 anchor to TP8 and checking
            # against the TP8 anchor actually measured (68.8 ms at 8192 tok):
            #
            #     law                                          err
            #     ideal sharding                             -21.9%
            #     whole-step ratio on everything              +4.9%
            #     simulator's split, both halves              +2.0%
            #     simulator's per-token ratio, cost held      +1.6%
            #     fitted from measured TP2 + TP4             -11.2%
            #     fitted from measured TP1 + TP4             +19.7%
            #
            # Two things fall out. Holding the per-step cost beats scaling it
            # by anything, including the simulator's own guess at it -- which
            # is not even self-consistent across models, claiming 1.034 here
            # and 0.520 on GLM-5.2-MXFP4 while measurement says it rises.
            #
            # And a second measured width does not help: it hurts. Worst of
            # all is the widest span, TP1 + TP4, because TP1 runs no
            # collectives whatsoever -- TP1 -> TP2 carries per-token work at
            # 1.135, costing *more* on two GPUs than on one. A fit through
            # that reads the one-off arrival of communication as a trend and
            # extrapolates it forever. TP2 + TP4 avoids that and is still
            # worse than the simulator, because two points cannot see a
            # threshold: GLM-5.2-MXFP4 shards near-ideally to TP4 and then
            # stalls, and nothing measured at TP4 or below contains that.
            lo, hi = None, None
            if probed and len(probed) >= 2 and probed[-1] > probed[0]:
                lo, hi = probed[0], probed[-1]
            elif at_n > 2048:
                lo, hi = 1024, at_n
            if lo is not None:
                s_lo = self._origami_steps(1, lo, "prefill")
                s_hi = self._origami_steps(1, hi, "prefill")
                if s_lo and s_hi:
                    (t_lo, b_lo), (t_hi, b_hi) = s_lo, s_hi
                    m_t = (t_hi - t_lo) / (hi - lo)
                    m_b = (b_hi - b_lo) / (hi - lo)
                    f_t, f_b = t_lo - m_t * lo, b_lo - m_b * lo
                    if m_t > 0.0 and m_b > 0.0 and f_b > 0.0:
                        print(
                            f"[inferasim:Inference] prefill width split, for "
                            f"the record only: TP{self._bench_tp} -> "
                            f"TP{self._tgt_tp} per-token x{m_t / m_b:.3f} "
                            f"({m_b * 1000:.3f} -> {m_t * 1000:.3f} us/tok), "
                            f"per-step x{f_t / f_b:.3f} ({f_b:.2f} -> "
                            f"{f_t:.2f} ms). Neither is used on its own; see "
                            f"_prefill_width_scale for why."
                        )
            steps = self._origami_steps(1, at_n, "prefill")
            if steps:
                s_tgt, s_bench = steps
                if s_bench > 0.0 and s_tgt > 0.0:
                    r = s_tgt / s_bench
                    floored = ""
                    at_len = self._measured_ratio_at(self._step_tokens(at_n))
                    if at_len is not None and at_len[0] > r:
                        # The same guard as below, but read at the prompt
                        # length being asked about instead of wherever the
                        # anchors were harvested, which is the only way the
                        # comparison means anything: the measured ratio moves
                        # from 0.936 to 0.564 across GLM-5.2-MXFP4's length
                        # range, so a single number for it is right at one
                        # length and wrong everywhere else.
                        #
                        # Verified against Llama-3.1-8B, measured at TP2, TP4
                        # and TP8. At every length the narrower jump carries
                        # less than the wider one, with margin:
                        #
                        #     tokens   TP2->TP4   TP4->TP8
                        #       1024      0.711      1.003
                        #       2048      0.685      0.837
                        #       4096      0.649      0.724
                        #       8192      0.603      0.703
                        #
                        # Note the top right: at 1024 tokens this model gains
                        # nothing at all from TP8 over TP4, because the step is
                        # almost entirely the cost that does not shard. Any law
                        # that reports sharding there is wrong, and the floor is
                        # what notices.
                        rf, lo, hi = at_len
                        r, floored = (
                            rf,
                            (
                                f" Raised to {rf:.3f}, which is what TP{lo} and "
                                f"TP{hi} measured for a {at_n}-token step and "
                                f"sharding cannot beat."
                            ),
                        )
                    elif measured is not None and measured > r:
                        # Sharding does not improve as a model is spread wider.
                        # Each doubling adds collectives and halves every GEMM
                        # again, so the fraction of a step that responds to
                        # width only falls: measured per-token prefill carries
                        # at 0.580 from TP2 to TP4 on Llama-3.1-8B and then
                        # 0.625 from TP4 to TP8, and GLM-5.2-MXFP4 goes 0.507
                        # and then 0.68 or worse. A ratio below what the anchor
                        # widths already demonstrated is therefore claiming a
                        # gain the hardware has never shown, so the measured
                        # one becomes a floor.
                        #
                        # This is a guard, not a correction. On both models
                        # where two widths exist the simulator is already above
                        # the floor -- 0.768 against 0.507, 0.676 against 0.580
                        # -- so it does not bind today. It exists because the
                        # opposite reduction is tempting and would be a
                        # disaster: taking the *lower* of the two hands GLM
                        # 0.507 against a true 0.68 to 0.825, which is the
                        # ideal-sharding error that read its TTFT 39% low.
                        r, floored = (
                            measured,
                            (
                                f" Raised to the {measured:.3f} its own anchor "
                                f"widths measured, which sharding cannot beat."
                            ),
                        )
                    print(
                        f"[inferasim:Inference] prefill width scaling from the "
                        f"simulator's whole-step TP{self._bench_tp} -> "
                        f"TP{self._tgt_tp} ratio: {r:.3f} (ideal sharding would "
                        f"be {ideal:.3f}).{floored}"
                    )
                    return r, r
        if measured is not None:
            print(
                f"[inferasim:Inference] prefill width scaling fitted through "
                f"two anchor widths: TP{self._bench_tp} -> TP{self._tgt_tp} "
                f"shards by {measured:.3f} (ideal would be {ideal:.3f})."
            )
            return measured, 1.0
        print(
            f"[inferasim:Inference] WARNING: per-token prefill work is assumed "
            f"to shard ideally from TP{self._bench_tp} to TP{self._tgt_tp} (by "
            f"{ideal:.3f}). That is an assumption, not a measurement, and one "
            f"anchor width cannot check it: GLM-5.2-MXFP4 actually carries its "
            f"prefill at 0.825, and assuming ideal read its TTFT 39% low and "
            f"its throughput 27% high across all 56 of its configs. Neither "
            f"the simulator ratio nor a second anchor width was available here."
        )
        return ideal, 1.0

    def _fit_tp_scaling(self, phase: str) -> None:
        """Fit ``compute(tp) = shardable / tp + invariant`` per batch size.

        Least squares in ``1/tp`` over the points registered by
        ``add_scaling_benchmark``; needs at least two parallelisms.
        """
        pts = self._bench_scaling_raw.get(phase) or {}
        fits = {}
        for batch, per_tp in pts.items():
            if len(per_tp) < 2:
                continue
            xs, ys = [], []
            for tp, (ms, ep, pp) in sorted(per_tp.items()):
                tokens = 1 if phase == "decode" else self._meas_ref_input
                cm = self._comm_model_at_tp(tp, ep, pp)
                dense = (
                    cm.layer_comm_ms(batch, tokens, is_moe=False).total_ms if self._n_dense else 0.0
                )
                moe = cm.layer_comm_ms(batch, tokens, is_moe=True).total_ms if self._n_moe else 0.0
                comm = self._n_dense * dense + self._n_moe * moe
                xs.append(1.0 / tp)
                ys.append(max(0.0, ms - comm))
            n = len(xs)
            sx, sy = sum(xs), sum(ys)
            sxx = sum(x * x for x in xs)
            sxy = sum(x * y for x, y in zip(xs, ys))
            det = n * sxx - sx * sx
            if abs(det) < 1e-12:
                continue
            shardable = (n * sxy - sx * sy) / det
            invariant = (sy - shardable * sx) / n
            if shardable <= 0.0:
                # No positive TP-shardable component — not a usable fit.
                continue
            if invariant < 0.0:
                # Near-/super-linear scaling. This is expected for a close pair
                # such as TP=1 + TP=2, where the TP-invariant remainder is still
                # masked by compute at low TP and the fit's intercept dips
                # slightly negative. Rather than discard the two measured anchors
                # (and fall back to a blind TP^-1), clamp the invariant to 0 and
                # refit ``shardable`` as least-squares through the origin. The law
                # then reproduces both measured anchors and scales as ~TP^-1 —
                # exact where measured, optimistic for TP well above the range.
                invariant = 0.0
                shardable = sxy / sxx if sxx > 0.0 else shardable
            fits[batch] = (shardable, invariant)
        if fits:
            self._bench_scaling_fit[phase] = fits

    @staticmethod
    def _anchor_floor_from(blob: dict) -> float:
        """The anchor's parallelism-invariant decode floor, from its own sweep.

        A decode step is a fixed per-step cost (kernel dispatch, occupancy,
        resident collectives) plus work that shards with parallelism. At the
        smallest batch the shardable part is nearly nothing, so the cheapest
        step in the sweep is essentially the fixed cost alone -- and being
        fixed, it is the same at the target parallelism, which is what lets a
        sub-node anchor reach a full-node target.

        Returns 0.0 unless the sweep has at least two batch points: a single
        measurement is not a floor, it is just that measurement, and treating
        it as one would pin the whole curve to it.
        """
        pts = [float(e["decode_ms"]) for e in (blob.get("sweep") or []) if e.get("decode_ms")]
        return min(pts) if len(pts) >= 2 else 0.0

    def _report_tp_scaling(self) -> None:
        """Print the TP-scaling law the restore will use."""
        if self._restore and getattr(self, "_anchor_decode_floor_ms", 0.0) > 0.0:
            print(
                f"[inferasim:Inference] TP scaling (decode): floor-preserving — "
                f"holding the anchor's measured {self._anchor_decode_floor_ms:.2f} ms "
                f"invariant floor fixed and sharding only the excess from TP="
                f"{self._bench_tp} to TP={self._tgt_tp}."
            )
        if (
            self._scaling_mode == "origami"
            and self._restore
            and getattr(self, "_lm_ratio_bench", None) is not None
        ):
            # Names the phases it still governs: when the decode floor law is
            # active the origami ratio carries prefill only, and saying
            # otherwise would credit it for a decode step it never touched.
            phases = (
                "prefill"
                if getattr(self, "_anchor_decode_floor_ms", 0.0) > 0.0
                else "prefill + decode"
            )
            print(
                f"[inferasim:Inference] TP scaling ({phases}): origami-ratio (simulate "
                f"vLLM-fused MoE) — scaling measured TP={self._bench_tp} anchor to TP="
                f"{self._tgt_tp} by sim(target)/sim(bench)."
            )
            return
        fits = self._bench_scaling_fit.get("decode") or {}
        if fits:
            # Largest batch: the concurrency the step is usually judged at.
            batch = max(fits)
            shardable, invariant = fits[batch]
            tps = sorted((self._bench_scaling_raw.get("decode") or {}).get(batch, {}))
            total = shardable + invariant
            print(
                f"[inferasim:Inference] TP scaling fitted from benchmark TP="
                f"{','.join(str(t) for t in tps)} at batch {batch}: "
                f"{shardable:.2f} ms shardable + {invariant:.2f} ms TP-invariant"
                f" ({invariant / total * 100:.0f}% does not shrink with TP)"
            )
            # Interpolation between measured anchors is exact; extrapolation
            # ABOVE the measured range is only as good as the fitted invariant,
            # which two low-TP anchors cannot fully resolve (the non-shrinking
            # floor is still masked at low TP). Flag it so an out-of-range TP is
            # treated as lower-confidence rather than trusted like a measurement.
            max_tp = max(tps) if tps else self._bench_tp
            if self._tgt_tp > max_tp:
                print(
                    f"[inferasim:Inference] WARNING: target TP={self._tgt_tp} is ABOVE the "
                    f"measured range (TP<={max_tp}); this decode step is EXTRAPOLATED and "
                    f"tends to under-predict latency / over-predict throughput. Add a "
                    f"benchmark at TP>={self._tgt_tp} to make it exact."
                )
            return
        if self._bench_tp != self._tgt_tp:
            print(
                f"[inferasim:Inference] WARNING: restoring benchmark TP="
                f"{self._bench_tp} to TP={self._tgt_tp} assuming the whole step is"
                " TP-shardable and scales as TP^-1. Real decode steps keep a"
                " TP-invariant remainder, so this under-predicts step latency."
                " Pass --load-benchmark-scaling with a run at another"
                " --benchmark-gpus to fit the split instead."
            )

    def add_scaling_benchmark(self, blob: dict) -> None:
        """Register a benchmark artifact taken at a different TP, for the fit only."""
        measured = blob.get("measured", blob)
        meta = blob.get("meta", {})
        tp = int(meta.get("benchmark_tp") or meta.get("tp") or 1)
        ep = int(meta.get("benchmark_ep") or meta.get("ep") or 1)
        pp = int(meta.get("benchmark_pp") or meta.get("pp") or 1)
        sweep = blob.get("sweep") or []
        model_step = measured.get("model") or {}
        ref_batch = int(meta.get("batch") or self.cfg.request_config.batch_size or 1)
        for phase, key in (("decode", "decode_ms"), ("prefill", "prefill_ms")):
            rows = [
                (int(e["batch"]), float(e[key]))
                for e in sweep
                if e.get("batch") is not None and e.get(key)
            ]
            if not rows and model_step.get(key):
                rows = [(ref_batch, float(model_step[key]))]
            for batch, ms in rows:
                self._bench_scaling_raw.setdefault(phase, {}).setdefault(batch, {})[tp] = (
                    ms,
                    ep,
                    pp,
                )
        # The length curve too, not just the step times. A ratio between two
        # measured widths is strongly prompt-length dependent -- GLM-5.2-MXFP4
        # carries TP2 -> TP4 at 0.936 for a 1024-token prompt and 0.564 for an
        # 8192-token one, because a short step is mostly the per-step cost that
        # does not shard -- so comparing it against anything requires
        # evaluating it at the length being asked about rather than wherever
        # the anchors happened to be harvested.
        curve = ((meta.get("prefill_anchor") or {}).get("curve_fit")) or None
        if curve and curve.get("ms_per_token"):
            self._bench_prefill_curves[tp] = curve

    def _restore_per_layer(self, ltype: str, ms_bench: float, batch: int, tokens: int) -> float:
        """Restore a per-layer time measured at the benchmark's (reduced) TP/EP to
        the target parallelism, training-style: the shardable compute scales by
        ``bench_tp / target_tp`` and the blocking collective at the target TP/EP
        is added back analytically. No-op when bench and target parallelism match
        or for the whole-model (vLLM) schema, which is non-decomposable."""
        if not self._restore or ms_bench <= 0.0:
            return ms_bench
        is_moe = ltype == "moe"
        comm_bench = self._comm_bench.layer_comm_ms(batch, tokens, is_moe=is_moe).total_ms
        comm_tgt = self._comm_tgt.layer_comm_ms(batch, tokens, is_moe=is_moe).total_ms
        compute = max(0.0, ms_bench - comm_bench) * (self._bench_tp / self._tgt_tp)
        return compute + comm_tgt

    def _restore_pp_ms(self, batch: int, tokens: int) -> float:
        """Per-*forward* pipeline P2P delta added when restoring to a target PP.
        PP distributes whole layers across stages, so it adds ``(pp-1)``
        send/recv hops to a forward pass without sharding compute — hence a
        single additive term per step, not a per-layer one. Reuses the shared
        ``cm.sendrecv`` primitive via ``InferenceCollectiveModel.pp_p2p_ms``."""
        if not self._restore:
            return 0.0
        return self._comm_tgt.pp_p2p_ms(batch, tokens) - self._comm_bench.pp_p2p_ms(batch, tokens)

    def _origami_steps(self, batch: int, tokens: int, phase: str) -> tuple | None:
        """Simulated whole-step time at the target and bench views, ``(tgt, bench)``.

        Reuses the analytical ``_forward_times`` at the bench and target views by
        temporarily swapping the profiler tree / comm model / view / sim backends
        (the analytical path is otherwise unused in benchmark mode). Returns
        ``None`` when origami is unavailable so the caller falls back to the fit.
        """
        if getattr(self, "_lm_ratio_bench", None) is None:
            return None
        if phase == "prefill":
            q_len, kv = max(1, tokens), max(1, tokens)
        else:
            q_len = 1
            kv = max(1, self._meas_ref_input or self.cfg.request_config.input_seq_len or 1024)
        # Explicit comm model per view when active; else builtin (comm=None) so
        # ``_forward_times`` derives it from the (swapped) view.
        comm_tgt = self._comm if self._comm is not None else None
        comm_bench = self._comm_bench if self._comm is not None else None

        # For decode, exclude communication from the scaling ratio: decode
        # collectives (TP all-reduce, EP all-to-all) are small-message,
        # latency-bound, and either overlapped with compute at high batch or
        # pipelined into the fixed per-step overhead at low batch — they are NOT
        # additive on top of a comm-free anchor. Charging them in the ratio
        # double-counts and, when restoring from a comm-free 1-GPU anchor,
        # explodes the ratio at low batch (the target grows a full A2A+AR the
        # anchor never had). The resident decode comm is instead captured by the
        # measured latency floor (``_decode_floor_ms``). Prefill comm is large-
        # message and genuinely exposed, so it stays in the ratio.
        comm_free = phase == "decode"

        def _step(lm, comm, view, imb) -> float:
            saved = (self._lm, self._comm, self._view, self._gemm, self._sdpa, self._moe_imbalance)
            self._lm, self._comm, self._view = lm, comm, view
            self._gemm, self._sdpa = self._gemm_sim, self._sdpa_sim
            # Per-view imbalance: bench-EP vs target-EP (see _setup_restoration).
            self._moe_imbalance = imb
            try:
                ft = self._forward_times(batch, q_len, phase, kv)
                if comm_free:
                    return max(
                        0.0,
                        ft.total_ms
                        - ft.comm.tp_allreduce_ms
                        - ft.comm.ep_a2a_ms
                        - ft.comm.pp_p2p_ms,
                    )
                return ft.total_ms
            finally:
                (self._lm, self._comm, self._view, self._gemm, self._sdpa, self._moe_imbalance) = (
                    saved
                )

        try:
            s_tgt = _step(self._lm_ratio_tgt, comm_tgt, self._view_tgt, self._imb_tgt)
            s_bench = _step(self._lm_ratio_bench, comm_bench, self._view_bench, self._imb_bench)
        except Exception:
            return None
        if s_bench <= 0.0 or s_tgt <= 0.0:
            return None
        return s_tgt, s_bench

    def _restore_whole(
        self, ms_bench: float, batch: int, tokens: int, phase: str = "decode"
    ) -> float:
        """Restore a whole-model (vLLM) step latency measured at the benchmark's
        reduced parallelism to the target TP/EP/PP, training-style and in the same
        ``pp -> ep -> tp`` order as the Megatron per-layer path:

          * strip the benchmark's per-layer communication (summed over layers) so
            only shardable compute remains,
          * scale that compute by ``bench_tp / target_tp`` (TP),
          * add the target communication back — which includes the target EP
            all-to-all and TP all-reduce (EP + TP), and
          * add the ``(pp-1)`` P2P delta for the target PP (PP).

        No-op when bench and target parallelism match, mirroring
        ``_restore_per_layer`` but applied to the full-model total (the whole-model
        vLLM step is not separable per layer, so comm is composed by layer count)."""
        if not self._restore or ms_bench <= 0.0:
            return ms_bench

        # Measured-floor sharding (preferred for decode): hold the anchor's own
        # invariant floor fixed and shard only the excess above it.
        #
        #     step(target) = floor + (step(bench) - floor) * bench_tp/target_tp
        #
        # This is the physics ``_decode_floor_ms`` already asserts -- above the
        # roofline knee the step is fixed per-step overhead and does not shrink
        # with parallelism -- applied as the transport itself rather than only
        # as a clamp after the fact. It needs nothing but the anchor's own batch
        # sweep, which is what makes a sub-node warmup worth having: the target
        # parallelism never has to be measured.
        #
        # Measured on MI355X, carrying a TP4 anchor of DeepSeek-V4-Pro (Atom,
        # ISL 8192) to TP8 against a TP8 ladder, over batches 1..64:
        #
        #     blind TP^-1              median 44.1% error   (max 51.7%)
        #     origami additive delta   median 18.5%         (max 29.3%)
        #     floor-preserving shard   median  0.9%         (max  3.5%)
        #
        # The two weaker laws fail for the same reason from opposite ends: both
        # let the fixed cost scale. Blind TP^-1 halves the whole step; the
        # origami delta subtracts a saving computed by a simulator whose own
        # floor is near zero (batch-1 decode of 4.20 ms at TP4 vs 2.33 at TP8,
        # near-perfect TP^-1, where the real steps are 15.34 and 15.89 -- flat).
        # Solving the two measured parallelisms for the split puts the invariant
        # at 14.4-15.5 ms at every batch from 1 to 64, i.e. 51-99% of the TP8
        # step, so a law that shrinks it cannot be close.
        if (
            phase == "decode"
            and self._tgt_tp != self._bench_tp
            and getattr(self, "_anchor_decode_floor_ms", 0.0) > 0.0
        ):
            floor = self._anchor_decode_floor_ms
            excess = max(0.0, ms_bench - floor)
            restored = floor + excess * (self._bench_tp / self._tgt_tp)
            if os.getenv("INFERASIM_DEBUG_RESTORE"):
                print(
                    f"[dbg-restore] phase=decode b={batch} floor={floor:.3f} "
                    f"ms_bench={ms_bench:.3f} restored={restored:.3f} "
                    f"(floor-preserving, TP{self._bench_tp}->TP{self._tgt_tp})"
                )
            return restored

        # Origami-ratio: scale the measured anchor by the simulator's
        # whole-step TP-scaling ratio sim(target)/sim(bench). The vLLM-fused MoE
        # cost model captures the saturating decode curve (compute sharding +
        # comm growth) better than a 2-point linear fit; the ~5x absolute origami
        # bias cancels in the ratio. Falls through to fit/blind if unavailable.
        if self._scaling_mode == "origami":
            steps = self._origami_steps(batch, tokens, phase)
            if steps is not None:
                s_tgt, s_bench = steps
                if phase == "decode":
                    # A decode step is a fixed per-step cost (kernel dispatch and
                    # occupancy, resident collectives) plus work that shards. Only
                    # the second part responds to parallelism, so move the anchor
                    # by the simulator's *difference*, not its ratio: scaling the
                    # whole measured step re-scales the fixed part too, which for
                    # an already floor-bound anchor (e.g. TP=8) inflates a
                    # less-sharded target several-fold.
                    restored = max(ms_bench * 0.05, ms_bench + (s_tgt - s_bench))
                else:
                    restored = ms_bench * (s_tgt / s_bench)
                if os.getenv("INFERASIM_DEBUG_RESTORE"):
                    print(
                        f"[dbg-restore] phase={phase} b={batch} "
                        f"sim_tgt={s_tgt:.3f} sim_bench={s_bench:.3f} "
                        f"ms_bench={ms_bench:.3f} restored={restored:.3f}"
                    )
                return restored

        def _comm_total(cm) -> float:
            dense = cm.layer_comm_ms(batch, tokens, is_moe=False).total_ms if self._n_dense else 0.0
            moe = cm.layer_comm_ms(batch, tokens, is_moe=True).total_ms if self._n_moe else 0.0
            return self._n_dense * dense + self._n_moe * moe

        comm_bench = _comm_total(self._comm_bench)
        comm_tgt = _comm_total(self._comm_tgt)
        compute = max(0.0, ms_bench - comm_bench)
        fit = (self._bench_scaling_fit.get(phase) or {}).get(batch)
        if fit:
            # Measured scaling: only the shardable part shrinks with TP.
            shardable, invariant = fit
            compute = shardable / self._tgt_tp + invariant
        else:
            compute *= self._bench_tp / self._tgt_tp
        # Apply the same compute-limited comm/compute overlap used on the
        # analytical path (``_overlap_keep``): the configured prefill/decode
        # overlap is a ceiling, but you can hide at most ``compute`` worth of
        # comm behind compute. No-op when the overlap knob is 0 (default).
        keep = self._overlap_keep(phase, comm_tgt, compute)
        return compute + comm_tgt * keep + self._restore_pp_ms(batch, tokens)

    # -- per-pass forward time -------------------------------------------------

    def _moe_imbalance_factor(self) -> float:
        """MoE expert-compute imbalance multiplier (>= 1.0) at the target EP."""
        return self._moe_imbalance_for_ep(
            max(1, self.cfg.model_parallel_config.expert_model_parallel_size)
        )

    def _moe_imbalance_for_ep(self, ep: int) -> float:
        """MoE expert-compute imbalance multiplier (>= 1.0) at an arbitrary EP.

        Only EP-sharded MoE models (``num_experts > 0`` and ``EP > 1``) see
        routing imbalance; for everything else this is a no-op (1.0).  The
        magnitude (and the ``redundant_experts`` mitigation) is resolved on the
        request config, given the model's expert count.  Evaluating this per-EP
        is what lets the origami ratio keep the EP sharding penalty instead of
        cancelling a single (target) imbalance value on both bench and target
        sides.
        """
        mc = self.cfg.model_config
        num_experts = int(getattr(mc, "num_experts", 0) or 0)
        if num_experts <= 0 or max(1, ep) <= 1:
            return 1.0
        return self.cfg.request_config.resolved_ep_imbalance(num_experts)

    def _forward_times(self, batch: int, q_len: int, phase: str, kv_len: int) -> PhaseForwardTimes:
        lm = self._lm
        # Sliding-window / local attention: cap the KV length each attention
        # layer reads at the window (blended across windowed/full layers for
        # interleaved models). Full attention leaves ``kv_len`` unchanged.
        #
        # Decode gets no such cap, because the engine does not deliver it. vLLM
        # only shrinks a windowed layer's KV through its hybrid KV-cache
        # manager, and on the runs behind this model that manager was never
        # active -- its own KV accounting showed every layer holding full
        # context rather than the windowed half capped at the window. The decode
        # step follows: crediting the window badly under-predicts how the step
        # grows with context, while charging the full read tracks the measured
        # growth, so that is what this model does.
        #
        # Prefill keeps the cap: there the window is a masking decision inside a
        # compute-bound kernel rather than a streaming cost, and nothing here
        # measured it.
        mc = self.cfg.model_config
        attn_kv = int(max(1, kv_len))
        if phase != "decode":
            attn_kv = self.cfg.request_config.effective_attn_kv(
                kv_len,
                model_window=getattr(mc, "sink_sliding_window", 0),
                even_layers_only=getattr(mc, "sink_window_even_layers_only", False),
            )
        # Linear / KDA / GDN layers are the architecture, not an engine
        # option: they keep a fixed-size state at decode as well as prefill.
        # Sliding-window stays decode-uncapped on the evidence in the comment
        # above; this blend is a different fact and applies to both phases.
        attn_kv = mc.blend_linear_attn_kv(attn_kv)
        # The attention profiler sizes its KV roofline from ``kv_cache_dtype`` on
        # the *model* config, and the serving request carries it on the request
        # config, so the two never met: every projection priced the cache at two
        # bytes an element no matter what was asked for. Quantising the KV cache
        # is one of the larger serving levers -- decode attention is a stream out
        # of that cache, so fp8 halves the traffic, and at long context that is
        # most of the step -- and the model was blind to it in both directions,
        # scoring an fp8 candidate as bf16 and a bf16 one as if it were fp8.
        mc.kv_cache_dtype = self.cfg.request_config.kv_cache_dtype
        lm.set_inference_phase(phase, attn_kv)

        dense_p = lm.sub_profilers.get("dense_transformer_layer")
        moe_p = lm.sub_profilers.get("moe_transformer_layer")

        has_dense = bool(self._n_dense and dense_p)
        has_moe = bool(self._n_moe and moe_p)

        dense_raw = dense_p.measured_forward_time(batch, q_len) if has_dense else 0.0
        moe_raw = moe_p.measured_forward_time(batch, q_len) if has_moe else 0.0

        # Split implicit per-layer comm out of the raw forward time so it can be
        # handled explicitly below.  Doing this unconditionally (not only when
        # the explicit comm model is active) lets the benchmark calibration
        # scale the *compute* part without disturbing communication cost.
        builtin_dense_comm = self._builtin_comm_ms("dense", batch, q_len) if has_dense else 0.0
        builtin_moe_comm = self._builtin_comm_ms("moe", batch, q_len) if has_moe else 0.0
        dense_compute = max(0.0, dense_raw - builtin_dense_comm) if has_dense else 0.0
        moe_compute = max(0.0, moe_raw - builtin_moe_comm) if has_moe else 0.0

        # MoE expert-MLP (grouped-GEMM) adjustments — applied only to the
        # expert-MLP portion of the layer (attention, router and comm are
        # unaffected):
        #   * routing imbalance (>= 1.0): the MoE step is gated by the busiest
        #     EP rank, which does ``imbalance``x the average expert work;
        #   * expert dtype speedup (<= 1.0): low-precision expert kernels
        #     (mxfp4 / fp8) run the grouped-GEMM faster.
        # These compose multiplicatively. No-op when balanced + bf16 / non-MoE.
        # In roofline mode the imbalance is applied inside the expert GEMM (M
        # scaling in moe_mlp), so the outer multiplier only carries the dtype
        # speedup here to avoid double-counting. When roofline mode is disabled
        # the outer multiplier applies the imbalance (legacy behaviour).
        outer_imb = 1.0 if self._imb_roofline else self._moe_imbalance
        if (
            has_moe
            and (outer_imb > 1.0 or self._moe_expert_speedup != 1.0)
            and hasattr(moe_p, "get_sub_profiler")
        ):
            mlp_p = moe_p.get_sub_profiler("mlp")
            if mlp_p is not None:
                expert_mlp_ms = mlp_p.measured_forward_time(batch, q_len)
                new_expert = expert_mlp_ms * outer_imb * self._moe_expert_speedup
                moe_compute += new_expert - expert_mlp_ms

        # Kernel-backend (attention library) + native-sparse-attention: adjust
        # only the attention sub-profiler's compute.  ``attn_mult`` scales the
        # whole attention forward (Triton baseline = 1.0); ``sparse_scale``
        # shrinks attention toward ``topk/context`` for long contexts (NSA).
        #
        # Decode gets no top-k credit, for the same reason and on the same
        # evidence as the sliding window above. Selecting a thousand keys out of
        # a hundred thousand is a decisive saving in a prefill, where attention
        # is quadratic and the kernel is compute-bound. A decode step is neither:
        # it is a streaming read of the cache, the indexer still has to score
        # every token to choose, and what it then gathers is scattered across
        # pages. Charging the discount there made the step twice as fast as it
        # measures on both DeepSeek-V4-Pro and MiniMax-M3, on both vendors, over
        # 96 trace rows at ~130k context -- and withdrawing it moves DeepSeek-V4
        # from 0.53x of measured TPOT to 0.83x on MI355X and 0.94x on GB300,
        # while the fixed-shape 8k sweep, where the floor barely binds, does not
        # move at all.
        # A model that records a per-layer compression schedule is costed from
        # it, weighting each branch by how many layers actually run it, rather
        # than from one top-k scale standing for the whole stack. Prefill only,
        # for the reason just given: decode's measured step does not shrink with
        # the selection, so the schedule is not what sets its cost either.
        # A model that selects the same way on every layer is costed from its
        # indexer geometry instead, for the same reason: one number cannot
        # hold a selection term that falls away with context and an indexing
        # term that does not. The floored top-k below stays as the fallback
        # for a model that declares a window but no indexer to choose with.
        sparse_scale = 1.0
        if phase != "decode":
            from infera.projection.core.projection.training_config import (
                hybrid_attention_scale,
                uniform_sparse_attention_scale,
            )

            sparse_scale = hybrid_attention_scale(self.cfg.model_config, kv_len)
            if sparse_scale is None:
                sparse_scale = uniform_sparse_attention_scale(
                    self.cfg.model_config,
                    kv_len,
                    self.cfg.request_config.sparse_attention_topk,
                    self.cfg.request_config.sparse_indexer_cost_scale,
                )
            if sparse_scale is None:
                sparse_scale = self.cfg.request_config.resolved_sparse_attention_scale(kv_len)
        else:
            # Decode keeps its dense charge -- the step does not shrink with
            # the selection, which is what the measured TPOT says -- but a
            # stack without a fused selection kernel pays for choosing on top
            # of it, and that is not in the dense charge. Only the excess
            # over a fused kernel is added, so a stack that has one is priced
            # exactly as the calibration that set the dense charge left it.
            from infera.projection.core.projection.training_config import (
                sparse_indexer_decode_overhead,
            )

            sparse_scale = 1.0 + sparse_indexer_decode_overhead(
                self.cfg.model_config,
                self.cfg.request_config.sparse_attention_topk,
                self.cfg.request_config.sparse_indexer_cost_scale,
            )
        # Attention-DP: the memory model has always known that a rank under DP
        # attention owns a subset of the *requests* rather than a slice of every
        # request's heads, but the time model did not, and charged every rank
        # the whole batch's attention. That is the difference between a rank
        # streaming the cache for 512 requests and for 64. Getting it wrong does
        # not merely mis-scale a number, it inverts a serving decision: measured
        # MLA on MI355X decodes 1.66x faster with DP attention on (4x at 8k and
        # concurrency 512), while the projector had it 1.7x slower, so a search
        # run on this model would have rejected the configuration the fleet
        # actually runs. Re-evaluating attention at the per-rank request count
        # rather than scaling by 1/dp keeps whatever curvature the profiler has
        # at small batch, which is where the decode floor lives.
        from .kv_cache import attention_dp_size

        dp = attention_dp_size(self.cfg)
        attn_batch = max(1, math.ceil(batch / dp)) if dp > 1 else batch
        factor = self._attn_backend_mult * sparse_scale
        if factor != 1.0 or attn_batch != batch:
            for has, prof, is_moe in ((has_dense, dense_p, False), (has_moe, moe_p, True)):
                if not has or not hasattr(prof, "get_sub_profiler"):
                    continue
                sub = prof.get_sub_profiler("self_attention")
                if sub is None:
                    continue
                charged = sub.measured_forward_time(batch, q_len)
                actual = (
                    sub.measured_forward_time(attn_batch, q_len) if attn_batch != batch else charged
                ) * factor
                if is_moe:
                    moe_compute = max(0.0, moe_compute + actual - charged)
                else:
                    dense_compute = max(0.0, dense_compute + actual - charged)

        comm = CommBreakdown()
        if self._comm is not None:
            # Feature B: explicit, knob-driven communication model with a
            # batch-dependent compute/comm overlap. The exposed comm is hidden
            # behind the SAME layer-type compute up to the configured ceiling
            # (see _overlap_keep), so dense and MoE layers get different exposure
            # and the residual at high batch is captured analytically.
            new_tp_ar = self._comm.tp_allreduce_ms(batch, q_len)
            new_ep_a2a = self._comm.ep_a2a_ms(batch, q_len)
            # Dense: 2 TP-AR (attention + MLP). MoE: the post-expert TP-AR is
            # only present while experts stay TP-sharded (EP=1); at EP==TP the
            # expert output is combined by the A2A, so MoE carries just the
            # attention AR (half of the dense 2-AR) plus the A2A. Charging the
            # full dense 2-AR + A2A on MoE at EP>1 double-counts (see
            # _moe_tp_allreduce_count).
            moe_tp_ar = new_tp_ar * (
                _moe_tp_allreduce_count(self._view) / _dense_tp_allreduce_count(self._view)
            )

            # Whether the expert All-to-All is hidden is a property of the
            # engine, not of the arithmetic sitting next to it. Hiding it
            # whenever expert compute merely exceeded it granted every
            # deployment an asynchronous dispatch/combine, including the ones
            # that issue it synchronously, and made expert parallelism free:
            # EP=TP dropped the post-expert reduction and paid nothing back, so
            # a mix search preferred wide EP everywhere. On MI355X that is
            # backwards -- across matched MiniMax-M3 pairs that differ only in
            # expert degree, EP8 measures 0.87x the throughput of EP1, and this
            # model called it 1.01x. Charging the collective the engine actually
            # exposes gives 0.84x.
            #
            # So the overlap now comes from the one place that knows whether the
            # engine can do it: ``deepep_overlap_efficiency``, already applied
            # inside ``ep_a2a_ms`` and zero unless DeepEP/SyncFree is on. What
            # remains after that is exposed like any other collective.
            keep_a2a = self._overlap_keep(phase, new_ep_a2a, moe_compute) if has_moe else 1.0
            ep_a2a_exposed = new_ep_a2a * keep_a2a

            dense_comm = new_tp_ar
            keep_dense = self._overlap_keep(phase, dense_comm, dense_compute) if has_dense else 1.0
            keep_moe_ar = self._overlap_keep(phase, moe_tp_ar, moe_compute) if has_moe else 1.0

            dense_fwd = dense_compute + (dense_comm * keep_dense if has_dense else 0.0)
            moe_fwd = moe_compute + (moe_tp_ar * keep_moe_ar + ep_a2a_exposed if has_moe else 0.0)

            # TP-AR appears in both layer types; charge each at its own exposure.
            comm.tp_allreduce_ms = (
                self._n_dense * new_tp_ar * keep_dense + self._n_moe * moe_tp_ar * keep_moe_ar
            )
            comm.ep_a2a_ms = self._n_moe * ep_a2a_exposed
            pp_keep = keep_moe_ar if has_moe else keep_dense
            comm.pp_p2p_ms = self._comm.pp_p2p_ms(batch, q_len) * pp_keep
        else:
            # Implicit comm: add the built-in cost back onto (calibrated)
            # compute. When DeepEP/SyncFree is enabled, the EP A2A overlaps
            # expert compute, so charge only the exposed (non-overlapped)
            # fraction of the raw A2A baked into the layer time.
            eff_moe_comm = builtin_moe_comm
            if self._deepep_overlap > 0 and has_moe:
                a2a_raw = _estimate_moe_a2a_time_ms(self._view, batch, q_len, self._gemm)
                eff_moe_comm = builtin_moe_comm - a2a_raw * self._deepep_overlap
            dense_fwd = dense_compute + builtin_dense_comm
            moe_fwd = moe_compute + eff_moe_comm

        layers = self._n_dense * dense_fwd + self._n_moe * moe_fwd

        emb = _safe_forward(lm.sub_profilers.get("embedding"), batch, q_len)
        # The final LayerNorm is element-wise and not separately timed by the
        # profiler (training does not measure it either) — treat as ~0.
        fnorm = _safe_forward(lm.sub_profilers.get("final_layernorm"), batch, q_len)
        # LM head only materialises logits for the token(s) being sampled.
        # Prefill samples 1 token; decode samples 1 per step.  Speculative
        # decode verifies q_len tokens, so size the head by q_len there.
        head_tokens = q_len if phase == "decode" else 1
        out = _safe_forward(lm.sub_profilers.get("output_layer"), batch, head_tokens)

        # Token sampling / logits post-processing: one sampled token per
        # sequence per step (``head_tokens`` per sequence for speculative
        # verification), a memory-bound reduction over the vocabulary.
        sampling_ms = 0.0
        if self._sampling_enabled:
            sampling_ms = self._sampler.forward_time_ms(batch * head_tokens)

        # Runtime activation quantization / cast, summed over all layers (same
        # per-layer-type accounting as ``layers`` above).
        quant_ms = 0.0
        if self._act_quant_dtype:
            dense_q = self._quant.dense_layer_ms(batch, q_len) if has_dense else 0.0
            moe_q = self._quant.moe_layer_ms(batch, q_len) if has_moe else 0.0
            quant_ms = self._n_dense * dense_q + self._n_moe * moe_q

        if os.getenv("INFERASIM_DEBUG_BREAKDOWN"):
            attn_ms = 0.0
            mlp_ms = 0.0
            if has_moe and hasattr(moe_p, "get_sub_profiler"):
                _a = moe_p.get_sub_profiler("self_attention")
                _m = moe_p.get_sub_profiler("mlp")
                attn_ms = self._n_moe * (_a.measured_forward_time(batch, q_len) if _a else 0.0)
                mlp_ms = self._n_moe * (_m.measured_forward_time(batch, q_len) if _m else 0.0)
            print(
                f"  [breakdown/{phase}] batch={batch} q_len={q_len} kv={kv_len}"
                f"  layers={layers:.3f} (moe_attn={attn_ms:.3f} moe_mlp={mlp_ms:.3f})"
                f"  emb={emb:.3f} head={out:.3f} sampling={sampling_ms:.3f}"
                f"  quant={quant_ms:.3f} comm={comm.total_ms:.3f}"
            )

        return PhaseForwardTimes(
            layers_ms=layers,
            embedding_ms=emb,
            final_norm_ms=fnorm,
            output_ms=out,
            sampling_ms=sampling_ms,
            quant_ms=quant_ms,
            dense_layer_ms=dense_fwd,
            moe_layer_ms=moe_fwd,
            comm=comm,
        )

    # -- prefill ---------------------------------------------------------------

    def _prefix_cached_tokens(self, input_len: int) -> int:
        """Prompt tokens served from the prefix cache (skip prefill compute).

        The non-cached suffix (``input_len - cached``) is what must actually be
        run through the network. At least one token always remains uncached so a
        fully-cached prompt still does one forward to emit its first token.
        """
        hit = self.cfg.request_config.resolved_prefix_cache_hit_rate()
        if hit <= 0.0 or input_len <= 1:
            return 0
        return max(0, min(int(input_len * hit), input_len - 1))

    def prefill_latency_ms(self, batch: int, input_len: int) -> float:
        """Time to process the prompt (→ first token).  Honors chunked prefill.

        With a prefix-cache hit rate the cached prefix is already resident, so
        only the non-cached suffix is prefilled; the suffix still attends over
        the full context (cached prefix + suffix).
        """
        cached = self._prefix_cached_tokens(input_len)
        new_tokens = input_len - cached

        # A prefix hit the host tier holds is staged back over the host link
        # instead of being recomputed; what stays in HBM is free. This is the
        # whole trade offload makes, and it only pays while the link is faster
        # than re-running the prompt.
        req = self.cfg.request_config
        fetch_ms = 0.0
        if cached and req.kv_offload_gb_per_gpu > 0 and req.kv_offload_bw_gbps > 0:
            from .kv_cache import estimate_kv_cache

            per_token_gb = estimate_kv_cache(
                self.cfg, _layers_on_rank(self.cfg), concurrency=1, context_len=1
            ).bytes_per_token / (1024.0**3)
            fetched_gb = min(cached * per_token_gb, req.kv_offload_gb_per_gpu)
            fetch_ms = fetched_gb / req.kv_offload_bw_gbps * 1000.0

        # Benchmark-based: use the measured full-prompt prefill step directly.
        if self._measured_mode:
            if self._meas_whole.get("prefill") or self._meas_layer:
                # Measured anchor is at ``ref_input``; scale by the per-token
                # prefill rate when the effective prompt differs in length.
                if self._meas_ref_input and input_len != self._meas_ref_input:
                    base = self._measured_prefill_tokens_ms(batch * input_len)
                else:
                    base = self._measured_full_prefill_ms(batch)
                # A cache-hit anchor already includes lookup plus the one-token
                # forward needed to produce the first output. Discounting it by
                # the uncached suffix would apply the hit a second time.
                if self._meas_prefill_cache_hit:
                    return base + fetch_ms
                # Prefix-cache hit: discount the SAME baseline proportionally to
                # the non-cached suffix. Scaling the chosen cost method (rather
                # than switching to a different one) keeps prefill continuous at
                # the first hit -- the token-rate and full-prefill anchors can
                # differ by the batch factor, so a method switch would cliff.
                if cached:
                    base *= new_tokens / max(1, input_len)
                return base + fetch_ms

        # Per-forward fixed cost. A prefill chunk is a forward pass over the same
        # graph a decode step runs: the same layers issuing the same kernels, so
        # it pays the same per-kernel device overhead (dispatch, wave launch,
        # drain) and the same per-step host overhead. The decode path has charged
        # both for a while; this path charged neither, which is not a modelling
        # choice about prefill but an asymmetry -- sharding does not remove
        # kernels and neither does having more tokens to feed them.
        #
        # It is a per-*chunk* cost, so it is negligible on a short prompt that
        # prefills in one forward and material on a long one that takes tens of
        # chunks, which is the shape of the residual it addresses.
        per_forward = self._decode_step_overhead_ms() + self._decode_occupancy_ms()
        scale = self._prefill_rate_scale()

        chunk = int(self.cfg.request_config.chunked_prefill_size or 0)
        if chunk <= 0 or chunk >= new_tokens:
            # Single forward over the ``new_tokens`` suffix; attention context is
            # the full prompt (``input_len``) since it also reads the cached KV.
            ft = self._forward_times(batch, new_tokens, "prefill", input_len)
            return ft.total_ms * scale + per_forward + fetch_ms

        # Chunked prefill: each chunk attends to all preceding context. The
        # cached prefix is already resident, so chunking starts after it.
        total = 0.0
        processed = cached
        while processed < input_len:
            this = min(chunk, input_len - processed)
            kv_len = processed + this
            ft = self._forward_times(batch, this, "prefill", kv_len)
            total += ft.total_ms * scale + per_forward
            processed += this
        return total + fetch_ms

    def _prefill_rate_scale(self) -> float:
        """Level correction bringing modelled prefill compute onto measurement.

        The anchor is a per-token TTFT slope measured by differencing two prompt
        lengths at concurrency 1. Comparing it against the model's *own* slope
        over the same two lengths isolates the level error: both sides are
        differences, so the per-request constant and the per-forward overheads
        cancel out of the ratio rather than contaminating it.

        Returned as a multiplier so the roofline keeps its shape -- prefill stays
        superlinear in context, which a slope fit over a short span could not
        reproduce if it were used as the cost directly.
        """
        req = self.cfg.request_config
        rate = float(req.prefill_rate_us_per_token or 0.0)
        lo, hi = int(req.prefill_rate_lo_tokens or 0), int(req.prefill_rate_hi_tokens or 0)
        if rate <= 0.0 or hi <= lo or lo <= 0:
            return 1.0
        if self._measured_mode:
            # Benchmark mode already prices prefill from measured kernels; a
            # second anchor on top would double-count the same correction.
            return 1.0
        cached = getattr(self, "_prefill_scale_cache", None)
        if cached is not None:
            return cached
        # The model's slope over the same span, in us per token. Batch 1 and no
        # prefix reuse, matching the concurrency-1 rows the fit came from.
        span_ms = (
            self._forward_times(1, hi, "prefill", hi).total_ms
            - self._forward_times(1, lo, "prefill", lo).total_ms
        )
        modelled = span_ms * 1000.0 / float(hi - lo)
        scale = (rate / modelled) if modelled > 0 else 1.0
        # A slope fit from two points on a handful of runs is not precise enough
        # to justify unbounded rescaling; clamp to the range the corpus supports
        # so one noisy pair cannot rewrite a projection.
        scale = max(0.5, min(3.0, scale))
        self._prefill_scale_cache = scale
        return scale

    # -- decode ----------------------------------------------------------------

    def _decode_step_overhead_ms(self) -> float:
        """Fixed per-step host/launch overhead (CUDA-graph-reducible)."""
        return max(0.0, self.cfg.request_config.resolved_decode_step_overhead_us()) / 1000.0

    def _decode_occupancy_ms(self) -> float:
        """Additive per-kernel GPU occupancy for a decode step (ms).

        Only for the pure-simulate path: a measured anchor already contains the
        occupancy of its own kernels, so adding it there would double-count.
        Because the term is identical at every parallelism, it cancels in the
        ``s_tgt - s_bench`` difference the anchor restore takes, which is the
        correct behaviour -- sharding does not remove kernels.

        Charged over every kernel the step issues, not only the elementwise
        ones. The per-kernel cost here is the device overhead paid around a
        kernel -- dispatch, wave launch, drain -- which sits outside the math a
        profiler times, so charging it on a GEMM as well as on a norm is not
        double-counting its compute. Restricting it to elementwise kernels left
        the step short by a fixed amount: measured against real vLLM decode
        steps it recovers 2.45 ms of missing time on gpt-oss-120b (36 layers)
        and 5.29 ms on DeepSeek-R1 (61 layers), where the full count charges
        2.72 and 4.59, and the elementwise count charges only 1.28 and 2.16.
        """
        from infera.projection.core.projection.training_config import (
            decode_kernels_per_layer,
        )

        return self.cfg.request_config.resolved_decode_occupancy_ms(
            self.cfg.model_config.num_layers,
            decode_kernels_per_layer(
                self.cfg.model_config,
                self.cfg.request_config.sparse_attention_topk,
            ),
        )

    def _launch_latency_floor_ms(self) -> float:
        """Small-tensor kernel-launch floor for the pure-simulate decode step.

        Depth-scaled launch-bound floor (``n_kernels * launch_latency_us``);
        0.0 when disabled or when CUDA-graph capture collapses the launches.
        """
        return self.cfg.request_config.resolved_kernel_launch_floor_ms(
            self.cfg.model_config.num_layers
        )

    def _decode_floor_ms(self, batch: int) -> float:
        """Hardware decode latency floor at ``batch`` from a sharded probe.

        Above the roofline knee the decode step is set by fixed per-step
        launch/dispatch overhead (parallelism-invariant), so a sharded probe's
        measured decode curve is the floor for any more-sharded target. Clamps
        to the probe's batch range and linearly interpolates between measured
        points. Returns 0.0 (no floor) when no probe was provided.
        """
        floor = self._decode_floor
        if not floor:
            return 0.0
        if batch in floor:
            return floor[batch]
        bs = sorted(floor)
        if batch <= bs[0]:
            return floor[bs[0]]
        if batch >= bs[-1]:
            return floor[bs[-1]]
        lo = max(b for b in bs if b <= batch)
        hi = min(b for b in bs if b >= batch)
        if hi == lo:
            return floor[lo]
        w = (batch - lo) / (hi - lo)
        return floor[lo] * (1.0 - w) + floor[hi] * w

    def _draft_overhead_ms(self, per_token_step_ms: float) -> float:
        """Speculative draft-model forward cost added to a verify step.

        The draft runs ``speculative_num_tokens`` times per verify step; each
        draft pass costs ``speculative_draft_cost_factor`` of one target decode
        token.  ``0`` for either knob is a no-op (legacy behaviour that only
        credited the accepted-token speedup).
        """
        req = self.cfg.request_config
        spec_k = int(req.speculative_num_tokens or 0)
        dcf = float(req.speculative_draft_cost_factor or 0.0)
        if spec_k > 0 and dcf > 0.0:
            return dcf * spec_k * max(0.0, per_token_step_ms)
        return 0.0

    def _measured_verify_step_scale(
        self, batch: int | None = None, context: float | None = None
    ) -> float:
        """Factor turning one measured decode number into one verify-step time.

        What the anchor measured decides this. Harvested *with* speculation, the
        differenced decode timing is a per-output-token latency with acceptance
        already folded in -- see the note in ``benchmark_vllm`` where the
        speculative config is applied -- so a step emitting
        ``_spec_tokens_per_step()`` tokens costs that many times the
        measurement.

        Scaling such an anchor by the verify width ``k + 1`` instead charges the
        draft twice. It inflated decode by ``(k + 1) / tokens_per_step`` -- 1.09x
        at ``k=1, accept=0.84``, 1.58x at ``k=3, accept=0.7``, exact only at
        perfect acceptance -- and understated throughput by the same ratio, so
        speculation read worse the better it was accepted.

        Harvested *without* speculation there is nothing folded in, and the
        anchor is a single-token step. Charging the verify width ``k + 1`` for
        it treats decode as compute-bound, which at these contexts it is not:
        the pass reads the same KV once however many positions ride along, so
        ``_verify_width_ratio`` prices the widening from the model and leaves
        the level to the anchor. The blind width remains the fallback for hosts
        where that ratio cannot be evaluated.

        Speculation stays regime-defining and the anchor store still refuses
        the pairing; this is what ``--load-benchmark`` and a relaxed axis get
        instead of a factor that scales with the draft depth.
        """
        spec_k = int(self.cfg.request_config.speculative_num_tokens or 0)
        if spec_k <= 0:
            return 1.0
        if self._bench_spec_k > 0:
            return max(1e-6, self._spec_tokens_per_step())
        ratio = self._verify_width_ratio(spec_k + 1, batch, context)
        return float(spec_k + 1) if ratio is None else ratio

    def _verify_width_ratio(
        self, width: int, batch: int | None, context: float | None
    ) -> float | None:
        """What a verify pass costs relative to the single-token step measured.

        The verify pass puts ``width`` query positions through the stack, but it
        reads the KV cache once for the same sequences. At the contexts these
        anchors are read at the read dominates, so the pass costs a little more
        than one step and nowhere near ``width`` of them; charging the width is
        what made a speculating target read slower the deeper its draft.

        Only the ratio between two step shapes comes from the analytical model
        -- the anchor still sets the level. That is the division of labour
        ``_restore_whole`` already uses to carry an anchor across tensor-parallel
        widths, applied to query width instead.

        ``None`` when the analytical path cannot be evaluated on this host,
        which leaves the caller on the blind width rather than on a ratio
        nothing computed.
        """
        if width <= 1:
            return 1.0
        b = max(1, int(batch or 1))
        kv = max(
            1,
            int(
                context
                or self._meas_ref_input
                or self.cfg.request_config.input_seq_len
                or 1024
            ),
        )
        key = (b, kv, int(width))
        if key in self._verify_ratio_cache:
            return self._verify_ratio_cache[key]
        try:
            one = self._forward_times(b, 1, "decode", kv).total_ms
            wide = self._forward_times(b, int(width), "decode", kv).total_ms
        except Exception:  # noqa: BLE001 - simulator availability is arch-dependent
            self._verify_ratio_cache[key] = None
            return None
        if one <= 0.0 or wide <= 0.0:
            self._verify_ratio_cache[key] = None
            return None
        # Bracketed by the two things it sits between: a pass carrying the extra
        # positions cannot beat the single-token step inside it, and cannot cost
        # more than running that step once per position.
        ratio = min(float(width), max(1.0, wide / one))
        self._verify_ratio_cache[key] = ratio
        return ratio

    def _decode_step_latency_ms(self, batch: int, kv_len: int, q_len: int = 1) -> float:
        # Benchmark-based: use the measured decode step directly (memory-bound,
        # ~flat in context over a generation, so no simulator context-scaling).
        if self._measured_mode:
            per_token = self._measured_decode_step_ms(batch, kv_len)
            step = per_token * self._measured_verify_step_scale(batch, kv_len)
            # A speculation-harvested anchor already paid for the draft pass, so
            # adding the modelled overhead would bill it a second time.
            if self._bench_spec_k <= 0:
                step += self._draft_overhead_ms(per_token)
            step += self._decode_step_overhead_ms()
            return max(step, self._decode_floor_ms(batch))
        ft = self._forward_times(batch, q_len, "decode", kv_len)
        per_token = ft.total_ms / max(1, q_len)
        step = ft.total_ms + self._draft_overhead_ms(per_token) + self._decode_step_overhead_ms()
        # Per-kernel GPU occupancy adds to the data-movement time rather than
        # capping it: the small latency-bound kernels of a decode layer run
        # alongside the large data-bound expert GEMMs, so both costs are paid.
        step += self._decode_occupancy_ms()
        # Pure-simulate: at low batch the roofline step underflows the real
        # launch-bound decode; apply the small-tensor launch-latency floor.
        return max(step, self._decode_floor_ms(batch), self._launch_latency_floor_ms())

    # -- DES event-duration kernel --------------------------------------------
    # Public wrappers used by the discrete-event simulator (``des.py``) so that
    # each simulated step's duration is drawn from this (possibly
    # benchmark-calibrated) cost model — i.e. "benchmark calibration inside a
    # DES". They mirror the pure/mixed step costs the steady-state
    # ``_continuous_decode_metrics`` blends analytically.

    def decode_step_latency_ms(self, batch: int, kv_len: int, q_len: int = 1) -> float:
        """One pure-decode step over ``batch`` resident sequences."""
        return self._decode_step_latency_ms(max(1, batch), max(1, kv_len), q_len)

    def mixed_step_latency_ms(
        self,
        num_decode: int,
        chunk_tokens: int,
        decode_ctx: int,
        prefill_kv_len: int,
        q_len: int = 1,
    ) -> float:
        """One scheduler step carrying a prefill chunk plus ``num_decode``
        concurrent decodes (``num_decode == 0`` → a pure prefill-chunk step)."""
        penalty = max(0.0, self.cfg.request_config.resolved_mixed_batch_penalty())
        ov = self._decode_step_overhead_ms()
        chunk_tokens = max(1, int(chunk_tokens))
        num_decode = max(0, int(num_decode))
        if self._measured_mode:
            # See ``_measured_verify_step_scale``: a speculation-harvested decode
            # number is per output token, so the step scales by the tokens it
            # emits rather than by the verify width ``q_len``.
            spec = self._measured_verify_step_scale(max(1, num_decode), decode_ctx)
            prefill_piece = self._measured_prefill_tokens_ms(chunk_tokens)
            dec_piece = (
                self._measured_decode_step_ms(num_decode, decode_ctx) * spec
                if num_decode > 0
                else 0.0
            )
            return (prefill_piece + dec_piece) * (1.0 + penalty) + ov
        prefill_piece = (
            self._forward_times(1, chunk_tokens, "prefill", max(1, prefill_kv_len)).total_ms
            * self._prefill_rate_scale()
        )
        dec_piece = (
            self._forward_times(num_decode, q_len, "decode", max(1, decode_ctx)).total_ms
            if num_decode > 0
            else 0.0
        )
        step = (prefill_piece + dec_piece) * (1.0 + penalty) + ov
        return max(step, self._launch_latency_floor_ms())

    def decode_total_ms(self, batch: int, input_len: int, output_len: int) -> float:
        """Integrate per-token decode latency over the growing KV cache.

        Per-step latency grows slowly with context, so we sample a handful of
        context lengths and trapezoid-integrate rather than simulating every
        one of ``output_len`` steps.
        """
        if output_len <= 0:
            return 0.0

        spec_k = int(self.cfg.request_config.speculative_num_tokens or 0)
        accept = float(self.cfg.request_config.speculative_acceptance_rate or 0.0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        # Expected accepted tokens per verify step (geometric series).
        if spec_k > 0 and 0.0 < accept < 1.0:
            tokens_per_step = (1.0 - accept ** (spec_k + 1)) / (1.0 - accept)
        elif spec_k > 0 and accept >= 1.0:
            tokens_per_step = spec_k + 1
        else:
            tokens_per_step = 1.0

        num_steps = max(1.0, output_len / tokens_per_step)

        # Sample step latency across [input_len, input_len + output_len].
        n_samples = min(8, max(2, int(output_len)))
        ctx_lo, ctx_hi = input_len, input_len + output_len
        samples = []
        for i in range(n_samples):
            frac = i / (n_samples - 1) if n_samples > 1 else 0.0
            ctx = int(ctx_lo + frac * (ctx_hi - ctx_lo))
            samples.append(self._decode_step_latency_ms(batch, ctx, q_len=q_len))
        avg_step = sum(samples) / len(samples)
        return avg_step * num_steps

    # -- continuous batching (steady-state TPOT) -------------------------------

    def _continuous_decode_metrics(
        self, input_len: int, output_len: int, concurrency: int
    ) -> dict[str, float]:
        """Steady-state decode under *continuous batching*.

        Real servers (vLLM, SGLang, ...) keep ``concurrency`` sequences resident
        and admit a new request's prefill the moment one finishes.  That makes a
        fraction of scheduler steps **mixed** (1 prefill chunk + ``C-1`` decode)
        which are far slower per token than a uniform **pure-decode** step — the
        "TPOT pollution" effect.  This models the blended steady state.

        Accounting (per admitted request, the ``R`` factor cancels):
          * a request's prefill is processed in ``n_chunks`` mixed steps;
          * pure steps emit ``C * tok/step`` decode tokens, mixed steps emit
            ``(C-1) * tok/step``;
          * total decode tokens per request = ``OSL``.
        From the per-request window time ``T`` we get
        ``TPOT = C * T / OSL`` and system ``throughput = 1000 * OSL / T``.
        """
        req = self.cfg.request_config
        ISL = max(1, input_len)
        OSL = max(1, output_len)
        C = max(1, int(concurrency))

        spec_k = int(req.speculative_num_tokens or 0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        tok_per_step = max(1e-6, self._spec_tokens_per_step())

        # Prefill of a newly-admitted request is split into chunks; with chunked
        # prefill each mixed step carries only one chunk (less pollution/step).
        # A prefix-cache hit skips the cached prefix, so only the non-cached
        # suffix (``prefill_span``) pollutes the decode stream.
        prefill_span = max(1, ISL - self._prefix_cached_tokens(ISL))
        chunk = int(req.chunked_prefill_size or 0)
        if chunk <= 0 or chunk >= prefill_span:
            n_chunks = 1
            chunk_tokens = prefill_span
        else:
            n_chunks = max(1, math.ceil(prefill_span / chunk))
            chunk_tokens = chunk

        # Scheduler per-step token budget (vLLM ``max_num_batched_tokens``). A
        # mixed step processes the prefill chunk PLUS the decode tokens of the
        # other ``C-1`` running sequences; that sum cannot exceed the cap. When
        # it would, the prefill admitted per step is bounded by the leftover
        # budget, so the prompt is split into more (smaller) prefill chunks →
        # more mixed steps → higher TPOT / lower throughput. First-order model:
        # clamp the effective prefill chunk to ``cap - decode_tokens`` and
        # recompute the chunk count. ``0`` = unlimited (path unchanged).
        cap = int(req.max_num_batched_tokens or 0)
        if cap > 0:
            decode_tokens_mixed = max(0, C - 1) * int(q_len)
            # Always make at least one token of prefill progress per step so the
            # model stays finite even if decode tokens alone saturate the cap.
            eff_chunk = min(chunk_tokens, max(1, cap - decode_tokens_mixed))
            if eff_chunk < chunk_tokens:
                chunk_tokens = eff_chunk
                n_chunks = max(1, math.ceil(prefill_span / eff_chunk))

        # The steps below are priced once and multiplied by ``n_chunks``, so the
        # chunk they are priced at has to be the *average*, not the cap. Chunking
        # is a ceiling division and the last chunk is a remainder: a prompt one
        # token over the budget takes two steps, the second carrying one token.
        # Charging both at the full budget bills twice the prompt -- which is
        # exactly the shape here, where the decode tokens shave the cap just
        # below the prompt and every prefill was billed ~2x.
        chunk_tokens = max(1, int(round(prefill_span / n_chunks)))

        penalty = max(0.0, req.resolved_mixed_batch_penalty())
        ov = self._decode_step_overhead_ms()

        if self._measured_mode:
            # Benchmark-based: average the pure/mixed step over the context window
            # [ISL, ISL+OSL]. The measured decode step carries its fitted KV term,
            # so this is flat only when that slope is ~0 (prior behaviour).
            # Same anchor semantics as ``_decode_step_latency_ms``: a
            # speculation-harvested decode number is per output token, so the
            # step scales by the tokens a step emits, not by the verify width.
            draft_billed_by_anchor = self._bench_spec_k > 0
            n_samples = min(8, max(2, int(OSL)))
            # Benchmark mode prices prefill from measured kernels, so the anchor
            # is inert there and the two steps coincide.
            pure, mixed, pf_only = [], [], []
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                ctx = int(ISL + frac * OSL)
                d_pure = self._measured_decode_step_ms(C, ctx)
                draft = 0.0 if draft_billed_by_anchor else self._draft_overhead_ms(d_pure)
                pure.append(d_pure * self._measured_verify_step_scale(C, ctx) + draft + ov)
                prefill_piece = self._measured_prefill_tokens_ms(chunk_tokens)
                dec_piece = self._measured_decode_step_ms(
                    max(1, C - 1), ctx
                ) * self._measured_verify_step_scale(max(1, C - 1), ctx)
                mixed.append((prefill_piece + dec_piece) * (1.0 + penalty) + ov)
                pf_only.append(prefill_piece * (1.0 + penalty))
            t_pure = sum(pure) / len(pure)
            t_mixed = t_mixed_ttft = sum(mixed) / len(mixed)
            t_prefill_only = sum(pf_only) / len(pf_only)
        else:
            # Simulation: average pure/mixed step latency over the (uniform)
            # context distribution [ISL, ISL+OSL].
            n_samples = min(8, max(2, int(OSL)))
            ctx_lo, ctx_hi = ISL, ISL + OSL
            # Per-kernel GPU occupancy is a property of the step's kernels, not
            # of which code path prices it, so both the pure and the mixed step
            # pay it -- a mixed step runs the same decode kernels and adds a
            # prefill chunk on top. Charging it only in
            # ``_decode_step_latency_ms`` left the continuous-batching path
            # (every vLLM workload) without it.
            occ = self._decode_occupancy_ms()
            # The measured prefill anchor is a slope on the *TTFT* path, and the
            # decode stream is evidence it does not belong on both: TPOT already
            # reads ~1.0x against this corpus, which it could not if the mixed
            # steps polluting it were mispriced by the anchor's factor. So the
            # scaled step prices what a waiting request sees, and the unscaled
            # one prices what the running decodes pay.
            scale = self._prefill_rate_scale()
            pure, mixed, mixed_ttft, pf_only = [], [], [], []
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                ctx = int(ctx_lo + frac * (ctx_hi - ctx_lo))
                pure_fwd = self._forward_times(C, q_len, "decode", ctx).total_ms
                t_pure = pure_fwd + self._draft_overhead_ms(pure_fwd / max(1, q_len)) + ov + occ
                prefill_piece = self._forward_times(
                    1, chunk_tokens, "prefill", min(ctx, ISL)
                ).total_ms
                dec_piece = self._forward_times(max(1, C - 1), q_len, "decode", ctx).total_ms
                t_mixed = (prefill_piece + dec_piece) * (1.0 + penalty) + ov + occ
                pure.append(t_pure)
                mixed.append(t_mixed)
                mixed_ttft.append((prefill_piece * scale + dec_piece) * (1.0 + penalty) + ov + occ)
                pf_only.append(prefill_piece * scale * (1.0 + penalty))
            t_pure = sum(pure) / len(pure)
            t_mixed = sum(mixed) / len(mixed)
            t_mixed_ttft = sum(mixed_ttft) / len(mixed_ttft)
            t_prefill_only = sum(pf_only) / len(pf_only)

        # Hardware decode latency floor (from a sharded probe): above the
        # roofline knee the pure-decode step can't drop below the fixed
        # per-step launch/dispatch overhead. A mixed step carries this decode
        # work plus a prefill chunk, so the same floor is a valid lower bound.
        # Pure-simulate also carries the small-tensor kernel-launch floor.
        floor = self._decode_floor_ms(C)
        if not self._measured_mode:
            floor = max(floor, self._launch_latency_floor_ms())
        if floor > 0.0:
            t_pure = max(t_pure, floor)
            t_mixed = max(t_mixed, floor)
            t_mixed_ttft = max(t_mixed_ttft, floor)

        # Under attention-DP a mixed step carries one prefill chunk *per rank*,
        # each for a different request, and the step is priced unsharded
        # because a rank under DP holds every head of its own chunk. So one
        # step of that length admits ``dp`` requests' chunks, not one: charging
        # every request the full chunk count against the longer step bills the
        # same prefill ``dp`` times over. The request's own service time is
        # still ``n_chunks`` such steps -- that is what TTFT sees below -- but
        # the shared window they ride holds ``dp`` of them at once.
        from .kv_cache import attention_dp_size

        _dp = attention_dp_size(self.cfg)
        n_mixed = (n_chunks / _dp) if _dp > 1 else n_chunks

        # Pure steps per request needed to make up the decode tokens the mixed
        # steps did not cover.
        mixed_tokens = n_mixed * (C - 1) * tok_per_step
        n_pure = max(0.0, (OSL - mixed_tokens) / (C * tok_per_step))
        window_ms = n_pure * t_pure + n_mixed * t_mixed
        if window_ms <= 0:
            window_ms = t_pure

        tpot_ms = C * window_ms / OSL
        system_tps = 1000.0 * OSL / window_ms
        decode_total_ms = tpot_ms * OSL
        total_steps = n_pure + n_mixed
        mixed_fraction = (n_mixed / total_steps) if total_steps > 0 else 0.0
        pollution_pct = (n_mixed * t_mixed / window_ms * 100.0) if window_ms > 0 else 0.0

        return {
            "tpot_ms": tpot_ms,
            "decode_total_ms": decode_total_ms,
            "system_tps": system_tps,
            "pure_step_ms": t_pure,
            "mixed_step_ms": t_mixed,
            # What a request waiting on admission sees, which the prefill anchor
            # moves; ``mixed_step_ms`` is what the decode stream pays, which it
            # does not. Equal unless the anchor is set.
            "mixed_step_ttft_ms": t_mixed_ttft,
            # What one queued prompt adds to another's wait: its own prefill
            # compute and nothing else. The steps it rides were going to run for
            # the decode batch regardless, so their decode work and per-step
            # overhead are not time a waiting prompt is behind. Distinct from
            # the elapsed ``mixed_step_ttft_ms`` above, which is what a prompt
            # being served experiences; see ``_closed_loop_wait_ms``.
            "prefill_demand_ms": float(n_chunks) * t_prefill_only,
            "mixed_step_fraction": mixed_fraction,
            "tpot_pollution_pct": pollution_pct,
            "concurrency": float(C),
            "prefill_chunks": float(n_chunks),
        }

    def _first_token_delay_ms(self, itl_ms: float, step_ms: float, output_len: int) -> tuple:
        """Scheduler delays between "prefill done" and "client sees token 1".

        Two engine knobs sit on this path and neither costs throughput, which is
        why they are set aggressively in production and why a FLOPs-only TTFT
        misses them by an order of magnitude:

        ``admit_ms``
            The decode scheduler polls its receive queue every
            ``decode_admission_steps`` decode steps, so a request that is ready
            mid-window waits the rest of it -- half the window on average.  This
            is real added wall time.

        ``buffered_ms``
            The server flushes the stream only every ``stream_interval`` tokens,
            so the client's first token arrives with the flush that carries it.
            This is *not* added wall time: it moves generation time the client
            already waits out of TPOT and into TTFT, which is why the caller
            deducts it from the decode span rather than adding it to end-to-end
            latency.
        """
        req = self.cfg.request_config
        admit_ms = 0.5 * max(0, int(req.decode_admission_steps)) * max(0.0, step_ms)
        buffered = min(max(0, int(req.stream_interval) - 1), max(0, output_len - 1))
        return admit_ms, buffered * max(0.0, itl_ms)

    @staticmethod
    def _closed_loop_wait_ms(
        service_ms: float, think_ms: float, clients: int, demand_ms: float = None
    ) -> float:
        """Mean response time of one shared server under a closed load.

        A serving benchmark run with ``--max-concurrency C`` is a closed
        network: ``C`` clients each alternate between waiting on the engine
        (``service_ms``, here the prefill) and generating (``think_ms``, the
        decode span, during which the client is not queued for prefill). Exact
        mean-value analysis walks the population up from 1 to ``C``; a request
        arriving into a population of ``k`` sees the queue that ``k-1`` others
        left behind.

        The two limits are the ones that matter. Lightly loaded (long
        generations, short prompts) it returns ``service_ms`` — no queueing.
        Saturated (prompts dominate) it approaches ``C * service_ms - think_ms``,
        the FIFO sweep. Pricing TTFT at either limit alone is wrong by the
        ratio between them, which across Hyperloom's workloads is ~20x.

        ``service_ms`` and ``demand_ms`` are the same number on a FIFO server and
        different ones here, which is the whole reason a continuous-batching
        engine does not behave like a queue of prompts. ``service_ms`` is what
        *my* prefill takes end to end: its chunks ride scheduler steps that are
        also carrying the running decodes, so the elapsed time includes that
        decode work. ``demand_ms`` is what each prompt queued ahead of me adds to
        my wait, which is only its own prefill compute -- those steps were going
        to run for the decode batch whether my prompt existed or not, so their
        decode work and per-step overhead are not a cost my prompt waits for.

        Charging the elapsed step to both roles is what makes the model diverge:
        the decode share grows with the client count, so a queue that should
        grow linearly in ``C`` grows like ``C^2`` and crosses into saturation
        that the real engine never reaches. Defaults to the FIFO reading when no
        demand is given.
        """
        service_ms = max(0.0, service_ms)
        think_ms = max(0.0, think_ms)
        demand_ms = service_ms if demand_ms is None else max(0.0, demand_ms)
        resp, queued = service_ms, 0.0
        for k in range(1, max(1, int(clients)) + 1):
            resp = service_ms + demand_ms * queued
            denom = resp + think_ms
            queued = (k * resp / denom) if denom > 0 else 0.0
        return resp

    def _request_rate_queueing(
        self, system_decode_tps: float, output_len: int, ttft_ms: float, request_latency_ms: float
    ) -> dict[str, float]:
        """First-order open-loop queueing delay for a given offered load.

        Closed-loop (``request_rate == 0`` or ``arrival_model == "closed"``) is
        the legacy behaviour and returns ``{}`` (no adjustment).  Otherwise the
        engine sustains a finite request-completion rate ``mu`` (decode-bound:
        ``system_decode_tps / OSL``); the offered rate ``lambda`` gives a
        utilization ``rho = lambda / mu`` and a queue-wait that is added to TTFT
        and end-to-end latency:

          * poisson      → M/M/1: ``Wq = rho/(1-rho) * (1/mu)``
          * deterministic→ ~D/M/1: roughly half the M/M/1 wait

        At/above saturation (``rho >= 1``) the queue is unbounded; we report a
        large finite penalty + a ``saturated`` flag so the agent ranks it last.
        """
        req = self.cfg.request_config
        rate = float(req.request_rate or 0.0)
        model = (req.arrival_model or "closed").lower()
        osl = max(1, output_len)
        # "none" is an alias of "closed"; "trace" has no closed-form rate and is
        # handled by the DES, so the analytical queue is a no-op for both.
        if rate <= 0.0 or model in ("closed", "none", "trace"):
            return {}
        mu = system_decode_tps / osl if system_decode_tps > 0 else 0.0
        if mu <= 0.0:
            return {}
        rho = rate / mu
        ts_ms = 1000.0 / mu  # mean service time per request
        out: dict[str, float] = {
            "offered_request_rate": rate,
            "max_sustainable_request_rate": mu,
            "utilization": rho,
        }
        if rho >= 1.0:
            out["saturated"] = 1.0
            wq_ms = ts_ms * 1000.0  # large but finite penalty
        else:
            out["saturated"] = 0.0
            wq_ms = rho / (1.0 - rho) * ts_ms
            if model == "deterministic":
                wq_ms *= 0.5
            wq_ms = min(wq_ms, ts_ms * 1000.0)
        out["queue_wait_ms"] = wq_ms
        out["ttft_with_queue_ms"] = ttft_ms + wq_ms
        out["request_latency_with_queue_ms"] = request_latency_ms + wq_ms
        return out

    def _use_continuous_batching(self, concurrency: int, output_len: int) -> bool:
        model = (self.cfg.request_config.serving_model or "continuous").lower()
        # With a single resident sequence there are no concurrent mixed batches,
        # so continuous batching degenerates to the static (pure-decode) case.
        return model == "continuous" and concurrency > 1 and output_len > 0

    # -- comm reporting --------------------------------------------------------

    def _spec_tokens_per_step(self) -> float:
        req = self.cfg.request_config
        spec_k = int(req.speculative_num_tokens or 0)
        accept = float(req.speculative_acceptance_rate or 0.0)
        if spec_k > 0 and 0.0 < accept < 1.0:
            return (1.0 - accept ** (spec_k + 1)) / (1.0 - accept)
        return float(spec_k + 1 if spec_k > 0 else 1)

    def _overlap_keep(self, phase: str, comm_ms: float, compute_ms: float | None) -> float:
        """Exposed-comm fraction (1 - hidden) after compute/comm overlap.

        The configured ``prefill_overlap`` / ``decode_overlap`` is the *ceiling*
        (max fraction of comm hideable behind compute). Physically you cannot
        hide more comm than there is compute to overlap it with, so the
        achievable overlap is ``min(ceiling, compute/comm)``. This makes the
        exposed comm naturally **batch-dependent** and fixes the residual seen at
        high batch: at small batch compute is small relative to the (partly
        fixed) comm, so more comm is exposed; as batch grows compute dominates
        and the overlap saturates at the ceiling.

        ``compute_ms=None`` (e.g. the benchmark-mode reporting path, where layer
        compute is not separately modelled) falls back to the constant ceiling,
        preserving the previous behaviour there.
        """
        ceiling = float(self._cc.prefill_overlap if phase == "prefill" else self._cc.decode_overlap)
        ceiling = min(max(ceiling, 0.0), 1.0)
        if ceiling <= 0.0:
            return 1.0
        if compute_ms is None or comm_ms <= 0.0:
            return 1.0 - ceiling
        hideable = min(ceiling, max(0.0, compute_ms) / comm_ms)
        return 1.0 - min(1.0, max(0.0, hideable))

    def _comm_breakdown(self, batch: int, q_len: int, phase: str) -> CommBreakdown:
        """Explicit per-phase communication breakdown (ms).

        Derived directly from the knob-driven communication model, without the
        analytical *compute* path (``_forward_times``).  This keeps the comm
        report available in **benchmark mode**, where the layer compute comes
        from measured silicon times and the GEMM/SDPA simulators are not built.
        """
        comm = CommBreakdown()
        if self._comm is None:
            return comm
        # Reporting path (no separated compute → constant ceiling, matching the
        # historical breakdown). The projection path applies the batch-dependent
        # overlap in _phase_forward_times.
        keep = self._overlap_keep(phase, 1.0, None)
        tp_ar = self._comm.tp_allreduce_ms(batch, q_len)
        ep_a2a = self._comm.ep_a2a_ms(batch, q_len)
        # MoE keeps only the attention AR when experts are fully EP-distributed
        # (see _moe_tp_allreduce_count); avoids double-counting with the A2A.
        moe_tp_ar = tp_ar * (
            _moe_tp_allreduce_count(self._view) / _dense_tp_allreduce_count(self._view)
        )
        comm.tp_allreduce_ms = (self._n_dense * tp_ar + self._n_moe * moe_tp_ar) * keep
        comm.ep_a2a_ms = self._n_moe * ep_a2a * keep
        comm.pp_p2p_ms = self._comm.pp_p2p_ms(batch, q_len) * keep
        return comm

    def _comm_extras(
        self, batch: int, input_len: int, output_len: int, prefill_batch: int | None = None
    ) -> dict[str, float]:
        """Representative per-phase comm breakdown (ms) for reporting.

        ``prefill_batch`` sizes the prefill breakdown; it defaults to ``batch``
        but is set to the per-request prefill batch (1 under continuous batching)
        so the reported prefill comm matches the per-request TTFT basis.
        """
        if self._comm is None:
            return {}
        pb = prefill_batch if prefill_batch else batch
        spec_k = int(self.cfg.request_config.speculative_num_tokens or 0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        if self._measured_mode:
            # Measured layer times already contain their own overlap, so the
            # step exposes no separable comm; fall back to the standalone cost.
            pre = self._comm_breakdown(pb, input_len, "prefill")
            dec = self._comm_breakdown(batch, q_len, "decode")
        else:
            # Report what the step charged, not what the collective costs in
            # isolation: most of the decode A2A hides behind expert compute, and
            # printing the un-overlapped figure invites reading it as a term of
            # the step latency printed beside it.
            pre = self._forward_times(pb, input_len, "prefill", input_len).comm
            dec = self._forward_times(batch, q_len, "decode", input_len + output_len).comm
        return {
            "comm_prefill_tp_allreduce_ms": pre.tp_allreduce_ms,
            "comm_prefill_ep_a2a_ms": pre.ep_a2a_ms,
            "comm_prefill_pp_p2p_ms": pre.pp_p2p_ms,
            "comm_prefill_total_ms": pre.total_ms,
            "comm_decode_tp_allreduce_ms": dec.tp_allreduce_ms,
            "comm_decode_ep_a2a_ms": dec.ep_a2a_ms,
            "comm_decode_pp_p2p_ms": dec.pp_p2p_ms,
            "comm_decode_total_ms": dec.total_ms,
        }

    # -- top level -------------------------------------------------------------

    def _resolve_hbm_gb(self) -> tuple[float, str]:
        """Per-GPU HBM capacity + where it came from, resolved in order:
        explicit ``--hbm-capacity-gb`` → live device query (GPU node). No default
        is assumed — if neither is available this raises, so the sustainable-
        concurrency number is never computed against a guessed memory size. The
        source string is surfaced so it always states the size it used."""
        args = self._args_ref
        hbm = getattr(args, "hbm_capacity_gb", None) if args else None
        if hbm:
            return float(hbm), "--hbm-capacity-gb"
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024.0**3), f"device({props.name})"
        except Exception:
            pass
        raise ValueError(
            "Per-GPU HBM capacity is required but was not provided: pass "
            "--hbm-capacity-gb (e.g. 192 for MI300X, 256 for MI325X, 288 for "
            "MI355X) or run on a GPU node where the device can be queried. "
            "No default is assumed."
        )

    def _sustainable_concurrency(self) -> tuple[int | None, float, str]:
        """KV-feasible max concurrent sequences at the target context length,
        i.e. how many sequences fit in the HBM left after weights + activations.
        Reuses the memory projection so there is a single sizing formula. Returns
        ``(max_conc_or_None, hbm_gb, hbm_source)``."""
        hbm_gb, source = self._resolve_hbm_gb()
        try:
            from .memory import project_inference_memory

            mem = project_inference_memory(self.cfg, hbm_capacity_gb=hbm_gb, verbose=False)
            return mem.max_concurrent_sequences, hbm_gb, source
        except Exception:
            return None, hbm_gb, source

    def _admission_ceiling(self) -> int:
        """How many requests of the target workload the KV pool can admit.

        Differs from ``_sustainable_concurrency`` in the context it sizes a
        resident request at. That one asks "how many sequences fit *at the
        context ceiling*", which is the right question for whether a config is
        legal and the wrong one for how many requests are resident under load:
        on an agentic replay whose prompts run from 400 tokens to 950k, sizing
        residency at the longest possible request puts the bound around 3-7
        where the hardware admits tens. This sizes it at what a request
        actually holds -- see ``resolved_resident_tokens``.

        The pool itself is taken from ``kv_pool_tokens`` when the caller knows
        what the engine allocated, and otherwise from the memory model.
        """
        req = self.cfg.request_config
        # If the engine told us how big its pool is, believe it rather than
        # reconstructing it. The bound is a ratio of pool to residency, so this
        # is the numerator measured instead of predicted.
        #
        # A KV offload tier deliberately does *not* raise this. The connectors
        # these runs use (vLLM's SimpleCPUOffloadConnector, LMCache) put evicted
        # blocks in host DRAM so a later request can re-read them; a request
        # being decoded still needs its blocks in HBM. Offload buys cache
        # capacity, not admission capacity.
        if req.kv_pool_tokens:
            resident = max(1, req.resolved_resident_tokens())
            return max(1, int(req.kv_pool_tokens) // resident)

        hbm_gb, _ = self._resolve_hbm_gb()
        try:
            from .kv_cache import max_concurrent_sequences
            from .memory import project_inference_memory

            mem = project_inference_memory(self.cfg, hbm_capacity_gb=hbm_gb, verbose=False)
            fraction = req.kv_cache_memory_fraction
            usable = mem.hbm_capacity_bytes or 0
            if fraction:
                usable = int(usable * float(fraction))
            free_for_kv = usable - mem.weight_bytes - mem.activation_bytes
            free_for_kv += req.kv_offload_gb_per_gpu * (1024.0**3)
            return max_concurrent_sequences(
                self.cfg,
                mem.layers_on_rank,
                free_for_kv,
                context_len=req.resolved_resident_tokens(),
            )
        except Exception:
            return 0

    def _admission_wait_ms(
        self, request_latency_ms: float, per_replica: int = 1
    ) -> tuple[float, int]:
        """How long a request waits to be admitted, and the bound it waits on.

        Below the bound this is zero: the pool holds every concurrent request,
        the closed-loop prefill queue already prices the contention, and TTFT is
        a service time. Above it the engine cannot hold all ``C`` clients, so
        only ``C_crit`` are resident and the rest wait outside for a resident
        request to *retire* -- not for a prefill slot to open. What they wait on
        is therefore a whole request latency, prefill plus the entire decode
        span, which is why measured TTFT past the knee grows faster than
        linearly in concurrency: across the published AgentX rows the median
        local slope ``d(log TTFT)/d(log C)`` climbs 0.18, 0.57, 1.35, 2.33
        through KV utilization bands of 0.60, 0.90, 0.99 and above.

        Throughput past the bound is pinned at ``C_crit / latency``, so by
        Little's law a request's sojourn is ``C * latency / C_crit`` and the
        admission wait is the excess over the latency it would have seen
        inside. Nothing here is fitted.

        Scored against 176 published AgentX measurements -- 6 models on 16
        model/stack curves -- this takes median per-curve TTFT rank correlation
        from 0.44 to 0.76 and the regret of picking on it from 3% to 0%, and it
        beats the discrete-event replay at 0.62. It does not cost accuracy in
        either direction: median absolute error is unchanged at ~0.8, because
        what the term fixes is the shape of TTFT in concurrency rather than its
        level, and the term is identically zero wherever the pool does not
        bind, which includes every fixed-sequence workpoint we run.
        """
        ceiling = self._admission_ceiling()
        requested = self.cfg.request_config.resolved_max_concurrency()
        # ``per_replica`` spreads the clients over a pool of replicas, so what
        # competes for one replica's pool is its share of them.
        if per_replica > 1:
            requested = max(1, int(math.ceil(requested / per_replica)))
        if ceiling <= 0 or requested <= ceiling:
            return 0.0, ceiling
        return max(0.0, request_latency_ms) * (requested / ceiling - 1.0), ceiling

    def _effective_concurrency(self) -> dict:
        """Concurrency that drives throughput, reconciled against the KV-feasible
        ceiling (cap + report):

          * no explicit ``max_concurrency`` → use the KV-derived sustainable max
            (instead of ``batch_size``), so a config that frees HBM is scored at
            the load it can actually serve;
          * explicit ``max_concurrency`` above the ceiling → clamp to it and flag;
          * HBM unknown / KV sizing unavailable → fall back to the prior
            ``resolved_max_concurrency()`` behaviour.

        Always records the sustainable max, the concurrency actually used, and
        the HBM capacity + source in ``extras``."""
        req = self.cfg.request_config
        explicit = req.max_concurrency
        sustainable, hbm_gb, hbm_source = self._sustainable_concurrency()
        capped = False
        if sustainable and sustainable > 0:
            if explicit is None:
                concurrency = sustainable
            else:
                concurrency = min(int(explicit), sustainable)
                capped = int(explicit) > sustainable
        else:
            concurrency = req.resolved_max_concurrency()
        concurrency = max(1, int(concurrency))
        extras = {
            "sustainable_concurrency": int(sustainable) if sustainable else 0,
            "concurrency_used": concurrency,
            "hbm_capacity_gb": float(hbm_gb),
            "hbm_capacity_source": hbm_source,
            "concurrency_capped": 1.0 if capped else 0.0,
        }
        return {"concurrency": concurrency, "extras": extras}

    def project(self) -> InferencePerfResult:
        if self.cfg.disaggregation_config and self.cfg.disaggregation_config.enabled:
            return self._project_disaggregated()
        return self._project_colocated()

    def _project_colocated(self) -> InferencePerfResult:
        req = self.cfg.request_config
        batch = max(1, req.batch_size)
        input_len = max(1, req.input_seq_len)
        output_len = max(0, req.output_seq_len)

        conc = self._effective_concurrency()
        concurrency = conc["concurrency"]
        spec_k = int(req.speculative_num_tokens or 0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        replica_gpus = _replica_gpus(self.cfg)

        # TTFT is a per-request quantity, but one engine prefills every
        # concurrent request, so a request also waits behind the prompts queued
        # ahead of it. Under continuous batching a prompt is admitted as chunks
        # riding mixed steps, so its own service time is ``chunks * mixed_step``
        # and the queue it waits in is the closed-loop one below.
        continuous = self._use_continuous_batching(concurrency, output_len)
        m = (
            self._continuous_decode_metrics(input_len, output_len, concurrency)
            if continuous
            else None
        )
        if continuous:
            prefill_service_ms = max(m["prefill_chunks"] * m["mixed_step_ttft_ms"], 0.0)
            # Under attention-DP each rank owns a subset of the requests and
            # runs its own prefill queue over them, so a prompt waits behind
            # its rank's share rather than the whole fleet's. The service time
            # above is already the per-rank one -- a rank under DP holds every
            # head of its own requests -- so charging the global concurrency
            # against it counts the same contention a second time. Same
            # per-replica reasoning the memory model applies under
            # disaggregation.
            from .kv_cache import attention_dp_size

            dp = attention_dp_size(self.cfg)
            queued_clients = max(1, math.ceil(concurrency / dp)) if dp > 1 else concurrency
            ttft = self._closed_loop_wait_ms(
                prefill_service_ms, m["decode_total_ms"], queued_clients, m["prefill_demand_ms"]
            )
            prefill_full_ms = self.prefill_latency_ms(batch, input_len)
        else:
            ttft = prefill_full_ms = self.prefill_latency_ms(batch, input_len)
        # Host prompt-tokenization cost (client sends text; server tokenizes it
        # after the TTFT clock starts). Latency-only, TTFT side -- symmetric with
        # the decode-side detokenization term below. Applied after prefill_full_ms
        # so it never leaks into prefill throughput.
        ttft += max(0.0, self.cfg.request_config.tokenize_overhead_us) / 1000.0 * max(0, input_len)
        # Fixed per-request host cost (accept, parse, admit, cache lookup, KV
        # allocation, stream open). Independent of prompt length, which is what
        # distinguishes it from the tokenization term above.
        ttft += max(0.0, self.cfg.request_config.request_overhead_ms)
        extras = {"speculative_tokens_per_step": self._spec_tokens_per_step()}
        extras.update(conc["extras"])

        if continuous:
            # Continuous batching: TPOT is the blended pure/mixed steady state.
            decode_total = m["decode_total_ms"]
            itl = m["tpot_ms"]
            step_latency = m["pure_step_ms"]
            decode_tps = m["system_tps"]
            per_req_decode_tps = (1000.0 / itl) if itl > 0 else 0.0
            extras.update(
                {
                    "serving_continuous_batching": 1.0,
                    "concurrency": m["concurrency"],
                    "pure_step_latency_ms": m["pure_step_ms"],
                    "mixed_step_latency_ms": m["mixed_step_ms"],
                    "mixed_step_fraction": m["mixed_step_fraction"],
                    "tpot_pollution_pct": m["tpot_pollution_pct"],
                }
            )
        else:
            decode_total = self.decode_total_ms(batch, input_len, output_len)
            mid_ctx = input_len + output_len // 2
            step_latency = self._decode_step_latency_ms(batch, mid_ctx, q_len=q_len)
            itl = (decode_total / output_len) if output_len > 0 else step_latency
            per_req_decode_tps = (1000.0 / itl) if itl > 0 else 0.0
            # One step emits ``_spec_tokens_per_step()`` tokens per sequence, not
            # one. ``step_latency`` is already the longer verify step, so without
            # the matching numerator speculation showed up as a throughput loss
            # here -- the bug the disaggregated path above was fixed for, which
            # this static branch kept.
            decode_tps = (
                (batch * self._spec_tokens_per_step() * 1000.0 / step_latency)
                if step_latency > 0
                else 0.0
            )

        # Per-token detokenization + streaming (client-side host cost). Serving
        # harnesses measure ITL client-side, so it carries this; the GPU decode
        # step does not. Latency-only: it overlaps the next server step, so
        # aggregate decode throughput is unchanged.
        detok_ms = max(0.0, self.cfg.request_config.detokenize_overhead_us) / 1000.0
        if detok_ms:
            itl += detok_ms
            decode_total += detok_ms * max(0, output_len)
            per_req_decode_tps = (1000.0 / itl) if itl > 0 else 0.0

        admit_ms, buffered_ms = self._first_token_delay_ms(itl, step_latency, output_len)
        ttft += admit_ms + buffered_ms
        decode_total = max(0.0, decode_total - buffered_ms)
        if output_len > 1:
            itl = decode_total / (output_len - 1)
            per_req_decode_tps = (1000.0 / itl) if itl > 0 else 0.0

        request_latency = ttft + decode_total

        # Admission. Everything above prices the engine's own queue, which
        # assumes every concurrent request is resident; the KV pool decides
        # whether it can be. Where the pool binds, the excess clients wait
        # outside for a resident request to retire, and that wait is the
        # dominant term in TTFT -- it is not a correction to it.
        admission_ms, admission_ceiling = self._admission_wait_ms(request_latency)
        if admission_ms > 0.0:
            ttft += admission_ms
            request_latency += admission_ms
        extras["admission_ceiling"] = float(admission_ceiling)
        extras["admission_wait_ms"] = float(admission_ms)
        extras["resident_tokens_per_request"] = float(
            self.cfg.request_config.resolved_resident_tokens()
        )

        decode_tps_per_gpu = decode_tps / replica_gpus if replica_gpus else 0.0
        prefill_tps = (batch * input_len * 1000.0 / prefill_full_ms) if prefill_full_ms > 0 else 0.0

        # Offered-load queueing (open-loop). The offered-load queue wait is the
        # client-side wait for a serving slot; like the vLLM / InferenceX harness
        # (whose TTFT clock starts after the request is admitted), we keep it OUT
        # of the primary TTFT and end-to-end latency and expose it separately
        # (queue_wait_ms, ttft_with_queue_ms, request_latency_with_queue_ms).
        # No-op unless a request rate is set. TPOT / throughput are steady-state
        # and unaffected either way.
        q = self._request_rate_queueing(decode_tps, output_len, ttft, request_latency)
        if q:
            extras.update(q)

        extras.update(
            self._comm_extras(
                batch, input_len, output_len, prefill_batch=(1 if continuous else batch)
            )
        )
        if self.is_benchmark_calibrated:
            extras["benchmark_calibrated"] = 1.0

        return InferencePerfResult(
            ttft_ms=ttft,
            decode_total_ms=decode_total,
            itl_ms=itl,
            request_latency_ms=request_latency,
            per_request_decode_tps=per_req_decode_tps,
            decode_throughput_tps=decode_tps,
            decode_throughput_tps_per_gpu=decode_tps_per_gpu,
            prefill_throughput_tps=prefill_tps,
            decode_step_latency_ms=step_latency,
            replica_gpus=replica_gpus,
            extras=extras,
        )

    def _kv_transfer_ms(
        self, decode_proj: InferencePerformanceProjector, batch: int, input_len: int
    ) -> float:
        """KV-cache transfer time prefill→decode worker, for ONE request's KV.

        Both callers are per-request quantities -- the TTFT this handoff delays,
        and the closed-loop think time, which is one client's round trip. Sizing
        the transfer at ``concurrency=batch`` charged every request for moving
        the whole resident batch's KV, which made TTFT scale linearly with
        concurrency (564 ms at batch 16 to 6813 ms at batch 224 on GLM-5.2 at
        100k) purely from a handoff that is the same size for each request.
        Contention between concurrent handoffs belongs in the pool's throughput,
        not multiplied into one request's first-token latency.
        """
        from .kv_cache import estimate_kv_cache

        disagg = self.cfg.disaggregation_config
        layers_on_rank = _layers_on_rank(decode_proj.cfg)
        kv = estimate_kv_cache(
            decode_proj.cfg, layers_on_rank, concurrency=1, context_len=input_len
        )
        comm = decode_proj._comm or self._comm
        if comm is None:
            # No collective model available; fall back to a direct bytes/bw calc.
            from .collectives import InferenceCollectiveModel

            comm = InferenceCollectiveModel(
                decode_proj.cfg.model_config,
                decode_proj.cfg.model_parallel_config,
                decode_proj.cfg.collective_config,
            )
        return comm.kv_transfer_ms(
            kv.bytes_total,
            bw_gbps=disagg.resolved_kv_transfer_bw_gbps(),
            latency_us=disagg.resolved_kv_transfer_latency_us(),
        )

    def pool_projectors(
        self,
    ) -> tuple[InferencePerformanceProjector, InferencePerformanceProjector]:
        """One projector per pool, each at its own parallelism and anchor.

        Exposed because the DES needs the same two cost models the closed-form
        path builds. A simulator handed the disaggregated projector alone would
        have to price both pools from the parent's parallelism, which is the
        colocated layout and belongs to neither pool.
        """
        from dataclasses import replace

        disagg = self.cfg.disaggregation_config
        mp = self.cfg.model_parallel_config
        # Disaggregation is disabled on the sub-configs to avoid recursion.
        prefill_cfg = replace(
            self.cfg,
            model_parallel_config=disagg.prefill_parallel(mp),
            disaggregation_config=replace(disagg, enabled=False),
        )
        decode_cfg = replace(
            self.cfg,
            model_parallel_config=disagg.decode_parallel(mp),
            disaggregation_config=replace(disagg, enabled=False),
        )
        return (
            InferencePerformanceProjector(
                prefill_cfg,
                args=self._args_ref,
                benchmark_layer_times=self._pool_anchor("prefill"),
            ),
            InferencePerformanceProjector(
                decode_cfg,
                args=self._args_ref,
                benchmark_layer_times=self._pool_anchor("decode"),
            ),
        )

    def kv_handoff_ms(self, decode_proj: InferencePerformanceProjector, context_len: int) -> float:
        """Prefill→decode KV handoff for one request of ``context_len`` tokens.

        The DES charges this per request at the moment prefill retires, which is
        where it actually lands; the closed-form path folds the same quantity
        into TTFT.
        """
        return self._kv_transfer_ms(decode_proj, 1, max(1, int(context_len)))

    def _pool_anchor(self, pool: str):
        """The measurement a disaggregated pool is calibrated against.

        The two pools are not the same experiment. They run at different
        parallelism and attention layout, and one is compute-bound on long
        prompts while the other is memory-bound on single tokens, so the single
        colocated artifact ``--load-benchmark`` supplies is a compromise that
        describes neither exactly. A pool that names its own measurement gets
        it; the shared anchor stays the fallback, so a run without the per-pool
        flags is unchanged.
        """
        return self._pool_benchmarks.get(pool) or self._bench_measured

    def _project_disaggregated(self) -> InferencePerfResult:
        req = self.cfg.request_config
        # PDD is a continuous serving topology: the load on the two pools is
        # the resolved in-flight concurrency, not the static microbatch field.
        # Using ``batch_size`` here made --max-concurrency a no-op only when
        # disaggregation was enabled, so the same requested load described two
        # different experiments on the colocated and PDD paths.
        conc = self._effective_concurrency()
        batch = conc["concurrency"]
        input_len = max(1, req.input_seq_len)
        output_len = max(0, req.output_seq_len)
        disagg = self.cfg.disaggregation_config

        for _pool in ("prefill", "decode"):
            if self._pool_benchmarks.get(_pool):
                print(f"[inferasim:Inference] {_pool} pool calibrated from its own anchor")
        prefill_proj, decode_proj = self.pool_projectors()
        prefill_cfg, decode_cfg = prefill_proj.cfg, decode_proj.cfg

        # Decode phase on the decode pool (drives ITL + decode throughput).
        # Computed first because the prefill queue below needs the generation
        # span as its think time.
        # ``batch`` is the system-wide concurrency; each decode replica holds
        # only its own share, exactly as the prefill queue below divides by
        # ``prefill_replicas``. Sizing the step at the full batch overstated the
        # step latency, and then scaling that step's throughput by the replica
        # count counted the same sequences once per replica.
        # A non-divisible population leaves one replica with the ceiling share.
        # Price that limiting replica; floor division silently dropped requests
        # and over-stated both latency and throughput scaling.
        decode_loads = _split_replica_loads(batch, disagg.decode_replicas)
        decode_batch = max(decode_loads)
        decode_total = decode_proj.decode_total_ms(decode_batch, input_len, output_len)
        mid_ctx = input_len + output_len // 2
        spec_k = int(req.speculative_num_tokens or 0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        step_latency = decode_proj._decode_step_latency_ms(decode_batch, mid_ctx, q_len=q_len)

        # Prefill phase on the prefill pool (drives TTFT + prefill throughput).
        # Requests that are decoding are not on these GPUs. A FIFO of
        # uncontended singles over the whole resident population tracked C * S
        # and saturated 100x high; a batched forward of C / prefill_replicas
        # did the same whenever S(n) is linear, because it still assumed every
        # in-flight client is prefilling at once. Closed-loop occupancy with
        # think time = generation puts only Little's N_p on the station:
        # long decode (MiniMax 8k/1k, agentic) collapses toward one uncontended
        # prompt; a prefill-heavy load still batches the people who are
        # actually waiting. Throughput below still prices the offered load --
        # occupancy is a latency quantity, not a capacity one.
        prefill_loads = _split_replica_loads(batch, disagg.prefill_replicas)
        per_replica = max(prefill_loads)
        kv_transfer = self._kv_transfer_ms(decode_proj, batch, input_len)
        s1 = prefill_proj.prefill_latency_ms(1, input_len)
        resp1 = self._closed_loop_wait_ms(s1, decode_total, per_replica)
        denom = resp1 + decode_total
        n_prefill = per_replica
        if denom > 0:
            n_prefill = max(1, min(per_replica, int(round(per_replica * resp1 / denom))))
        prefill_full_ms = prefill_proj.prefill_latency_ms(n_prefill, input_len)
        ttft_compute = prefill_full_ms
        # Host prompt-tokenization cost (latency-only, TTFT side).
        tok_ms = max(0.0, req.tokenize_overhead_us) / 1000.0 * max(0, input_len)
        # Fixed per-request host cost; see the co-located path for the rationale.
        ttft = ttft_compute + kv_transfer + tok_ms + max(0.0, req.request_overhead_ms)

        itl = (decode_total / output_len) if output_len > 0 else step_latency
        # Per-token detokenization + streaming (latency-only; see the co-located
        # projection path for the rationale).
        detok_ms = max(0.0, req.detokenize_overhead_us) / 1000.0
        if detok_ms:
            itl += detok_ms
            decode_total += detok_ms * max(0, output_len)

        admit_ms, buffered_ms = self._first_token_delay_ms(itl, step_latency, output_len)
        ttft += admit_ms + buffered_ms
        decode_total = max(0.0, decode_total - buffered_ms)
        if output_len > 1:
            itl = decode_total / (output_len - 1)
        request_latency = ttft + decode_total

        # Admission, which this path did not price at all. The KV pool binds on
        # the decode side -- that is where a sequence's blocks live for the
        # whole generation, and the memory pass already re-binds to that pool --
        # so a split is no less capable of over-subscribing it than a colocated
        # engine is. Charged per decode replica, because the clients are spread
        # across the pool and it is one replica's residency that binds.
        #
        # On the published disaggregated DeepSeek rows this term is identically
        # zero, and that is the finding rather than a failure of it: moving
        # prefill off the decode GPUs leaves a pool that admits ~770 sequences
        # against the 256 ever offered, so the split never reaches its knee.
        # Their TTFT still grows 9x over that range, which means for a split
        # the binding queue is the *prefill* pool's and not the KV pool's --
        # the closed-form knee ranks those seven rows at rho 0.286 where the
        # replay, which simulates that queue, gets 0.71-0.96. Price a split's
        # TTFT from the replay; this term is here for the case where the decode
        # pool is genuinely over-subscribed, which one replica with a small
        # pool still can be.
        admission_ms, admission_ceiling = self._admission_wait_ms(
            request_latency, per_replica=max(1, int(disagg.decode_replicas or 1))
        )
        if admission_ms > 0.0:
            ttft += admission_ms
            request_latency += admission_ms

        per_req_decode_tps = (1000.0 / itl) if itl > 0 else 0.0

        # Per-replica decode throughput, scaled by the decode-pool replica count.
        # A speculative step verifies ``spec_k + 1`` tokens and keeps however many
        # the target accepts, so it emits ``_spec_tokens_per_step()`` tokens rather
        # than one. ``step_latency`` above is already the longer verify step; without
        # the matching numerator, turning MTP on lengthened the step, credited no
        # extra tokens, and reported speculation as a throughput *loss* even as
        # per-request TPOT improved. The co-located path gets this from the
        # continuous-batching model's ``system_tps``.
        spec_tokens = self._spec_tokens_per_step()
        decode_tps = 0.0
        for load in decode_loads:
            latency = decode_proj._decode_step_latency_ms(load, mid_ctx, q_len=q_len)
            if latency > 0:
                decode_tps += load * spec_tokens * 1000.0 / latency

        # Capacity of the prefill pool at the offered load, not at the
        # occupancy that sets TTFT. Using the occupancy batch here would
        # report a nearly-idle station as unable to feed decode.
        prefill_tps = 0.0
        for load in prefill_loads:
            prefill_cap_ms = prefill_proj.prefill_latency_ms(load, input_len)
            if prefill_cap_ms > 0:
                prefill_tps += load * input_len * 1000.0 / prefill_cap_ms

        # In steady state the decode pool can only run what the prefill pool
        # hands it, so the system request rate is the smaller of the two.
        # Without this an under-provisioned prefill pool reported the decode
        # pool's unfed ceiling: low prefill:decode ratios came out optimistic
        # and throughput did not respond to the prefix-cache hit rate at all.
        if input_len > 0 and output_len > 0:
            supply = prefill_tps / input_len  # requests/s the prefill pool feeds
            demand = decode_tps / output_len  # requests/s the decode pool could run
            if 0.0 < supply < demand:
                decode_tps = supply * output_len

        decode_replica_gpus = _replica_gpus(decode_cfg)
        prefill_replica_gpus = _replica_gpus(prefill_cfg)
        total_decode_gpus = decode_replica_gpus * max(1, disagg.decode_replicas)
        decode_tps_per_gpu = decode_tps / total_decode_gpus if total_decode_gpus else 0.0

        extras = {"speculative_tokens_per_step": self._spec_tokens_per_step()}
        extras.update(conc["extras"])
        extras.update(
            decode_proj._comm_extras(decode_batch, input_len, output_len, prefill_batch=1)
        )
        if self.is_benchmark_calibrated:
            extras["benchmark_calibrated"] = 1.0
        extras["prefill_compute_ttft_ms"] = ttft_compute
        extras["admission_ceiling"] = float(admission_ceiling)
        extras["admission_wait_ms"] = float(admission_ms)
        extras["resident_tokens_per_request"] = float(req.resolved_resident_tokens())
        extras["prefill_replicas"] = float(disagg.prefill_replicas)
        extras["decode_replicas"] = float(disagg.decode_replicas)
        extras["prefill_occupancy"] = float(n_prefill)
        # The step decomposition, which this path reported as nothing at all.
        # A disaggregated decode pool is continuously batched -- the load on it
        # is the resolved in-flight concurrency, sized above -- and no prefill
        # chunk ever lands in one of its batches, because moving prefill off
        # these GPUs is the entire point of the topology. So the honest reading
        # is a mixed-step fraction of exactly zero, not an absent one.
        #
        # Reporting nothing dropped ``decode_step_ms_pure``,
        # ``mixed_step_fraction_pct`` and ``tpot_pollution_pct`` from every
        # disaggregated row, which are precisely the objectives that show what
        # the split buys: the colocated projection at the same shape spends
        # 6.2% of its steps mixed and pays 33% TPOT pollution for it, and the
        # search could not credit the alternative because the alternative
        # reported no number to compare.
        #
        # The cost of a mixed step is the pure step rather than zero. There is
        # no such step to price, and a zero would hand the topology a free win
        # on a minimized objective instead of saying the step never happens.
        extras["serving_continuous_batching"] = 1.0
        extras["concurrency"] = float(batch)
        extras["pure_step_latency_ms"] = step_latency
        extras["mixed_step_latency_ms"] = step_latency
        extras["mixed_step_fraction"] = 0.0
        extras["tpot_pollution_pct"] = 0.0

        return InferencePerfResult(
            ttft_ms=ttft,
            decode_total_ms=decode_total,
            itl_ms=itl,
            request_latency_ms=request_latency,
            per_request_decode_tps=per_req_decode_tps,
            decode_throughput_tps=decode_tps,
            decode_throughput_tps_per_gpu=decode_tps_per_gpu,
            prefill_throughput_tps=prefill_tps,
            decode_step_latency_ms=step_latency,
            replica_gpus=decode_replica_gpus,
            is_disaggregated=True,
            kv_transfer_ms=kv_transfer,
            prefill_replica_gpus=prefill_replica_gpus,
            decode_replica_gpus=decode_replica_gpus,
            extras=extras,
        )


def project_inference_performance(
    inference_config: InferenceConfig, args=None, benchmark_layer_times=None
) -> InferencePerfResult:
    return InferencePerformanceProjector(
        inference_config, args=args, benchmark_layer_times=benchmark_layer_times
    ).project()
