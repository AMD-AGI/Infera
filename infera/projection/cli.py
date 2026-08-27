###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""inferasim inference/serving projection CLI.

Analytical + GPU-calibrated projection of TTFT / ITL / throughput / KV-cache
for a serving recipe, plus discrete-event serving simulation. This is the
``inferasim`` product surface over the first-class ``infera.projection.*``
modules.

Examples
--------
Project a recipe from a saved 1-GPU anchor (no GPU needed)::

    python -m infera.projection inference --config <model.yaml> \
        --input-len 1024 --output-len 1024 --inference-batch-size 32 \
        --gpu-arch mi355x --load-benchmark anchor.json

Harvest a GPU-calibrated anchor (needs a ROCm GPU + vLLM), then project::

    python -m infera.projection anchor --model <hf-or-model-name> \
        --benchmark-gpus 1 --save anchor.json
"""
import argparse
import sys

from infera.projection.core.launcher.parser import add_pretrain_parser
from infera.projection.core.projection.inference_projection import (
    launch_projection_from_cli,
)

import argparse as _argparse  # noqa: F401  (used by arg helpers)

def _add_pipeline_schedule_algorithm_arg(parser):
    parser.add_argument(
        "--pipeline-schedule-algorithm",
        type=str,
        required=False,
        default="auto",
        choices=[
            "auto",
            "zerobubble",
            "zerobubble-heuristic",
            "zbv-formatted",
            "zbv-greedy-half",
            "zbv-greedy-min",
            "seaailab-ilp",
            "all",
        ],
        help=(
            "Pipeline schedule for validation and (perf) simulation. "
            "Must not be combined with activation recompute — split-wgrad "
            "schedules pin inputs that recompute cannot free."
        ),
    )


def _add_save_benchmark_arg(parser):
    """``--save-benchmark`` (with deprecated ``--save-profiling`` alias).

    The bench artifact (timing + memory) is shared between perf and
    memory projections; both subparsers accept the same save flag.
    """
    parser.add_argument(
        "--save-benchmark",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to write the bench artifact JSON (timing + memory). "
            "The artifact is shareable: a single bench run can feed both "
            "perf and memory projections via --load-benchmark."
        ),
    )
    parser.add_argument(
        "--save-profiling",
        type=str,
        required=False,
        default=None,
        help=_argparse.SUPPRESS,  # deprecated alias for --save-benchmark
    )


def _add_load_benchmark_arg(parser, *, include_compute_baseline_alias: bool):
    """``--load-benchmark`` (skip bench, project from saved artifact).

    Memory subparser: accept ``--compute-baseline`` as a deprecated alias
    (same semantic — "load a saved bench artifact").

    Perf subparser: do NOT alias ``--compute-baseline`` — that flag has a
    different meaning on the perf side (the bg=1 hybrid baseline), and
    it is registered separately.
    """
    parser.add_argument(
        "--load-benchmark",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to a previously saved bench artifact JSON. When provided, "
            "the bench is skipped and the projection runs directly from the "
            "loaded measurements."
        ),
    )
    parser.add_argument(
        "--load-benchmark-scaling",
        type=str,
        action="append",
        required=False,
        default=None,
        metavar="PATH[,PATH...]",
        help=(
            "Extra bench artifacts for the same model captured at other "
            "--benchmark-gpus values, used to fit how the step scales with TP "
            "instead of assuming TP^-1. Repeatable or comma-separated; only "
            "matters when the benchmark parallelism differs from the target."
        ),
    )
    parser.add_argument(
        "--decode-floor-benchmark",
        type=str,
        required=False,
        default=None,
        metavar="PATH",
        help=(
            "Path to a sharded (e.g. 2-GPU EP-sharded) bench artifact whose "
            "measured decode step defines the hardware latency floor. The "
            "restored decode step is capped from below at this floor per batch "
            "(decode = max(restored, floor)). Above the roofline knee, decode "
            "latency is fixed by per-step launch/dispatch overhead and is "
            "parallelism-invariant, so a single sharded probe supplies the "
            "floor for all more-sharded targets. No-op below the floor."
        ),
    )
    if include_compute_baseline_alias:
        parser.add_argument(
            "--compute-baseline",
            type=str,
            required=False,
            default=None,
            help=_argparse.SUPPRESS,  # deprecated memory-side alias for --load-benchmark
        )


def _add_perf_compute_baseline_arg(parser):
    """Perf-side ``--compute-baseline`` (bg=1 hybrid baseline).

    Distinct semantic from ``--load-benchmark``: this is a *secondary*
    artifact used to source clean compute timings during EP-reduced
    benches.  Kept hidden from --help (internal/subprocess use).
    """
    parser.add_argument(
        "--compute-baseline",
        type=str,
        required=False,
        default=None,
        help=_argparse.SUPPRESS,
    )


def _add_topology_args(parser):
    """``--target-nodes`` / ``--benchmark-gpus`` — shared between memory and perf."""
    parser.add_argument(
        "--target-nodes",
        type=int,
        required=False,
        default=None,
        help=(
            "Target number of nodes for projection. "
            "If not specified, defaults to the minimum nodes required by "
            "the parallelism config (TP × PP × CP / GPUs_per_node)."
        ),
    )
    parser.add_argument(
        "--benchmark-gpus",
        type=int,
        required=False,
        default=None,
        help=(
            "Number of GPUs to use for the underlying bench. When set lower "
            "than GPUS_PER_NODE, enables sub-node benchmarking with "
            "analytical upscaling. Defaults to GPUS_PER_NODE."
        ),
    )


def _add_performance_args(parser):
    """All the perf-specific knobs (profiling-mode, gpu-arch, schedules, etc.)."""
    parser.add_argument(
        "--hardware-config",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to YAML file with hardware configuration for collective communication modeling. "
            "If not provided, uses default cluster parameters.\n\n"
        ),
    )
    parser.add_argument(
        "--profiling-mode",
        type=str,
        required=False,
        default="benchmark",
        choices=["benchmark", "simulate", "both"],
        help=(
            "Profiling mode for layer timing:\n"
            "  benchmark  - Measure on real GPUs (needs GPUs, a serving engine\n"
            "               and --bench-model), then calibrate to what was measured\n"
            "  simulate   - Use simulation backends (origami for GEMM,\n"
            "               analytical model for SDPA). No GPU required.\n"
            "  both       - Run both benchmark and simulation, report side-by-side\n"
            "'inference' defaults to 'simulate', so a projection needs no GPU\n"
            "unless you ask for one; 'anchor' always measures.\n"
        ),
    )
    parser.add_argument(
        "--bench-model",
        type=str,
        required=False,
        default=None,
        help=(
            "Checkpoint to serve while measuring (HF id or local path), for\n"
            "--profiling-mode benchmark. Required there because a structural\n"
            "config describes an architecture, not a checkpoint. Weights are\n"
            "served random, so the id only has to name the right shape.\n"
            "Env: INFERASIM_BENCH_MODEL.\n"
        ),
    )
    parser.add_argument(
        "--gemm-backend",
        type=str,
        required=False,
        default=None,
        choices=["origami"],
        help=(
            "GEMM simulation backend (only used when --profiling-mode is 'simulate' or 'both').\n"
            "  origami  - Open-source GEMM performance model (default)\n"
        ),
    )
    parser.add_argument(
        "--gpu-arch",
        type=str,
        required=False,
        default=None,
        help=(
            "Target GPU architecture for simulation (e.g. 'mi300x', 'gfx942', 'mi355x', 'gfx950').\n"
            "If not specified, auto-detected or uses INFERASIM_GPU_ARCH env var.\n"
        ),
    )
    parser.add_argument(
        "--gpu-clock-mhz",
        type=int,
        required=False,
        default=None,
        help=(
            "Override the GPU compute clock frequency in MHz for simulation.\n"
            "If not specified, uses the default from the hardware profile for the\n"
            "given --gpu-arch (e.g. 2100 MHz for MI300X/MI325X).\n"
            "Can also be set via the INFERASIM_GPU_CLOCK_MHZ env var.\n"
            "Example: --gpu-clock-mhz 1500\n"
        ),
    )
    _add_pipeline_schedule_algorithm_arg(parser)
    # Projection-specific overrides.
    parser.add_argument(
        "--target-num-nodes",
        type=int,
        required=False,
        default=None,
        help="Target number of nodes for multinode projection (alias for --target-nodes).",
    )
    parser.add_argument(
        "--target-ep-size",
        type=int,
        required=False,
        default=None,
        help="Override expert_model_parallel_size for projection target.",
    )
    parser.add_argument(
        "--enable-zero-bubble",
        action="store_true",
        default=False,
        help="Enable zero-bubble pipeline scheduling.",
    )
    parser.add_argument(
        "--enable-deepep",
        action="store_true",
        default=False,
        help="Enable DeepEP (async All-to-All overlap with compute).",
    )
    parser.add_argument(
        "--sync-free-stage",
        type=int,
        required=False,
        default=0,
        help="SyncFree MoE stage (0=off, 1=fused router, 2=+DeepEP+grouped, 3=+fused act). Auto-enables DeepEP.",
    )
    parser.add_argument(
        "--num-virtual-stages-per-pipeline-rank",
        type=int,
        required=False,
        default=None,
        help="Override virtual_pipeline_model_parallel_size (VPP) for projection.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        required=False,
        default=None,
        help="Override micro_batch_size for projection.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        required=False,
        default=None,
        help="Override global_batch_size for projection.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        default=False,
        help=_argparse.SUPPRESS,
    )


def _add_inference_args(parser):
    """Inference / serving projection knobs.

    Reuses ``--gpu-arch`` / ``--gpu-clock-mhz`` / ``--gemm-backend`` from the
    perf arg group (added separately) for the simulation backends.
    """
    parser.add_argument(
        "--inference-mode",
        type=str,
        required=False,
        default="both",
        choices=["performance", "memory", "both"],
        help="Which inference projection to run (default: both).",
    )
    # ---- Request / serving workload ----
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Prompt length in tokens (prefill). Defaults to the config seq_length.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Number of tokens to generate (decode steps).",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help="Number of sequences processed together per decode forward.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Max resident sequences for KV-cache sizing (default: batch size).",
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=None,
        help="Largest context (prompt+generated) for KV sizing (default: input+output).",
    )
    # ---- Precision ----
    parser.add_argument(
        "--weight-dtype",
        type=str,
        default=None,
        help="Resident weight precision (bf16 | fp8 | ...). Default: bf16.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default=None,
        help="KV-cache precision (bf16 | fp8 | int8 | ...). Default: bf16.",
    )
    # ---- Serving features ----
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=None,
        help="Chunked-prefill chunk size in tokens (0 disables).",
    )
    parser.add_argument(
        "--prefix-cache-hit-rate",
        "--prefix-hit-fraction",
        dest="prefix_cache_hit_rate",
        type=float,
        default=None,
        help="Prefix-cache hit rate in [0,1): fraction of each prompt already "
        "resident in the KV cache (automatic prefix caching / shared prefix "
        "reuse). Cached tokens skip prefill, so TTFT scales with (1 - rate). "
        "Default: 0 (cold cache).",
    )
    parser.add_argument(
        "--speculative-num-tokens",
        type=int,
        default=None,
        help="Draft tokens proposed per speculative verify step (0 disables).",
    )
    parser.add_argument(
        "--speculative-acceptance-rate",
        type=float,
        default=None,
        help="Expected per-token acceptance rate for speculative decoding [0,1].",
    )
    # ---- Token sampling / logits post-processing ----
    parser.add_argument(
        "--no-sampling",
        dest="no_sampling",
        action="store_true",
        help="Drop the token-sampling / logits post-processing cost from each "
        "step (compare against a logits-free forward). Default: sampling modelled.",
    )
    parser.add_argument(
        "--sampling-top-k",
        type=int,
        default=None,
        help="Top-k sampling width (adds a partial-sort pass over the logits). "
        "0 = greedy/argmax (cheapest). Default: 0.",
    )
    parser.add_argument(
        "--sampling-top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling threshold in (0,1]. <1 adds a "
        "threshold+renormalize pass over the logits. Default: 1.0 (off).",
    )
    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=None,
        help="Sampling temperature. !=1 adds a fused logits-scale pass. "
        "Default: 1.0.",
    )
    # ---- Runtime activation quantization / cast ----
    parser.add_argument(
        "--act-quant-dtype",
        choices=["none", "bf16", "fp8", "mxfp4"],
        default=None,
        help="Precision the runtime casts activations to before each "
        "low-precision GEMM (memory-bound cast overhead). Default: auto-detect "
        "from --weight-dtype / model fp8; 'none'/'bf16' drops the cast.",
    )
    # ---- Capacity ----
    parser.add_argument(
        "--hbm-capacity-gb",
        type=float,
        default=None,
        help="Per-GPU HBM capacity (GB). When set, reports fit + max concurrency.",
    )
    parser.add_argument(
        "--kv-cache-memory-fraction",
        type=float,
        default=None,
        help="Fraction of HBM the engine may use (vLLM gpu_memory_utilization / "
        "SGLang mem_fraction_static). Bounds usable HBM + max concurrency. Default: full HBM.",
    )
    parser.add_argument(
        "--kv-block-size",
        type=int,
        default=None,
        help="Paged-KV block (page) size in tokens (vLLM block_size, e.g. 16). "
        "Per-sequence context is rounded up to whole blocks, inflating KV bytes "
        "and lowering max concurrency. Default: 0 (no paging / contiguous).",
    )
    parser.add_argument(
        "--kv-offload-gb-per-gpu",
        type=float,
        default=None,
        help="Host DRAM per GPU used as a second KV tier (TRT-LLM native / "
        "SGLang HiCache 'dram' offload). Holds KV for idle sessions and prefix "
        "blocks evicted from HBM, raising sustainable concurrency; a prefix hit "
        "it holds is fetched over the host link instead of recomputed. "
        "Default: 0 (offload disabled).",
    )
    parser.add_argument(
        "--kv-offload-bw-gbps",
        type=float,
        default=None,
        help="Host<->device bandwidth for the KV offload tier in GB/s. "
        "PCIe 5 x16 is ~64; Grace-Blackwell NVLink-C2C is ~900. Default: 64.",
    )
    # ---- Feature B: custom collective ops ----
    coll = parser.add_argument_group("inference collectives (feature B)")
    coll.add_argument(
        "--comm-model",
        type=str,
        default=None,
        choices=["explicit", "builtin"],
        help="Communication model: 'explicit' (knob-driven breakdown, default) "
        "or 'builtin' (folded into layer time, no breakdown).",
    )
    coll.add_argument(
        "--tp-allreduce-algo",
        type=str,
        default=None,
        choices=["auto", "ring", "one_shot", "two_shot", "hierarchical"],
        help="Force the TP AllReduce algorithm (default: auto = fastest).",
    )
    coll.add_argument(
        "--ep-a2a-algo",
        type=str,
        default=None,
        choices=["auto", "direct", "single_shot", "hierarchical"],
        help="Force the EP AllToAll algorithm (default: auto = fastest).",
    )
    coll.add_argument(
        "--prefill-comm-overlap",
        type=float,
        default=None,
        help="Fraction of prefill comm hidden behind compute [0,1] (default 0).",
    )
    coll.add_argument(
        "--decode-comm-overlap",
        type=float,
        default=None,
        help="Fraction of decode comm hidden behind compute [0,1] (default 0).",
    )
    coll.add_argument(
        "--tp-allreduce-efficiency",
        type=float,
        default=None,
        help="TP AllReduce time multiplier (<1 = fused-op speedup, default 1.0).",
    )
    coll.add_argument(
        "--ep-a2a-efficiency",
        type=float,
        default=None,
        help="EP AllToAll time multiplier (<1 = fused/overlapped speedup, default 1.0).",
    )
    coll.add_argument(
        "--quick-reduce",
        action="store_true",
        default=False,
        help="ROCm 'quick reduce': low-latency quantized all-reduce for small "
        "messages (extra TP AllReduce speedup).",
    )
    coll.add_argument(
        "--fuse-rmsnorm-allreduce",
        action="store_true",
        default=False,
        help="Fused RMSNorm + AllReduce: hides part of the TP all-reduce latency "
        "behind the norm (extra TP AllReduce speedup).",
    )
    coll.add_argument(
        "--ep-load-balance",
        type=float,
        default=None,
        help="MoE expert routing imbalance: hottest-rank / mean token-load ratio "
        "(1.0 = perfectly balanced). Inflates MoE expert-compute time on EP>1. Default 1.0.",
    )
    coll.add_argument(
        "--redundant-experts",
        type=int,
        default=None,
        help="Extra replicated expert slots (EPLB) that reduce realized MoE routing "
        "imbalance. Default 0.",
    )
    par = parser.add_argument_group("inference parallelism")
    par.add_argument(
        "--attention-dp-size",
        type=int,
        default=None,
        help="Run attention data-parallel across this many ranks, splitting the "
        "running requests between them while the MLP/MoE stay tensor/expert "
        "parallel. Must divide the tensor-parallel size, which it subdivides "
        "rather than adds to. This is how MLA models are served: tensor "
        "parallelism replicates their compressed KV latent instead of sharding "
        "it, so only splitting by request shrinks the cache a rank holds.",
    )
    # ---- Feature A: prefill/decode disaggregation ----
    dis = parser.add_argument_group("inference disaggregation (feature A)")
    dis.add_argument(
        "--disaggregate",
        action="store_true",
        help="Enable prefill/decode disaggregation (separate worker pools).",
    )
    dis.add_argument("--prefill-tp", type=int, default=None, help="Prefill-pool tensor parallelism.")
    dis.add_argument("--prefill-pp", type=int, default=None, help="Prefill-pool pipeline parallelism.")
    dis.add_argument("--prefill-ep", type=int, default=None, help="Prefill-pool expert parallelism.")
    dis.add_argument("--decode-tp", type=int, default=None, help="Decode-pool tensor parallelism.")
    dis.add_argument("--decode-pp", type=int, default=None, help="Decode-pool pipeline parallelism.")
    dis.add_argument("--decode-ep", type=int, default=None, help="Decode-pool expert parallelism.")
    dis.add_argument(
        "--prefill-attention-dp",
        type=int,
        default=None,
        help="Prefill-pool attention-DP degree, overriding --attention-dp-size for "
        "that pool. Must divide --prefill-tp.",
    )
    dis.add_argument(
        "--decode-attention-dp",
        type=int,
        default=None,
        help="Decode-pool attention-DP degree, overriding --attention-dp-size for "
        "that pool. Must divide --decode-tp. Deployments commonly run DP attention "
        "on one pool only, which a single global degree cannot express.",
    )
    dis.add_argument(
        "--prefill-replicas", type=int, default=None, help="Number of prefill-pool replicas."
    )
    dis.add_argument(
        "--decode-replicas", type=int, default=None, help="Number of decode-pool replicas."
    )
    dis.add_argument(
        "--kv-transfer-bw-gbps",
        type=float,
        default=None,
        help="KV-cache transfer bandwidth GB/s (default: inter-node pod BW).",
    )
    dis.add_argument(
        "--kv-transfer-latency-us",
        type=float,
        default=None,
        help="Fixed KV-cache transfer latency overhead (us).",
    )
    dis.add_argument(
        "--transfer-backend",
        type=str,
        default=None,
        choices=["nixl", "mooncake", "mori"],
        help="KV-transfer engine preset (sets link BW + latency unless overridden "
        "by --kv-transfer-bw-gbps / --kv-transfer-latency-us).",
    )
    # ---- Serving / continuous-batching dynamics ----
    serv = parser.add_argument_group("inference serving dynamics")
    serv.add_argument(
        "--gpu-cost-per-hour",
        type=float,
        default=None,
        help="Price of one GPU-hour. Given it, the report prices the projected "
        "throughput as cost per million tokens -- the unit a serving budget is "
        "quoted in. Costs the whole replica, so a recipe that needs more GPUs "
        "for the same tokens is charged for them.",
    )
    serv.add_argument(
        "--serving-model",
        type=str,
        default=None,
        choices=["continuous", "static"],
        help=(
            "Decode latency model: 'continuous' (continuous batching with mixed "
            "prefill+decode steps → models TPOT pollution; default) or 'static' "
            "(idealized pure-decode batch; prefill charged once as TTFT)."
        ),
    )
    serv.add_argument(
        "--decode-step-overhead-us",
        type=float,
        default=None,
        help="Fixed per-decode-step host/launch overhead (us). CUDA graphs reduce this. Default 0.",
    )
    serv.add_argument(
        "--kernel-launch-latency-us",
        type=float,
        default=None,
        help="Per-kernel launch latency (us) for the small-tensor launch-bound "
        "decode floor in pure-simulate mode: step = max(step, n_kernels*lat). "
        "Disabled by CUDA-graph capture. Default 0 (off).",
    )
    serv.add_argument(
        "--kernels-per-layer",
        type=int,
        default=None,
        help="Representative kernel launches per transformer layer, used only "
        "by the launch-latency floor. Default 12.",
    )
    serv.add_argument(
        "--decode-kernel-occupancy-us",
        type=float,
        default=None,
        help="Per-kernel GPU occupancy (us): the minimum time a kernel holds "
        "the device regardless of how little data it touches. Added to the "
        "decode step (not a max) and not cancelled by CUDA-graph capture, "
        "which removes host dispatch but not kernel execution. Default 4.35 "
        "from the MI355X TP ladder; 0 disables.",
    )
    serv.add_argument(
        "--anchor-store",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory of warmup measurements to reuse. When set and no "
        "explicit --load-benchmark is given, the closest anchor measured in the "
        "same regime (model, dtypes, attention backend, cudagraph, speculation) "
        "calibrates this projection. An anchor from a different regime is "
        "reported and NOT used, because it describes different kernels. "
        "Defaults to $INFERASIM_ANCHOR_STORE.",
    )
    serv.add_argument(
        "--moe-router-coverage",
        type=str,
        default=None,
        metavar="FILE",
        help="JSON of measured router coverage: how many distinct experts the "
        "real router reaches per step, as a fraction of what independent "
        "per-token routing predicts, keyed by batch. Produced from the "
        "checkpoint's router weights (CPU only). Without it, routing is "
        "assumed independent, which over-counts experts at mid batch.",
    )
    serv.add_argument(
        "--moe-routing-skew",
        type=float,
        default=None,
        help="Zipf exponent of the MoE router's popularity law (0 = uniform). "
        "Sets how many distinct experts a decode step touches, which is what a "
        "weight-bandwidth-bound MoE step costs. Converts from a measured "
        "max/mean expert load by I(s) = N / H_N(s).",
    )
    serv.add_argument(
        "--detokenize-overhead-us",
        type=float,
        default=None,
        help="Per-output-token host detokenization + streaming cost (us/token). Added to "
             "ITL/TPOT and end-to-end latency only (not throughput), matching client-side "
             "serving-harness ITL. Default 0.",
    )
    serv.add_argument(
        "--tokenize-overhead-us",
        type=float,
        default=None,
        help="Per-prompt-token host tokenization cost (us/token). The prompt is sent as text "
             "and tokenized server-side after the TTFT clock starts, so it is added to TTFT "
             "and end-to-end latency only (not throughput). Default 0.",
    )
    serv.add_argument(
        "--stream-interval",
        type=int,
        default=None,
        help="Output tokens buffered per streaming flush (vLLM/SGLang --stream-interval). "
             "The client's first token, and so the measured TTFT, arrives only after this "
             "many tokens are decoded. Default 1 (flush every token).",
    )
    serv.add_argument(
        "--decode-admission-steps",
        type=int,
        default=None,
        help="Decode-scheduler admission granularity in decode steps, i.e. "
             "--num-continuous-decode-steps x --scheduler-recv-interval. A prefilled request "
             "waits part of this window before joining the running batch; TTFT-only. Default 0.",
    )
    serv.add_argument(
        "--mixed-batch-penalty",
        type=float,
        default=None,
        help="Extra cost fraction for mixed prefill+decode steps (PIECEWISE vs FULL CUDA graph). Default 0.",
    )
    serv.add_argument(
        "--cudagraph-mode",
        type=str,
        default=None,
        choices=["none", "piecewise", "full"],
        help="CUDA-graph capture preset: 'none' (eager), 'piecewise', or 'full'. "
        "Sets per-step overhead + mixed-batch penalty unless those are given explicitly.",
    )
    serv.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Scheduler per-step token budget (vLLM --max-num-batched-tokens). Caps "
        "prefill-chunk + concurrent-decode tokens per step; oversized steps split, "
        "raising TPOT. Default: 0 (unlimited).",
    )
    # ---- Offered load / request rate (open-loop arrivals) ----
    serv.add_argument(
        "--request-rate",
        type=float,
        default=None,
        help="Offered load in requests/sec (open-loop). Adds a first-order queueing "
        "delay to TTFT as load approaches the engine's max sustainable rate. "
        "Default: 0 (closed-loop, no queue).",
    )
    serv.add_argument(
        "--arrival-model",
        type=str,
        default=None,
        choices=["closed", "none", "poisson", "deterministic"],
        help="Arrival process for --request-rate. 'closed'/'none' (default): no "
        "queue, steady-state only. 'poisson'/'deterministic': run the "
        "discrete-event simulator (DES) for TTFT/TPOT/ITL percentiles. Both also "
        "still report the analytical M/M/1 / D/M/1 mean.",
    )
    serv.add_argument(
        "--des-num-requests",
        type=int,
        default=400,
        help="DES: number of requests to simulate at the configured offered load "
        "(--arrival-model poisson/deterministic). Default: 400.",
    )
    serv.add_argument(
        "--des-seed",
        type=int,
        default=0,
        help="DES: RNG seed for arrival/acceptance sampling (reproducible). Default: 0.",
    )
    serv.add_argument(
        "--des-sweep",
        action="store_true",
        default=False,
        help="DES: also sweep offered load (fractions of max-sustainable rate) and "
        "emit a throughput-vs-latency curve (p50/p99 TTFT & TPOT per load).",
    )
    serv.add_argument(
        "--des-burstiness",
        type=float,
        default=None,
        help="DES: gamma-arrival shape for --arrival-model poisson. 1.0 = Poisson "
        "(default), <1 = burstier, >1 = smoother/more regular.",
    )
    serv.add_argument(
        "--des-range-ratio",
        type=float,
        default=None,
        help="DES: per-request length heterogeneity. Actual ISL/OSL sampled "
        "uniformly from [ratio*len, len]. 1.0 = homogeneous (default), e.g. 0.5 "
        "= lengths vary down to half of --input-len/--output-len.",
    )
    serv.add_argument(
        "--des-kv-cache-tokens",
        type=int,
        default=None,
        help="DES: total KV token-slot pool. Admission reserves full ISL+OSL per "
        "request (head-of-line blocks on shortage). Default: 0 (unlimited; "
        "concurrency-bound only).",
    )
    serv.add_argument(
        "--des-workload-file",
        type=str,
        default=None,
        help="DES: replay a workload from JSON (list of dicts) or CSV with columns "
        "arrival(ms),isl,osl instead of synthetic sampling. Enables the DES even "
        "without --request-rate.",
    )
    serv.add_argument(
        "--des-dump-steps",
        type=str,
        default=None,
        help="DES: write per-step batch-composition records (query/KV shapes per "
        "request per step) + packing summary to this JSON path.",
    )
    # ---- DES multi-instance routing + shared prefix pool ----
    serv.add_argument(
        "--des-instances",
        type=int,
        default=None,
        help="DES: number of engine replicas behind a router (data-parallel "
        "copies of this recipe). >1 enables the multi-instance fleet model. "
        "Default: 1 (single engine).",
    )
    serv.add_argument(
        "--des-routing",
        type=str,
        default=None,
        choices=["round_robin", "random", "prefix_aware", "kv"],
        help="DES: request routing policy across instances. 'kv' is the serving "
        "router's KV-aware policy, trading cache overlap against load by the same "
        "cost function (see --des-overlap-weight); 'prefix_aware' consistently "
        "hashes the leading block so same-prefix requests co-locate; "
        "'round_robin'/'random' ignore locality (each instance re-warms). "
        "Default: round_robin.",
    )
    serv.add_argument(
        "--des-overlap-weight",
        type=float,
        default=None,
        help="DES: how much a cached KV block is worth against a block of load "
        "in 'kv' routing, matching the serving router's overlap weight. 0 routes "
        "purely by load; high values pin prefixes to whichever instance holds "
        "them. Default: 1.0.",
    )
    serv.add_argument(
        "--des-num-prefixes",
        type=int,
        default=None,
        help="DES: number of distinct shared prefixes in the workload (e.g. "
        "system-prompt / template variants). Requests sharing a prefix reuse its "
        "KV on a cache hit. 0 = no shared prefixes (every request unique).",
    )
    serv.add_argument(
        "--des-prefix-len",
        type=int,
        default=None,
        help="DES: shared-prefix length in tokens (the cached span on a hit). "
        "0 (with --des-num-prefixes>0) defaults to half the prompt.",
    )
    serv.add_argument(
        "--des-prefix-zipf",
        type=float,
        default=None,
        help="DES: prefix popularity skew. 0 = uniform; >0 = power-law (a few hot "
        "prefixes dominate, as with shared system prompts). Default: uniform.",
    )
    serv.add_argument(
        "--des-cache-slots",
        type=int,
        default=None,
        help="DES: legacy alias for --des-kv-blocks (per-instance block-cache "
        "capacity). 0 = unbounded within the run. Default: 0.",
    )
    serv.add_argument(
        "--des-block-size",
        type=int,
        default=None,
        help="DES: KV paged-block size in tokens for the content-addressed prefix "
        "cache. A cache hit reuses whole leading blocks. Default: 512 (Mooncake "
        "convention).",
    )
    serv.add_argument(
        "--des-kv-blocks",
        type=int,
        default=None,
        help="DES: per-instance KV block-cache capacity (blocks kept resident, "
        "LRU-evicted under pressure). 0 = unbounded within the run. Default: 0.",
    )
    serv.add_argument(
        "--des-mooncake-trace",
        type=str,
        default=None,
        help="DES: replay a Mooncake-format trace (JSONL/JSON with timestamp, "
        "input_length, output_length, hash_ids). The hash_ids drive real "
        "content-addressed prefix-cache matching. Enables the DES without "
        "--request-rate; combine with --des-instances/--des-routing to study a "
        "fleet.",
    )
    # ---- Kernel backend + fused ops + sparse attention + expert precision ----
    kern = parser.add_argument_group("inference kernel backend & ops")
    kern.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        choices=["aiter", "triton", "ck", "hip"],
        help="Attention kernel library (ROCm). Representative compute multiplier "
        "vs the Triton baseline. Default: engine default (1.0).",
    )
    kern.add_argument(
        "--sparse-attention-topk",
        type=int,
        default=None,
        help="Native sparse attention (DeepSeek V3.2/V4 NSA) top-k KV tokens per "
        "query. Attention scales toward topk/context for long contexts. "
        "Default: 0 (dense).",
    )
    kern.add_argument(
        "--sliding-window",
        type=int,
        default=None,
        help="Sliding-window (local) attention size in KV tokens each query "
        "attends to (0 = force full attention). Default: follow the model's "
        "sink_sliding_window. Bounds decode attention compute and KV-cache "
        "footprint at long context.",
    )
    kern.add_argument(
        "--sliding-window-layer-fraction",
        type=float,
        default=None,
        help="Fraction of attention layers that are windowed for models that "
        "interleave local/global layers. Default: 0.5 when the model is "
        "even-layers-only, else 1.0 (every layer windowed).",
    )
    kern.add_argument(
        "--moe-expert-dtype",
        type=str,
        default=None,
        choices=["bf16", "fp8", "mxfp4"],
        help="Expert grouped-GEMM compute precision (separate from --weight-dtype). "
        "Models the expert-MLP speedup of low-precision expert kernels.",
    )
    kern.add_argument(
        "--fused-kernels",
        action="store_true",
        default=False,
        help="Fused elementwise kernels (RMSNorm / RoPE / quant / KV-store+quant) "
        "that cut per-decode-step launch overhead.",
    )
    kern.add_argument(
        "--speculative-draft-cost-factor",
        type=float,
        default=None,
        help="Draft-model forward cost per proposed draft token, as a fraction of "
        "one target decode step. Default: 0 (ignore draft cost).",
    )
    parser.add_argument(
        "--inference-bench-layers",
        type=int,
        default=None,
        help=(
            "Benchmark mode: number of same-type transformer layers to build and "
            "time as a chained stack per phase (per-layer time = stack time / N). "
            "Larger N averages out per-layer jitter and captures inter-layer "
            "effects. Default: 4."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infera.projection",
        description="Infera inference/serving projection (TTFT, ITL, throughput, KV cache).",
    )
    sub = parser.add_subparsers(dest="suite", required=True)

    inference = sub.add_parser(
        "inference",
        help="Inference / serving projection (analytical or anchor-calibrated).",
    )
    add_pretrain_parser(inference)
    _add_topology_args(inference)
    _add_performance_args(inference)
    _add_save_benchmark_arg(inference)
    _add_load_benchmark_arg(inference, include_compute_baseline_alias=False)
    _add_inference_args(inference)
    inference.set_defaults(profiling_mode="simulate", func="inference")

    # anchor-harvest is a thin shim over the vLLM harness
    # Registered only so it appears in `--help`. Its flags are parsed by the
    # harness itself (see the dispatch in main()), so none are declared here.
    sub.add_parser(
        "anchor",
        help="Harvest a GPU-calibrated anchor via the vLLM benchmark harness "
             "(`inferasim anchor --help` lists its flags).",
        add_help=False,
    )
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # `anchor` forwards its flags verbatim to the vLLM harness, and several of
    # them (--model, --input-len, --tp) also exist on the projection parser.
    # Split it off before argparse sees it: sharing the tokens leaves the
    # harness with an argv the two parsers have already torn in half.
    if argv and argv[0] == "anchor":
        from infera.projection.core.projection.inference_projection import (
            benchmark_vllm,
        )
        return benchmark_vllm.main(argv[1:])

    parser = build_parser()
    args, overrides = parser.parse_known_args(argv)

    # inference projection (simulate by default; --load-benchmark uses an anchor)
    launch_projection_from_cli(args, overrides)


if __name__ == "__main__":
    main()
