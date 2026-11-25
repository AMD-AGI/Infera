###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
# Provenance: port of AKLR `aklr/benchmarks/models/{glm52,deepseekv4}/model_bench.py` @ 33689332
# (copies in research/aklr_src/) to DeepSeek-V4.1-Flash on sglang 6be0d75ae. DSv4 specifics
# (sub-pools, capacity refusal, hash/compression layer kinds, rank-max under EP) come from the
# deepseekv4 port; the MTP path (spec worker, draft row, simulated acceptance) from the glm52 port,
# rebuilt for DSPARK. DSv4.1-only additions are marked "DSv4.1:".

"""Shared plumbing for the DeepSeek-V4.1 micro-benchmarks: sglang model setup, per-op timing and layer roll-up."""

import argparse
import contextlib
import importlib
import multiprocessing
import os
from collections.abc import Callable
from multiprocessing.connection import wait
from typing import NamedTuple

import torch
from sglang_bench import (
    GRAPH_REPS,
    A2aImpl,
    PrefillStep,
    _balanced_forward,
    _partitions,
    agreed,
    cfg,
    local_conc,
    moe_parallel,
    parallel_state,
    prepare_decode,
    publish_once,
    require_aiter,
    synthetic_reqs,
    table,
)

# Point --model-path at a local checkpoint to avoid downloading weights.
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
SUPPORTED_ARCHS = ("DeepseekV41ForCausalLM", "DeepseekV4ForCausalLM")


def model_tag(server_args) -> str:
    """Checkpoint basename, keeping Flash/Pro reports separate."""
    return server_args.model_path.rstrip("/").rpartition("/")[2].replace(".", "")


def make_parser() -> argparse.ArgumentParser:
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    # ServerArgs marks --model-path required; default it instead.
    for action in parser._actions:
        if action.dest == "model_path":
            action.required = False
            action.default = DEFAULT_MODEL
    parser.add_argument("--ISL", type=int, default=4096, help="input length per request")
    parser.add_argument("--reps", type=int, default=50, help="graph replays per op")
    parser.add_argument("--repeat", type=int, default=1, help="timing passes per op, fastest one wins")
    parser.add_argument(
        "--full-step",
        type=int,
        default=0,
        metavar="N",
        help="time N+2 whole steps under sglang's own cuda graphs and exit (fastest wins), "
        "instead of profiling per op. Read by build_server_args, so it lives here.",
    )
    parser.add_argument("--out", help="markdown report path (default: auto-named)")
    parser.add_argument(
        "--routing",
        choices=("balanced", "native"),
        default="balanced",
        help="balanced (default, as upstream): seeded uniform experts on every MoE layer; native: the model's own"
        " router on the synthetic hidden states (a diagnostic for comparing with a server, not the bench's design)",
    )
    parser.add_argument(
        "--kv-fill",
        choices=("random", "prefill"),
        default="random",
        help="decode context: random KV bytes over a natively allocated context (default, as upstream), "
        "or one real prefill forward of the synthetic prompt",
    )
    return parser


DUMMY_WARNING = (
    "--load-format dummy. Operators whose cost depends on values (indexer top-k, routing) are mistimed. "
    "Do not act on these numbers."
)


def warn_if_dummy_weights(load_format: str) -> None:
    if load_format == "dummy":
        print(f"[BENCH] WARNING: {DUMMY_WARNING}", flush=True)


def dummy_banner(server_args) -> str:
    if server_args.load_format != "dummy":
        return ""
    return f"\n> **WARNING: {DUMMY_WARNING}**\n"


def weights_line(model_runner) -> str:
    """The ``weights`` row: how the checkpoint is quantized, read off its config."""
    sa = model_runner.server_args
    hf = model_runner.model_config.hf_config
    q = getattr(hf, "quantization_config", None) or {}
    if not isinstance(q, dict):
        q = getattr(q, "__dict__", {})
    method = cfg(sa).quantization or q.get("quant_method")
    dtype = str(model_runner.model_config.dtype).replace("torch.", "")
    if not method:
        return f"`{sa.load_format}` load, {dtype} throughout"
    block = q.get("weight_block_size")
    detail = method + (f" block {'x'.join(str(b) for b in block)}" if block else "")
    experts = q.get("expert_dtype")
    return f"`{sa.load_format}` load, {detail} quantized" + (f", experts {experts}" if experts else "") + f", {dtype} elsewhere"


def parallel_tags(model_runner, layers: "Layers") -> list[str]:
    """Report-name parts for the layout, e.g. ``tp4 dp4 ep4``."""
    ps, moe = parallel_state(model_runner), moe_parallel(model_runner)
    tags = [f"tp{ps.tp_size}"]
    if ps.attn_dp_size > 1:
        tags.append(f"dp{ps.attn_dp_size}")
    if moe.ep_size > 1:
        tags.append(f"ep{moe.ep_size}")
    if moe.a2a:
        tags += [moe.a2a, str(cfg(model_runner.server_args).deepep_mode)]
        impl = a2a_impl(model_runner, layers)
        if impl is not None:
            tags.append(f"cap{impl.cap}")
    return tags


def parallel_line(model_runner) -> str:
    ps, moe = parallel_state(model_runner), moe_parallel(model_runner)
    line = f"TP {ps.tp_size}, attn-DP {ps.attn_dp_size}, attn-TP {ps.attn_tp_size}, MoE-EP {moe.ep_size}"
    if moe.ep_size > 1 and moe.tp_size > 1:
        line += f" x TP {moe.tp_size}"
    if moe.a2a:
        line += f", MoE tokens {moe.a2a} all-to-all"
    return line


def a2a_impl(model_runner, layers: "Layers") -> A2aImpl | None:
    if moe_parallel(model_runner).a2a is None:
        return None
    experts = layers.layers[0].mlp.experts
    try:
        dispatcher = experts.dispatcher._inners[0]
        impl = (
            dispatcher._normal_dispatcher
            if cfg(model_runner.server_args).deepep_mode == "normal"
            else dispatcher._low_latency_dispatcher
        )
        return A2aImpl(
            cap=int(impl.num_max_dispatch_tokens_per_rank),
            dispatch_dtype=impl.dispatch_dtype.name,
            combine_dtype=impl.combine_dtype.name,
        )
    except (AttributeError, IndexError):
        return None


def moe_line(model_runner, layers: "Layers") -> str | None:
    moe = moe_parallel(model_runner)
    if moe.a2a is None:
        return None
    impl = a2a_impl(model_runner, layers)
    mode = cfg(model_runner.server_args).deepep_mode
    if impl is None:
        return f"{moe.a2a} all-to-all, mode {mode}; dispatch cap and dtypes not found on the dispatcher"
    knob = " (SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK)" if moe.a2a == "mori" else ""
    return (
        f"{moe.a2a} all-to-all, mode {mode}, dispatch {impl.dispatch_dtype}, combine"
        f" {impl.combine_dtype}, dispatch cap {impl.cap:,} tokens/rank{knob}:"
        f" `mlp.experts` is handed up to {impl.cap * moe.ep_size:,} rows per call"
    )


def world_times(times: dict[str, float], model_runner) -> dict[str, float]:
    """Per-key ms over world ranks: the slowest under EP, whose routed load differs per rank, else the fastest."""
    from sglang_bench import gather_obj

    pick = max if moe_parallel(model_runner).ep_size > 1 else min
    gathered = gather_obj(times)
    return {name: pick(t[name] for t in gathered) for name in times}


def build_server_args(args: argparse.Namespace, cuda_graph: bool = False):
    """Validate the workload, then resolve native server flags the way the launcher does."""
    from sglang.srt.server_args import ServerArgs

    if getattr(args, "CONC", 1) <= 0:
        raise ValueError("--CONC must be positive")
    if args.ISL <= 0 or args.reps <= 0 or args.repeat <= 0 or args.full_step < 0:
        raise ValueError("--ISL, --reps and --repeat must be positive; --full-step must be >= 0")
    require_aiter()
    warn_if_dummy_weights(args.load_format)
    # Module hooks need Python forwards; each op gets its own graph instead.
    args.disable_cuda_graph = not (args.full_step or cuda_graph)
    # `auto` is bf16, doubling attention bytes per token; production runs fp8.
    if args.kv_cache_dtype == "auto":
        args.kv_cache_dtype = "fp8_e4m3"
    server_args = ServerArgs.from_cli_args(args)
    dp = server_args.dp_size if server_args.enable_dp_attention else 1
    if server_args.enable_dp_attention and args.CONC % dp:
        raise ValueError(f"--CONC {args.CONC} is the global batch and must divide evenly across --dp-size {dp}")
    args.dp_raw = dp
    if not args.disable_cuda_graph and getattr(args, "cuda_graph_max_bs_decode", None) in (None, 0):
        # Same mechanism as one_batch.main: the decode graph must cover the local batch, else decode runs eager.
        _declare_decode_graph_bs(server_args, max(getattr(args, "concs", [args.CONC])) // dp)
    # Resolve first, as one_batch.main does: unresolved, mori gets deepep_mode=auto and picks the wrong kernel.
    server_args.resolve_once()
    return server_args


def _declare_decode_graph_bs(server_args, local_bs: int) -> None:
    from sglang.srt.arg_groups.overrides import declare_resolution
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, Phase

    explicit = server_args.cuda_graph_config
    if isinstance(explicit, CudaGraphConfig):
        explicit = explicit.to_dict()
    gcfg = dict(explicit or {})
    dec = dict(gcfg.get(Phase.DECODE) or {})
    dec["max_bs"] = max(1, local_bs)
    gcfg[Phase.DECODE] = dec
    declare_resolution(server_args, "inproc_bench.tool2_perop", cuda_graph_config=gcfg)


def validate_model(server_args):
    """Reject other architectures from config, before loading model weights."""
    from sglang.srt.configs.model_config import ModelConfig

    hf = ModelConfig.from_server_args(server_args).hf_config
    archs = getattr(hf, "architectures", None) or []
    if not set(archs) & set(SUPPORTED_ARCHS):
        raise ValueError(f"This benchmark requires a DeepSeek-V4/V4.1 checkpoint ({SUPPORTED_ARCHS}); got {archs}")


def run(work_func: Callable, server_args, args) -> None:
    """Run ``work_func`` per rank, mirroring ``one_batch.main`` (TP>1 forks one process per rank)."""
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs
    from sglang.srt.utils import kill_process_tree, maybe_reindex_device_id

    # The default of 32 per process times tp_size is a few hundred compile workers.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

    validate_model(server_args)
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, args, 0, 0)
        return

    nranks_per_node = server_args.tp_size // server_args.nnodes
    lo = server_args.node_rank * nranks_per_node
    procs = []
    try:
        for tp_rank in range(lo, lo + nranks_per_node):
            with maybe_reindex_device_id(tp_rank - lo) as gpu_id:
                p = multiprocessing.Process(
                    target=work_func,
                    args=(server_args, port_args, args, gpu_id, tp_rank),
                )
                p.start()
                procs.append(p)
        alive = {p.sentinel: (rank, p) for rank, p in enumerate(procs, lo)}
        while alive:
            for sentinel in wait(alive):
                rank, p = alive.pop(sentinel)
                p.join()  # already exited; reaps it and sets exitcode
                if p.exitcode:
                    # A dead rank leaves the others blocked in its collectives; the finally kills them.
                    raise SystemExit(f"rank {rank} exited {p.exitcode}; aborting {len(alive)} ranks")
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def setup(server_args, port_args, gpu_id: int, tp_rank: int, tag: str = "BENCH"):
    """Build the model (and the DSPARK worker under --speculative-algorithm). Returns (model_runner, spec, rank_print)."""
    from sglang.benchmark.one_batch import _init_process_global_configs, load_model
    from sglang.srt.plugins import load_plugins
    from sglang.srt.utils import configure_logger

    # The initializers read the published runtime context, and a spawned child arrives with nothing published.
    publish_once(server_args, gpu_id, tp_rank)
    _init_process_global_configs()
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    # As the scheduler process does: sglang.srt.plugins hooks (AKLR's dispatch, when installed) attach.
    load_plugins()
    if importlib.util.find_spec("aklr") is not None:
        # load_routes proxies aiter's entry points only: proxy sglang's too, so AKLR_EXPORT records them.
        from aklr.core.runtime import seam
        from aklr.ops import load_routes

        load_routes()
        seam.aklr_proxy()

    spec = build_spec_worker(server_args, port_args, gpu_id, tp_rank)
    if spec is None:
        model_runner = load_model(server_args, port_args, gpu_id, tp_rank)[0].torch_runner
    else:
        # TpModelWorker builds its own ModelRunner; a second load_model would not fit in memory.
        model_runner = spec.target_worker.model_runner

    if tp_rank != 0:
        return model_runner, spec, lambda *a, **k: None

    def rank_print(msg=""):
        print("\n".join(f"[{tag}] {ln}" for ln in str(msg).split("\n")), flush=True)

    return model_runner, spec, rank_print


# ----------------------------------------------------------------------------------------------
# MTP (DSPARK). glm52 builds sglang's own EAGLEWorkerV2; DSv4.1 serves DSPARK, so the same role is
# played by DSparkWorkerV2, built in the scheduler's order (target worker, spec worker, pools,
# attention backends, graphs).


def build_spec_worker(server_args, port_args, gpu_id: int, tp_rank: int):
    """sglang's own DSPARK worker, or None when --speculative-algorithm is unset."""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = cfg(server_args).speculative_algorithm
    if algo is None or SpeculativeAlgorithm.from_string(algo).is_none():
        return None
    import spec_dspark

    return spec_dspark.build(server_args, port_args, gpu_id, tp_rank)


def accept_len():
    """SGLANG_SIMULATE_ACC_LEN, the same knob the server runs under; None when unset."""
    from sglang.srt.speculative.spec_utils import SIMULATE_ACC_LEN

    return SIMULATE_ACC_LEN if SIMULATE_ACC_LEN and SIMULATE_ACC_LEN > 0 else None


def spec_note(server_args) -> str:
    import spec_dspark

    return spec_dspark.note(server_args)


# ----------------------------------------------------------------------------------------------
# Requests, capacity, batches.


def _pages(tokens: int, page: int) -> int:
    return -(-tokens // page) * page


def capacities(model_runner) -> dict[str, int]:
    """Token / slot capacities of this rank's pools, read off the native allocators."""
    out = {"full KV": int(model_runner.max_total_num_tokens)}
    allocator = model_runner.token_to_kv_pool_allocator
    swa = getattr(allocator, "swa_attn_allocator", None)
    if swa is not None and getattr(swa, "size", None):
        out["SWA"] = int(swa.size)
    out["request slots"] = int(model_runner.req_to_token_pool.size)
    return out


def needs(args, model_runner, conc: int, factor: int = 1, decode_tokens: int = 1) -> dict[str, int]:
    page = cfg(model_runner.server_args).page_size or 1
    want = _pages(args.ISL + decode_tokens, page) * conc * factor
    return {"full KV": want, "SWA": want, "request slots": conc * factor}


def check_capacity(args, model_runner, factor: int = 1, decode_tokens: int = 1) -> None:
    """[BLOCKED_CAPACITY] instead of shrinking ISL or CONC (deepseekv4 port's rule)."""
    conc = local_conc(args, model_runner)
    have, want = capacities(model_runner), needs(args, model_runner, conc, factor, decode_tokens)
    short = [k for k in have if want.get(k, 0) > have[k]]
    if short:
        raise ValueError(
            f"[BLOCKED_CAPACITY] --ISL {args.ISL:,} x --CONC {args.CONC} ({conc} requests/rank, factor {factor})"
            f" needs {', '.join(f'{k}={want[k]:,}' for k in have)}; capacity {', '.join(f'{k}={v:,}' for k, v in have.items())}."
            f" Insufficient {', '.join(short)}. Provision capacity (--mem-fraction-static, --max-running-requests,"
            " --swa-full-tokens-ratio); benchmark parameters were not reduced."
        )


def decode_capacity(args, model_runner, layers: "Layers") -> tuple[int, str]:
    """Largest global decode CONC that fits at this ISL, and the limit that binds it."""
    page = cfg(model_runner.server_args).page_size or 1
    have = capacities(model_runner)
    per_req = _pages(args.ISL + 1, page)
    limits = {k: (v // per_req if k != "request slots" else v) for k, v in have.items()}
    if (impl := a2a_impl(model_runner, layers)) is not None:
        limits["a2a dispatch cap"] = impl.cap
    why = min(limits, key=limits.get)
    return limits[why] * parallel_state(model_runner).attn_dp_size, why


class _TreeCache:
    """one_batch.TreeCacheNamespace plus prefix_device_indices: a new request with prefix_len > 0 reads its
    cached prefix off the tree (mem_cache/allocation._prefix_kv_indices), so the bench hands it its slots."""

    def __new__(cls, model_runner, prefixes: dict | None = None):
        from sglang.benchmark.one_batch import TreeCacheNamespace

        prefixes = prefixes or {}
        ns = TreeCacheNamespace(
            page_size=cfg(model_runner.server_args).page_size,
            device=model_runner.device,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        )
        ns.prefix_device_indices = lambda req: prefixes[req.rid][: req.prefix_len]
        return ns


def make_extend_batch(reqs, model_runner, reset: bool = True, prefixes: dict | None = None):
    """A prefill ScheduleBatch, up to but not including the forward."""
    from sglang.benchmark.one_batch import _maybe_prepare_mlp_sync_batch
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if reset:
        model_runner.token_to_kv_pool_allocator.clear()
        model_runner.req_to_token_pool.clear()
    available = model_runner.req_to_token_pool.available_size()
    new = sum(not getattr(getattr(r, "kv", None), "holds_kv", False) for r in reqs)
    if new > available:
        raise ValueError(
            f"[BLOCKED_CAPACITY] request slots: need {new} new slots/rank, available={available}."
            " Set --max-running-requests >= the global concurrency (before DP division)."
        )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=_TreeCache(model_runner, prefixes),
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    if batch.input_ids is None and getattr(batch, "prefill_input_ids_cpu", None) is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        batch.prefill_input_ids_cpu = None
    return batch


def make_reqs(args, model_runner, factor: int = 1, prefix: str = "bench"):
    check_capacity(args, model_runner, factor=factor)
    rank = parallel_state(model_runner).attn_dp_rank or 0
    return synthetic_reqs(local_conc(args, model_runner), args.ISL, prefix=f"{prefix}-dp{rank}")


def make_decode_batch(args, model_runner, kv_fill: str = "random"):
    """Synthetic decode batch: an ISL context natively allocated (random KV) or prefilled, then one decode step."""
    from sglang_bench import forward_batch_of

    batch = make_extend_batch(make_reqs(args, model_runner), model_runner)
    if kv_fill == "prefill":
        model_runner.forward(forward_batch_of(batch, model_runner))
    prepare_decode(batch, model_runner)
    return batch


# ----------------------------------------------------------------------------------------------
# Synthetic KV.


def sub_pools(pool):
    """DSv4/4.1 keep no buffers on the top pool: SWA, compressed-KV and indexer caches are sub-pools."""
    seen = set()
    for name in ("swa_kv_pool", "c4_kv_pool", "c128_kv_pool", "c4_indexer_kv_pool", "unified_kv_pool"):
        child = getattr(pool, name, None)
        if child is not None and id(child) not in seen:
            seen.add(id(child))
            yield name, child
    # DSv4.1: compressed pools keyed by ratio (1/2/4/128) and indexer pools keyed by ratio.
    for attr in ("kv_pools", "index_pools"):
        for ratio, child in sorted((getattr(pool, attr, None) or {}).items()):
            if child is not None and id(child) not in seen:
                seen.add(id(child))
                yield f"{attr}[{ratio}]", child


def pool_buffers(pool):
    """(sub-pool, attr, tensor) for every cache buffer: attributes named *buffer* holding tensors or lists of them."""
    for pname, child in sub_pools(pool):
        for attr, value in sorted(vars(child).items()):
            if "buffer" not in attr:
                continue
            items = value if isinstance(value, (list, tuple)) else [value]
            for t in items:
                if isinstance(t, torch.Tensor) and t.is_cuda and t.numel():
                    yield pname, attr, t


def randomize_kv(model_runner, spec=None) -> dict:
    """Synthetic KV performance data, not a numerically valid prefilled context. Returns bytes filled per kind."""
    runners = [model_runner]
    if spec is not None:
        import spec_dspark

        runners = list(spec_dspark.runners(spec).values())
    filled = {}
    for runner in runners:
        for pname, attr, buf in pool_buffers(runner.token_to_kv_pool):
            raw = buf.view(torch.uint8) if buf.element_size() == 1 else None
            if "scale" in attr and raw is not None:
                # ue8m0 / fp8 scales near 1 (0x7E..0x80), so scores are not crushed to zero.
                raw.random_(0x7E, 0x81)
            elif raw is not None:
                # below 0x7C: no fp8 e4m3 NaN, no fp4 issue, finite ue8m0
                raw.random_(0, 0x7C)
            elif buf.is_floating_point():
                buf.uniform_(-1, 1)
            else:
                continue
            key = f"{pname}.{attr}"
            filled[key] = filled.get(key, 0) + buf.numel() * buf.element_size()
    return filled


# ----------------------------------------------------------------------------------------------
# Prefill timeline.


def chunk_tokens(args, model_runner) -> int:
    """New tokens the scheduler admits per forward, on this rank."""
    # The resolved view already divides --chunked-prefill-size by attn_dp_size (arg_groups/parallel_hook.py).
    size = cfg(model_runner.server_args).chunked_prefill_size
    return size if size and size > 0 else args.ISL


def chunk_flag(args, model_runner) -> int:
    if args.chunked_prefill_size:
        return args.chunked_prefill_size
    return chunk_tokens(args, model_runner) * parallel_state(model_runner).attn_dp_size


def plan_prefill(args, model_runner) -> list[PrefillStep]:
    """The chunk timeline the scheduler would walk for one request."""
    page = cfg(model_runner.server_args).page_size or 1
    cached = int(args.ISL * args.prefix_cache_hit) // page * page
    budget = chunk_tokens(args, model_runner)
    steps, prefix = [], cached
    while prefix < args.ISL:
        extend = min(budget, args.ISL - prefix)
        steps.append(PrefillStep(prefix, extend))
        prefix += extend
    if not steps:
        raise ValueError(f"--prefix-cache-hit {args.prefix_cache_hit} leaves nothing to prefill at --ISL {args.ISL}")
    return steps


def make_prefill_context(args, model_runner) -> list[torch.Tensor]:
    """KV slots holding each request's context; a step takes its prefix from these."""
    batch = make_extend_batch(make_reqs(args, model_runner, factor=2, prefix="ctx"), model_runner)
    randomize_kv(model_runner)
    rows = model_runner.req_to_token_pool.req_to_token
    # Clone: a table row is a view and every extend rewrites the table.
    return [rows[idx, : args.ISL].clone().to(torch.int64) for idx in batch.req_pool_indices_cpu.tolist()]


def make_prefill_batch(model_runner, context, step: PrefillStep, n: int = 0):
    """A ScheduleBatch for one prefill step, its prefix already in the pool (read through the tree cache)."""
    reqs = synthetic_reqs(len(context), step.prefix + step.extend, prefix=f"chunk{n}")
    prefixes = {}
    for req, slots in zip(reqs, context):
        req.prefix_len = step.prefix
        req.extend_end = step.prefix + step.extend
        prefixes[req.rid] = slots
    return make_extend_batch(reqs, model_runner, reset=False, prefixes=prefixes)


def free_req_rows(batch, model_runner) -> None:
    """Release request rows between chunks, retaining the shared KV context."""
    pool = model_runner.req_to_token_pool
    rows = batch.req_pool_indices_cpu.tolist()
    if hasattr(pool, "free_rows"):
        pool.free_rows(rows)
        return
    for req in batch.reqs:
        pool.free(req)


# ----------------------------------------------------------------------------------------------
# Recording and replay.


class Recorded(NamedTuple):
    """One op's captured call site: arguments plus the ambient state it reads."""

    args: tuple
    kwargs: dict
    flags: dict  # every scoped forward flag (glm52 kept attn_inputs + the two MLP flags; 6be0d75ae has more)
    fctx: object  # the ForwardContext active at the call (moe_num_token_non_padded lives there)
    fn: Callable | None = None  # set when the op is a call site, not a module __call__ (DSv4.1 layer rows)
    # DSv4.1 prefill: ((forward_batch, tail_source_kv_layer_id), (forward_batch, dp counts) | None) when the op ran
    # inside the decoder SWA bounded-replay tail, whose backend metadata and DP counts the loop swaps in.
    late: tuple | None = None


def _flags_now() -> dict:
    from sglang.srt.runtime_context import ForwardFlags, get_forward

    fwd = get_forward()
    return {f: getattr(fwd, f) for f in ForwardFlags._DEFAULTS}


def _fctx_now():
    from sglang.srt.model_executor.forward_context import get_forward_context, has_forward_context

    return get_forward_context() if has_forward_context() else None


def snapshot(value, memo: dict | None = None):
    """Clone tensors (once per id) so in-place mutation after the hook cannot change the recorded input.
    Descends into tuples/lists/dicts/NamedTuples and msgspec Structs (DSv4.1 HcState, HcPending); other
    objects (ForwardBatch, modules) are shared, as upstream shares native batch metadata."""
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            memo[id(value)] = value.detach().clone()
        return memo[id(value)]
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(snapshot(v, memo) for v in value))
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(v, memo) for v in value)
    if isinstance(value, dict):
        return {k: snapshot(v, memo) for k, v in value.items()}
    fields = getattr(type(value), "__struct_fields__", None)
    if fields is not None:
        return type(value)(**{f: snapshot(getattr(value, f), memo) for f in fields})
    return value


def residual_owner(name: str) -> str:
    """The `<rest>` row that absorbs ``name``'s cost when it has no row of its own."""
    kind, _, op = name.rpartition("|")
    if not kind or kind == "draft" or op == "layer":
        return ""
    parent = op.rpartition(".")[0] or "layer"
    if op in OUTSIDE_LAYER:
        return ""
    if parent not in RESIDUALS:
        return ""
    return f"{kind}|{parent}.<rest>"


def skipped_note(skipped: dict) -> str:
    if not skipped:
        return ""
    rows = []
    for name, err in sorted(skipped.items()):
        # Split on the bar: a `kind|op` key carries the character markdown uses to end a cell.
        kind, _, op = name.rpartition("|")
        owner = residual_owner(name)
        if kind == "draft":
            where = "the `draft model` row"
        elif op in RESIDUALS:
            where = "`layer.<rest>`; its children leave the roll-up too"
        elif owner:
            where = f"`{owner.split('|', 1)[1]}`"
        else:
            where = "nowhere -- the roll-up is short by it"
        rows.append((kind or "--", f"`{op}`", where, err.replace("|", "\\|")))
    return table(rows, ("layer kind", "op", "its ms now sits in", "why it could not be replayed")) + (
        "\n\nThese ops ran in the recording forward, so their cost is real and is"
        " in the parent row; what is missing is the row that would separate it"
        " out. Read the named `<rest>` row as an upper bound, not a measurement."
    )


def coverage_note(ops: dict, saved: dict) -> str:
    """Hooked modules bypassed by inline/fused code; globals have no residual."""
    missing = sorted(n for n in ops if n not in saved)
    if not missing:
        return ""
    rows = []
    for name in missing:
        kind, _, op = name.rpartition("|")
        owner = residual_owner(name)
        where = owner.split("|", 1)[1] if owner else "unattributed (no parent residual)"
        if op in ("attn.compressor", "attn.indexer") and f"{kind}|attn.sources" in saved:
            where = "attn.sources (timed through the backend call site)"
        rows.append((kind or "--", op, where))
    return table(rows, ("layer kind", "hook not called", "possible containing row")) + (
        "\n\nHooked but never called during the recording forward. A name is missing either because this"
        " layer kind does not run that op, or because a fused kernel reads the module's weight directly"
        " (DSv4.1 on gfx950: the mHC boundary kernel applies `input_layernorm` / `post_attention_layernorm`"
        " itself). A parent residual can contain this work; missing globals are not covered by layer residuals."
    )


# DSv4.1: call sites that are not an nn.Module __call__. On gfx950 the model loop runs every layer through
# `forward_layer_fused_boundary(model, i, HcState, ...)` (hc_boundary_fused), never `layer(...)`, so a layer
# pre-hook would never fire; the model head runs `mhc_combine` (a function). Both are wrapped at their
# module attribute during the recording forward and replayed from the recorded arguments.
LAYER_FN = ("sglang.srt.models.deepseek_v4", "forward_layer_fused_boundary")
HEAD_FN = ("sglang.kernels.ops.layernorm.mhc", "mhc_combine")


@contextlib.contextmanager
def _patched(module_name: str, attr: str, make):
    mod = importlib.import_module(module_name)
    orig = getattr(mod, attr)
    setattr(mod, attr, make(orig))
    try:
        yield orig
    finally:
        setattr(mod, attr, orig)


@contextlib.contextmanager
def record_inputs(ops: dict, into: dict, layers: "Layers | None" = None):
    """Capture first-call inputs (snapshotted before any in-place mutation) plus ambient forward state."""
    handles, stack = [], contextlib.ExitStack()
    late = {"tail": None, "dp": None}

    def keep(name, args, kwargs, fn=None):
        if name not in into:
            memo = {}
            scope = (late["tail"], late["dp"]) if late["tail"] is not None else None
            into[name] = Recorded(snapshot(args, memo), snapshot(kwargs, memo), _flags_now(), _fctx_now(), fn, scope)

    try:
        for name, mod in ops.items():

            def hook(_m, args, kwargs, _n=name):
                keep(_n, args, kwargs)

            handles.append(mod.register_forward_pre_hook(hook, with_kwargs=True))
        if layers is not None:
            by_layer = {}
            for name, mod in ops.items():
                if name.endswith("|layer"):
                    by_layer[id(mod)] = name

            def layer_fn(orig):
                def wrapped(model, i, state, **kw):
                    name = by_layer.get(id(model.layers[i]))
                    if name is not None:
                        keep(name, (model, i, state), kw, fn=orig)
                    return orig(model, i, state, **kw)

                return wrapped

            stack.enter_context(_patched(*LAYER_FN, layer_fn))
            _track_late_scope(stack, late)
            # Non-fused HIP path: the loop calls layer.forward_hc_pre_from_prev(...).
            for name, mod in ops.items():
                if name.endswith("|layer") and hasattr(mod, "forward_hc_pre_from_prev"):
                    orig = mod.forward_hc_pre_from_prev

                    def method(*a, _n=name, _o=orig, **kw):
                        keep(_n, a, kw, fn=_o)
                        return _o(*a, **kw)

                    mod.forward_hc_pre_from_prev = method
                    stack.callback(lambda m=mod: m.__dict__.pop("forward_hc_pre_from_prev", None))
            sources = {id(ops[n.rpartition("|")[0] + "|attn"]): n for n in ops if n.endswith("|attn.sources")}
            if sources:
                from sglang.srt.model_executor.forward_context import get_attn_backend

                backend = get_attn_backend()
                orig_src = backend.forward_low_ratio_sources

                def src_method(*a, _o=orig_src, **kw):
                    name = sources.get(id(kw.get("layer")))
                    if name is not None:
                        keep(name, a, kw, fn=_o)
                    return _o(*a, **kw)

                backend.forward_low_ratio_sources = src_method
                stack.callback(lambda b=backend: b.__dict__.pop("forward_low_ratio_sources", None))
            if "hc_head" in ops:

                def head_fn(orig):
                    def wrapped(*a, **kw):
                        keep("hc_head", a, kw, fn=orig)
                        return orig(*a, **kw)

                    return wrapped

                stack.enter_context(_patched(*HEAD_FN, head_fn))
        yield into
    finally:
        for h in handles:
            h.remove()
        stack.close()


@contextlib.contextmanager
def fn_child_ranges(active: bool):
    """Kineto child scope around the backend sources call site, so a parent `attn` kernel sample excludes it
    the way kernel_breakdown.child_ranges excludes module children (a function site has no module hook)."""
    if not active:
        yield
        return
    import kernel_breakdown
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from torch.profiler import record_function

    backend = get_attn_backend()
    orig = backend.forward_low_ratio_sources

    def wrapped(*a, **kw):
        with record_function(kernel_breakdown.CHILD):
            return orig(*a, **kw)

    backend.forward_low_ratio_sources = wrapped
    try:
        yield
    finally:
        backend.__dict__.pop("forward_low_ratio_sources", None)


LATE_DP_FNS = ("sglang.srt.models.deepseek_v4", "enter_late_layer_dp", "exit_late_layer_dp")


def _track_late_scope(stack, late: dict) -> None:
    """Follow the model loop into and out of the late-layer SWA tail (prefill only)."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    if not hasattr(backend, "enter_late_layer_tail"):
        return
    enter, leave = backend.enter_late_layer_tail, backend.exit_late_layer_tail

    def tail_in(forward_batch):
        late["tail"] = (forward_batch, getattr(backend, "tail_source_kv_layer_id", None))
        return enter(forward_batch)

    def tail_out(saved, forward_batch):
        late["tail"], late["dp"] = None, None
        return leave(saved, forward_batch)

    backend.enter_late_layer_tail, backend.exit_late_layer_tail = tail_in, tail_out
    stack.callback(lambda: [backend.__dict__.pop(n, None) for n in ("enter_late_layer_tail", "exit_late_layer_tail")])

    def dp_in(orig):
        def wrapped(forward_batch, counts):
            late["dp"] = (forward_batch, list(counts))
            return orig(forward_batch, counts)

        return wrapped

    stack.enter_context(_patched(LATE_DP_FNS[0], LATE_DP_FNS[1], dp_in))


@contextlib.contextmanager
def _late_scope(scope):
    """Re-enter the tail metadata (and the late DP counts) around a replay of an op recorded inside it."""
    if scope is None:
        yield
        return
    from sglang.srt.model_executor.forward_context import get_attn_backend

    (tail_fb, source_id), dp = scope
    backend = get_attn_backend()
    if hasattr(backend, "tail_source_kv_layer_id"):
        backend.tail_source_kv_layer_id = source_id
    saved = backend.enter_late_layer_tail(tail_fb)
    saved_dp = None
    mod = importlib.import_module(LATE_DP_FNS[0])
    try:
        if dp is not None:
            saved_dp = getattr(mod, LATE_DP_FNS[1])(dp[0], dp[1])
        yield
    finally:
        if saved_dp is not None:
            getattr(mod, LATE_DP_FNS[2])(dp[0], saved_dp)
        backend.exit_late_layer_tail(saved, tail_fb)


def make_call(mod, entry, layer_index: int | None = None) -> Callable:
    """Replay an op with its recorded arguments, forward flags and forward context.
    ``layer_index`` re-targets a layer call site at another layer of the same kind."""
    from sglang.srt.model_executor.forward_context import forward_context
    from sglang.srt.runtime_context import get_forward

    args, kwargs, fn = entry.args, entry.kwargs, entry.fn
    if fn is not None and layer_index is not None:
        if getattr(fn, "__name__", "") == LAYER_FN[1]:
            args = (args[0], layer_index, *args[2:])
        else:  # bound forward_hc_pre_from_prev of the representative
            fn = type(mod).forward_hc_pre_from_prev.__get__(mod)
    target = fn if fn is not None else mod

    def call():
        ctx = forward_context(entry.fctx) if entry.fctx is not None else contextlib.nullcontext()
        with ctx, get_forward().scoped(**entry.flags), _late_scope(entry.late):
            return target(*args, **kwargs)

    return call


# MI355X's last-level cache.
MALL_BYTES = 256 << 20
# Twice the cache, so a slot is evicted before its next read.
ROTATION_BYTES = 2 * MALL_BYTES
MAX_ROTATION_SLOTS = 256


def rotation_plan(nbytes: int) -> int:
    """Ring slots for a weight of ``nbytes``; 0 = no rotation (too big for the MALL to matter, or too small)."""
    slots = min(ROTATION_BYTES // max(nbytes, 1), MAX_ROTATION_SLOTS)
    if nbytes >= MALL_BYTES or slots < 2 or slots * nbytes <= MALL_BYTES:
        return 0
    return slots


@contextlib.contextmanager
def rotated_calls(mod, entry):
    """Yield ``(fns, inner)`` reading a different copy of the weight each call."""
    call = make_call(mod, entry)
    w = getattr(mod, "weight", None) if entry.fn is None else None
    if not isinstance(w, torch.Tensor) or w.dim() != 2:
        yield [call], GRAPH_REPS
        return
    nbytes = w.numel() * w.element_size()
    slots = rotation_plan(nbytes)
    if not slots:
        yield [call], GRAPH_REPS
        return
    try:
        ring = torch.empty((slots, *w.shape), dtype=w.dtype, device=w.device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        ring = None
    # Agreed across ranks: a rank that fell back alone would issue mismatched collectives and wedge the world.
    if not agreed(ring is not None):
        if ring is not None:
            del ring
            torch.cuda.empty_cache()
        yield [call], GRAPH_REPS
        return
    ring.copy_(w)

    def rotate_to(slot):
        def fn():
            mod.weight = slot
            return call()

        return fn

    params = [torch.nn.Parameter(ring[i], requires_grad=False) for i in range(slots)]
    try:
        # The assignment runs during capture, so each launch records its own slot's pointer.
        yield [rotate_to(p) for p in params], max(GRAPH_REPS, slots)
    finally:
        mod.weight = w
        del params, ring
        torch.cuda.empty_cache()


def balance_modules(modules) -> int:
    """Install the seeded balanced router on every MoE layer of a module list (hash layers included)."""
    n = 0
    for layer in modules:
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if experts is not None and not getattr(experts.forward, "bench_balanced", False):
            experts.forward = _balanced_forward(experts)
            n += 1
    return n


def balance_moe_routing(layers, routing: str = "balanced") -> int:
    if routing == "native":
        return 0
    return balance_modules(layers.layers[layers.start : layers.end])


def balance_draft_routing(spec, routing: str = "balanced") -> int:
    if spec is None or routing == "native":
        return 0
    import spec_dspark

    return balance_modules(spec_dspark.draft_layers(spec))


# ----------------------------------------------------------------------------------------------
# Layer kinds and the op tree.


def layer_kind(layer, i: int, late_start: int | None) -> str:
    """DSv4.1 layers differ by compression ratio (0/1/2; DSv4: 0/4/128), hash vs gate routing, whether the
    layer owns a compressor (kv source) / indexer (index source), carries an Engram, and whether it sits in
    the late-layer SWA tail (prefill runs those layers on the tail rows only)."""
    attn, mlp = layer.self_attn, layer.mlp
    ratio = attn.compress_ratio
    kind = "moe" + ("+hash" if getattr(mlp, "is_hash", False) else "") + f"+c{ratio}"
    if attn.compressor is not None and ratio in (1, 2):
        kind += "+kvsrc"
    if attn.indexer is not None:
        kind += "+indexer"
    if getattr(layer, "engram", None) is not None:
        kind += "+engram"
    if late_start is not None and i >= late_start:
        kind += "+late"
    return kind


class Layers:
    """Which layers pay for what, so the roll-up multiplies by the right counts."""

    def __init__(self, model_runner):
        model = model_runner.model.model
        self.model = model
        self.layers = model.layers
        self.start, self.end = model.start_layer, model.end_layer
        # Excludes the nextn / DSPARK draft layers, which live in the draft model.
        self.n_all = self.end - self.start
        self.late_start = getattr(model, "late_layer_start", None)
        self.kinds: dict[str, list[int]] = {}
        for i in range(self.start, self.end):
            self.kinds.setdefault(layer_kind(self.layers[i], i, self.late_start), []).append(i)

    @property
    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.kinds.items()}

    def describe(self) -> str:
        kinds = ", ".join(f"{k} {len(v)} ({_ids(v)})" for k, v in self.kinds.items())
        return f"{self.n_all} layers -- {kinds}"

    def experts_per_rank(self) -> int:
        experts = self.layers[self.start].mlp.experts
        return int(experts.num_local_experts - getattr(experts, "num_fused_shared_experts", 0))


def _ids(ids: list[int]) -> str:
    out, run = [], [ids[0]]
    for x in ids[1:]:
        if x == run[-1] + 1:
            run.append(x)
        else:
            out.append(run)
            run = [x]
    out.append(run)
    return ",".join(f"{r[0]}" if len(r) == 1 else f"{r[0]}-{r[-1]}" for r in out)


class _FnOp(torch.nn.Module):
    """Placeholder for a function call site (no module to hook); the recorder fills its call."""

    def __init__(self, label: str):
        super().__init__()
        self.label = label


TREE = [
    ("layer", 0),
    ("input_layernorm", 1),
    ("attn", 1),
    ("attn.wqkv_a", 2),
    ("attn.wq_a", 2),
    ("attn.wkv", 2),
    ("attn.wq_b", 2),
    ("attn.wo_a", 2),
    ("attn.wo_b", 2),
    ("attn.compressor", 2),
    ("attn.indexer", 2),
    # DSv4.1: ratio-1/2 layers run their kv-source compressor and indexer through the attention backend's
    # forward_low_ratio_sources(layer=...), not through the modules (they appear in Coverage); this row is
    # that backend call site, recorded and replayed like a module.
    ("attn.sources", 2),
    ("attn.<rest>", 2),
    ("post_attention_layernorm", 1),
    ("mlp", 1),
    ("mlp.gate", 2),
    ("mlp.topk", 2),
    ("mlp.experts", 2),
    ("mlp.shared_experts", 2),
    ("mlp.<rest>", 2),
    ("layer.<rest>", 1),
    # DSv4.1: the Engram runs from the model loop before the layer call, so it is a sibling of `layer`,
    # outside its residual, still multiplied by the kind's layer count.
    ("engram", 0),
]

OUTSIDE_LAYER = ("engram",)

# parent -> children subtracted from it to get its `<rest>` row; mHC and collectives live in `layer` itself.
RESIDUALS = {
    parent: [n for n, d in TREE if d == 2 and n.startswith(f"{parent}.") and not n.endswith("<rest>")]
    for parent in ("attn", "mlp")
}
RESIDUALS["layer"] = ["input_layernorm", "attn", "post_attention_layernorm", "mlp"]

# Rows that partition a layer without overlap; their sum is the roll-up exactly.
LEAVES = [n for n, d in TREE if d == 2] + [n for n, d in TREE if d == 1 and n not in RESIDUALS] + list(OUTSIDE_LAYER)

# TREE names are attribute paths, except `attn`, which is `self_attn`.
_ATTR = {"attn": "self_attn"}


def layer_ops(layer) -> dict[str, torch.nn.Module]:
    """The hookable modules of one decoder layer, keyed by TREE name."""
    owners = {"": layer, "attn": getattr(layer, "self_attn", None), "mlp": getattr(layer, "mlp", None)}
    ops = {"layer": layer}
    for name, _ in TREE:
        if name in ops or name.endswith("<rest>"):
            continue
        parent, _, attr = name.rpartition(".")
        if name == "attn.sources":
            attn = layer.self_attn
            if getattr(attn, "compress_ratio", None) in (1, 2) and (
                getattr(attn, "compressor", None) is not None or getattr(attn, "indexer", None) is not None
            ):
                ops[name] = _FnOp("forward_low_ratio_sources")
            continue
        mod = getattr(owners[parent], _ATTR.get(name, attr), None)
        if isinstance(mod, torch.nn.Module):
            ops[name] = mod
    return ops


# Outside the layer stack, run once per forward. DSv4.1 adds the Engram hasher and the mHC head
# (`hc_head` = mhc_combine under hc_pre_from_prev_sublayer); the upstream deepseekv4 port had no head row.
GLOBAL_OPS = ("embed_tokens", "engram_hasher", "hc_head", "norm", "logits")


def collect_ops(layers: "Layers", model) -> dict:
    """Everything to time: `kind|op` per layer kind, plus the global ops."""
    ops = {}
    for kind, ids in layers.kinds.items():
        for name, mod in layer_ops(layers.layers[ids[0]]).items():
            ops[f"{kind}|{name}"] = mod
    inner = model.model
    ops["embed_tokens"] = inner.embed_tokens
    if isinstance(getattr(inner, "engram_hasher", None), torch.nn.Module):
        ops["engram_hasher"] = inner.engram_hasher
    if getattr(inner, "hc_pre_from_prev_sublayer", False):
        ops["hc_head"] = _FnOp("mhc_combine")
    ops["norm"] = inner.norm
    # LogitsProcessor, not lm_head: it matmuls the head's weight directly, so lm_head hooks never fire.
    ops["logits"] = model.logits_processor
    return ops


def add_residuals(times: dict, counts: dict) -> None:
    """Give every parent a `<rest>` row: itself minus its measurable children."""
    for kind in counts:
        for parent, kids in RESIDUALS.items():
            if f"{kind}|{parent}" in times:
                got = sum(times[f"{kind}|{k}"] for k in kids if f"{kind}|{k}" in times)
                times[f"{kind}|{parent}.<rest>"] = times[f"{kind}|{parent}"] - got


def leaf_totals(times: dict, counts: dict) -> dict:
    """ms per forward for the rows that partition a layer without overlapping."""
    totals = {}
    for name in LEAVES:
        ms = sum(
            times[f"{k}|{name}"] * n
            for k, n in counts.items()
            if f"{k}|{name}" in times and (name in OUTSIDE_LAYER or _partitions(times, k, name))
        )
        if ms:
            totals[name] = ms
    totals.update({n: times[n] for n in GLOBAL_OPS if n in times})
    return totals


def kernel_totals(profiles: dict, counts: dict, times: dict) -> dict:
    """Roll representative kernel samples up with the same layer multiplicities."""
    import kernel_breakdown

    totals = {}
    for label, kernels in profiles.items():
        kind, _, name = label.rpartition("|")
        if label not in times or (kind and kind not in counts):
            continue
        if kind and name not in OUTSIDE_LAYER and not _partitions(times, kind, name):
            continue
        row = name + ".<rest>" if name in RESIDUALS else name
        dest = totals.setdefault(row, {})
        multiplier = counts[kind] if kind else 1
        for kernel, (ms, calls, frames) in kernels.items():
            kernel_breakdown.add_sample(dest, kernel, ms * multiplier, calls * multiplier, frames)
    return totals


GB = 1 << 30


def memory_section(args, model_runner, layers, spec=None, kv_factor: int = 1, filled: dict | None = None) -> str:
    """Weights and KV pool, measured off the real buffers rather than a formula."""
    mr = model_runner
    pool = mr.token_to_kv_pool
    groups: dict[str, list] = {}
    for pname, attr, t in pool_buffers(pool):
        groups.setdefault(pname, []).append(t)
    cap = mr.max_total_num_tokens
    used = local_conc(args, mr) * args.ISL * kv_factor
    dp = parallel_state(mr).attn_dp_size

    rows, total = [], 0
    for pname, bufs in groups.items():
        nbytes = sum(t.numel() * t.element_size() for t in bufs)
        total += nbytes
        rows.append((pname, len(bufs), round(nbytes / cap / max(len(bufs), 1)), round(nbytes / cap), nbytes / GB))
    if spec is None:
        models = [mr.model]
    else:
        import spec_dspark

        models = [r.model for r in spec_dspark.runners(spec).values()]
    tensors = {id(t): t for m in models for t in (*m.parameters(), *m.buffers())}.values()
    weights = sum(t.numel() * t.element_size() for t in tensors if t.is_cuda)
    host = sum(t.numel() * t.element_size() for t in tensors if not t.is_cuda)
    hbm = torch.cuda.mem_get_info()[1]
    frac = cfg(mr.server_args).mem_fraction_static
    rows += [
        ("**kv total**", "", "", round(total / cap), total / GB),
        ("weights" + (" (target + draft)" if spec is not None else ""), "", "", "", weights / GB),
        ("**HBM**", "", "", "", hbm / GB),
    ]
    if host:
        rows.append(("host-resident params/buffers (not HBM)", "", "", "", host / GB))
    caps = capacities(mr)
    if dp > 1:
        capacity = (
            f"**capacity {cap * dp:,} full-KV tokens cluster-wide** = {cap:,} per rank x"
            f" {dp} DP ranks = {cap * dp // args.ISL:,} requests at ISL {args.ISL}"
        )
    else:
        capacity = (
            f"**capacity {cap:,} full-KV tokens** = {cap // args.ISL:,} requests at ISL {args.ISL};"
            " the MQA KV is replicated on every TP rank"
        )
    notes = [
        capacity,
        "native pools: " + ", ".join(f"{k} {v:,}" for k, v in caps.items())
        + "; the B/tok columns divide every buffer by the full-KV token capacity (SWA holds its own, smaller count)",
        f"`--mem-fraction-static {frac}` caps weights + kv at {frac * hbm / GB:.0f} GB of the {hbm / GB:.0f} GB HBM;"
        f" this run lands at {(weights + total) / GB:.0f} GB",
        f"this run uses {used:,} context tokens, {used / cap * 100:.1f}% of this rank's full-KV pool",
    ]
    if filled:
        notes.append(
            "synthetic KV filled: " + ", ".join(f"`{k}` {v / GB:.2f} GB" for k, v in sorted(filled.items()))
        )
    moe = moe_parallel(mr)
    if moe.ep_size > 1:
        notes.append(
            f"EP {moe.ep_size}: this rank holds {layers.experts_per_rank()} routed experts"
            + (f", each with its intermediate split {moe.tp_size} ways" if moe.tp_size > 1 else " at full intermediate width")
        )
    return "\n".join(
        [table(rows, ("component", "buffers", "B/tok/buffer", "B/tok", "GB"), fmt=".2f"), "", *(f"- {n}" for n in notes)]
    )


# Inline and fused DSv4.1 work with no independently hookable module.
RESIDUAL_CONTENTS = {
    "attn.<rest>": "q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel",
    "attn.sources": "kv-source compressor + indexer (backend forward_low_ratio_sources)",
    "mlp.<rest>": "shared+routed add, post-expert collective",
    "layer.<rest>": "mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives",
    "engram": "Engram n-gram embedding gate (runs before the layer)",
    "hc_head": "final mHC combine",
}


def residual_contents(model_runner) -> dict[str, str]:
    contents = dict(RESIDUAL_CONTENTS)
    if (a2a := moe_parallel(model_runner).a2a) is not None:
        contents["mlp.experts"] = f"{a2a} dispatch+experts+combine"
        contents["mlp.<rest>"] = "shared+routed add"
    return contents


def index_topk(model_runner):
    hf = model_runner.model_config.hf_text_config if hasattr(model_runner.model_config, "hf_text_config") else model_runner.model_config.hf_config
    return getattr(hf, "index_topk", None)


def tree_name(name: str, depth: int, contents: dict[str, str] | None = None) -> str:
    row_contents = (RESIDUAL_CONTENTS if contents is None else contents).get(name)
    label = f"{name} :: {row_contents}" if row_contents else name
    return f"`{'  ' * depth}{label}`"
