"""Shared plumbing for DeepSeek-V4 Flash/Pro micro-benchmarks."""

import argparse
import contextlib
import multiprocessing
import os
from collections.abc import Callable
from multiprocessing.connection import wait
from typing import NamedTuple

import kernel_breakdown
import torch
from sglang_bench import (
    GRAPH_REPS,
    A2aImpl,
    PrefillStep,
    _balanced_forward,
    _partitions,
    agreed,
    local_conc,
    moe_parallel,
    parallel_state,
    prepare_decode,
    publish_once,
    require_aiter,
    table,
)

# Point --model at a local checkpoint to avoid downloading weights.
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


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
    parser.set_defaults(full_step=0)
    parser.add_argument("--out", help="markdown report path (default: auto-named)")
    return parser


DUMMY_WARNING = (
    "--load-format dummy. Operators whose cost depends on values are mistimed. Do not act on these numbers."
)


def warn_if_dummy_weights(load_format: str) -> None:
    """Warn that dummy weights mistime ops whose cost depends on the values."""
    if load_format != "dummy":
        return
    print(f"[BENCH] WARNING: {DUMMY_WARNING}", flush=True)


def dummy_banner(server_args) -> str:
    """The dummy-weight warning as markdown, or empty on a real load."""
    if server_args.load_format != "dummy":
        return ""
    return f"\n> **WARNING: {DUMMY_WARNING}**\n"


def weights_line(model_runner) -> str:
    """The ``weights`` row: how the checkpoint is quantized, read off its config."""
    sa = model_runner.server_args
    hf = model_runner.model_config.hf_config
    q = getattr(hf, "quantization_config", None) or {}
    method = sa.quantization or q.get("quant_method")
    dtype = str(model_runner.model_config.dtype).replace("torch.", "")
    if not method:
        return f"`{sa.load_format}` load, {dtype} throughout"
    block = q.get("weight_block_size")
    detail = method + (f" block {'x'.join(str(b) for b in block)}" if block else "")
    return f"`{sa.load_format}` load, {detail} quantized, {dtype} elsewhere"


def parallel_tags(model_runner, layers: "Layers") -> list[str]:
    """Report-name parts for the layout, e.g. ``tp8 dp8 ep8 mori normal cap4096``."""
    ps, moe = parallel_state(model_runner), moe_parallel(model_runner)
    tags = [f"tp{ps.tp_size}"]
    if ps.attn_dp_size > 1:
        tags.append(f"dp{ps.attn_dp_size}")
    if moe.ep_size > 1:
        tags.append(f"ep{moe.ep_size}")
    if moe.a2a:
        tags += [moe.a2a, cfg(model_runner.server_args).deepep_mode]
        impl = a2a_impl(model_runner, layers)
        if impl is not None:
            tags.append(f"cap{impl.cap}")
    return tags


def parallel_line(model_runner) -> str:
    """The ``parallel`` row of the config table."""
    ps, moe = parallel_state(model_runner), moe_parallel(model_runner)
    line = f"TP {ps.tp_size}, attn-DP {ps.attn_dp_size}, MoE-EP {moe.ep_size}"
    if moe.ep_size > 1 and moe.tp_size > 1:
        line += f" x TP {moe.tp_size}"
    if moe.a2a:
        line += f", MoE tokens {moe.a2a} all-to-all"
    return line


def a2a_impl(model_runner, layers: "Layers") -> A2aImpl | None:
    """The all-to-all's cap and dtypes, or None without one."""
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
    """The ``moe`` row of the config table, or None when nothing needs saying."""
    moe = moe_parallel(model_runner)
    if moe.a2a is None:
        return None
    impl = a2a_impl(model_runner, layers)
    # The mode picks the kernel; the wrong one made decode 5x slower.
    mode = cfg(model_runner.server_args).deepep_mode
    if impl is None:
        return (
            f"{moe.a2a} all-to-all, mode {mode}; dispatch cap and dtypes not found"
            " on the dispatcher, see the `[MORI init]` line in the log"
        )
    knob = " (SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK)" if moe.a2a == "mori" else ""
    return (
        f"{moe.a2a} all-to-all, mode {mode}, dispatch {impl.dispatch_dtype}, combine"
        f" {impl.combine_dtype}, dispatch cap {impl.cap:,} tokens/rank{knob}:"
        f" `mlp.experts` is handed up to {impl.cap * moe.ep_size:,} rows per call"
    )


def world_times(times: dict[str, float], model_runner) -> dict[str, float]:
    """Per-key ms over world ranks: the slowest under EP, whose routed load differs per rank, else the fastest."""
    from sglang.srt.distributed import get_world_group

    try:
        group = get_world_group()
    except Exception:  # noqa: BLE001 -- single-rank runs have no world
        return times
    pick = max if moe_parallel(model_runner).ep_size > 1 else min
    gathered = group.all_gather_object(times)
    return {name: pick(t[name] for t in gathered) for name in times}


def cfg(server_args):
    """SGLang's resolved values, not the raw CLI fields."""
    from sglang.srt.arg_groups.overrides import resolving_view

    return resolving_view(server_args)


def build_server_args(args: argparse.Namespace, cuda_graph: bool = False):
    """Validate the supported V4 workload before resolving native server flags."""
    from sglang.srt.server_args import ServerArgs

    if args.speculative_algorithm is not None:
        raise ValueError(
            "DeepSeek-V4 micro-bench supports ordinary decode/prefill only; "
            "--speculative-algorithm is not supported"
        )
    if args.CONC <= 0:
        raise ValueError("--CONC must be positive")
    if args.ISL <= 0 or args.reps <= 0 or args.repeat <= 0 or args.full_step < 0:
        raise ValueError("--ISL, --reps and --repeat must be positive; --full-step must be >= 0")
    require_aiter()
    warn_if_dummy_weights(args.load_format)
    # Module hooks need Python forwards; each op gets its own graph instead.
    args.disable_cuda_graph = not (args.full_step or cuda_graph)
    if args.kv_cache_dtype == "auto":
        args.kv_cache_dtype = "fp8_e4m3"
    server_args = ServerArgs.from_cli_args(args)
    server_args.resolve_once()
    resolved = cfg(server_args)
    if resolved.enable_dp_attention and args.CONC % resolved.dp_size:
        raise ValueError(
            f"--CONC {args.CONC} is the global batch and must divide evenly "
            f"across --dp-size {resolved.dp_size}"
        )
    return server_args


def validate_model(server_args):
    """Reject other architectures from config, before loading model weights."""
    from sglang.srt.configs.model_config import ModelConfig

    hf = ModelConfig.from_server_args(server_args).hf_config
    if getattr(hf, "architectures", None) != ["DeepseekV4ForCausalLM"]:
        raise ValueError("This benchmark requires a DeepSeek-V4 Flash/Pro checkpoint (DeepseekV4ForCausalLM)")


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
    """Build the model. Returns (model_runner, rank_print)."""
    from sglang.benchmark.one_batch import load_model
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.plugins import load_plugins
    from sglang.srt.utils import configure_logger

    # The initializers read the published runtime context, and a spawned child arrives with nothing published.
    publish_once(server_args, gpu_id, tp_rank)
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    # as the scheduler process does: sglang.srt.plugins hooks (AKLR's dispatch) attach
    load_plugins()
    # load_routes proxies aiter's entry points only: proxy sglang's too, so AKLR_EXPORT records them
    from aklr.core.runtime import seam
    from aklr.ops import load_routes

    load_routes()
    seam.aklr_proxy()

    model_runner = load_model(server_args, port_args, gpu_id, tp_rank)[0].torch_runner

    if tp_rank != 0:
        return model_runner, lambda *a, **k: None

    def rank_print(msg=""):
        print("\n".join(f"[{tag}] {ln}" for ln in str(msg).split("\n")), flush=True)

    return model_runner, rank_print


def make_reqs(args, model_runner, factor: int = 1, *, decode_tokens: int = 0, swa_prefix: int = 0):
    from sglang.benchmark.one_batch import prepare_synthetic_inputs_for_latency_test

    conc = local_conc(args, model_runner)
    page = cfg(model_runner.server_args).page_size
    # ``factor`` is the peak: prefill holds its context while chunks allocate over it.
    want = -(-(args.ISL + decode_tokens) // page) * page * conc * factor
    if not 0 <= swa_prefix < args.ISL or swa_prefix % page:
        raise ValueError("SWA prefix must be page-aligned and smaller than ISL")
    capacities = {"full KV": model_runner.max_total_num_tokens}
    needed = {"full KV": want}
    allocator = model_runner.token_to_kv_pool_allocator
    # Total sizes: the previous timing batch may still own pages here.
    capacities["SWA"] = allocator.swa_attn_allocator.size
    needed["SWA"] = -(-(args.ISL - swa_prefix + decode_tokens) // page) * page * conc * factor
    short = [name for name, size in capacities.items() if needed[name] > size]
    if short:
        have = ", ".join(f"{name}={size:,}" for name, size in capacities.items())
        requirements = ", ".join(f"{name}={size:,}" for name, size in needed.items())
        raise ValueError(
            f"[BLOCKED_CAPACITY] --ISL {args.ISL:,} x --CONC {args.CONC} "
            f"({conc} requests/rank, page_size={page}, factor={factor}, "
            f"decode_tokens={decode_tokens}) needs {want:,} KV tokens per rank; "
            f"requirements: {requirements}; capacity: {have}. "
            f"Insufficient {', '.join(short)} capacity. "
            "Provision sufficient KV/SWA capacity; benchmark parameters were not reduced."
        )

    return prepare_synthetic_inputs_for_latency_test(conc, args.ISL)


def decode_capacity(args, model_runner, layers: "Layers") -> tuple[int, str]:
    """Largest global decode CONC that fits at this ISL, and the limit that binds it."""
    page = cfg(model_runner.server_args).page_size
    swa_prefix = decode_swa_prefix(args, model_runner)
    allocator = model_runner.token_to_kv_pool_allocator

    def pages(tokens):
        return -(-tokens // page) * page

    limits = {
        "full KV": model_runner.max_total_num_tokens // pages(args.ISL + 1),
        "SWA": allocator.swa_attn_allocator.size // pages(args.ISL - swa_prefix + 1),
        "request slots": model_runner.req_to_token_pool.size,
    }
    if (impl := a2a_impl(model_runner, layers)) is not None:
        limits["a2a dispatch cap"] = impl.cap
    why = min(limits, key=limits.get)
    return limits[why] * parallel_state(model_runner).attn_dp_size, why


def decode_swa_prefix(args, model_runner) -> int:
    """Page-aligned expired SWA prefix; full/compressed context is retained."""
    window = model_runner.model_config.window_size
    page = cfg(model_runner.server_args).page_size
    if not window or window <= 0 or page <= 1:
        raise ValueError("DeepSeek-V4 requires a positive SWA window and paged KV")
    if not callable(model_runner.token_to_kv_pool_allocator.alloc_extend_swa_tail):
        raise TypeError("DeepSeek-V4 requires SGLang's native SWA-tail allocator")
    return max(0, (args.ISL - 1 - window) // page * page)


def make_decode_batch(args, model_runner):
    """Synthetic performance inputs with native full-KV / SWA-tail allocation."""
    swa_prefix = decode_swa_prefix(args, model_runner)
    reqs = make_reqs(args, model_runner, decode_tokens=1, swa_prefix=swa_prefix)
    if swa_prefix:
        allocator = model_runner.token_to_kv_pool_allocator
        allocator.clear()
        model_runner.req_to_token_pool.clear()
        # Native alloc_extend_swa_tail accepts bs=1, so allocate per request, then assemble one batch.
        prefix_len = args.ISL - 1
        zero_cpu = torch.zeros(1, dtype=torch.int64)
        seq_cpu = torch.tensor([prefix_len], dtype=torch.int64)
        zero = zero_cpu.to(model_runner.device)
        seq = seq_cpu.to(model_runner.device)
        last = torch.full((1,), -1, dtype=torch.int64, device=model_runner.device)
        for req in reqs:
            indices = allocator.alloc_extend_swa_tail(
                zero,
                zero_cpu,
                seq,
                seq_cpu,
                last,
                prefix_len,
                swa_tail_len=prefix_len - swa_prefix,
            )
            if indices is None:
                raise RuntimeError("Native SWA-tail allocation failed after capacity check")
            req.prefix_indices = indices
            req.set_extend_range(prefix_len, args.ISL)
        batch = make_extend_batch(reqs, model_runner, reset=False)
    else:
        batch = make_extend_batch(reqs, model_runner)
    prepare_decode(batch, model_runner)
    return batch


def make_extend_batch(reqs, model_runner, reset: bool = True):
    """A prefill ScheduleBatch, up to but not including the forward."""
    from sglang.benchmark.one_batch import (
        TreeCacheNamespace,
        _maybe_prepare_mlp_sync_batch,
    )
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if reset:
        model_runner.token_to_kv_pool_allocator.clear()
        model_runner.req_to_token_pool.clear()

    needed = sum(getattr(req, "req_pool_idx", None) is None for req in reqs)
    available = model_runner.req_to_token_pool.available_size()
    if needed > available:
        global_capacity = len(reqs) * parallel_state(model_runner).attn_dp_size
        raise ValueError(
            f"[BLOCKED_CAPACITY] request slots: need {needed} new slots/rank, "
            f"available={available}. Set --max-running-requests >= "
            f"{global_capacity} (global concurrency, before DP division); "
            "also account for any other live batches."
        )

    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=TreeCacheNamespace(
            page_size=cfg(model_runner.server_args).page_size,
            device=model_runner.device,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        ),
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        batch.prefill_input_ids_cpu = None
    return batch


def randomize_kv(model_runner) -> None:
    """Synthetic KV performance data, not a numerically valid prefilled context."""
    _randomize_pool(model_runner.token_to_kv_pool)


# V4 keeps no buffers on the pool itself: SWA, compressed and indexer caches are separate sub-pools.
SUB_POOLS = ("swa_kv_pool", "c4_kv_pool", "c128_kv_pool", "c4_indexer_kv_pool")


def sub_pools(pool):
    """V4's SWA, compressed KV and indexer pools (no top-level KV buffers)."""
    for name in SUB_POOLS:
        child = getattr(pool, name)
        if child is not None:
            yield child


def _randomize_pool(pool) -> None:
    for p in sub_pools(pool):
        for buf in getattr(p, "kv_buffer", []):
            if buf.element_size() == 1:
                buf.view(torch.uint8).random_(0, 0x7C)
            else:
                buf.uniform_(-1, 1)
        for buf in getattr(p, "index_k_with_scale_buffer", []):
            keys = p.page_size * p.index_head_dim
            buf[:, :keys].random_(0, 0x7C)
            buf[:, keys:].view(torch.float32).uniform_(0.5, 2.0)


def chunk_tokens(args, model_runner) -> int:
    """New tokens the scheduler admits per forward, on this rank."""
    size = cfg(model_runner.server_args).chunked_prefill_size
    return size if size > 0 else args.ISL


def chunk_flag(args, model_runner) -> int:
    """The --chunked-prefill-size behind that budget, before the DP division."""
    if args.chunked_prefill_size:
        return args.chunked_prefill_size
    return chunk_tokens(args, model_runner) * parallel_state(model_runner).attn_dp_size


def plan_prefill(args, model_runner) -> list[PrefillStep]:
    """The chunk timeline the scheduler would walk for one request."""
    page = cfg(model_runner.server_args).page_size
    cached = int(args.ISL * args.prefix_cache_hit) // page * page
    budget = chunk_tokens(args, model_runner)
    steps, prefix = [], cached
    while prefix < args.ISL:
        extend = min(budget, args.ISL - prefix)
        steps.append(PrefillStep(prefix, extend))
        prefix += extend
    if not steps:
        raise ValueError(
            f"--prefix-cache-hit {args.prefix_cache_hit} leaves nothing to prefill at --ISL {args.ISL}"
        )
    return steps


def make_prefill_context(args, model_runner) -> list[torch.Tensor]:
    """KV slots holding each request's context; a step takes its prefix from these."""
    batch = make_extend_batch(make_reqs(args, model_runner, factor=2), model_runner)
    randomize_kv(model_runner)
    table = model_runner.req_to_token_pool.req_to_token
    # Clone: a table row is a view and every extend rewrites the table.
    return [table[idx, : args.ISL].clone().to(torch.int64) for idx in batch.req_pool_indices_cpu.tolist()]


def make_prefill_batch(model_runner, context, step: PrefillStep):
    """A ScheduleBatch for one prefill step, its prefix already in the pool."""
    from sglang.benchmark.one_batch import prepare_synthetic_inputs_for_latency_test

    reqs = prepare_synthetic_inputs_for_latency_test(len(context), step.prefix + step.extend)
    for req, slots in zip(reqs, context):
        req.prefix_indices = slots[: step.prefix]
        req.set_extend_range(step.prefix, step.prefix + step.extend)
    return make_extend_batch(reqs, model_runner, reset=False)


def free_req_rows(batch, model_runner) -> None:
    """Release request rows between chunks, retaining the shared KV context."""
    model_runner.req_to_token_pool.free_rows(batch.req_pool_indices_cpu.tolist())


MLP_FLAGS = ("fuse_mlp_allreduce", "mlp_reduce_scatter")


class Recorded(NamedTuple):
    """One op's captured call site: arguments plus the ambient state it reads."""

    args: tuple
    kwargs: dict
    mlp_flags: dict[str, bool]


def residual_owner(name: str) -> str:
    """The `<rest>` row that absorbs ``name``'s cost when it has no row of its own."""
    kind, _, op = name.rpartition("|")
    if not kind:
        return ""
    if op == "layer":
        return ""
    parent = op.rpartition(".")[0] or "layer"
    if parent not in RESIDUALS:
        return ""
    return f"{kind}|{parent}.<rest>"


def skipped_note(skipped: dict) -> str:
    """Ops that fired but could not be replayed on their own, and where they went."""
    if not skipped:
        return ""
    rows = []
    for name, err in sorted(skipped.items()):
        # Split on the bar: a `kind|op` key carries the character markdown uses to end a cell.
        kind, _, op = name.rpartition("|")
        owner = residual_owner(name)
        if op in RESIDUALS:
            where = "`layer.<rest>`; its children leave the roll-up too"
        elif owner:
            where = f"`{owner.split('|', 1)[1]}`"
        else:
            where = "nowhere -- the roll-up is short by it"
        rows.append((kind or "--", f"`{op}`", where, err))
    return table(rows, ("layer kind", "op", "its ms now sits in", "why it could not be replayed")) + (
        "\n\nThese ops ran in the recording forward, so their cost is real and is"
        " in the parent row; what is missing is the row that would separate it"
        " out. Read the named `<rest>` row as an upper bound, not a measurement."
    )


def coverage_note(ops: dict, saved: dict) -> str:
    """Hooked modules bypassed by inline/fused V4 code; globals have no residual."""
    missing = sorted(n for n in ops if n not in saved)
    if not missing:
        return ""
    rows = []
    for name in missing:
        kind, _, op = name.rpartition("|")
        owner = residual_owner(name)
        where = owner.split("|", 1)[1] if owner else "unattributed (no parent residual)"
        rows.append((kind or "--", op, where))
    return table(rows, ("layer kind", "hook not called", "possible containing row")) + (
        "\n\nHooks do not observe direct weight reads or helper calls. A parent residual "
        "can contain this work; missing globals are not covered by layer residuals."
    )


@contextlib.contextmanager
def record_inputs(ops: dict, into: dict):
    """Capture first-call V4 inputs and scoped MoE flags, outside timing clocks."""
    from sglang.srt.runtime_context import get_forward
    from torch.utils._pytree import tree_map

    handles = []
    try:
        for name, mod in ops.items():

            def hook(_m, args, kwargs, _n=name):
                if _n not in into:
                    fwd = get_forward()
                    snapshots = {}

                    def snapshot(value):
                        if not isinstance(value, torch.Tensor):
                            return value
                        if id(value) not in snapshots:
                            snapshots[id(value)] = value.detach().clone()
                        return snapshots[id(value)]

                    args, kwargs = tree_map(snapshot, (args, kwargs))
                    into[_n] = Recorded(args, kwargs, {f: getattr(fwd, f) for f in MLP_FLAGS})

            handles.append(mod.register_forward_pre_hook(hook, with_kwargs=True))
        yield into
    finally:
        for handle in handles:
            handle.remove()


def make_call(mod, entry) -> Callable:
    """Replay a V4 module with its recorded arguments and scoped MoE flags."""
    from sglang.srt.runtime_context import get_forward

    def call():
        with get_forward().scoped(**entry.mlp_flags):
            return mod(*entry.args, **entry.kwargs)

    return call


# MI355X's last-level cache.
MALL_BYTES = 256 << 20
# Twice the cache, so a slot is evicted before its next read.
ROTATION_BYTES = 2 * MALL_BYTES
MAX_ROTATION_SLOTS = 256


@contextlib.contextmanager
def rotated_calls(mod, entry):
    """Yield ``(fns, inner)`` reading a different copy of the weight each call."""
    call = make_call(mod, entry)
    w = getattr(mod, "weight", None)
    if w is None or not isinstance(w, torch.Tensor) or w.dim() != 2:
        yield [call], GRAPH_REPS
        return

    nbytes = w.numel() * w.element_size()
    slots = min(ROTATION_BYTES // max(nbytes, 1), MAX_ROTATION_SLOTS)
    if nbytes >= MALL_BYTES or slots < 2 or slots * nbytes <= MALL_BYTES:
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
    assert ring is not None
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


def balance_moe_routing(layers) -> None:
    """Synthetic seeded expert routing for every V4 layer, including hash layers."""
    for layer in layers.layers[: layers.n_all]:
        experts = layer.mlp.experts
        experts.forward = _balanced_forward(experts)


def layer_kind(layer) -> str:
    """V4 layers differ by KV compression (0/4/128) and hash versus gate routing."""
    attn, mlp = layer.self_attn, layer.mlp
    ratio = attn.compress_ratio
    if ratio not in (0, 4, 128):
        raise ValueError(f"Unsupported DeepSeek-V4 compression ratio: {ratio}")
    kind = "moe" + ("+hash" if mlp.is_hash else "") + f"+c{ratio}"
    return kind + ("+indexer" if ratio == 4 else "")


class Layers:
    """Which layers pay for what, so the roll-up multiplies by the right counts."""

    def __init__(self, model_runner):
        self.layers = model_runner.model.model.layers
        # Excludes the nextn draft layer, which sits past the end of the stack.
        self.n_all = model_runner.model_config.hf_config.num_hidden_layers

        self.kinds: dict[str, list[int]] = {}
        for i in range(self.n_all):
            self.kinds.setdefault(layer_kind(self.layers[i]), []).append(i)

    @property
    def counts(self) -> dict[str, int]:
        """How many layers of each kind."""
        return {k: len(v) for k, v in self.kinds.items()}

    def describe(self) -> str:
        kinds = ", ".join(f"{k} {len(v)}" for k, v in self.kinds.items())
        return f"{self.n_all} layers -- {kinds}"

    def experts_per_rank(self) -> int:
        """Routed experts this rank holds, read off the first MoE layer."""
        experts = self.layers[0].mlp.experts
        return int(experts.num_local_experts - experts.num_fused_shared_experts)


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
    ("attn.<rest>", 2),
    ("post_attention_layernorm", 1),
    ("mlp", 1),
    ("mlp.gate", 2),
    ("mlp.topk", 2),
    ("mlp.experts", 2),
    ("mlp.shared_experts", 2),
    ("mlp.<rest>", 2),
    ("layer.<rest>", 1),
]

# parent -> children subtracted from it to get its `<rest>` row; mHC and collectives live in `layer` itself.
RESIDUALS = {
    parent: [n for n, d in TREE if d == 2 and n.startswith(f"{parent}.") and not n.endswith("<rest>")]
    for parent in ("attn", "mlp")
}
RESIDUALS["layer"] = ["input_layernorm", "attn", "post_attention_layernorm", "mlp"]

# Rows that partition a layer without overlap; their sum is the roll-up exactly.
LEAVES = [n for n, d in TREE if d == 2] + [n for n, d in TREE if d == 1 and n not in RESIDUALS]

# TREE names are attribute paths, except `attn`, which is `self_attn`.
_ATTR = {"attn": "self_attn"}


def layer_ops(layer) -> dict[str, torch.nn.Module]:
    """The hookable modules of one decoder layer, keyed by TREE name."""
    owners = {"": layer, "attn": layer.self_attn, "mlp": layer.mlp}
    ops = {"layer": layer}
    for name, _ in TREE:
        if name in ops or name.endswith("<rest>"):
            continue
        parent, _, attr = name.rpartition(".")
        mod = getattr(owners[parent], _ATTR.get(name, attr), None)
        if mod is not None:
            ops[name] = mod
    return ops


# Outside the layer stack, run once per forward.
GLOBAL_OPS = ("embed_tokens", "norm", "logits")


def collect_ops(layers: "Layers", model) -> dict:
    """Everything to time: `kind|op` per layer kind, plus the global ops."""
    ops = {}
    for kind, ids in layers.kinds.items():
        for name, mod in layer_ops(layers.layers[ids[0]]).items():
            ops[f"{kind}|{name}"] = mod
    ops["embed_tokens"] = model.model.embed_tokens
    ops["norm"] = model.model.norm
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
            if f"{k}|{name}" in times and _partitions(times, k, name)
        )
        if ms:
            totals[name] = ms
    totals.update({n: times[n] for n in GLOBAL_OPS if n in times})
    return totals


def kernel_totals(profiles: dict, counts: dict, times: dict) -> dict:
    """Roll representative kernel samples up with the same layer multiplicities."""
    totals = {}
    for label, kernels in profiles.items():
        kind, _, name = label.rpartition("|")
        if label not in times or (kind and kind not in counts):
            continue
        if kind and not _partitions(times, kind, name):
            continue
        row = name + ".<rest>" if name in RESIDUALS else name
        dest = totals.setdefault(row, {})
        multiplier = counts[kind] if kind else 1
        for kernel, (ms, calls, frames) in kernels.items():
            kernel_breakdown.add_sample(dest, kernel, ms * multiplier, calls * multiplier, frames)
    return totals


GB = 1 << 30


def memory_section(args, model_runner, layers, kv_factor: int = 1) -> str:
    """Weights and KV pool, measured off the real buffers rather than a formula."""
    mr = model_runner
    pool = mr.token_to_kv_pool
    # Only entries holding bytes: the pool keeps a slot per layer whether or not it runs the indexer.
    pools = list(sub_pools(pool))
    kv_bufs = [t for p in pools for t in getattr(p, "kv_buffer", [])]
    idx_bufs = [t for p in pools for t in getattr(p, "index_k_with_scale_buffer", []) if t.numel()]
    idx = sum(t.numel() * t.element_size() for t in idx_bufs)
    total = idx + sum(t.numel() * t.element_size() for t in kv_bufs)
    cap = mr.max_total_num_tokens
    # kv_factor is the peak, not the reservation.
    used = local_conc(args, mr) * args.ISL * kv_factor
    dp = parallel_state(mr).attn_dp_size

    def row(name, bufs, nbytes):
        per_tok = nbytes / cap
        return (
            name,
            len(bufs),
            round(per_tok / max(len(bufs), 1)),
            round(per_tok),
            nbytes / GB,
        )

    # Summed, since weight_load_mem_usage is a free-memory delta that swings between runs.
    weights = sum(t.numel() * t.element_size() for t in (*mr.model.parameters(), *mr.model.buffers()))
    hbm = torch.cuda.mem_get_info()[1]
    frac = cfg(mr.server_args).mem_fraction_static
    rows = [
        row("SWA/compressed KV", kv_bufs, total - idx),
        row("indexer k", idx_bufs, idx),
        ("**kv total**", "", "", round(total / cap), total / GB),
        ("weights", "", "", "", weights / GB),
        ("**HBM**", "", "", "", hbm / GB),
    ]
    # Lead with real serving capacity: under DP attention the cluster holds dp_size times a rank's pool.
    if dp > 1:
        capacity = (
            f"**capacity {cap * dp:,} tokens cluster-wide** = {cap:,} per rank x"
            f" {dp} DP ranks = {cap * dp // args.ISL:,} requests at ISL {args.ISL}"
        )
    else:
        capacity = (
            f"**capacity {cap:,} tokens** = {cap // args.ISL:,} requests at ISL"
            f" {args.ISL}; the MQA KV is replicated on every TP rank, so the"
            " cluster holds these same tokens, not tp_size times them"
        )
    notes = [
        capacity,
        (
            f"`--mem-fraction-static {frac}` caps weights + kv at"
            f" {frac * hbm / GB:.0f} GB of the {hbm / GB:.0f} GB HBM; this run lands at"
            f" {(weights + total) / GB:.0f} GB"
        ),
        f"this run uses {used:,} tokens, {used / cap * 100:.1f}% of this rank's pool",
    ]
    moe = moe_parallel(mr)
    if moe.ep_size > 1:
        notes.append(
            f"EP {moe.ep_size}: this rank holds {layers.experts_per_rank()} routed"
            " experts"
            + (
                f", each with its intermediate split {moe.tp_size} ways"
                if moe.tp_size > 1
                else " at full intermediate width"
            )
            + (
                f"; under {moe.a2a} the shared expert is not sharded, and the"
                " all-to-all buffers (sized by the dispatch cap, allocated"
                " outside torch) are not in this table or the"
                " `--mem-fraction-static` budget"
                if moe.a2a
                else ""
            )
        )
    return "\n".join(
        [
            table(rows, ("component", "layers", "B/tok/layer", "B/tok", "GB"), fmt=".2f"),
            "",
            *(f"- {n}" for n in notes),
        ]
    )


# Inline and fused V4 work with no independently hookable module.
RESIDUAL_CONTENTS = {
    "attn.<rest>": "qk_norm+rope+compressor+wo_a+kv_write+attn_kernel",
    "mlp.<rest>": "shared+routed add, post-expert collective",
    "layer.<rest>": "mhc_pre+hc_post+collectives",
}


def residual_contents(model_runner) -> dict[str, str]:
    """V4 residual labels; all-to-all dispatch/combine live inside experts."""
    contents = dict(RESIDUAL_CONTENTS)
    if (a2a := moe_parallel(model_runner).a2a) is not None:
        contents["mlp.experts"] = f"{a2a} dispatch+experts+combine"
        contents["mlp.<rest>"] = "shared+routed add"
    return contents


def index_topk(model_runner):
    """Compressed C4 tokens retained per query by the V4 indexer."""
    return model_runner.model_config.hf_config.index_topk


def tree_name(name: str, depth: int, contents: dict[str, str] | None = None) -> str:
    """Indent an op name for a markdown table, tagging rows named in `contents`."""
    row_contents = (RESIDUAL_CONTENTS if contents is None else contents).get(name)
    label = f"{name} :: {row_contents}" if row_contents else name
    return f"`{'  ' * depth}{label}`"
