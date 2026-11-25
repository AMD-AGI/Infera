"""Shared plumbing for the GLM-5.2 micro-benchmarks: sglang model setup, per-op timing and layer roll-up."""

import argparse
import contextlib
import inspect
import multiprocessing
import os
from collections.abc import Callable
from typing import NamedTuple

import torch
from sglang_bench import (
    GRAPH_REPS,
    A2aImpl,
    PrefillStep,
    _balanced_forward,
    _partitions,
    agreed,
    bench,
    local_conc,
    moe_parallel,
    parallel_state,
    publish_once,
    require_aiter,
    table,
)

# Hub id keeps the default working off this machine; point --model at a local copy to avoid fetching weights.
DEFAULT_MODEL = "amd/GLM-5.2-MXFP4"

# Constant report prefix: the hub id or a local dir does not normalize into a readable name.
MODEL_TAG = "glm52_mxfp4"


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
        help="time N whole steps under sglang's own cuda graphs and exit, "
        "instead of profiling per op; the only server-comparable step time. "
        "Read by build_server_args, so it lives here rather than with the "
        "decode profile that acts on it.",
    )
    parser.add_argument("--out", help="markdown report path (default: auto-named)")
    return parser


DUMMY_WARNING = (
    "--load-format dummy. Operators whose cost depends on values are mistimed. "
    "Measured on GLM-5.2 decode, attn.indexer reports 24.1% of the step at ISL "
    "100k and 65.9% at ISL 1M, against 9.5% and 16.0% on real weights. "
    "Do not act on these numbers."
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
    experts = next(
        (layer.mlp.experts for layer in layers.layers if hasattr(layer.mlp, "experts")),
        None,
    )
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


def cfg(server_args):
    """What resolution decided, which is not what the raw ServerArgs fields say."""
    try:
        from sglang.srt.arg_groups.overrides import resolving_view
    except ImportError:
        return server_args

    return resolving_view(server_args)


def build_server_args(args: argparse.Namespace, cuda_graph: bool = False):
    """Fill in defaults suited to a micro-bench run."""
    from sglang.srt.server_args import ServerArgs

    require_aiter()

    warn_if_dummy_weights(args.load_format)
    # Eager is required: with sglang's graph on, record_inputs() pre-hooks never fire.
    args.disable_cuda_graph = not (args.full_step or cuda_graph)
    # `auto` is bf16, doubling attention bytes per token; production runs fp8.
    if args.kv_cache_dtype == "auto":
        args.kv_cache_dtype = "fp8_e4m3"

    server_args = ServerArgs.from_cli_args(args)
    # Resolve first, as one_batch.main does: unresolved, mori gets deepep_mode=auto and picks the wrong kernel.
    if hasattr(server_args, "resolve_once"):
        server_args.resolve_once()
    if cfg(server_args).enable_dp_attention and args.CONC % server_args.dp_size:
        raise ValueError(
            f"--CONC {args.CONC} is the global batch and must divide evenly "
            f"across --dp-size {server_args.dp_size}"
        )
    return server_args


def run(work_func: Callable, server_args, args) -> None:
    """Run ``work_func`` per rank, mirroring ``one_batch.main`` (TP>1 forks one process per rank)."""
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs
    from sglang.srt.utils import kill_process_tree, maybe_reindex_device_id

    # The default of 32 per process times tp_size is a few hundred compile workers.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

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
        for p in procs:
            p.join()
        failed = [p.exitcode for p in procs if p.exitcode]
        if failed:
            # A dead rank must not leave the parent at exit 0.
            raise SystemExit(f"{len(failed)} of {len(procs)} ranks exited non-zero: {failed}")
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

    # Dispatch on the signature, since the sglang trees declare this differently and none agree on a version.
    if not inspect.signature(initialize_moe_config).parameters:
        publish_once(server_args, gpu_id, tp_rank)
    for fn in (
        initialize_moe_config,
        initialize_fp8_gemm_config,
        initialize_fp4_gemm_config,
    ):
        fn(server_args) if inspect.signature(fn).parameters else fn()
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    # as the scheduler process does: sglang.srt.plugins hooks (AKLR's dispatch) attach
    load_plugins()

    spec = _build_spec_worker(server_args, port_args, gpu_id, tp_rank)
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


def _parallel_state(server_args, gpu_id: int, tp_rank: int):
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.layers.dp_attention import compute_dp_attention_world_info

    # Use the resolved view, as one_batch.load_model does; the raw record builds a different world under mori.
    c = cfg(server_args)
    attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = compute_dp_attention_world_info(
        c.enable_dp_attention,
        tp_rank,
        c.tp_size,
        c.dp_size,
        c.attn_cp_size,
    )
    return ParallelState(
        tp_rank=tp_rank,
        tp_size=c.tp_size,
        pp_rank=0,
        pp_size=1,
        dp_rank=None,
        dp_size=c.dp_size,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_cp_rank=0,
        attn_cp_size=c.attn_cp_size,
        attn_dcp_rank=tp_rank % c.dcp_size,
        attn_dcp_size=c.dcp_size,
        attn_dp_rank=attn_dp_rank,
        attn_dp_size=attn_dp_size,
        moe_ep_rank=tp_rank // (c.tp_size // c.ep_size),
        moe_ep_size=c.ep_size,
        moe_dp_rank=None,
        moe_dp_size=c.moe_dp_size,
        gpu_id=gpu_id,
    )


def _build_spec_worker(server_args, port_args, gpu_id: int, tp_rank: int):
    """sglang's own EAGLE worker, or None when --speculative-algorithm is unset."""
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if SpeculativeAlgorithm.from_string(server_args.speculative_algorithm).is_none():
        return None

    # The bags must exist before TpModelWorker builds a ModelRunner; the scheduler publishes first for the same reason.
    publish_once(server_args, gpu_id, tp_rank)

    ps = _parallel_state(server_args, gpu_id, tp_rank)
    target = TpModelWorker(
        server_args=server_args,
        gpu_id=gpu_id,
        ps=ps,
        nccl_port=port_args.nccl_port,
    )
    spec = EAGLEWorkerV2(
        server_args=server_args,
        gpu_id=gpu_id,
        ps=ps,
        nccl_port=port_args.nccl_port,
        target_worker=target,
    )

    # Same order as the scheduler's init_memory_pools / attention backends / cuda graphs.
    target.alloc_memory_pool()
    pool, allocator = target.get_memory_pool()
    spec.alloc_memory_pool(
        memory_pool_config=target.model_runner.memory_pool_config,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
    )
    target.init_attention_backends()
    spec.init_attention_backends()
    target.init_cuda_graphs()
    spec.init_cuda_graphs()
    return spec


def accept_len():
    """SGLANG_SIMULATE_ACC_LEN, the same knob the server runs under."""
    from sglang.srt.speculative.spec_utils import SIMULATE_ACC_LEN

    return SIMULATE_ACC_LEN if SIMULATE_ACC_LEN > 0 else None


def spec_draft_decodes(server_args) -> int:
    """Draft-decode forwards in one step."""
    return max(0, (cfg(server_args).speculative_num_steps or 1) - 1)


def spec_forward_count(server_args) -> int:
    """Forwards in one step: draft decodes, draft extend, verify."""
    return spec_draft_decodes(server_args) + 2


def spec_note(server_args) -> str:
    sa = cfg(server_args)
    return (
        f"{sa.speculative_algorithm}, {sa.speculative_num_steps} draft steps,"
        f" topk {sa.speculative_eagle_topk}, {sa.speculative_num_draft_tokens}"
        f" draft tokens. One step is the draft loop, the target verify and the"
        f" draft extend -- the per-op rows below are the verify's share of it,"
        f" measured on the target stack only."
    )


def seed_draft_input(spec, batch) -> None:
    """Give the batch the draft input a previous step's draft-extend would have left."""
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.runtime_context import get_spec
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.eagle_utils import (
        get_draft_recurrent_hidden_state_spec,
    )

    bs = len(batch.reqs)
    device = batch.device
    hidden_size, dtype = get_draft_recurrent_hidden_state_spec(spec.draft_worker.draft_runner)
    vocab = spec.target_worker.model_config.vocab_size
    topk = spec.topk

    batch.spec_info = EagleDraftInput(
        bonus_tokens=torch.randint(0, vocab, (bs,), device=device, dtype=torch.int32),
        hidden_states=torch.randn((bs, hidden_size), device=device, dtype=dtype),
        topk_p=torch.rand((bs, topk), device=device, dtype=torch.float32),
        topk_index=torch.randint(0, vocab, (bs, topk), device=device, dtype=torch.int64),
        # Present exactly when create_idle_input would make it; torch.stack rejects a None.
        draft_probs=(
            torch.rand((bs, vocab), device=device, dtype=torch.float32)
            if get_spec().speculative_use_rejection_sampling
            else None
        ),
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )


def run_step(spec, new_batch, forward_batch, model_runner):
    """One decode step: sglang's whole MTP step when speculating, else a forward."""
    if spec is None:
        return model_runner.forward(forward_batch)
    batch = new_batch()
    seed_draft_input(spec, batch)
    # Seed the DSA top-k a previous draft-extend left, so the draft's first step matches the server's steady state.
    carried = getattr(spec, "_bench_dsa_seed", None)
    if carried is not None:
        batch.spec_info.dsa_topk_indices = carried
    out = spec.forward_batch_generation(batch)
    spec._bench_dsa_seed = getattr(out.next_draft_input, "dsa_topk_indices", None)
    return out


def make_reqs(args, model_runner, factor: int = 1):
    from sglang.benchmark.one_batch import prepare_synthetic_inputs_for_latency_test

    conc = local_conc(args, model_runner)
    page = cfg(model_runner.server_args).page_size
    # ``factor`` is the peak: prefill holds its context and every chunk allocates on top of it.
    want = -(-args.ISL // page) * page * conc * factor
    have = model_runner.max_total_num_tokens
    if want > have:
        # Fail here; sglang's own out-of-memory path raises while formatting the message.
        raise ValueError(
            f"--ISL {args.ISL:,} x --CONC {args.CONC}"
            + (
                f" x {factor} (prefill holds its context while the chunks allocate over it)"
                if factor > 1
                else ""
            )
            + f" needs {want:,} KV tokens "
            f"per rank, and the pool holds {have:,}. Lower one of them, or "
            f"raise --mem-fraction-static."
        )

    return prepare_synthetic_inputs_for_latency_test(conc, args.ISL)


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


def randomize_kv(model_runner, spec=None) -> None:
    """Fill the KV pool with random values, standing in for the extend forward."""
    runners = [model_runner]
    if spec is not None:
        runners = list(spec_runners(spec).values())
    for runner in runners:
        _randomize_pool(runner.token_to_kv_pool)


def _randomize_pool(pool) -> None:
    for buf in pool.kv_buffer:
        if buf.element_size() == 1:
            buf.view(torch.uint8).random_(0, 0x7C)
        else:
            buf.uniform_(-1, 1)
    for buf in getattr(pool, "index_k_with_scale_buffer", []):
        keys = pool.page_size * pool.index_head_dim
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
    """Give a chunk's page-table rows back."""
    pool = model_runner.req_to_token_pool
    if hasattr(pool, "free_rows"):
        pool.free_rows(batch.req_pool_indices_cpu.tolist())
        return
    for req in batch.reqs:
        if getattr(req, "req_pool_idx", None) is not None:
            pool.free(req)


@contextlib.contextmanager
def time_calls(targets, into: dict):
    """CUDA-event timing around chosen methods, one event pair per call."""
    saved = []
    for label, obj, name in targets:
        orig = getattr(obj, name)
        patched_forward = False
        if isinstance(orig, torch.nn.Module):
            # nn.Module.__setattr__ refuses a plain function on a submodule; patch its forward via the instance instead.
            obj, name, orig = orig, "forward", orig.forward
            patched_forward = True

        def wrapped(*a, _l=label, _o=orig, **kw):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            out = _o(*a, **kw)
            end.record()
            into.setdefault(_l, []).append((start, end))
            return out

        setattr(obj, name, wrapped)
        saved.append((obj, name, orig, patched_forward))
    try:
        yield
    finally:
        for obj, name, orig, patched_forward in saved:
            if patched_forward:
                delattr(obj, name)
            else:
                setattr(obj, name, orig)


def elapsed(into: dict) -> dict:
    """(ms, call count) per label, after a synchronize."""
    return {k: (sum(s.elapsed_time(e) for s, e in pairs), len(pairs)) for k, pairs in into.items()}


def spec_phases(spec) -> list:
    """The phases of one speculative step, and the model forwards inside them."""
    models = spec_models(spec)
    return [
        ("draft", spec.draft_worker, "draft"),
        ("verify", spec, "verify"),
        ("draft_extend", spec.draft_worker, "_draft_extend_for_decode"),
        ("  forward:draft", models["draft"], "forward"),
        ("  forward:target", models["target"], "forward"),
    ]


@contextlib.contextmanager
def bench_forwards(models: dict, reps: int, rounds: int, into: dict):
    """Time every top-level forward of one step, each with its own graph."""
    originals = {name: model.forward for name, model in models.items()}
    for name, model in models.items():

        def wrapped(*a, _n=name, _orig=originals[name], **kw):
            into.setdefault(_n, []).append(bench(lambda: _orig(*a, **kw), reps, inner=1, rounds=rounds))
            return _orig(*a, **kw)

        model.forward = wrapped
    try:
        yield
    finally:
        for name, model in models.items():
            model.forward = originals[name]


def spec_runners(spec) -> dict:
    return {
        "target": spec.target_worker.model_runner,
        "draft": spec.draft_worker.draft_runner,
    }


def spec_models(spec) -> dict:
    return {k: r.model for k, r in spec_runners(spec).items()}


def draft_ops(spec) -> dict:
    """The draft model's hookable modules, keyed `draft|<name>`."""
    model = spec.draft_worker.draft_runner.model
    inner = model.model
    ops = {f"draft|{n}": m for n, m in layer_ops(inner.decoder).items()}
    for name in ("embed_tokens", "enorm", "hnorm", "eh_proj"):
        mod = getattr(inner, name, None)
        if mod is not None:
            ops[f"draft|{name}"] = mod
    head = getattr(inner, "shared_head", None)
    if getattr(head, "norm", None) is not None:
        ops["draft|final_norm"] = head.norm
    # lm_head's weight is matmul'd directly, so a hook on it never fires.
    if getattr(model, "logits_processor", None) is not None:
        ops["draft|logits"] = model.logits_processor
    return ops


MLP_FLAGS = ("fuse_mlp_allreduce", "mlp_reduce_scatter")


class Recorded(NamedTuple):
    """One op's captured call site: arguments plus the ambient state it reads."""

    args: tuple
    kwargs: dict
    attn_inputs: object
    mlp_flags: dict[str, bool]


def residual_owner(name: str) -> str:
    """The `<rest>` row that absorbs ``name``'s cost when it has no row of its own."""
    kind, _, op = name.rpartition("|")
    if not kind or kind == "draft":
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
        rows.append((kind or "--", f"`{op}`", where, err))
    return table(rows, ("layer kind", "op", "its ms now sits in", "why it could not be replayed")) + (
        "\n\nThese ops ran in the recording forward, so their cost is real and is"
        " in the parent row; what is missing is the row that would separate it"
        " out. Read the named `<rest>` row as an upper bound, not a measurement."
    )


def coverage_note(ops: dict, saved: dict) -> str:
    """The hooked names that never fired, so they have no row anywhere."""
    missing = sorted(n for n in ops if n not in saved)
    if not missing:
        return ""
    return (
        "Hooked but never called during the recording forward, so absent from"
        " every table and folded into whichever `<rest>` row contains them: "
        + ", ".join(f"`{n}`" for n in missing)
        + ". A name is missing either because this layer kind does not have that"
        " op, or because a fused kernel does its work without going through the"
        " module."
    )


@contextlib.contextmanager
def record_inputs(ops: dict, into: dict):
    """Record each op's real inputs from one real forward."""
    from sglang.srt.runtime_context import get_forward

    handles = []
    for name, mod in ops.items():

        def hook(_m, args, kwargs, _n=name):
            if _n not in into:
                fwd = get_forward()
                into[_n] = Recorded(
                    args,
                    kwargs,
                    getattr(fwd, "attn_inputs", None),
                    {f: getattr(fwd, f) for f in MLP_FLAGS},
                )

        handles.append(mod.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield into
    finally:
        for h in handles:
            h.remove()


def make_call(mod, entry) -> Callable:
    """A callable that runs `mod` on its recorded inputs, ambient state included."""
    from sglang.srt.layers.communicator import ATTN_TP_CONTEXT, AttentionInputs
    from sglang.srt.runtime_context import get_forward

    src = entry.attn_inputs

    def call():
        if src is not None:
            ATTN_TP_CONTEXT.set_attn_inputs(
                AttentionInputs(src.hidden_states_local, src.forward_batch, src.qkv_latent_func)
            )
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
    if not isinstance(w, torch.Tensor) or w.dim() != 2:
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
    """Make every MoE layer spread its tokens over experts, as a trained router does."""
    balance_modules(layers.layers)


def balance_modules(modules) -> None:
    """Install the balanced router on any module list, not just the target stack."""
    for layer in modules:
        experts = getattr(layer.mlp, "experts", None)
        if experts is not None:
            experts.forward = _balanced_forward(experts)


def balance_draft_routing(spec) -> None:
    if spec is None:
        return
    balance_modules([spec.draft_worker.draft_runner.model.model.decoder])


class Layers:
    """Which layers pay for what, so the roll-up multiplies by the right counts."""

    def __init__(self, model_runner):
        self.layers = model_runner.model.model.layers
        # Excludes the nextn draft layer, which sits past the end of the stack.
        self.n_all = model_runner.model_config.hf_config.num_hidden_layers

        # skip_topk layers reuse the previous result but still have an Indexer module, so `is not None` would overcount.
        self.kinds: dict[str, list[int]] = {}
        for i in range(self.n_all):
            layer = self.layers[i]
            kind = "moe" if hasattr(layer.mlp, "experts") else "dense"
            if not getattr(layer.self_attn, "skip_topk", True):
                kind += "+indexer"
            self.kinds.setdefault(kind, []).append(i)

    @property
    def counts(self) -> dict[str, int]:
        """How many layers of each kind."""
        return {k: len(v) for k, v in self.kinds.items()}

    def describe(self) -> str:
        kinds = ", ".join(f"{k} {len(v)}" for k, v in self.kinds.items())
        return f"{self.n_all} layers -- {kinds}"

    def experts_per_rank(self) -> int:
        for layer in self.layers:
            experts = getattr(layer.mlp, "experts", None)
            if experts is not None:
                return int(experts.num_local_experts - experts.num_fused_shared_experts)
        return 0


TREE = [
    ("layer", 0),
    ("input_layernorm", 1),
    ("attn", 1),
    ("attn.fused_qkv_a_proj_with_mqa", 2),
    ("attn.q_b_proj", 2),
    # Only dense-MHA chunks reach this; the absorbed MLA path never calls the module.
    ("attn.kv_b_proj", 2),
    ("attn.o_proj", 2),
    ("attn.indexer", 2),
    ("attn.<rest>", 2),
    ("post_attention_layernorm", 1),
    ("mlp", 1),
    ("mlp.gate", 2),
    ("mlp.topk", 2),
    ("mlp.experts", 2),
    ("mlp.shared_experts", 2),
    ("mlp.gate_up_proj", 2),
    ("mlp.down_proj", 2),
    ("mlp.<rest>", 2),
    ("layer.<rest>", 1),
]

# parent -> children subtracted from it to get its `<rest>` row.
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


GB = 1 << 30


def memory_section(args, model_runner, layers, spec=None, kv_factor: int = 1) -> str:
    """Weights and KV pool, measured off the real buffers rather than a formula."""
    mr = model_runner
    pool = mr.token_to_kv_pool
    idx_bufs = [t for t in getattr(pool, "index_k_with_scale_buffer", []) if t.numel()]
    idx = sum(t.numel() * t.element_size() for t in idx_bufs)
    total = pool.get_kv_size_bytes()
    cap = mr.max_total_num_tokens
    # kv_factor is the peak, not the reservation.
    used = local_conc(args, mr) * args.ISL * kv_factor
    dp = parallel_state(mr).attn_dp_size
    n_indexer = sum(len(v) for k, v in layers.kinds.items() if "indexer" in k)

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
    weights = sum(
        t.numel() * t.element_size()
        for m in ([mr.model] if spec is None else list(spec_models(spec).values()))
        for t in (*m.parameters(), *m.buffers())
    )
    hbm = torch.cuda.mem_get_info()[1]
    frac = cfg(mr.server_args).mem_fraction_static
    rows = [
        row("latent kv", getattr(pool, "kv_buffer", []), total - idx),
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
            f" {args.ISL}; the MLA latent is replicated on every TP rank, so the"
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
    if idx and len(idx_bufs) > n_indexer:
        dead = idx / len(idx_bufs) * (len(idx_bufs) - n_indexer) / GB
        notes.append(
            f"the indexer K cache is allocated for all {len(idx_bufs)} layers but only "
            f"{n_indexer} run the indexer, so ~{dead:.1f} GB of it is never written"
        )
    return "\n".join(
        [
            table(rows, ("component", "layers", "B/tok/layer", "B/tok", "GB"), fmt=".2f"),
            "",
            *(f"- {n}" for n in notes),
        ]
    )


# What lands in each `<rest>` row: inline code or fused kernels with no module to hook.
RESIDUAL_CONTENTS = {
    "attn.<rest>": "qk_norm+rope+absorb_bmm+kv_write+attn_kernel",
    "mlp.<rest>": "act_fn (dense); shared+routed add, post-expert all-reduce (moe)",
    "layer.<rest>": "collectives+residual_add",
}

MHA_RESIDUAL_CONTENTS = {
    **RESIDUAL_CONTENTS,
    "attn.<rest>": "qk_norm+rope+kv_write+attn_kernel",
}


def attn_path(model_runner) -> str:
    """Which attention the extend forward just took: `"mha"` or `"mla"`."""
    backend = getattr(model_runner, "attn_backend", None)
    backend = getattr(backend, "primary", backend)
    return "mha" if getattr(backend, "use_mha", False) else "mla"


def residual_contents(model_runner, path: str) -> dict[str, str]:
    """Labels for the `<rest>` rows, by attention path and MoE layout."""
    contents = dict(MHA_RESIDUAL_CONTENTS if path == "mha" else RESIDUAL_CONTENTS)
    a2a = moe_parallel(model_runner).a2a
    if a2a is None:
        return contents
    contents["mlp.experts"] = f"{a2a} dispatch+experts+combine"
    contents["mlp.<rest>"] = "act_fn (dense); shared+routed add (moe)"
    if parallel_state(model_runner).attn_tp_size > 1:
        contents["layer.<rest>"] = "attn all-gather+reduce-scatter"
    elif cfg(model_runner.server_args).moe_dense_tp_size == 1:
        contents["layer.<rest>"] = "nothing unhooked, noise"
    else:
        contents["layer.<rest>"] = "dense-layer gather+scatter; noise on moe"
    return contents


def index_topk(model_runner):
    """Tokens the DSA indexer keeps per query, or None on a dense model."""
    return getattr(model_runner.model_config.hf_config, "index_topk", None)


def tree_name(name: str, depth: int, contents: dict[str, str] | None = None) -> str:
    """Indent an op name for a markdown table, tagging rows named in `contents`."""
    contents = (RESIDUAL_CONTENTS if contents is None else contents).get(name)
    label = f"{name} :: {contents}" if contents else name
    return f"`{'  ' * depth}{label}`"
