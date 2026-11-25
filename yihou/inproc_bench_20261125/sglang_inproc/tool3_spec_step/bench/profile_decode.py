#!/usr/bin/env python3
"""Pinned SGLang DeepSeek-V4.1 internal DSPARK speculative-decode benchmark; no scheduler or serving layer.

Port of the GLM-5.2 tool 3 (TpModelWorker + EAGLEWorkerV2) to sglang 6be0d75ae
(lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111) and DeepSeek-V4.1-Flash: target TpModelWorker
+ DSparkWorkerV2, built in the Scheduler's construction order, driven by hand on one ScheduleBatch.

Use plain python: this entry point spawns its own TP processes. --help requires
only the standard library. Additional arguments pass through to ServerArgs.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time

from batch_state import (
    DecodeAccounting,
    count_graph_executions,
    module_source_paths,
    required_token_capacity,
    validate_kv_layout,
    validate_worker_progress,
)

from topology import (
    DecodeTopology,
    aggregate_rank_summaries,
    compare_parallel_state,
    prepare_dp_metadata,
    rank_progress_signature,
    validate_rank_progress,
)

# Stdlib-only, like the two modules above: this is imported in the parent before spawning.
from isl_spec import parse_isl_spec

# Opt-in profiling. This module is stdlib-only at import time (torch and SGLang are
# imported lazily inside it) and nothing it owns is constructed unless --profile.
from profiling_yihou import (
    ProfileSession,
    add_profile_cli_args,
    configure_profile_env,
    validate_profile_args,
)

PINNED_SGLANG = "6be0d75ae060699396a6cab5f356ef499b90de5a"
DEFAULT_INPUT_LEN = 8192
VERIFY_WINDOW = 6  # DSPARK block 5 = gamma 5 -> verify window gamma + 1 = 6 (bonus-inclusive accept bound)
OWNED_FLAGS = ("--dp-size", "--data-parallel-size", "--ep-size", "--expert-parallel-size", "--tp-size",
               "--tensor-parallel-size", "--enable-dp-attention", "--moe-a2a-backend",
               "--cuda-graph-bs-decode", "--cuda-graph-max-bs-decode")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    # Default stays None so that passing both this and --input-len-spec is detectable; an
    # unspecified run still resolves to DEFAULT_INPUT_LEN.
    parser.add_argument("--input-len", type=int, default=None, help="decimal tokens per request, uniform across the batch")
    parser.add_argument("--input-len-spec", default=None,
                        help="heterogeneous ISL, per attention-DP rank: uniform:N | bimodal:A,B,ratio "
                             "| normal:mean,std[,lo,hi] | list:a,b,... | list:@file")
    parser.add_argument("--output-len", type=int, default=1024, help="useful emitted tokens per request")
    parser.add_argument("--accept-length", type=float, default=3.51, help="expected bonus-inclusive length, not probability")
    parser.add_argument("--accept-method", choices=("match-expected", "multinomial"), default="match-expected")
    parser.add_argument("--accept-token-mode", choices=("real-draft-token", "fixed"), default="real-draft-token")
    parser.add_argument("--accept-greedy", action="store_true",
                        help="SGLANG_SIMULATE_ACC_GREEDY (DSPARK opt-in; off in #5034)")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--enable-dp-attention", action="store_true")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means full OSL; positive values are incomplete smoke runs")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-interval", type=int, default=100)
    # Port-only knobs (the GLM-5.2 tool had no prefill; see bootstrap()).
    parser.add_argument("--prefill-tokens", type=int, default=None,
                        help="max prompt tokens per bootstrap prefill forward (whole requests are grouped, one at minimum); "
                             "default: the engine's resolved chunked_prefill_size (16384 under DPA4 for #5034)")
    parser.add_argument("--dp-metadata", choices=("gather", "static"), default="gather",
                        help="gather: the engine's MLP-sync all-gather each forward (as the Scheduler); static: set by hand (GLM-5.2 style)")
    add_profile_cli_args(parser)
    return parser


def validate_args(args):
    DecodeTopology(args.tp_size, args.ep_size, args.batch_size, args.enable_dp_attention)
    if min(args.batch_size, args.output_len, args.tp_size) <= 0:
        raise ValueError("batch size, output length and TP size must be positive")
    if args.max_steps < 0 or args.warmup_steps < 0:
        raise ValueError("max-steps and warmup-steps must be nonnegative")
    if not 1 <= args.accept_length <= VERIFY_WINDOW:
        raise ValueError(f"accept-length must be in [1, {VERIFY_WINDOW}] for DSPARK block 5 (verify window 6)")
    if args.input_len is not None and args.input_len <= 0:
        raise ValueError("input-len must be positive")
    if args.input_len is not None and args.input_len_spec is not None:
        raise ValueError("Pass either --input-len or --input-len-spec, not both")
    if args.prefill_tokens is not None and args.prefill_tokens <= 0:
        raise ValueError("prefill-tokens must be positive")
    validate_profile_args(args)


def resolve_input_lens(args):
    """Resolve the ISL spec ONCE in the parent, then ship the vector to every rank."""
    topology = DecodeTopology(args.tp_size, args.ep_size, args.batch_size, args.enable_dp_attention)
    text = args.input_len_spec if args.input_len_spec is not None else str(
        DEFAULT_INPUT_LEN if args.input_len is None else args.input_len)
    spec = parse_isl_spec(text)
    lens = spec.generate(topology.local_batch_size, args.seed)
    if min(lens) < 2:
        raise ValueError("every input length must be at least 2 for a populated-prefix bootstrap")
    args.input_lens = lens
    return spec


def server_cli(args, extra):
    topology = DecodeTopology(args.tp_size, args.ep_size, args.batch_size, args.enable_dp_attention)
    flags = {item.split("=", 1)[0] for item in extra if item.startswith("--")}
    if flags & set(OWNED_FLAGS):
        raise ValueError(f"Benchmark controls these topology/graph flags: {sorted(flags & set(OWNED_FLAGS))}")
    result = ["--model-path", args.model_path, "--tp-size", str(args.tp_size),
              "--ep-size", str(args.ep_size), "--dp-size", str(topology.dp_size),
              "--moe-a2a-backend", "none"] + list(extra)
    if args.enable_dp_attention:
        result.append("--enable-dp-attention")
    defaults = {
        "--speculative-algorithm": "DSPARK",
        "--speculative-dspark-block-size": "5",
        "--cuda-graph-bs-decode": str(topology.local_batch_size),
        "--cuda-graph-max-bs-decode": str(topology.local_batch_size),
        "--random-seed": str(args.seed),
    }
    for name, value in defaults.items():
        if name not in flags:
            result.extend([name, value])
    for flag in ("--disable-radix-cache", "--skip-tokenizer-init", "--disable-overlap-schedule", "--trust-remote-code"):
        if flag not in flags:
            result.append(flag)
    if args.disable_cuda_graph:
        result.append("--disable-cuda-graph")
    return result


def configure_acceptance(args):
    # Must precede importing spec_utils, whose simulation constants are import-time; the DSPARK
    # worker reads envs.SGLANG_SIMULATE_ACC_LEN / _GREEDY at construction.
    values = {"SGLANG_SIMULATE_ACC_LEN": str(args.accept_length),
              "SGLANG_SIMULATE_ACC_METHOD": args.accept_method,
              "SGLANG_SIMULATE_ACC_TOKEN_MODE": args.accept_token_mode,
              "SGLANG_SIMULATE_ACC_GREEDY": "1" if args.accept_greedy else "0"}
    os.environ.update(values)
    if "sglang" in sys.modules:
        from sglang.srt.environ import envs
        envs.SGLANG_SIMULATE_ACC_LEN.set(args.accept_length)
        envs.SGLANG_SIMULATE_ACC_METHOD.set(args.accept_method)
        envs.SGLANG_SIMULATE_ACC_TOKEN_MODE.set(args.accept_token_mode)
        envs.SGLANG_SIMULATE_ACC_GREEDY.set(bool(args.accept_greedy))
    return values


def import_redact_argv():
    """The kit's common/redact.py (also found from a bench_snapshot copy, via the container's /kit)."""
    for candidate in (Path(__file__).resolve().parents[3] / "common", Path("/kit/common")):
        if (candidate / "redact.py").is_file():
            sys.path.insert(0, str(candidate))
            break
    from redact import redact_argv
    return redact_argv


def log(rank, message):
    print(f"[INTERNAL-DECODE TP{rank}] {message}", flush=True)


class Engine:
    """The objects the Scheduler would own for this rank (target, spec worker, pools, tree cache)."""

    def __init__(self, server_args, port_args, gpu_id, rank, args, topology):
        import torch
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed import bootstrap
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
        from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
        from sglang.srt.managers.tp_worker import TpModelWorker
        from sglang.srt.mem_cache import kv_cache_builder
        from sglang.srt.runtime_context import get_device, get_model, get_parallel, get_schedule, get_spec
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.args, self.topology, self.rank = args, topology, rank
        self.forward_ct = 0
        # Scheduler.__init__ order (sglang 6be0d75ae managers/scheduler.py:596-626).
        self.model_config = ModelConfig.from_server_args(server_args)
        bootstrap.init_parallel_runtime(server_args=server_args, device=get_device().device,
                                        dist_port=port_args.nccl_port)
        bootstrap.init_layer_runtime(model_config=self.model_config)
        initialize_moe_config()
        initialize_fp8_gemm_config()
        initialize_fp4_gemm_config()
        initialize_bf16_gemm_config()
        self.parallel_mismatch = compare_parallel_state(topology, rank, get_parallel())
        if self.parallel_mismatch:
            raise RuntimeError(f"Resolved parallel state differs from benchmark topology: {self.parallel_mismatch}")
        self.spec_algorithm = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
        if not self.spec_algorithm.is_dspark():
            raise RuntimeError(f"Expected DSPARK, got {get_spec().speculative_algorithm}")
        # Scheduler.init_model_worker (managers/scheduler.py:1192-1270).
        log(rank, "loading target weights")
        self.target = TpModelWorker(server_args=server_args, gpu_id=gpu_id, nccl_port=port_args.nccl_port)
        if get_model().is_startup_weight_load_overlap:
            self.target.start_startup_weight_load()
        log(rank, "loading DSPARK draft weights")
        worker_cls = self.spec_algorithm.create_worker(server_args)
        self.worker = worker_cls(server_args=server_args, gpu_id=gpu_id, nccl_port=port_args.nccl_port,
                                 target_worker=self.target)
        log(rank, "allocating target and draft KV pools")
        runner = self.target.model_runner
        runner.account_preloaded_weights(self.target.preloaded_weights_bytes + self.worker.preloaded_weights_bytes)
        self.target.alloc_memory_pool(draft_parallel_layout=self.worker.draft_parallel_layout)
        kv_cache_builder.resolve_decode_retraction_backup(tp_worker=self.target)
        pool, allocator = self.target.get_memory_pool()
        self.worker.alloc_memory_pool(memory_pool_config=runner.memory_pool_config,
                                      req_to_token_pool=pool, token_to_kv_pool_allocator=allocator)
        self.worker.init_hicache_draft_plan()
        self.target.init_attention_backends()
        self.worker.init_attention_backends()
        log(rank, "capturing target and draft graphs (startup may take minutes)")
        with torch.no_grad():
            self.target.init_cuda_graphs()
            self.worker.init_cuda_graphs()
        self.worker.prewarm_sampling()
        if runner.token_to_kv_pool.post_capture_active:
            runner.post_capture_resize_kv_pool(draft_runners=self.worker._draft_model_runners())
        kv_cache_builder.agree_on_stage_request_capacity(tp_worker=self.target)
        if get_model().is_startup_weight_load_overlap:
            self.target.finalize_startup_weight_load()
        # Scheduler builds the tree cache after the worker (managers/scheduler.py:631-677).
        self.page_size = get_schedule().page_size
        self.tree_cache_kind = "server"
        try:
            built = kv_cache_builder.build_kv_cache(
                server_args=server_args, model_config=self.model_config, tp_worker=self.target,
                page_size=self.page_size, spec_algorithm=self.spec_algorithm, enable_metrics=False,
                enable_kv_cache_events=False, enable_hierarchical_cache=False,
                hicache_draft_plan=getattr(self.worker, "hicache_draft_plan", None),
                draft_window_reaches=getattr(self.worker, "draft_window_reaches", None))
            self.tree_cache = built.tree_cache
            self.req_to_token_pool = built.req_to_token_pool
            self.allocator = built.token_to_kv_pool_allocator
        except Exception as error:  # noqa: BLE001 - recorded and replaced by bench_one_batch's dummy
            from sglang.benchmark.one_batch import TreeCacheNamespace
            log(rank, f"build_kv_cache failed ({error!r}); using bench_one_batch TreeCacheNamespace")
            self.tree_cache_kind = f"bench_one_batch.TreeCacheNamespace (build_kv_cache: {error!r})"
            self.req_to_token_pool, self.allocator = pool, allocator
            self.tree_cache = TreeCacheNamespace(page_size=self.page_size, device=runner.device,
                                                 req_to_token_pool=pool, token_to_kv_pool_allocator=allocator)

    @property
    def target_runner(self):
        return self.target.model_runner

    @property
    def draft_runner(self):
        return self.worker.draft_model_runner

    def graph_runner(self, runner):
        graph = getattr(runner, "decode_cuda_graph_runner", None)
        if graph is None or graph is getattr(runner, "eager_runner", object()):
            return None
        return graph

    def sync_fn(self):
        if self.args.dp_metadata == "static":
            return None
        from sglang.srt.layers.dp_attention import require_mlp_sync, require_mlp_tp_gather
        from sglang.srt.managers.scheduler_components.dp_attn import prepare_mlp_sync_batch_raw
        from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
        from sglang.srt.runtime_context import get_schedule
        if not require_mlp_sync():
            return lambda batch: batch

        def no_idle():
            raise RuntimeError("bench never runs an idle DP rank")

        def sync(batch):
            return prepare_mlp_sync_batch_raw(
                batch, model_runner=self.target_runner, get_idle_batch=no_idle,
                disable_cuda_graph=cuda_graph_fully_disabled(), require_mlp_tp_gather=require_mlp_tp_gather(),
                disable_overlap_schedule=get_schedule().disable_overlap_schedule, offload_tags=set())
        return sync

    def kv_inventory(self):
        inventory = {}
        for label, runner in (("target", self.target_runner), ("draft", self.draft_runner)):
            pool = runner.token_to_kv_pool
            entry = {"class": type(pool).__name__}
            infos = getattr(pool, "get_contiguous_buf_infos", None)
            if infos is not None:
                ptrs, lens, _items = infos()
                entry.update(num_buffers=len(ptrs), bytes=int(sum(lens)))
            inventory[label] = entry
        target = inventory["target"]
        validate_kv_layout(target["class"], target.get("num_buffers", 0))
        return inventory


def make_reqs(engine, lengths, osl, rank):
    """Synthetic prompts: uniform random ids in [0, 10000), identical on every rank (same seed)."""
    from array import array
    import numpy as np
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    rng = np.random.default_rng(engine.args.seed)
    offset = engine.topology.request_id_offset(rank)
    reqs = []
    for index, length in enumerate(lengths):
        ids = rng.integers(0, 10000, length, dtype=np.int64)
        # String rids: DP-attention sync calls rid.startswith (VALIDATION.md, step_bench fix).
        req = Req(rid=f"bench-{offset + index}", origin_input_text="", origin_input_ids=array("q", ids.tolist()),
                  sampling_params=SamplingParams(temperature=0, max_new_tokens=osl + 2 * VERIFY_WINDOW, ignore_eos=True))
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.extend_end = len(req.origin_input_ids)
        reqs.append(req)
    return reqs


def forward(engine, batch):
    """Scheduler.run_batch non-overlap spec-v2 branch (managers/scheduler.py:5418-5442)."""
    from sglang.srt.managers.overlap_utils import resolve_forward_inputs
    # Scheduler.run_batch stamps the forward counter (managers/scheduler.py:4961); DSPARK observers read it.
    engine.forward_ct += 1
    batch.forward_iter = engine.forward_ct
    if batch.prefill_input_ids_cpu is not None:
        # Prefill staging only; decode input ids come from the DSPARK draft input (no FutureMap here,
        # since the non-overlap path installs next_draft_input directly).
        resolve_forward_inputs(batch, None)
    snapshot = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
    if batch.sampling_info is not None:
        batch.sampling_info = batch.sampling_info.copy_for_forward()
    try:
        result = engine.worker.forward_batch_generation(batch)
    finally:
        for name, value in snapshot.items():
            setattr(batch, name, value)
    batch.spec_info = result.next_draft_input
    new_seq_lens = result.new_seq_lens
    if new_seq_lens is not None and new_seq_lens is not batch.seq_lens:
        batch.seq_lens = new_seq_lens
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_cpu = new_seq_lens.to("cpu")
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
    batch.input_ids = None
    return result


def bootstrap(engine, rank):
    """Real prefill of the synthetic prompts, grouped, then merged into one running batch.

    Adapted from the GLM-5.2 one-token bootstrap over random KV bytes: the DeepSeek-V4.1 pool
    (SWA ring, C4/C128 compressed pools with stateful compressors, FP4/FP8 indexer, engram host
    table) is written by the engine's own prefill kernels here, so the prefix KV is physically
    valid by construction and the DSPARK draft KV is injected from the target hidden states
    exactly as in serving (DSparkWorkerV2._forward_prefill).
    """
    import torch
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    args, topology = engine.args, engine.topology
    reqs = make_reqs(engine, args.input_lens, args.output_len, rank)
    groups, current, tokens = [], [], 0
    for req in reqs:
        if current and tokens + len(req.origin_input_ids) > args.prefill_tokens:
            groups.append(current)
            current, tokens = [], 0
        current.append(req)
        tokens += len(req.origin_input_ids)
    groups.append(current)
    sync = engine.sync_fn()
    batch = None
    for group in groups:
        part = ScheduleBatch.init_new(group, engine.req_to_token_pool, engine.allocator, engine.tree_cache,
                                      engine.model_config, False, engine.spec_algorithm)
        part.prepare_for_extend()
        prepare_dp_metadata(part, topology, is_extend=True, disable_cuda_graph=args.disable_cuda_graph, sync=sync)
        result = forward(engine, part)
        torch.cuda.synchronize()
        first = result.next_token_ids.cpu().tolist()
        if len(first) != len(group):
            raise RuntimeError(f"Prefill returned {len(first)} tokens for {len(group)} requests")
        for req, token in zip(group, first):
            req.output_ids.append(int(token))
        info = result.next_draft_input
        if info is None or info.bonus_tokens is None or info.bonus_tokens.shape[0] != len(group):
            raise RuntimeError(f"Bootstrap produced invalid draft input: {info}")
        if batch is None:
            batch = part
        else:
            batch.merge_batch(part)
    if list(batch.seq_lens_cpu.tolist()) != list(args.input_lens):
        raise RuntimeError(f"Post-prefill seq_lens {batch.seq_lens_cpu.tolist()} != ISL {args.input_lens}")
    return batch, len(groups)


def release(engine, batch):
    """Free every request's KV row through the server path (common.release_kv_cache)."""
    from sglang.srt.mem_cache.common import release_kv_cache
    for req in batch.reqs:
        if req.kv.holds_kv:
            release_kv_cache(req, engine.tree_cache, checkpoint=False)
    engine.worker.clear_cache_pool()


def reset_acceptance_rng(engine, seed):
    """Re-seed the engine's simulated-acceptance generator to its construction state.

    GLM-5.2's coins came from the process CPU RNG, reset by torch.manual_seed. DSPARK draws them
    from a private generator seeded from the server seed (spec_utils.simulated_acc_len_generator,
    dspark_verify.py:172); re-creating it makes realized acceptance independent of warmup length.
    """
    from sglang.srt.speculative.spec_utils import simulated_acc_len_generator
    executor = engine.worker._verify_executor
    if executor._simulated_acc_len_rng is None:
        raise RuntimeError("DSPARK simulated acceptance is not active (SGLANG_SIMULATE_ACC_LEN not set?)")
    executor._simulated_acc_len_rng = simulated_acc_len_generator(seed)


def decode_step(engine, batch, sync):
    """prepare_for_decode (engine allocation), DP metadata, one full DSPARK iteration, commit."""
    import torch
    args, topology = engine.args, engine.topology
    batch.prepare_for_decode()
    prepare_dp_metadata(batch, topology, is_extend=False, disable_cuda_graph=args.disable_cuda_graph, sync=sync)
    previous = [int(req.kv.kv_committed_len) for req in batch.reqs]
    result = forward(engine, batch)
    torch.cuda.synchronize()
    accept = [int(value) for value in result.accept_lens.cpu().tolist()]
    new_lens = [int(value) for value in result.new_seq_lens.cpu().tolist()]
    tokens = result.next_token_ids.cpu().tolist()
    stride = int(result.speculative_num_draft_tokens or VERIFY_WINDOW)
    for index, (req, count) in enumerate(zip(batch.reqs, accept)):
        req.kv.commit_decode(count)  # batch_result_processor.py:989
        req.output_ids.extend(int(t) for t in tokens[index * stride: index * stride + count])
    return result, previous, accept, new_lens


def check_rank_progress(iteration, previous_lens, accept_lens, new_lens, final_len, buffers):
    import torch
    import torch.distributed as dist
    signature = rank_progress_signature(iteration, previous_lens, accept_lens, new_lens, final_len,
                                        max_accept=VERIFY_WINDOW)
    local, gathered = buffers
    local.copy_(torch.tensor(signature, dtype=torch.int64, device=local.device))
    dist.all_gather(gathered, local)
    validate_rank_progress([value.cpu().tolist() for value in gathered])


def rank_main(rank, gpu_id, server_args, port_args, args):
    try:
        run_rank(rank, gpu_id, server_args, port_args, args)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise


def run_rank(rank, gpu_id, server_args, port_args, args):
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, spawn_world_rank
    from sglang.srt.utils import configure_logger
    rank_start = time.perf_counter()
    publish(server_args, role="scheduler",
            ranks=SpawnRanks(world_rank=spawn_world_rank(server_args, tp_rank=rank, pp_rank=0), gpu_id=gpu_id))
    configure_logger(server_args, prefix=f" TP{rank}")
    topology = DecodeTopology(args.tp_size, args.ep_size, args.batch_size, args.enable_dp_attention)
    phase_seconds = {}
    session = ProfileSession.create(args, rank, log)  # None unless --profile
    configure_acceptance(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    phase_start = time.perf_counter()
    engine = Engine(server_args, port_args, gpu_id, rank, args, topology)
    phase_seconds["load_pool_capture"] = time.perf_counter() - phase_start
    target_graph = engine.graph_runner(engine.target_runner)
    draft_graph = engine.graph_runner(engine.draft_runner)
    captured = {label: sorted(int(x) for x in (getattr(graph, "capture_bs", None) or []))
                for label, graph in (("target", target_graph), ("draft", draft_graph)) if graph is not None}
    log(rank, f"topology={topology.parallel_state_kwargs(rank)} local_batch={topology.local_batch_size} "
              f"global_batch={args.batch_size} moe_a2a={args.resolved['moe_a2a_backend']} "
              f"moe_runner={args.resolved['moe_runner_backend']} captured_bs={captured} "
              f"tree_cache={engine.tree_cache_kind}")
    progress_buffers = (torch.empty(7, dtype=torch.int64, device="cuda"),
                        [torch.empty(7, dtype=torch.int64, device="cuda") for _ in range(args.tp_size)])
    final_lens = [length + args.output_len for length in args.input_lens]
    import sglang.srt.server_args as server_args_module
    import sglang.srt.speculative.dspark_components.dspark_worker_v2 as dspark_module
    import sglang.srt.managers.tp_worker as tp_worker_module
    source_paths = module_source_paths([server_args_module, dspark_module, tp_worker_module])
    log(rank, f"runtime source paths={source_paths}")
    result_dir = Path(args.result_dir)
    sync = engine.sync_fn()
    with torch.inference_mode():
        phase_start = time.perf_counter()
        runner = engine.target_runner
        reserve = 2 * VERIFY_WINDOW  # DFlashDraftInputV2.prepare_for_decode: reserve = 2 * block
        max_input_len = max(args.input_lens)
        memory = {"page_size": engine.page_size, "speculative_reserve": reserve,
                  "max_input_len": max_input_len, "min_input_len": min(args.input_lens),
                  "reserved_tokens": required_token_capacity(topology.local_batch_size, max_input_len,
                                                             args.output_len, engine.page_size, reserve),
                  "available_tokens_before_prefill": int(engine.allocator.available_size()),
                  "max_total_num_tokens": int(getattr(runner, "max_total_num_tokens", 0) or 0),
                  "req_to_token_row": int(engine.req_to_token_pool.req_to_token.shape[1]),
                  "tree_cache": engine.tree_cache_kind, "kv_pools": engine.kv_inventory()}
        if memory["available_tokens_before_prefill"] < memory["reserved_tokens"]:
            raise RuntimeError(f"KV capacity insufficient: {memory}")
        per_request = max_input_len + args.output_len + reserve
        if memory["req_to_token_row"] < per_request:
            raise RuntimeError(f"Request row too short: need {per_request}, have {memory['req_to_token_row']}")
        if runner.model_config.context_len < per_request:
            raise RuntimeError("Model context length does not cover requested progression plus speculative reserve")
        phase_seconds["physical_initialization"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        log(rank, f"bootstrap prefill of {topology.local_batch_size} synthetic prompts; memory={memory}")
        batch, groups = bootstrap(engine, rank)
        memory["prefill_groups"] = groups
        phase_seconds["bootstrap"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        log(rank, f"warming up {args.warmup_steps} complete speculative iterations")
        for iteration in range(args.warmup_steps):
            result, previous, accept, new_lens = decode_step(engine, batch, sync)
            check_rank_progress(iteration, previous, accept, new_lens, [10 ** 12] * len(new_lens), progress_buffers)
        if args.warmup_steps:
            release(engine, batch)
            batch, _ = bootstrap(engine, rank)
        phase_seconds["warmup_and_reset"] = time.perf_counter() - phase_start
        reset_acceptance_rng(engine, args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        dist.barrier()
        accounting = DecodeAccounting(topology.local_batch_size, args.input_lens, args.output_len,
                                      max_accept_len=VERIFY_WINDOW)
        steps = []
        graph_steps = 0
        graph_execution_counts = count_graph_executions({
            "target": target_graph, "draft": draft_graph,
            "draft_extend": None,  # DSPARK has no draft-extend forward; draft KV is injected from target hidden
        })
        if session is not None:
            session.attach_device_timer({"target": engine.target_runner, "draft": engine.draft_runner})
        log(rank, "measured decode loop begins at exact requested ISL")
        start = time.perf_counter()
        while not accounting.complete and (not args.max_steps or accounting.verify_ct < args.max_steps):
            if session is not None:
                session.step_boundary(accounting.verify_ct)  # outside the per-step tick
            tick = time.perf_counter()
            result, previous, accept_lens, new_lens = decode_step(engine, batch, sync)
            check_rank_progress(accounting.verify_ct, accounting.seq_lens, accept_lens, new_lens, final_lens,
                                progress_buffers)
            validate_worker_progress(accounting.seq_lens, accept_lens, new_lens)
            step = accounting.record(accept_lens, time.perf_counter() - tick)
            step["seconds"] = time.perf_counter() - tick
            step["target_graph"] = bool(result.can_run_cuda_graph)
            graph_steps += int(step["target_graph"])
            steps.append(step)
            if rank == 0 and (accounting.verify_ct <= 3 or accounting.verify_ct % max(1, args.log_interval) == 0):
                contexts = accounting.seq_lens
                log(rank, f"iteration={accounting.verify_ct} context_min={min(contexts)} context_max={max(contexts)} "
                          f"useful={sum(accounting.emitted)} target_graph={step['target_graph']} "
                          f"graphs={dict(graph_execution_counts)}")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        phase_seconds["decode"] = elapsed
        phase_seconds["rank_total_to_result"] = time.perf_counter() - rank_start
        summary = accounting.summary(elapsed)
        summary.update(topology.summary())
        worker = engine.worker
        summary.update({
            "expected_accept_length": args.accept_length, "accept_method": args.accept_method,
            "accept_token_mode": args.accept_token_mode, "accept_greedy": bool(args.accept_greedy),
            "tp_size": args.tp_size,
            "output_tokens_per_second_per_gpu": summary["output_tokens_per_second"] / args.tp_size,
            "target_graph_iterations": graph_steps, "warmup_steps_excluded": args.warmup_steps,
            "graph_execution_counts": graph_execution_counts,
            "bootstrap_tokens_excluded": topology.local_batch_size,
            "rank": rank, "parallel_state": topology.parallel_state_kwargs(rank),
            "report_scope": "local_attention_dp_shard", "graph_execution_counts_scope": "local_rank",
            "phase_seconds": phase_seconds, "moe_a2a_backend": args.resolved["moe_a2a_backend"],
            "moe_runner_backend": args.resolved["moe_runner_backend"],
            "cross_rank_progress_check": "every warmup and measured iteration; included in timing",
            "dp_metadata": args.dp_metadata,
            "synthetic_prefix": True, "prefix_kv_source": "real prefill of synthetic random token ids",
            "simulated_acceptance": True, "real_model_weights": True,
            "real_moe_routing": True, "scheduler_used": False, "pinned_sglang": PINNED_SGLANG,
            "speculative_algorithm": "DSPARK", "dspark_gamma": int(worker.gamma),
            "verify_num_draft_tokens": int(worker.verify_num_draft_tokens),
            "dspark_verify_mode": worker._verify_planner.mode_value,
            "module_source_paths": source_paths,
            "timing_boundary": "complete internal speculative loop (prepare_for_decode + DP sync + draft/verify/accept/commit) + synchronization + bookkeeping; excludes bootstrap/warmup/load/capture",
            "terminal_policy": "count only remaining useful tokens; pay full terminal computation; no subsequent forward",
            "memory": memory, "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "model_path": args.model_path,
            "target_backend": type(engine.target_runner.attn_backend).__name__,
            "draft_backend": type(engine.draft_runner.attn_backend).__name__,
            "captured_decode_bs": captured,
            "draft_graph_available": draft_graph is not None,
            "draft_extend_graph_available": None,
            "draft_extend_note": "DSPARK has no draft-extend forward (draft KV is injected from target hidden states in commit_hidden)",
        })
        if session is not None:
            summary["profiled"] = True
            summary["is_performance_measurement"] = False
            session.finish(get_parallel(), {
                "measured_iterations": accounting.verify_ct,
                "measured_elapsed_seconds": elapsed,
                "pinned_sglang": PINNED_SGLANG,
                "module_source_paths": source_paths,
                "topology": topology.summary(),
                "result_dir": args.result_dir,
            })
        (result_dir / f"rank_{rank}_yihou.json").write_text(json.dumps(summary, indent=2) + "\n")
        (result_dir / f"steps_rank_{rank}_yihou.jsonl").write_text("".join(json.dumps(step) + "\n" for step in steps))
        reports = [None] * args.tp_size
        dist.all_gather_object(reports, summary)
        aggregate = aggregate_rank_summaries(reports, topology)
        if rank == 0:
            (result_dir / "result_yihou.json").write_text(json.dumps(aggregate, indent=2) + "\n")
            (result_dir / "steps_yihou.jsonl").write_text("".join(json.dumps(step) + "\n" for step in steps))
            log(rank, json.dumps({key: aggregate[key] for key in (
                "complete", "batch_size", "input_len", "output_len", "verify_iterations",
                "realized_accept_length", "effective_token_latency_ms_per_user", "output_tokens_per_second",
                "graph_execution_counts")}, sort_keys=True))
    dist.barrier()
    from sglang.srt.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    destroy_model_parallel()
    destroy_distributed_environment()


def build_server_args(args, extra):
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.server_args import ServerArgs
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    cli = server_cli(args, extra)
    server_args = ServerArgs.from_cli_args(parser.parse_args(cli))
    server_args.resolve_once()
    return server_args, resolving_view(server_args), cli


def check_server_args(cfg, topology):
    # sglang 6be0d75ae resolves --enable-dp-attention --dp-size N into attn_dp_size=N (and dp_size=1
    # data-parallel replicas); the raw ServerArgs keeps the CLI view. Check the resolved one.
    resolved = (cfg.tp_size, cfg.ep_size, cfg.attn_dp_size, cfg.dp_size)
    if resolved != (topology.tp_size, topology.ep_size, topology.dp_size, 1):
        raise ValueError(f"Resolved server topology {resolved} differs from benchmark topology {topology}")
    if getattr(cfg, "pp_size", 1) != 1 or getattr(cfg, "nnodes", 1) != 1 or getattr(cfg, "attn_cp_size", 1) != 1 \
            or getattr(cfg, "dcp_size", 1) not in (1, None) or getattr(cfg, "moe_dp_size", 1) != 1:
        raise ValueError("This wrapper requires one node, PP1/CP1/DCP1/MoEDP1")
    if cfg.moe_a2a_backend != "none":
        raise ValueError("Pinned EP benchmark requires moe-a2a-backend=none")
    if (cfg.speculative_algorithm, cfg.speculative_num_draft_tokens) != ("DSPARK", VERIFY_WINDOW):
        raise ValueError(f"This benchmark requires DSPARK block 5 (verify window {VERIFY_WINDOW}); "
                         f"got {cfg.speculative_algorithm} / {cfg.speculative_num_draft_tokens}")
    if cfg.load_format == "dummy":
        raise ValueError("Dummy weights are not permitted for this performance benchmark")


def main():
    args, extra = make_parser().parse_known_args()
    validate_args(args)
    isl_spec = resolve_input_lens(args)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Run with plain python, not torchrun; this driver spawns TP ranks")
    configure_acceptance(args)  # before any sglang import: spec_utils constants are import-time
    redact_argv = import_redact_argv()
    import multiprocessing
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs
    from sglang.srt.utils import kill_process_tree, maybe_reindex_device_id
    configure_acceptance(args)
    topology = DecodeTopology(args.tp_size, args.ep_size, args.batch_size, args.enable_dp_attention)
    server_args, cfg, cli = build_server_args(args, extra)
    check_server_args(cfg, topology)
    if args.prefill_tokens is None:
        # A single 131072-token extend overflowed an int32 kernel argument in the FlyDSL MXFP8 GEMM
        # (tp4_ep4_dpa_on_c64_yihou, first attempt); the server never runs one larger than this.
        args.prefill_tokens = int(cfg.chunked_prefill_size)
    args.resolved = {name: str(getattr(cfg, name, None)) for name in (
        "moe_a2a_backend", "moe_runner_backend", "attention_backend", "kv_cache_dtype",
        "mem_fraction_static", "max_running_requests", "page_size", "speculative_num_draft_tokens",
        "attn_dp_size", "cuda_graph_bs_decode")}
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir = str(result_dir)
    if args.profile:
        # Must precede spawning: SGLANG_TORCH_PROFILER_DIR has to be in the children's
        # environment before CUDA-graph capture, which runs inside each rank.
        profile_env = configure_profile_env(args)
        print(f"[INTERNAL-DECODE] profiling enabled; env={profile_env}", flush=True)
    heterogeneous = args.input_len_spec is not None
    recorded = {key: value for key, value in vars(args).items()
                if (args.profile or not (key == "profile" or key.startswith("profile_") or key == "device_timer"))
                and (heterogeneous or key not in ("input_len_spec", "input_lens"))}
    payload = {"benchmark": recorded, "server_cli": redact_argv(cli), "expected_sglang_commit": PINNED_SGLANG,
               "simulated_acceptance_env": configure_acceptance(args)}
    if heterogeneous:
        payload["input_len_spec_resolved"] = isl_spec.describe()
        payload["input_lens_realized"] = list(args.input_lens)
    (result_dir / "config_yihou.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    if args.tp_size == 1:
        rank_main(0, 0, server_args, port_args, args)
        return 0
    procs = []
    try:
        for rank in range(args.tp_size):
            with maybe_reindex_device_id(rank) as gpu_id:
                proc = multiprocessing.Process(target=rank_main, args=(rank, gpu_id, server_args, port_args, args))
                proc.start()
                procs.append(proc)
        # A rank that dies leaves the others blocked in a collective: stop waiting once any failed.
        while any(proc.is_alive() for proc in procs):
            if any(proc.exitcode not in (None, 0) for proc in procs):
                print("[INTERNAL-DECODE] a rank failed; stopping the remaining ranks", flush=True)
                break
            time.sleep(2)
        for proc in procs:
            proc.join(timeout=1)
        return 1 if any(proc.exitcode != 0 for proc in procs) else 0
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    sys.exit(main())
