#!/usr/bin/env python3
"""Pinned vLLM DeepSeek-V4.1 internal DSpark whole-step decode benchmark; no scheduler, no server.

vLLM port of the GLM-5.2 "tool 3" (SGLang TpModelWorker + EAGLEWorkerV2 driver). Each spawned rank
builds the engine's own LLMEngine -> EngineCore -> Worker -> GPUModelRunner (V2) exactly as `vllm serve`
does (same EngineArgs, OPENAI_API_SERVER usage defaults, KV cache sizing, CUDA-graph capture), then the
bench drives `Worker.execute_model` / `Worker.sample_tokens` itself with hand-built SchedulerOutputs: one
fixed batch, synthetic but physical KV, the engine's own synthetic rejection sampler (simulated
acceptance), the DSpark drafter, FULL CUDA graphs. The vLLM Scheduler is constructed by the engine but
never called.

Use plain python: this entry point spawns its own ranks (one per DP x TP rank, vLLM external launcher).
--help requires only the standard library. Additional arguments pass through to vLLM EngineArgs.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

from batch_state import (
    KV_INIT_MODES,
    DecodeAccounting,
    accept_bounds,
    blocks_for_request,
    count_graph_executions,
    kv_fill_rule,
    length_to_rates,
    module_source_paths,
    required_tokens_per_request,
    unconditional_to_conditional,
    validate_worker_progress,
)
from topology import (
    DecodeTopology,
    aggregate_rank_summaries,
    compare_resolved_parallel,
    rank_progress_signature,
    validate_rank_progress,
)
from isl_spec import parse_isl_spec
from profiling_yihou import (
    ProfileSession,
    add_profile_cli_args,
    configure_profile_env,
    validate_profile_args,
)

PINNED_VLLM = "0.34.0rc1.dev897+g9c1d45c21"
PINNED_IMAGE = "infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2"
DEFAULT_INPUT_LEN = 173000
# Engine flags the bench owns (topology, launcher, speculative config, seed).
OWNED_FLAGS = {
    "--tensor-parallel-size", "-tp", "--data-parallel-size", "-dp", "--enable-expert-parallel", "-ep",
    "--no-enable-expert-parallel", "--distributed-executor-backend", "--data-parallel-rank", "-dpn",
    "--data-parallel-size-local", "-dpl", "--speculative-config", "-sc", "--seed", "--model",
    "--pipeline-parallel-size", "-pp", "--decode-context-parallel-size", "-dcp",
    "--prefill-context-parallel-size", "-pcp",
}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=16, help="GLOBAL concurrency C (all DP ranks)")
    parser.add_argument("--input-len", type=int, default=None, help="decimal tokens per request, uniform across the batch")
    parser.add_argument("--input-len-spec", default=None,
                        help="heterogeneous ISL, per DP rank: uniform:N | bimodal:A,B,ratio "
                             "| normal:mean,std[,lo,hi] | list:a,b,... | list:@file")
    parser.add_argument("--output-len", type=int, default=1024, help="useful emitted tokens per request")
    parser.add_argument("--accept-length", type=float, default=3.51,
                        help="expected bonus-inclusive accepted tokens per step (vLLM synthetic_acceptance_length)")
    parser.add_argument("--accept-method", choices=("length", "rates"), default="length",
                        help="length: vLLM's minimum-variance schedule from --accept-length; "
                             "rates: --accept-rates (per-position unconditional rates)")
    parser.add_argument("--accept-rates", default=None, help="comma list of K unconditional acceptance rates")
    parser.add_argument("--accept-token-mode", choices=("real-draft-token",), default="real-draft-token",
                        help="vLLM's synthetic sampler always emits the real draft token on accept")
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--spec-method", default="dspark")
    parser.add_argument("--draft-sample-method", choices=("probabilistic", "greedy"), default="probabilistic")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=1, help="1 (MoE tensor-parallel over DPxTP) or DP*TP (expert parallel)")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--disable-cuda-graph", action="store_true", help="eager diagnosis (vLLM --enforce-eager)")
    parser.add_argument("--graph-sizes", choices=("engine", "local"), default="engine",
                        help="engine: the engine's capture list (as served); local: capture exactly local_batch*(1+K)")
    parser.add_argument("--kv-init", choices=KV_INIT_MODES, default="random",
                        help="random: GLM rule (finite random bytes / uniform[-1,1]); decode-bench: vLLM "
                             "DecodeBenchConnector fill (constant 0.015, packed layouts zero); zeros: vllm bench forward")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means full OSL; positive values are incomplete smoke runs")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--master-port", type=int, default=0, help="torch.distributed port (0 = pick a free one)")
    add_profile_cli_args(parser)
    return parser


def topology_of(args):
    return DecodeTopology(args.tp_size, args.dp_size, args.ep_size, args.batch_size)


def resolve_rates(args):
    if args.accept_method == "rates":
        if not args.accept_rates:
            raise ValueError("--accept-method rates needs --accept-rates")
        rates = [float(x) for x in args.accept_rates.split(",")]
        if len(rates) != args.num_speculative_tokens:
            raise ValueError(f"--accept-rates needs {args.num_speculative_tokens} values")
        return rates
    if args.accept_rates:
        raise ValueError("--accept-rates is only valid with --accept-method rates")
    return length_to_rates(args.accept_length, args.num_speculative_tokens)


def validate_args(args):
    topology_of(args)
    if min(args.batch_size, args.output_len, args.tp_size, args.dp_size) <= 0:
        raise ValueError("batch size, output length, TP and DP sizes must be positive")
    if args.max_steps < 0 or args.warmup_steps < 0:
        raise ValueError("max-steps and warmup-steps must be nonnegative")
    if not 1 <= args.num_speculative_tokens <= 16:
        raise ValueError("num-speculative-tokens must be in [1, 16]")
    if not 1 <= args.accept_length <= args.num_speculative_tokens + 1:
        raise ValueError(f"accept-length must be in [1, {args.num_speculative_tokens + 1}] for K={args.num_speculative_tokens}")
    if args.input_len is not None and args.input_len <= 0:
        raise ValueError("input-len must be positive")
    if args.input_len is not None and args.input_len_spec is not None:
        raise ValueError("Pass either --input-len or --input-len-spec, not both")
    resolve_rates(args)
    validate_profile_args(args)


def resolve_input_lens(args):
    """Resolve the ISL spec ONCE in the parent, then ship the vector to every rank (Plan 1)."""
    topology = topology_of(args)
    text = args.input_len_spec if args.input_len_spec is not None else str(
        DEFAULT_INPUT_LEN if args.input_len is None else args.input_len)
    spec = parse_isl_spec(text)
    lens = spec.generate(topology.local_batch_size, args.seed)
    if min(lens) < 2:
        raise ValueError("every input length must be at least 2 for a populated-prefix bootstrap")
    args.input_lens = lens
    return spec


def speculative_config(args):
    config = {"method": args.spec_method, "num_speculative_tokens": args.num_speculative_tokens,
              "draft_sample_method": args.draft_sample_method, "rejection_sample_method": "synthetic",
              "enable_adaptive_verification": False}
    if args.accept_method == "rates":
        config["synthetic_acceptance_rates"] = resolve_rates(args)
    else:
        config["synthetic_acceptance_length"] = args.accept_length
    return config


def engine_cli(args, extra):
    topology = topology_of(args)
    flags = {item.split("=", 1)[0] for item in extra if item.startswith("-")}
    if flags & OWNED_FLAGS:
        raise ValueError(f"Benchmark controls these engine flags: {sorted(flags & OWNED_FLAGS)}")
    result = ["--model", args.model_path,
              "--tensor-parallel-size", str(topology.tp_size),
              "--data-parallel-size", str(topology.dp_size),
              "--distributed-executor-backend", "external_launcher",
              "--speculative-config", json.dumps(speculative_config(args), separators=(",", ":")),
              "--seed", str(args.seed)] + list(extra)
    if topology.enable_expert_parallel:
        result.append("--enable-expert-parallel")
    if "--enable-prefix-caching" not in flags and "--no-enable-prefix-caching" not in flags:
        result.append("--no-enable-prefix-caching")
    if "--trust-remote-code" not in flags:
        result.append("--trust-remote-code")
    if args.disable_cuda_graph and "--enforce-eager" not in flags:
        result.append("--enforce-eager")
    if args.graph_sizes == "local" and "--cudagraph-capture-sizes" not in flags:
        result += ["--cudagraph-capture-sizes", str(topology.local_batch_size * (1 + args.num_speculative_tokens))]
    return result


def log(rank, message):
    print(f"[INTERNAL-DECODE R{rank}] {message}", flush=True)


# ---------------------------------------------------------------- engine construction (per rank)

def create_engine(args, rank, cli):
    """Build the engine as `vllm serve` would, on this rank (external launcher, in-process EngineCore)."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.engine.llm_engine import LLMEngine
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    engine_args = EngineArgs.from_cli_args(parser.parse_args(cli))
    if engine_args.load_format == "dummy":
        raise ValueError("Dummy weights are not permitted for this performance benchmark")
    log(rank, "building LLMEngine (load, KV sizing, CUDA-graph capture; may take many minutes)")
    engine = LLMEngine.from_engine_args(engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
    worker = engine.model_executor.driver_worker.worker
    return engine, worker, worker.model_runner


def resolved_engine_summary(runner, worker):
    from vllm.distributed.parallel_state import get_tp_group
    config = runner.vllm_config
    parallel, cache, comp = config.parallel_config, config.cache_config, config.compilation_config
    spec = config.speculative_config
    return {
        "parallel": dict(dp_rank=parallel.data_parallel_rank, dp_size=parallel.data_parallel_size,
                         tp_rank=get_tp_group().rank_in_group, tp_size=parallel.tensor_parallel_size,
                         pp_size=parallel.pipeline_parallel_size,
                         enable_expert_parallel=bool(parallel.enable_expert_parallel),
                         dcp_size=parallel.decode_context_parallel_size,
                         backend=parallel.distributed_executor_backend),
        "cache": dict(block_size=cache.block_size, cache_dtype=str(cache.cache_dtype),
                      gpu_memory_utilization=cache.gpu_memory_utilization,
                      num_blocks=runner.kv_cache_config.num_blocks,
                      enable_prefix_caching=cache.enable_prefix_caching),
        "scheduler": dict(max_num_seqs=config.scheduler_config.max_num_seqs,
                          max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
                          async_scheduling=getattr(config.scheduler_config, "async_scheduling", None)),
        "model": dict(max_model_len=config.model_config.max_model_len, dtype=str(config.model_config.dtype),
                      architecture=getattr(config.model_config, "architecture", None)),
        "compilation": dict(cudagraph_mode=str(comp.cudagraph_mode),
                            cudagraph_capture_sizes=list(comp.cudagraph_capture_sizes or []),
                            max_cudagraph_capture_size=comp.max_cudagraph_capture_size),
        "speculative": None if spec is None else dict(
            method=spec.method, num_speculative_tokens=spec.num_speculative_tokens,
            draft_sample_method=spec.draft_sample_method,
            rejection_sample_method=spec.rejection_sample_method,
            synthetic_acceptance_rates=list(spec.synthetic_acceptance_rates or []),
            enable_adaptive_verification=spec.enable_adaptive_verification),
        "runner": dict(cls=type(runner).__module__ + "." + type(runner).__name__,
                       speculator=type(runner.speculator).__name__ if runner.speculator is not None else None,
                       max_num_reqs=runner.max_num_reqs, num_speculative_steps=runner.num_speculative_steps,
                       decode_query_len=runner.decode_query_len,
                       cudagraph_manager=type(runner.cudagraph_manager).__name__ if runner.cudagraph_manager else None),
        "kv_cache_groups": kv_group_inventory(runner),
    }


def kv_group_inventory(runner):
    from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size
    dcp = runner.parallel_config.decode_context_parallel_size
    out = []
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        kind, window, num_spec_blocks = group_kind(group)
        layers = list(group.layer_names)
        tensors = [runner.kv_caches_by_layer.get(name) for name in layers]
        out.append(dict(group=gid, kind=kind, window=window, num_spec_blocks=num_spec_blocks,
                        manager=group.manager_cls.__name__, specs=sorted({type(s).__name__ for s in group.distinct_specs}),
                        block_size_tokens=resolve_dcp_kv_block_size(group, dcp), num_layers=len(layers),
                        dtypes=sorted({str(t.dtype) for t in tensors if hasattr(t, "dtype")}),
                        bytes=sum(t.numel() * t.element_size() for t in tensors if hasattr(t, "numel"))))
    return out


def group_kind(group):
    from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager, MambaManager, SlidingWindowManager
    params = group.manager_config.params
    if isinstance(params, MambaManager.Params):
        return "mamba", None, params.num_speculative_blocks
    if issubclass(group.manager_cls, CircularBufferManager):
        return "circular", None, 0
    if isinstance(params, SlidingWindowManager.Params):
        return "sliding", params.sliding_window, 0
    return "full", None, 0


# ---------------------------------------------------------------- batch allocation and KV init

class BatchPlan:
    """Per-request block lists for every KV cache group, reserved ONCE up front."""

    def __init__(self, runner, args, topology, rank, min_accept, max_accept):
        from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size
        self.local = topology.local_batch_size
        self.offset = topology.request_id_offset(rank)
        dcp = runner.parallel_config.decode_context_parallel_size
        groups = runner.kv_cache_config.kv_cache_groups
        num_blocks = runner.kv_cache_config.num_blocks
        K = args.num_speculative_tokens
        self.reserved_tokens = [required_tokens_per_request(isl, args.output_len, min_accept, max_accept, K)
                                for isl in args.input_lens]
        max_len = runner.model_config.max_model_len
        if max(self.reserved_tokens) > max_len:
            raise RuntimeError(f"max-model-len {max_len} < reserved request length {max(self.reserved_tokens)} "
                               "(ISL + OSL growth bound + verify/draft windows)")
        if self.local > runner.max_num_reqs:
            raise RuntimeError(f"max_num_seqs {runner.max_num_reqs} < local batch {self.local} (pass --max-num-seqs)")
        tokens = self.local * (1 + K)
        if tokens > runner.scheduler_config.max_num_batched_tokens:
            raise RuntimeError(f"max-num-batched-tokens {runner.scheduler_config.max_num_batched_tokens} < "
                               f"one verify step {self.local}x{1 + K}={tokens}")
        cursor = 1  # block 0 is the null block
        self.block_ids = []      # per request: tuple(list per group)
        self.group_ranges = []   # per request: list of (begin, end) per group
        self.per_group_blocks = [0] * len(groups)
        for isl, total in zip(args.input_lens, self.reserved_tokens):
            ids, ranges = [], []
            for gid, group in enumerate(groups):
                kind, window, num_spec_blocks = group_kind(group)
                block_size = resolve_dcp_kv_block_size(group, dcp)
                skipped, live = blocks_for_request(kind, block_size, isl - 1, total, window=window,
                                                   num_spec_blocks=num_spec_blocks)
                ids.append([0] * skipped + list(range(cursor, cursor + live)))
                ranges.append((cursor, cursor + live))
                self.per_group_blocks[gid] += live
                cursor += live
            self.block_ids.append(tuple(ids))
            self.group_ranges.append(ranges)
        self.blocks_used = cursor
        self.num_blocks = num_blocks
        if cursor > num_blocks:
            raise RuntimeError(f"KV capacity insufficient: need {cursor} blocks, the engine allocated {num_blocks} "
                               f"(per group {self.per_group_blocks})")

    def summary(self):
        return {"blocks_reserved": self.blocks_used, "kv_num_blocks": self.num_blocks,
                "blocks_per_group": self.per_group_blocks,
                "reserved_tokens_per_request_max": max(self.reserved_tokens),
                "reserved_tokens_per_request_min": min(self.reserved_tokens),
                "kv_block_utilization": self.blocks_used / self.num_blocks}


def initialize_kv(runner, plan, mode, seed):
    """Physically fill every KV tensor row this batch owns, outside measurement. Returns bytes written."""
    import torch
    groups = runner.kv_cache_config.kv_cache_groups
    generator = torch.Generator(device=runner.device).manual_seed(seed)
    written = 0
    inventory = {}
    for gid, group in enumerate(groups):
        kind, _, _ = group_kind(group)
        ids = []
        for ranges in plan.group_ranges:
            begin, end = ranges[gid]
            ids.extend(range(begin, end))
        if not ids:
            continue
        index = torch.tensor(ids, dtype=torch.long, device=runner.device)
        for name in group.layer_names:
            tensor = runner.kv_caches_by_layer.get(name)
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"KV cache of {name} is {type(tensor).__name__}, not a block-indexed tensor")
            if tensor.shape[0] != plan.num_blocks:
                raise RuntimeError(f"KV cache of {name} has first dim {tensor.shape[0]} != num_blocks {plan.num_blocks}")
            rule = "zeros" if mode == "zeros" else kv_fill_rule(str(tensor.dtype), tensor.element_size(),
                                                                tensor.dtype.is_floating_point, kind == "circular")
            if mode == "decode-bench" and rule != "zeros":
                rule = "decode_bench_constant"
            inventory.setdefault(f"group{gid}:{kind}:{tensor.dtype}", {"layers": 0, "rule": rule})["layers"] += 1
            rows_per_chunk = max(1, (1 << 26) // max(1, tensor[0].numel() * tensor.element_size()))
            for chunk in index.split(rows_per_chunk):
                shape = (len(chunk),) + tuple(tensor.shape[1:])
                if rule == "zeros":
                    values = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
                elif rule == "bytes_0_0x7c":
                    values = torch.randint(0, 0x7C, shape, dtype=torch.uint8, device=tensor.device,
                                           generator=generator).view(tensor.dtype)
                elif rule == "uniform_-1_1":
                    values = torch.empty(shape, dtype=tensor.dtype, device=tensor.device).uniform_(-1, 1, generator=generator)
                elif rule == "decode_bench_constant":
                    values = decode_bench_values(runner, name, tensor, shape)
                else:
                    raise RuntimeError(f"unknown fill rule {rule}")
                tensor[chunk] = values
                written += values.numel() * values.element_size()
    return written, inventory


def decode_bench_values(runner, name, tensor, shape):
    """vLLM DecodeBenchConnector default fill: constant 0.015 in the stored float type; packed layouts 0."""
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import _get_fp8_dtype
    spec = None
    for group in runner.kv_cache_config.kv_cache_groups:
        if name in group.kv_cache_specs:
            spec = group.kv_cache_specs[name]
    dtype = _get_fp8_dtype(spec, runner.vllm_config.cache_config.cache_dtype) if spec is not None else None
    dtype = dtype or tensor.dtype
    if not dtype.is_floating_point:
        return torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    return torch.full(shape, 0.015, dtype=dtype, device=tensor.device).view(tensor.dtype)


# ---------------------------------------------------------------- scheduler outputs (what the Scheduler would send)

def default_sampling_params(runner, args, reserved):
    from vllm.sampling_params import SamplingParams
    defaults = runner.model_config.get_diff_sampling_param() or {}
    params = SamplingParams.from_optional(**defaults) if defaults else SamplingParams()
    params.max_tokens = reserved  # never let the runner trim drafts near max_tokens; the bench ends the run
    params.ignore_eos = True
    return params


def bootstrap_output(plan, args, epoch, prompts, sampling):
    """One-token admission (GLM-5.2 bootstrap): a new request with ISL-1 tokens of KV in place, scheduled
    exactly its last prompt token, no drafts. The runner runs it as a decode row ("one new prompt token over
    existing context"), samples the bonus token and the DSpark drafter proposes K drafts, so the measured loop
    starts at exactly ISL with a valid draft state. (The serving scheduler instead pads an admitted request to
    1 + K tokens; the V2 runner zero-initializes a new request's draft slots, so that padded step accepts
    synthetic drafts of token 0 and the context would not start at exactly ISL. Admission is untimed either way.)"""
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    K = args.num_speculative_tokens
    output = SchedulerOutput.make_empty()
    req_ids = []
    for index, (isl, blocks) in enumerate(zip(args.input_lens, plan.block_ids)):
        req_id = f"bench-{epoch}-{plan.offset + index}"
        prompt = prompts[index]
        output.scheduled_new_reqs.append(NewRequestData(
            req_id=req_id, prompt_token_ids=prompt, mm_features=[], sampling_params=sampling[index],
            pooling_params=None, block_ids=blocks, num_computed_tokens=isl - 1, lora_request=None,
            prefill_token_ids=prompt))
        output.num_scheduled_tokens[req_id] = 1
        req_ids.append(req_id)
    output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
    output.num_spec_tokens_to_schedule = K
    return output, req_ids


def decode_output(req_ids, computed, emitted, args, finished=()):
    """A verify step of the running batch: 1 + K tokens per request, K drafts (values live on device)."""
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
    K = args.num_speculative_tokens
    output = SchedulerOutput.make_empty()
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=list(req_ids), new_token_ids=[[] for _ in req_ids], new_block_ids=[None for _ in req_ids],
        num_computed_tokens=list(computed), num_output_tokens=list(emitted))
    for req_id in req_ids:
        output.num_scheduled_tokens[req_id] = 1 + K
        output.scheduled_spec_decode_tokens[req_id] = [-1] * K
    output.total_num_scheduled_tokens = len(req_ids) * (1 + K)
    output.num_spec_tokens_to_schedule = K
    output.finished_req_ids = set(finished)
    return output


def run_step(worker, scheduler_output, req_ids, tokens=None):
    """execute_model -> sample_tokens -> output copy -> drafts, as EngineCore.step does. Returns
    (accept lens in req_ids order, host timestamps); `tokens` (a list) receives the sampled ids if given."""
    import torch
    t0 = time.perf_counter()
    worker.execute_model(scheduler_output)
    t1 = time.perf_counter()
    async_output = worker.sample_tokens(None)
    t2 = time.perf_counter()
    output = async_output.get_output() if hasattr(async_output, "get_output") else async_output
    worker.take_draft_token_ids()
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    position = {req_id: index for index, req_id in enumerate(output.req_ids)}
    accept = [len(output.sampled_token_ids[position[req_id]]) for req_id in req_ids]
    if tokens is not None:
        tokens.extend(list(output.sampled_token_ids[position[req_id]]) for req_id in req_ids)
    return accept, {"execute_s": t1 - t0, "sample_s": t2 - t1, "output_sync_s": t3 - t2}


def worker_computed(runner, req_ids):
    states = runner.req_states
    index = [states.req_id_to_index[req_id] for req_id in req_ids]
    return states.num_computed_tokens.gpu[index].cpu().tolist()


def check_rank_progress(iteration, previous, accept, new, final_lens, max_accept, group):
    import torch
    import torch.distributed as dist
    signature = rank_progress_signature(iteration, previous, accept, new, final_lens, max_accept)
    local = torch.tensor(signature, dtype=torch.int64)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    validate_rank_progress([value.tolist() for value in gathered])


def bootstrap(worker, runner, plan, args, epoch, prompts, sampling, finished=()):
    """One untimed admission step: a real forward of the last prompt token over the synthetic prefix, target
    sampling, and a DSpark proposal, so the drafter holds a valid state."""
    import numpy as np
    np.random.seed(args.seed)  # per-request sampler seeds are np.random draws: identical on every rank/epoch
    output, req_ids = bootstrap_output(plan, args, epoch, prompts, sampling)
    output.finished_req_ids = set(finished)
    accept, _ = run_step(worker, output, req_ids)
    if accept != [1] * len(req_ids):
        raise RuntimeError(f"Bootstrap emitted {set(accept)} tokens per request, expected exactly 1 (no drafts scheduled)")
    computed = worker_computed(runner, req_ids)
    if computed != list(args.input_lens):
        raise RuntimeError(f"Post-bootstrap context {computed[:4]}... != ISL {list(args.input_lens)[:4]}...")
    return req_ids


def run_rank(rank, args, cli, port):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(topology_of(args).world_size),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), VLLM_ENABLE_V1_MULTIPROCESSING="0")
    import numpy as np
    import random
    import torch
    rank_start = time.perf_counter()
    topology = topology_of(args)
    phase_seconds = {}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.profile:
        from profiling_yihou import install_capture_trace
        install_capture_trace(rank, log)
    session = ProfileSession.create(args, rank, log)
    phase_start = time.perf_counter()
    engine, worker, runner = create_engine(args, rank, cli)
    phase_seconds["load_pool_capture"] = time.perf_counter() - phase_start
    from vllm.distributed.parallel_state import get_world_group
    group = get_world_group().cpu_group
    resolved = resolved_engine_summary(runner, worker)
    state = compare_resolved_parallel(topology, rank, resolved["parallel"])
    spec = resolved["speculative"]
    if spec is None or spec["rejection_sample_method"] != "synthetic" or spec["enable_adaptive_verification"]:
        raise RuntimeError(f"Engine resolved a speculative config the bench cannot account for: {spec}")
    if runner.speculator is None:
        raise RuntimeError("Engine has no speculator; the whole-step bench needs the drafter")
    K = runner.num_speculative_steps
    if K != args.num_speculative_tokens:
        raise RuntimeError(f"runner K={K} != --num-speculative-tokens {args.num_speculative_tokens}")
    conditional = unconditional_to_conditional(spec["synthetic_acceptance_rates"])
    min_accept, max_accept = accept_bounds(conditional)
    log(rank, f"topology={state} local_batch={topology.local_batch_size} global_batch={args.batch_size} "
              f"accept_bounds=({min_accept},{max_accept}) kv_num_blocks={resolved['cache']['num_blocks']}")
    import vllm.v1.worker.gpu.model_runner as runner_module
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler as rejection_module
    source_paths = module_source_paths([runner_module, rejection_module,
                                        sys.modules[type(runner.speculator).__module__]])
    result_dir = Path(args.result_dir)
    if rank == 0:
        (result_dir / "engine_resolved_yihou.json").write_text(json.dumps(resolved, indent=2, default=str) + "\n")
    final_lens = [isl + args.output_len for isl in args.input_lens]
    with torch.inference_mode():
        phase_start = time.perf_counter()
        plan = BatchPlan(runner, args, topology, rank, min_accept, max_accept)
        memory = plan.summary()
        log(rank, f"physically initializing KV ({args.kv_init}); plan={memory}")
        memory["kv_initialized_bytes"], memory["kv_init_inventory"] = initialize_kv(runner, plan, args.kv_init, args.seed)
        rng = np.random.default_rng(args.seed)  # same prompts on every DP rank (Plan 1)
        vocab = runner.model_config.get_vocab_size()
        prompts = [rng.integers(0, vocab, isl, dtype=np.int64).tolist() for isl in args.input_lens]
        sampling = [default_sampling_params(runner, args, total) for total in plan.reserved_tokens]
        torch.cuda.synchronize()
        phase_seconds["physical_initialization"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        log(rank, "running untimed admission (bootstrap) step")
        req_ids = bootstrap(worker, runner, plan, args, 0, prompts, sampling)
        phase_seconds["bootstrap"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        log(rank, f"warming up {args.warmup_steps} complete speculative iterations")
        warm = DecodeAccounting(topology.local_batch_size, args.input_lens, args.output_len, max_accept_len=K + 1)
        for iteration in range(args.warmup_steps):
            if warm.complete:
                break
            previous = warm.seq_lens
            accept, _ = run_step(worker, decode_output(req_ids, previous, warm.emitted, args), req_ids)
            new = worker_computed(runner, req_ids)
            check_rank_progress(iteration, previous, accept, new, final_lens, K + 1, group)
            validate_worker_progress(previous, accept, new)
            warm.record(accept, 0.0)
        if args.warmup_steps:
            # Reset to exactly ISL: finish the warmed requests and re-admit fresh ones on the same blocks.
            req_ids = bootstrap(worker, runner, plan, args, 1, prompts, sampling, finished=req_ids)
        phase_seconds["warmup_and_reset"] = time.perf_counter() - phase_start
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
        accounting = DecodeAccounting(topology.local_batch_size, args.input_lens, args.output_len, max_accept_len=K + 1)
        steps = []
        graph_steps = draft_graph_steps = 0
        graph_execution_counts = count_graph_executions({
            "target": runner.cudagraph_manager,
            "draft": getattr(runner.speculator, "query_cudagraph_manager", None),
        })
        graph_execution_counts["draft_extend"] = None  # DSpark has no draft-extend forward (context KV is in the draft graph)
        if session is not None:
            stages = {"target_forward": (runner.cudagraph_manager, "run_fullgraph") if runner.cudagraph_manager else None,
                      "target_sample": (runner, "sample"),
                      "draft_propose": (runner.speculator, "propose")}
            if args.disable_cuda_graph or runner.cudagraph_manager is None:
                stages["target_forward"] = (runner.model, "forward")
            session.attach_device_timer({k: v for k, v in stages.items() if v is not None})
        log(rank, "measured decode loop begins at exact requested ISL")
        start = time.perf_counter()
        while not accounting.complete and (not args.max_steps or accounting.verify_ct < args.max_steps):
            if session is not None:
                session.step_boundary(accounting.verify_ct)
            tick = time.perf_counter()
            before = (graph_execution_counts["target"], graph_execution_counts["draft"])
            previous = accounting.seq_lens
            accept, host = run_step(worker, decode_output(req_ids, previous, accounting.emitted, args), req_ids)
            new = worker_computed(runner, req_ids)
            check_rank_progress(accounting.verify_ct, previous, accept, new, final_lens, K + 1, group)
            validate_worker_progress(previous, accept, new)
            step = accounting.record(accept, time.perf_counter() - tick)
            step.pop("seq_lens")
            step["seconds"] = time.perf_counter() - tick
            step.update(host)
            step["target_graph"] = graph_execution_counts["target"] > before[0]
            step["draft_graph"] = graph_execution_counts["draft"] > before[1]
            graph_steps += int(step["target_graph"])
            draft_graph_steps += int(step["draft_graph"])
            steps.append(step)
            if rank == 0 and (accounting.verify_ct <= 3 or accounting.verify_ct % max(1, args.log_interval) == 0):
                contexts = accounting.seq_lens
                log(rank, f"iteration={accounting.verify_ct} context_min={min(contexts)} context_max={max(contexts)} "
                          f"useful={sum(accounting.emitted)} step_ms={step['seconds'] * 1000:.2f} "
                          f"target_graph={step['target_graph']} draft_graph={step['draft_graph']}")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        phase_seconds["decode"] = elapsed
        phase_seconds["rank_total_to_result"] = time.perf_counter() - rank_start
        summary = accounting.summary(elapsed)
        summary.update(topology.summary())
        seconds = [s["seconds"] for s in steps]
        summary.update({
            "expected_accept_length": args.accept_length, "accept_method": args.accept_method,
            "accept_token_mode": args.accept_token_mode,
            "synthetic_acceptance_rates": spec["synthetic_acceptance_rates"],
            "synthetic_conditional_rates": conditional, "accept_bounds": [min_accept, max_accept],
            "speculative_method": spec["method"], "num_speculative_tokens": K,
            "draft_sample_method": spec["draft_sample_method"],
            "output_tokens_per_second_per_gpu": summary["output_tokens_per_second"] / topology.world_size,
            "target_graph_iterations": graph_steps, "draft_graph_iterations": draft_graph_steps,
            "warmup_steps_excluded": args.warmup_steps,
            "graph_execution_counts": graph_execution_counts,
            "bootstrap_tokens_excluded": topology.local_batch_size,
            "step_seconds_p50": statistics.median(seconds) if seconds else None,
            "step_seconds_p10": statistics.quantiles(seconds, n=10)[0] if len(seconds) > 1 else None,
            "step_seconds_p90": statistics.quantiles(seconds, n=10)[-1] if len(seconds) > 1 else None,
            "host_execute_seconds_p50": statistics.median(s["execute_s"] for s in steps) if steps else None,
            "host_sample_seconds_p50": statistics.median(s["sample_s"] for s in steps) if steps else None,
            "rank": rank, "parallel_state": state,
            "report_scope": "local_dp_shard", "graph_execution_counts_scope": "local_rank",
            "phase_seconds": phase_seconds, "kv_init": args.kv_init, "graph_sizes": args.graph_sizes,
            "cross_rank_progress_check": "every warmup and measured iteration (gloo, CPU); included in timing",
            "synthetic_prefix": True, "simulated_acceptance": True, "real_model_weights": True,
            "real_moe_routing": True, "scheduler_used": False, "prefix_kv_source": f"synthetic fill ({args.kv_init})",
            "pinned_vllm": PINNED_VLLM, "pinned_image": PINNED_IMAGE, "module_source_paths": source_paths,
            "timing_boundary": "complete speculative step (SchedulerOutput build + execute_model + sample_tokens "
                               "+ output copy + drafts + synchronize + progress checks); excludes admission/warmup/load/capture",
            "terminal_policy": "fixed batch: a request that reached OSL keeps decoding; its tokens are paid, not counted",
            "memory": memory, "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "model_path": args.model_path,
            "target_backend": resolved["runner"]["cudagraph_manager"], "draft_backend": resolved["runner"]["speculator"],
            "draft_graph_available": getattr(runner.speculator, "query_cudagraph_manager", None) is not None,
            "draft_extend_graph_available": None,
            "cudagraph_mode": resolved["compilation"]["cudagraph_mode"],
        })
        if session is not None:
            summary["profiled"] = True
            summary["is_performance_measurement"] = False
            session.finish(state, {
                "measured_iterations": accounting.verify_ct, "measured_elapsed_seconds": elapsed,
                "pinned_vllm": PINNED_VLLM, "module_source_paths": source_paths,
                "topology": topology.summary(), "result_dir": args.result_dir})
        (result_dir / f"rank_{rank}_yihou.json").write_text(json.dumps(summary, indent=2) + "\n")
        (result_dir / f"steps_rank_{rank}_yihou.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
        reports = [None] * topology.world_size
        torch.distributed.all_gather_object(reports, summary, group=group)
        aggregate = aggregate_rank_summaries(reports, topology)
        if rank == 0:
            (result_dir / "result_yihou.json").write_text(json.dumps(aggregate, indent=2) + "\n")
            (result_dir / "steps_yihou.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
            log(rank, json.dumps({k: aggregate[k] for k in (
                "effective_token_latency_ms_per_user", "output_tokens_per_second", "output_tokens_per_second_per_gpu",
                "realized_accept_length", "verify_iterations", "step_seconds_p50", "target_graph_iterations",
                "draft_graph_iterations")}, sort_keys=True))
    torch.distributed.barrier(group=group)
    os._exit(0)  # skip engine teardown (DP/NCCL shutdown can hang); results are already on disk


def free_port():
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    args, extra = make_parser().parse_known_args()
    validate_args(args)
    isl_spec = resolve_input_lens(args)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ and "TORCHELASTIC_RUN_ID" in os.environ:
        raise RuntimeError("Run with plain python, not torchrun; this driver spawns its ranks")
    cli = engine_cli(args, extra)
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir = str(result_dir)
    if args.profile:
        profile_env = configure_profile_env(args)
        print(f"[INTERNAL-DECODE] profiling enabled; env={profile_env}", flush=True)
    heterogeneous = args.input_len_spec is not None
    recorded = {key: value for key, value in vars(args).items()
                if (args.profile or not (key == "profile" or key.startswith("profile_") or key == "device_timer"))
                and (heterogeneous or key not in ("input_len_spec", "input_lens"))}
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
    try:
        from redact import redact_argv
        recorded_cli = redact_argv(cli)
    except ImportError:
        recorded_cli = cli
    payload = {"benchmark": recorded, "engine_cli": recorded_cli, "expected_vllm_version": PINNED_VLLM,
               "expected_image": PINNED_IMAGE, "speculative_config": speculative_config(args)}
    if heterogeneous:
        payload["input_len_spec_resolved"] = isl_spec.describe()
        payload["input_lens_realized"] = list(args.input_lens)
    (result_dir / "config_yihou.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    port = args.master_port or free_port()
    world = topology_of(args).world_size
    import torch.multiprocessing as mp
    mp.spawn(run_rank, args=(args, cli, port), nprocs=world, join=True)


if __name__ == "__main__":
    main()
