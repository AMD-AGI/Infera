#!/usr/bin/env python3
"""Pinned SGLang GLM-5.2 internal MTP benchmark; no scheduler or serving layer.

Use plain python: this entry point spawns its own TP processes. --help requires
only the standard library. Additional arguments pass through to ServerArgs.
"""
import argparse
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

PINNED_SGLANG = "402df1e1e453e1e85ec0f5ac4052d36598cc691a"


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=70000, help="decimal tokens per request")
    parser.add_argument("--output-len", type=int, default=10000, help="useful emitted tokens per request")
    parser.add_argument("--accept-length", type=float, default=3.61, help="expected bonus-inclusive length, not probability")
    parser.add_argument("--accept-method", choices=("match-expected", "multinomial"), default="match-expected")
    parser.add_argument("--accept-token-mode", choices=("real-draft-token", "fixed"), default="real-draft-token")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means full OSL; positive values are incomplete smoke runs")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser


def validate_args(args):
    if min(args.batch_size, args.input_len, args.output_len, args.tp_size) <= 0:
        raise ValueError("batch size, lengths and TP size must be positive")
    if args.max_steps < 0 or args.warmup_steps < 0:
        raise ValueError("max-steps and warmup-steps must be nonnegative")
    if not 1 <= args.accept_length <= 6:
        raise ValueError("accept-length must be in [1, 6] for steps=5/draft=6")
    if args.input_len < 2:
        raise ValueError("input-len must be at least 2 for a populated-prefix bootstrap")


def server_cli(args, extra):
    flags = {item.split("=", 1)[0] for item in extra if item.startswith("--")}
    result = ["--model-path", args.model_path, "--tp-size", str(args.tp_size)] + list(extra)
    defaults = {
        "--speculative-algorithm": "EAGLE",
        "--speculative-num-steps": "5",
        "--speculative-num-draft-tokens": "6",
        "--speculative-eagle-topk": "1",
        "--kv-cache-dtype": "fp8_e4m3",
        "--dsa-decode-backend": "flydsl",
        "--dsa-prefill-backend": "flydsl",
        "--dsa-topk-backend": "aiter",
        "--cuda-graph-bs-decode": str(args.batch_size),
        "--cuda-graph-max-bs-decode": str(args.batch_size),
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
    # Must precede importing spec_utils, whose simulation constants are import-time.
    from sglang.srt.environ import envs
    envs.SGLANG_SIMULATE_ACC_LEN.set(args.accept_length)
    envs.SGLANG_SIMULATE_ACC_METHOD.set(args.accept_method)
    envs.SGLANG_SIMULATE_ACC_TOKEN_MODE.set(args.accept_token_mode)


def create_workers(server_args, port_args, rank, args):
    import torch
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import set_global_server_args_for_scheduler
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
    from sglang.srt.utils import configure_logger

    torch.cuda.set_device(rank)
    set_global_server_args_for_scheduler(server_args)
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    configure_logger(server_args, prefix=f" TP{rank}")
    ps = ParallelState(
        tp_rank=rank, tp_size=args.tp_size, pp_rank=0, pp_size=1,
        dp_rank=None, dp_size=1, attn_tp_rank=rank, attn_tp_size=args.tp_size,
        attn_cp_rank=0, attn_cp_size=1, attn_dcp_rank=0, attn_dcp_size=1,
        attn_dp_rank=0, attn_dp_size=1, moe_ep_rank=0, moe_ep_size=1,
        moe_dp_rank=None, moe_dp_size=1, gpu_id=rank,
    )
    log(rank, "loading target weights")
    target = TpModelWorker(server_args, rank, ps, port_args.nccl_port)
    if server_args.is_startup_weight_load_overlap:
        target.start_startup_weight_load()
    log(rank, "loading draft weights")
    worker = EAGLEWorkerV2(server_args, rank, ps, port_args.nccl_port, target)
    log(rank, "allocating target and draft KV pools")
    target.alloc_memory_pool()
    pool, allocator = target.get_memory_pool()
    worker.alloc_memory_pool(
        memory_pool_config=target.model_runner.memory_pool_config,
        req_to_token_pool=pool, token_to_kv_pool_allocator=allocator,
    )
    target.init_attention_backends()
    worker.init_attention_backends()
    log(rank, "capturing target and required draft graphs (startup may take minutes)")
    target.init_cuda_graphs()
    worker.init_cuda_graphs()
    if server_args.is_startup_weight_load_overlap:
        target.finalize_startup_weight_load()
    return target, worker


def log(rank, message):
    print(f"[INTERNAL-DECODE TP{rank}] {message}", flush=True)


def initialize_kv(model_runner, seed):
    """Physically fill target/draft KV and DSA keys/scales, outside measurement."""
    import torch
    pool = model_runner.token_to_kv_pool
    if not getattr(pool, "kv_buffer", None):
        raise RuntimeError(f"Unsupported KV pool for physical initialization: {type(pool).__name__}")
    validate_kv_layout(
        pool.kv_cache_dim, pool.kv_lora_rank, pool.qk_rope_head_dim,
        pool.dsa_kv_cache_store_fp8,
    )
    generator = torch.Generator(device=model_runner.device).manual_seed(seed)
    num_bytes = 0
    for buffer in pool.kv_buffer:
        if buffer.element_size() == 1:
            buffer.view(torch.uint8).random_(0, 0x7C, generator=generator)
        else:
            buffer.uniform_(-1, 1, generator=generator)
        num_bytes += buffer.numel() * buffer.element_size()
    index_buffers = getattr(pool, "index_k_with_scale_buffer", None)
    if index_buffers is None:
        raise RuntimeError("GLM DSA pool has no index_k_with_scale_buffer; refusing metadata-only KV")
    for buffer in index_buffers:
        num_key_bytes = pool.page_size * pool.index_head_dim
        buffer[:, :num_key_bytes].random_(0, 0x7C, generator=generator)
        buffer[:, num_key_bytes:].view(torch.float32).uniform_(0.5, 2.0, generator=generator)
        num_bytes += buffer.numel() * buffer.element_size()
    return num_bytes


def allocate_batch(target, args):
    import torch
    from sglang.benchmark.one_batch import TreeCacheNamespace, prepare_synthetic_inputs_for_latency_test
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.allocation_sizing import get_alloc_reserve_per_decode
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    runner = target.model_runner
    page_size = runner.token_to_kv_pool_allocator.page_size
    reserve = get_alloc_reserve_per_decode()
    num_tokens = required_token_capacity(args.batch_size, args.input_len, args.output_len, page_size, reserve)
    reserved_len = num_tokens // args.batch_size
    pool = runner.req_to_token_pool
    allocator = runner.token_to_kv_pool_allocator
    if allocator.available_size() < num_tokens:
        raise RuntimeError(f"KV capacity insufficient: need {num_tokens} tokens, available {allocator.available_size()}")
    if pool.req_to_token.shape[1] < reserved_len:
        raise RuntimeError(f"Request row too short: need {reserved_len}, have {pool.req_to_token.shape[1]}")
    if runner.model_config.context_len < args.input_len + args.output_len + reserve:
        raise RuntimeError("Model context length does not cover requested progression plus speculative reserve")
    reqs = prepare_synthetic_inputs_for_latency_test(args.batch_size, reserved_len)
    for req in reqs:
        req.sampling_params.max_new_tokens = args.output_len
        req.sampling_params.ignore_eos = True
    batch = ScheduleBatch.init_new(
        reqs=reqs, req_to_token_pool=pool, token_to_kv_pool_allocator=allocator,
        tree_cache=TreeCacheNamespace(page_size=page_size, device=runner.device, token_to_kv_pool_allocator=allocator),
        model_config=runner.model_config, enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.EAGLE,
    )
    batch.prepare_for_extend()  # allocation and page mappings only; no model prefill
    # Prefix contents remain synthetic. All mapped future pages are reserved once,
    # so speculative prepare_for_decode cannot allocate or recycle a live page.
    for req in reqs:
        req.origin_input_ids = req.origin_input_ids[:args.input_len]
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.set_extend_range(args.input_len - 1, args.input_len)
        req.kv_committed_len = args.input_len
    prompt_tokens = torch.tensor([req.origin_input_ids[-1] for req in reqs], dtype=torch.int64, device=runner.device)
    return batch, prompt_tokens, {"reserved_tokens": num_tokens, "reserved_tokens_per_request": reserved_len, "page_size": page_size, "speculative_reserve": reserve}


def bootstrap(batch, prompt_tokens, worker, args):
    """One real last-prompt-token target+draft forward over the synthetic prefix."""
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    size, length = args.batch_size, args.input_len
    batch.forward_mode = ForwardMode.EXTEND
    batch.is_extend_in_batch = False
    batch.spec_info = None
    batch.input_ids = prompt_tokens.clone()
    batch.prefill_input_ids_cpu = None
    batch.seq_lens_cpu = torch.full((size,), length, dtype=torch.int64)
    batch.seq_lens = batch.seq_lens_cpu.to(prompt_tokens.device)
    batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
    batch.seq_lens_sum = size * length
    batch.prefix_lens = [length - 1] * size
    batch.extend_lens = [1] * size
    batch.extend_num_tokens = size
    batch.extend_logprob_start_lens = [1] * size
    batch.out_cache_loc = batch.req_to_token_pool.req_to_token[batch.req_pool_indices.long(), length - 1].to(torch.int64)
    for req in batch.reqs:
        req.kv_committed_len = length
    result = worker.forward_batch_generation(batch)
    torch.cuda.synchronize()
    info = result.next_draft_input
    for name in ("hidden_states", "topk_p", "topk_index", "bonus_tokens"):
        value = getattr(info, name)
        if value is None or value.shape[0] != size:
            raise RuntimeError(f"Bootstrap produced invalid {name}: {getattr(value, 'shape', None)}")
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise RuntimeError(f"Non-finite bootstrap {name}")
    batch.spec_info = info
    batch.input_ids = None
    batch.forward_mode = ForwardMode.DECODE
    return result


def commit_result(batch, result, seq_lens):
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    batch.spec_info = result.next_draft_input
    batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    batch.seq_lens = batch.seq_lens_cpu.to(result.new_seq_lens.device)
    batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
    batch.seq_lens_sum = sum(seq_lens)
    batch.input_ids = None
    batch.forward_mode = ForwardMode.DECODE
    batch.is_extend_in_batch = False
    for req, length in zip(batch.reqs, seq_lens):
        req.kv_committed_len = length
        if length > req.kv.kv_allocated_len:
            raise RuntimeError("Committed sequence exceeds physical allocation")


def run_rank(rank, server_args, port_args, args):
    import torch
    import torch.distributed as dist
    import numpy as np
    import random
    configure_acceptance(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    target, worker = create_workers(server_args, port_args, rank, args)
    import sglang.srt.server_args as server_args_module
    import sglang.srt.speculative.eagle_worker_v2 as eagle_worker_module
    source_paths = module_source_paths([server_args_module, eagle_worker_module])
    log(rank, f"runtime source paths={source_paths}")
    result_dir = Path(args.result_dir)
    with torch.inference_mode():
        batch, prompt_tokens, memory = allocate_batch(target, args)
        log(rank, f"physically initializing pools; mapping={memory}")
        memory["target_initialized_bytes"] = initialize_kv(target.model_runner, args.seed)
        memory["draft_initialized_bytes"] = initialize_kv(worker.draft_worker.draft_runner, args.seed + 1)
        log(rank, "running one-token untimed target+draft bootstrap")
        bootstrap(batch, prompt_tokens, worker, args)
        log(rank, f"warming up {args.warmup_steps} complete speculative iterations")
        for _ in range(args.warmup_steps):
            batch.prepare_for_decode()
            result = worker.forward_batch_generation(batch)
            torch.cuda.synchronize()
            commit_result(batch, result, result.new_seq_lens.cpu().tolist())
        if args.warmup_steps:
            bootstrap(batch, prompt_tokens, worker, args)
        # Acceptance uses CPU torch RNG; reseed after weight/init/warmup work so
        # every TP rank consumes identical coins independent of startup branches.
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        dist.barrier()
        accounting = DecodeAccounting(args.batch_size, args.input_len, args.output_len)
        steps = []
        graph_steps = 0
        graph_execution_counts = count_graph_executions({
            "target": target.model_runner.decode_cuda_graph_runner,
            "draft": worker.draft_worker.cuda_graph_runner,
            "draft_extend": worker.draft_worker.cuda_graph_runner_for_draft_extend,
        })
        log(rank, "measured decode loop begins at exact requested ISL")
        start = time.perf_counter()
        while not accounting.complete and (not args.max_steps or accounting.verify_ct < args.max_steps):
            tick = time.perf_counter()
            batch.prepare_for_decode()
            result = worker.forward_batch_generation(batch)
            torch.cuda.synchronize()  # result tensors and cross-stream keep-alives stay live through here
            accept_lens = result.accept_lens.cpu().tolist()
            if len(set(accept_lens)) != 1:
                raise RuntimeError("Pinned simulation must use one shared acceptance coin per batch; unequal lengths need active-request compaction")
            validate_worker_progress(
                accounting.seq_lens, accept_lens, result.new_seq_lens.cpu().tolist()
            )
            step = accounting.record(accept_lens, time.perf_counter() - tick)
            # Only the terminal iteration may clip useful emissions. Its full
            # draft/verify/extension compute is paid; beyond-cap tokens are not
            # counted. No next forward consumes the clipped recurrent state.
            commit_result(batch, result, accounting.seq_lens)
            step["seconds"] = time.perf_counter() - tick
            step["target_graph"] = bool(result.can_run_cuda_graph)
            graph_steps += int(step["target_graph"])
            steps.append(step)
            if rank == 0 and (accounting.verify_ct <= 3 or accounting.verify_ct % max(1, args.log_interval) == 0):
                log(rank, f"iteration={accounting.verify_ct} context={accounting.seq_lens[0]} useful={sum(accounting.emitted)} target_graph={step['target_graph']}")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rank_elapsed = torch.tensor(elapsed, dtype=torch.float64, device="cuda")
        dist.all_reduce(rank_elapsed, op=dist.ReduceOp.MAX)
        summary = accounting.summary(rank_elapsed.item())
        summary.update({
            "expected_accept_length": args.accept_length, "accept_method": args.accept_method,
            "accept_token_mode": args.accept_token_mode, "tp_size": args.tp_size,
            "output_tokens_per_second_per_gpu": summary["output_tokens_per_second"] / args.tp_size,
            "target_graph_iterations": graph_steps, "warmup_steps_excluded": args.warmup_steps,
            "graph_execution_counts": graph_execution_counts,
            "bootstrap_tokens_excluded": args.batch_size,
            "synthetic_prefix": True, "simulated_acceptance": True, "real_model_weights": True,
            "real_moe_routing": True, "scheduler_used": False, "pinned_sglang": PINNED_SGLANG,
            "module_source_paths": source_paths,
            "timing_boundary": "complete internal speculative loop + synchronization + bookkeeping; excludes bootstrap/warmup/load/capture",
            "terminal_policy": "count only remaining useful tokens; pay full terminal computation; no subsequent forward",
            "memory": memory, "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "model_path": args.model_path,
            "target_backend": type(target.model_runner.attn_backend).__name__,
            "draft_backend": type(worker.draft_worker.draft_runner.attn_backend).__name__,
            "draft_graph_available": worker.draft_worker.cuda_graph_runner is not None,
            "draft_extend_graph_available": worker.draft_worker.cuda_graph_runner_for_draft_extend is not None,
        })
        (result_dir / f"rank_{rank}_yihou.json").write_text(json.dumps(summary, indent=2) + "\n")
        if rank == 0:
            (result_dir / "result_yihou.json").write_text(json.dumps(summary, indent=2) + "\n")
            (result_dir / "steps_yihou.jsonl").write_text("".join(json.dumps(step) + "\n" for step in steps))
            log(rank, json.dumps(summary, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


def main():
    args, extra = make_parser().parse_known_args()
    validate_args(args)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Run with plain python, not torchrun; this driver spawns TP ranks")
    configure_acceptance(args)
    import torch.multiprocessing as mp
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    cli = server_cli(args, extra)
    server_args = ServerArgs.from_cli_args(parser.parse_args(cli))
    if server_args.dp_size != 1 or server_args.ep_size != 1 or server_args.pp_size != 1 or server_args.nnodes != 1:
        raise ValueError("This fixed-batch wrapper supports only one node, DP1/EP1/PP1")
    if (server_args.speculative_algorithm, server_args.speculative_num_steps, server_args.speculative_num_draft_tokens, server_args.speculative_eagle_topk) != ("EAGLE", 5, 6, 1):
        raise ValueError("This benchmark requires EAGLE steps5/draft6/topk1")
    if server_args.load_format == "dummy":
        raise ValueError("Dummy weights are not permitted for this performance benchmark")
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir = str(result_dir)
    (result_dir / "config_yihou.json").write_text(json.dumps({"benchmark": vars(args), "server_cli": cli, "expected_sglang_commit": PINNED_SGLANG}, indent=2, sort_keys=True) + "\n")
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    if args.tp_size == 1:
        run_rank(0, server_args, port_args, args)
    else:
        mp.spawn(run_rank, args=(server_args, port_args, args), nprocs=args.tp_size, join=True)


if __name__ == "__main__":
    main()
