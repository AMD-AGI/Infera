#!/usr/bin/env python3
"""Reference comparison using fixed waves, never Scheduler or a server."""
import argparse
from copy import copy
import json
from pathlib import Path
import statistics
import time

import profile_decode as base
from batch_state import DecodeAccounting, count_graph_executions, validate_worker_progress
from topology import DecodeTopology, prepare_dp_metadata


def wave_sizes(num_requests, concurrency):
    if num_requests <= 0 or concurrency <= 0:
        raise ValueError('Request counts must be positive')
    if num_requests < concurrency:
        return [num_requests]
    if num_requests % concurrency:
        raise ValueError('Measured request count must divide into full fixed waves')
    return [concurrency] * (num_requests // concurrency)


def decode_budget(output_len, initial_state):
    budget = output_len - int(initial_state == 'fake-server')
    if budget <= 0:
        raise ValueError('At least one computed decode token is required')
    return budget


def summarize_waves(waves):
    requests = sum(w['batch_size'] for w in waves)
    outputs = sum(w['output_tokens'] for w in waves)
    decoded = sum(w['decode_tokens'] for w in waves)
    seconds = sum(w['seconds'] for w in waves)
    exposure = sum(w['batch_size'] * w['verify_iterations'] for w in waves)
    return dict(num_requests=requests, output_tokens=outputs, decode_tokens=decoded,
                decode_seconds=seconds, output_tokens_per_second=outputs / seconds,
                decode_tokens_per_second=decoded / seconds,
                output_amortized_tpot_ms=seconds * 1000 * requests / outputs,
                decode_tpot_ms=seconds * 1000 * requests / decoded,
                wave_tpot_median_ms=statistics.median(w['seconds'] * 1000 * w['batch_size'] / w['decode_tokens'] for w in waves),
                realized_accept_length=sum(w['raw_accept_tokens'] for w in waves) / exposure)


def initialize_state(target, worker, args, size, rank):
    import torch
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.eagle_utils import get_draft_recurrent_hidden_state_spec
    target.model_runner.req_to_token_pool.clear()
    target.model_runner.token_to_kv_pool_allocator.clear()
    local = copy(args)
    local.batch_size = size
    topology = DecodeTopology(args.tp_size, args.ep_size, size, False)
    batch, prompt, memory = base.allocate_batch(target, local, topology, rank)
    if args.initial_state == 'sikl':
        base.initialize_kv(target.model_runner, args.seed)
        base.initialize_kv(worker.draft_worker.draft_runner, args.seed + 1)
        base.bootstrap(batch, prompt, worker, local, topology)
    else:
        # The fake receiver does not write KV; reproduce fresh zero pools rather
        # than claiming byte-equivalence to a server's recycled pages.
        for runner in (target.model_runner, worker.draft_worker.draft_runner):
            pool = runner.token_to_kv_pool
            for buffer in pool.kv_buffer:
                buffer.zero_()
            for buffer in pool.index_k_with_scale_buffer:
                buffer.zero_()
        hidden_size, dtype = get_draft_recurrent_hidden_state_spec(worker.draft_worker.draft_runner)
        probabilities = torch.zeros((size, target.model_config.vocab_size), dtype=torch.float32, device='cuda')
        probabilities[:, 0] = 1
        batch.spec_info = EagleDraftInput(
            topk_p=torch.ones((size, 1), dtype=torch.float32, device='cuda'),
            topk_index=torch.zeros((size, 1), dtype=torch.int64, device='cuda'),
            draft_probs=probabilities,
            hidden_states=torch.zeros((size, hidden_size), dtype=dtype, device='cuda'),
            capture_hidden_mode=CaptureHiddenMode.LAST,
            bonus_tokens=torch.zeros(size, dtype=torch.int64, device='cuda'))
        batch.seq_lens_cpu = torch.full((size,), args.input_len, dtype=torch.int64)
        batch.seq_lens = batch.seq_lens_cpu.cuda()
        batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
        batch.seq_lens_sum = size * args.input_len
        batch.forward_mode = ForwardMode.DECODE
        batch.input_ids = None
        for req in batch.reqs:
            req.output_ids = [0]
            req.kv_committed_len = args.input_len
    prepare_dp_metadata(batch, topology, is_extend=False, disable_cuda_graph=args.disable_cuda_graph)
    torch.cuda.synchronize()
    return batch, topology, memory


def run_wave(target, worker, args, size, output_len, rank, wave_index, progress_buffers):
    import torch
    import torch.distributed as dist
    setup = time.perf_counter()
    batch, topology, memory = initialize_state(target, worker, args, size, rank)
    setup = time.perf_counter() - setup
    budget = decode_budget(output_len, args.initial_state)
    accounting = DecodeAccounting(size, args.input_len, budget)
    counts = count_graph_executions({'target': target.model_runner.decode_cuda_graph_runner,
        'draft': worker.draft_worker.cuda_graph_runner,
        'draft_extend': worker.draft_worker.cuda_graph_runner_for_draft_extend})
    steps = []
    dist.barrier()
    start = time.perf_counter()
    while not accounting.complete:
        tick = time.perf_counter()
        batch.prepare_for_decode()
        prepare_dp_metadata(batch, topology, is_extend=False, disable_cuda_graph=args.disable_cuda_graph)
        result = worker.forward_batch_generation(batch)
        torch.cuda.synchronize()
        lens = result.accept_lens.cpu().tolist()
        new_lens = result.new_seq_lens.cpu().tolist()
        base.check_rank_progress(accounting.verify_ct, accounting.seq_lens, lens, new_lens,
                                args.input_len + budget, progress_buffers)
        validate_worker_progress(accounting.seq_lens, lens, new_lens)
        step = accounting.record(lens, time.perf_counter() - tick)
        base.commit_result(batch, result, accounting.seq_lens)
        step['seconds'] = time.perf_counter() - tick
        steps.append(step)
    torch.cuda.synchronize()
    seconds = torch.tensor(time.perf_counter() - start, dtype=torch.float64, device='cuda')
    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    summary = accounting.summary(seconds.item())
    summary.update(wave=wave_index, batch_size=size, output_tokens=size * output_len,
                   decode_tokens=size * budget, handoff_tokens=size if args.initial_state == 'fake-server' else 0,
                   seconds=seconds.item(), setup_seconds=setup, graph_execution_counts=counts,
                   memory=memory)
    return summary, steps


def rank_main(rank, server_args, port_args, args):
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    base.configure_acceptance(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    target, worker = base.create_workers(server_args, port_args, rank, args)
    buffers = (torch.empty(7, dtype=torch.int64, device='cuda'),
               [torch.empty(7, dtype=torch.int64, device='cuda') for _ in range(args.tp_size)])
    torch.manual_seed(args.seed)
    out = Path(args.result_dir)
    with torch.inference_mode():
        if args.warmup_requests:
            for i, size in enumerate(wave_sizes(args.warmup_requests, args.batch_size)):
                summary, _ = run_wave(target, worker, args, size, min(args.output_len, 32), rank, -i-1, buffers)
                base.log(rank, f'request warmup completed: {size} requests, {summary["seconds"]:.4f}s')
        waves = []
        wall_start = time.perf_counter()
        for i, size in enumerate(wave_sizes(args.num_requests, args.batch_size)):
            summary, steps = run_wave(target, worker, args, size, args.output_len, rank, i, buffers)
            waves.append(summary)
            (out / f'wave_{i:02d}_rank_{rank}_yihou.json').write_text(json.dumps(summary, indent=2)+'\n')
            if rank == 0:
                (out / f'wave_{i:02d}_steps_yihou.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in steps))
                base.log(rank, f'wave {i+1}/{len(wave_sizes(args.num_requests,args.batch_size))} complete: {summary["seconds"]:.6f}s')
        result = summarize_waves(waves)
        result.update(initial_state=args.initial_state, batch_size=args.batch_size,
                      warmup_requests=args.warmup_requests,warmup_output_len=min(args.output_len,32),
                      input_len=args.input_len,output_len=args.output_len,tp_size=args.tp_size,ep_size=args.ep_size,
                      wave_orchestration_wall_seconds=time.perf_counter()-wall_start,
                      seed=args.seed,complete=True,synthetic_prefix=True,simulated_acceptance=True,
                      scheduler_used=False,overlap_schedule=False,
                      reset_policy='fresh physical prefix per wave; no server recycled-KV equivalence claim',
                      timing_boundary='sum of complete decode loops; wave setup and file writes excluded')
        (out / f'comparison_rank_{rank}_yihou.json').write_text(json.dumps(result,indent=2)+'\n')
        if rank == 0:
            (out/'comparison_yihou.json').write_text(json.dumps(result,indent=2)+'\n')
            base.log(rank,json.dumps(result))
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = base.make_parser()
    parser.set_defaults(tp_size=4,ep_size=4,batch_size=16,input_len=10000,output_len=500)
    parser.add_argument('--num-requests',type=int,default=128)
    parser.add_argument('--warmup-requests',type=int,default=16)
    parser.add_argument('--initial-state',choices=['sikl','fake-server'],default='fake-server')
    args, extra = parser.parse_known_args()
    base.validate_args(args)
    wave_sizes(args.num_requests,args.batch_size)
    if args.warmup_requests:wave_sizes(args.warmup_requests,args.batch_size)
    decode_budget(args.output_len,args.initial_state)
    if args.enable_dp_attention:raise ValueError('Server reference comparison requires DP1')
    base.configure_acceptance(args)
    import torch.multiprocessing as mp
    from sglang.srt.server_args import ServerArgs,PortArgs
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    cli=base.server_cli(args,extra)
    # Capture actual warmup and measurement batch sizes, not a padded C32 warmup.
    pos=cli.index('--cuda-graph-bs-decode')
    cli[pos+1:pos+2]=[str(x) for x in sorted(set([args.batch_size]+wave_sizes(args.warmup_requests,args.batch_size) if args.warmup_requests else [args.batch_size]))]
    config_parser=argparse.ArgumentParser()
    ServerArgs.add_cli_args(config_parser)
    server_args=ServerArgs.from_cli_args(config_parser.parse_args(cli))
    out=Path(args.result_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'config_yihou.json').write_text(json.dumps({'benchmark':vars(args),'server_cli':cli},indent=2)+'\n')
    _set_envs_and_config(server_args)
    mp.spawn(rank_main,args=(server_args,PortArgs.init_new(server_args),args),nprocs=args.tp_size,join=True)


if __name__=='__main__':
    main()
