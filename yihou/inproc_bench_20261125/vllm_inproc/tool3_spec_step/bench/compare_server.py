#!/usr/bin/env python3
"""Reference comparison using fixed waves, never the Scheduler or a server (vLLM port of GLM-5.2's).

Runs `--num-requests` requests as consecutive full waves of `--batch-size` (global) requests through the same
engine objects as profile_decode.py, so its TPOT is directly comparable to a server run of the same number of
requests at max concurrency C (scripts/server_reference.sh): every wave admits fresh requests, decodes them to
completion and releases them.

--initial-state:
  synthetic    the bench's own state: --kv-init fill, the untimed admission step is not counted; OSL decode tokens.
  server-like  what a request of the vLLM DecodeBenchConnector server sees: KV filled as that connector does
               (--kv-init decode-bench), the admission step emits output token 1 (counted as the first token, outside
               TPOT as the client counts it), then OSL-1 decode tokens.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import profile_decode as base
from batch_state import DecodeAccounting, accept_bounds, unconditional_to_conditional, validate_worker_progress
from topology import DecodeTopology


def wave_sizes(num_requests, concurrency):
    if num_requests <= 0 or concurrency <= 0:
        raise ValueError('Request counts must be positive')
    if num_requests < concurrency:
        return [num_requests]
    if num_requests % concurrency:
        raise ValueError('Measured request count must divide into full fixed waves')
    return [concurrency] * (num_requests // concurrency)


def decode_budget(output_len, initial_state):
    budget = output_len - int(initial_state == 'server-like')
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
                output_amortized_tpot_ms=sum(w['seconds'] * 1000 * w['batch_size'] for w in waves) / outputs,
                decode_tpot_ms=sum(w['seconds'] * 1000 * w['batch_size'] for w in waves) / decoded,
                wave_tpot_median_ms=statistics.median(w['seconds'] * 1000 * w['batch_size'] / w['decode_tokens'] for w in waves),
                realized_accept_length=sum(w['raw_accept_tokens'] for w in waves) / exposure)


def make_parser():
    parser = base.make_parser()
    parser.add_argument('--num-requests', type=int, default=None, help='global requests (default: one wave)')
    parser.add_argument('--initial-state', choices=('synthetic', 'server-like'), default='server-like')
    return parser


def run_rank(rank, args, cli, port):
    import os
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(base.topology_of(args).world_size),
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), VLLM_ENABLE_V1_MULTIPROCESSING='0')
    import numpy as np
    import torch
    from vllm.distributed.parallel_state import get_world_group
    torch.manual_seed(args.seed)
    engine, worker, runner = base.create_engine(args, rank, cli)
    group = get_world_group().cpu_group
    spec = base.resolved_engine_summary(runner, worker)['speculative']
    min_accept, max_accept = accept_bounds(unconditional_to_conditional(spec['synthetic_acceptance_rates']))
    K = runner.num_speculative_steps
    budget = decode_budget(args.output_len, args.initial_state)
    waves = []
    previous_ids = []
    sizes = wave_sizes(args.num_requests or args.batch_size, args.batch_size)
    with torch.inference_mode():
        for wave, size in enumerate(sizes):
            topology = DecodeTopology(args.tp_size, args.dp_size, args.ep_size, size)
            args.input_lens = args.input_lens[:topology.local_batch_size]
            plan = base.BatchPlan(runner, args, topology, rank, min_accept, max_accept)
            base.initialize_kv(runner, plan, args.kv_init, args.seed + wave)
            rng = np.random.default_rng(args.seed + wave)
            vocab = runner.model_config.get_vocab_size()
            prompts = [rng.integers(0, vocab, isl, dtype=np.int64).tolist() for isl in args.input_lens]
            sampling = [base.default_sampling_params(runner, args, t) for t in plan.reserved_tokens]
            req_ids = base.bootstrap(worker, runner, plan, args, 100 + wave, prompts, sampling, finished=previous_ids)
            accounting = DecodeAccounting(topology.local_batch_size, args.input_lens, budget, max_accept_len=K + 1)
            torch.distributed.barrier(group=group)
            start = time.perf_counter()
            while not accounting.complete:
                previous = accounting.seq_lens
                accept, _ = base.run_step(worker, base.decode_output(req_ids, previous, accounting.emitted, args), req_ids)
                new = base.worker_computed(runner, req_ids)
                base.check_rank_progress(accounting.verify_ct, previous, accept, new,
                                         [i + budget for i in args.input_lens], K + 1, group)
                validate_worker_progress(previous, accept, new)
                accounting.record(accept, 0.0)
            seconds = time.perf_counter() - start
            local = topology.local_batch_size
            waves.append(dict(wave=wave, batch_size=local, seconds=seconds, verify_iterations=accounting.verify_ct,
                              decode_tokens=sum(accounting.emitted),
                              output_tokens=sum(accounting.emitted) + (local if args.initial_state == 'server-like' else 0),
                              raw_accept_tokens=accounting.raw_accept_tokens))
            previous_ids = req_ids
            base.log(rank, f'wave {wave}: {seconds:.3f} s, {accounting.verify_ct} steps')
        reports = [None] * base.topology_of(args).world_size
        torch.distributed.all_gather_object(reports, {'rank': rank, 'waves': waves}, group=group)
        if rank == 0:
            topology = base.topology_of(args)
            replicas = [r for r in reports if r['rank'] in topology.representative_ranks]
            merged = []
            for index in range(len(waves)):
                parts = [r['waves'][index] for r in replicas]
                merged.append(dict(wave=index, batch_size=sum(p['batch_size'] for p in parts),
                                   seconds=max(r['waves'][index]['seconds'] for r in reports),
                                   verify_iterations=parts[0]['verify_iterations'],
                                   decode_tokens=sum(p['decode_tokens'] for p in parts),
                                   output_tokens=sum(p['output_tokens'] for p in parts),
                                   raw_accept_tokens=sum(p['raw_accept_tokens'] for p in parts)))
            payload = {'initial_state': args.initial_state, 'kv_init': args.kv_init, 'waves': merged,
                       'summary': summarize_waves(merged), 'topology': topology.summary(),
                       'scheduler_used': False, 'note': 'decode_tpot_ms is the analogue of the client TPOT '
                       '(first token excluded under server-like); seconds = max over ranks per wave'}
            (Path(args.result_dir) / 'compare_yihou.json').write_text(json.dumps(payload, indent=2) + '\n')
            base.log(rank, json.dumps(payload['summary']))
    torch.distributed.barrier(group=group)
    os._exit(0)


def main():
    args, extra = make_parser().parse_known_args()
    base.validate_args(args)
    if args.input_len_spec is not None:
        raise ValueError('compare_server.py does not support --input-len-spec (the client sends one fixed ISL)')
    base.resolve_input_lens(args)
    if args.initial_state == 'server-like' and args.kv_init != 'decode-bench':
        raise ValueError('--initial-state server-like needs --kv-init decode-bench (the server connector fill)')
    cli = base.engine_cli(args, extra)
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir = str(result_dir)
    (result_dir / 'config_yihou.json').write_text(json.dumps(
        {'benchmark': {k: v for k, v in vars(args).items() if k != 'input_lens'}, 'engine_cli': cli},
        indent=2, sort_keys=True) + '\n')
    import torch.multiprocessing as mp
    mp.spawn(run_rank, args=(args, cli, args.master_port or base.free_port()),
             nprocs=base.topology_of(args).world_size, join=True)


if __name__ == '__main__':
    main()
