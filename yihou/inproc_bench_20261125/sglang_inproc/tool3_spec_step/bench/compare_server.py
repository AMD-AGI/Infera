#!/usr/bin/env python3
"""Reference comparison using fixed waves, never Scheduler or a server (internal side).

Port of the GLM-5.2 compare_server.py: the same engine (profile_decode.Engine) runs `--num-requests`
requests as fixed waves of `--batch-size`, each wave a fresh prefill of its prompts followed by a
full decode loop, and reports per-wave and aggregate TPOT in the units a serving client reports.

Adapted:
- initial_state: 'sikl' (as profile_decode: the prefill token is excluded and OSL decode tokens are
  computed) or 'server-like' (the prefill emits output token 1 as a served request does, so each
  request computes OSL-1 decode tokens; replaces GLM-5.2's 'fake-server', whose zero-KV / one-hot
  EAGLE state mimicked a fake-input decode server that does not exist for DSv4.1 -- tool 1 waived).
- DP attention is supported (the #5034 reference server runs DPA4); GLM-5.2 required DP1.
The real-server side is scripts/server_reference.sh + scripts/server_tpot.py.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import profile_decode as base
from batch_state import DecodeAccounting, validate_worker_progress
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


def run_wave(engine, args, size, output_len, rank, wave_index, buffers, sync):
    import torch
    import torch.distributed as dist
    topology = DecodeTopology(args.tp_size, args.ep_size, size, args.enable_dp_attention)
    engine.topology = topology
    args.input_lens = [args.input_len] * topology.local_batch_size
    setup = time.perf_counter()
    batch, _ = base.bootstrap(engine, rank)
    torch.cuda.synchronize()
    setup = time.perf_counter() - setup
    budget = decode_budget(output_len, args.initial_state)
    accounting = DecodeAccounting(topology.local_batch_size, args.input_len, budget, max_accept_len=base.VERIFY_WINDOW)
    steps = []
    dist.barrier()
    start = time.perf_counter()
    while not accounting.complete:
        tick = time.perf_counter()
        result, _previous, lens, new_lens = base.decode_step(engine, batch, sync)
        base.check_rank_progress(accounting.verify_ct, accounting.seq_lens, lens, new_lens,
                                 args.input_len + budget, buffers)
        validate_worker_progress(accounting.seq_lens, lens, new_lens)
        step = accounting.record(lens, time.perf_counter() - tick)
        step['seconds'] = time.perf_counter() - tick
        step['target_graph'] = bool(result.can_run_cuda_graph)
        steps.append(step)
    torch.cuda.synchronize()
    seconds = torch.tensor(time.perf_counter() - start, dtype=torch.float64, device='cuda')
    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    base.release(engine, batch)
    summary = accounting.summary(seconds.item())
    # Global counts: every DP replica runs local_batch_size requests of the wave.
    summary.update(wave=wave_index, batch_size=size, local_batch_size=topology.local_batch_size,
                   output_tokens=size * output_len, decode_tokens=size * budget,
                   raw_accept_tokens=summary['raw_accept_tokens'] * topology.dp_size,
                   handoff_tokens=size if args.initial_state == 'server-like' else 0,
                   seconds=seconds.item(), setup_seconds=setup,
                   target_graph_iterations=sum(int(s['target_graph']) for s in steps))
    return summary, steps


def rank_main(rank, gpu_id, server_args, port_args, args):
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank
    from sglang.srt.utils import configure_logger
    publish(server_args, role="scheduler",
            ranks=SpawnRanks(world_rank=spawn_world_rank(server_args, tp_rank=rank, pp_rank=0), gpu_id=gpu_id))
    configure_logger(server_args, prefix=f" TP{rank}")
    base.configure_acceptance(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    topology = DecodeTopology(args.tp_size, args.ep_size, args.batch_size, args.enable_dp_attention)
    engine = base.Engine(server_args, port_args, gpu_id, rank, args, topology)
    sync = engine.sync_fn()
    buffers = (torch.empty(7, dtype=torch.int64, device='cuda'),
               [torch.empty(7, dtype=torch.int64, device='cuda') for _ in range(args.tp_size)])
    torch.manual_seed(args.seed)
    out = Path(args.result_dir)
    with torch.inference_mode():
        if args.warmup_requests:
            for i, size in enumerate(wave_sizes(args.warmup_requests, args.batch_size)):
                summary, _ = run_wave(engine, args, size, min(args.output_len, 32), rank, -i - 1, buffers, sync)
                base.log(rank, f'request warmup completed: {size} requests, {summary["seconds"]:.4f}s')
        base.reset_acceptance_rng(engine, args.seed)
        waves = []
        wall_start = time.perf_counter()
        sizes = wave_sizes(args.num_requests, args.batch_size)
        for i, size in enumerate(sizes):
            summary, steps = run_wave(engine, args, size, args.output_len, rank, i, buffers, sync)
            waves.append(summary)
            (out / f'wave_{i:02d}_rank_{rank}_yihou.json').write_text(json.dumps(summary, indent=2) + '\n')
            if rank == 0:
                (out / f'wave_{i:02d}_steps_yihou.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in steps))
                base.log(rank, f'wave {i + 1}/{len(sizes)} complete: {summary["seconds"]:.6f}s')
        result = summarize_waves(waves)
        result.update(initial_state=args.initial_state, batch_size=args.batch_size,
                      warmup_requests=args.warmup_requests, warmup_output_len=min(args.output_len, 32),
                      input_len=args.input_len, output_len=args.output_len, tp_size=args.tp_size, ep_size=args.ep_size,
                      enable_dp_attention=args.enable_dp_attention,
                      target_graph_iterations=sum(w['target_graph_iterations'] for w in waves),
                      verify_iterations=sum(w['verify_iterations'] for w in waves),
                      wave_orchestration_wall_seconds=time.perf_counter() - wall_start,
                      seed=args.seed, complete=True, synthetic_prefix=True, simulated_acceptance=True,
                      scheduler_used=False, overlap_schedule=False,
                      reset_policy='fresh real prefill per wave (release_kv_cache between waves); no server recycled-KV equivalence claim',
                      timing_boundary='sum of complete decode loops; wave prefill and file writes excluded')
        (out / f'comparison_rank_{rank}_yihou.json').write_text(json.dumps(result, indent=2) + '\n')
        if rank == 0:
            (out / 'comparison_yihou.json').write_text(json.dumps(result, indent=2) + '\n')
            base.log(rank, json.dumps(result))
    dist.barrier()


def main():
    parser = base.make_parser()
    parser.set_defaults(tp_size=4, ep_size=4, batch_size=16, input_len=8192, output_len=1024)
    parser.add_argument('--num-requests', type=int, default=128)
    parser.add_argument('--warmup-requests', type=int, default=16)
    parser.add_argument('--initial-state', choices=['sikl', 'server-like'], default='server-like')
    args, extra = parser.parse_known_args()
    # This tool inherits profile_decode's parser, so --input-len-spec is accepted by the parser
    # but nothing here honours it. Rejecting it explicitly beats silently measuring a uniform batch.
    if getattr(args, 'input_len_spec', None) is not None:
        raise ValueError('Server reference comparison supports only a uniform --input-len; use profile_decode.py for --input-len-spec')
    base.validate_args(args)
    sizes = wave_sizes(args.num_requests, args.batch_size)
    warm = wave_sizes(args.warmup_requests, args.batch_size) if args.warmup_requests else []
    decode_budget(args.output_len, args.initial_state)
    for size in set(sizes + warm):
        DecodeTopology(args.tp_size, args.ep_size, size, args.enable_dp_attention)
    base.configure_acceptance(args)
    import multiprocessing
    import os
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs
    from sglang.srt.utils import kill_process_tree, maybe_reindex_device_id
    topology = DecodeTopology(args.tp_size, args.ep_size, args.batch_size, args.enable_dp_attention)
    # Capture the actual warmup and measurement local batch sizes, not a padded one.
    local_sizes = sorted({size // topology.dp_size for size in sizes + warm})
    extra = list(extra)
    server_args, cfg, cli = base.build_server_args(args, extra)
    pos = cli.index('--cuda-graph-bs-decode')
    cli[pos + 1:pos + 2] = [str(x) for x in local_sizes]
    cli[cli.index('--cuda-graph-max-bs-decode') + 1] = str(max(local_sizes))
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.server_args import ServerArgs
    config_parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(config_parser)
    server_args = ServerArgs.from_cli_args(config_parser.parse_args(cli))
    server_args.resolve_once()
    cfg = resolving_view(server_args)
    base.check_server_args(cfg, topology)
    if args.prefill_tokens is None:
        args.prefill_tokens = int(cfg.chunked_prefill_size)
    args.resolved = {name: str(getattr(cfg, name, None)) for name in ("moe_a2a_backend", "moe_runner_backend")}
    out = Path(args.result_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.result_dir = str(out)
    redact_argv = base.import_redact_argv()
    (out / 'config_yihou.json').write_text(json.dumps({'benchmark': vars(args), 'server_cli': redact_argv(cli)},
                                                      indent=2, default=str) + '\n')
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    procs = []
    try:
        for rank in range(args.tp_size):
            with maybe_reindex_device_id(rank) as gpu_id:
                proc = multiprocessing.Process(target=rank_main, args=(rank, gpu_id, server_args, port_args, args))
                proc.start()
                procs.append(proc)
        while any(proc.is_alive() for proc in procs):
            if any(proc.exitcode not in (None, 0) for proc in procs):
                break
            time.sleep(2)
        return 1 if any(proc.exitcode != 0 for proc in procs) else 0
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == '__main__':
    sys.exit(main())
