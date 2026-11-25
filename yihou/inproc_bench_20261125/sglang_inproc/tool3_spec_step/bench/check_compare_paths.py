#!/usr/bin/env python3
"""Bounded graph/eager differential check from an identical initial state.

Port of the GLM-5.2 check_compare_paths.py. Four modes (graph, graph_repeat, eager, eager_repeat),
each starting from a fresh real prefill of the same synthetic prompts and a re-seeded acceptance
generator, run `--path-steps` DSPARK iterations; pairs are compared for exact equality of integer
outputs (accept lens, new seq lens, emitted token ids, bonus tokens) and max-abs / relative-L2 of the
verify logits. Execution success is not numerical equivalence: the report says which.
Adapted: GLM-5.2 compared EAGLE hidden_states / topk_p / topk_index; DSPARK's next draft input
carries only bonus tokens and seq lens, so the float comparison is on the target verify logits.
Eager is forced by detaching the target and draft decode graph runners (ModelRunner.forward and the
DSPARK proposer replay only when `decode_cuda_graph_runner` is set).
"""
import json
from pathlib import Path
import sys

import profile_decode as base
from topology import DecodeTopology


def compare(left, right):
    import torch
    row = {}
    for key in ('accept_lens', 'new_seq_lens', 'next_token_ids', 'bonus_tokens'):
        row[key + '_equal'] = torch.equal(left[key], right[key])
    a, b = left['logits'], right['logits']
    if a is None or b is None or a.shape != b.shape:
        row['logits_comparable'] = False
        return row
    row['logits_comparable'] = True
    row['logits_finite'] = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    row['logits_max_abs_diff'] = (a - b).abs().max().item()
    row['logits_relative_l2'] = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    row['argmax_equal'] = torch.equal(a.argmax(-1), b.argmax(-1))
    return row


def rank_main(rank, gpu_id, server_args, port_args, args):
    import torch
    import torch.distributed as dist
    from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank
    from sglang.srt.utils import configure_logger
    publish(server_args, role="scheduler",
            ranks=SpawnRanks(world_rank=spawn_world_rank(server_args, tp_rank=rank, pp_rank=0), gpu_id=gpu_id))
    configure_logger(server_args, prefix=f" TP{rank}")
    base.configure_acceptance(args)
    torch.manual_seed(args.seed)
    topology = DecodeTopology(args.tp_size, args.ep_size, args.batch_size, args.enable_dp_attention)
    engine = base.Engine(server_args, port_args, gpu_id, rank, args, topology)
    sync = engine.sync_fn()
    owners = [engine.target_runner, engine.draft_runner]
    graphs = [owner.decode_cuda_graph_runner for owner in owners]
    outputs = {}
    with torch.inference_mode():
        for mode in ['graph', 'graph_repeat', 'eager', 'eager_repeat']:
            for owner, graph in zip(owners, graphs):
                owner.decode_cuda_graph_runner = graph if mode.startswith('graph') else None
            batch, _ = base.bootstrap(engine, rank)
            base.reset_acceptance_rng(engine, args.seed)
            torch.manual_seed(args.seed)
            outputs[mode] = []
            for _step in range(args.path_steps):
                result, _prev, accept, new_lens = base.decode_step(engine, batch, sync)
                logits = getattr(result.logits_output, 'next_token_logits', None) if result.logits_output is not None else None
                outputs[mode].append({
                    'accept_lens': torch.tensor(accept), 'new_seq_lens': torch.tensor(new_lens),
                    'next_token_ids': result.next_token_ids.detach().cpu().clone(),
                    'bonus_tokens': result.next_draft_input.bonus_tokens.detach().cpu().clone(),
                    'logits': None if logits is None else logits.detach().float().cpu().clone(),
                    'target_graph': bool(result.can_run_cuda_graph),
                })
            base.release(engine, batch)
        for owner, graph in zip(owners, graphs):
            owner.decode_cuda_graph_runner = graph
        report = []
        for left, right in [('graph', 'graph_repeat'), ('eager', 'eager_repeat'), ('graph', 'eager')]:
            for step, (a, b) in enumerate(zip(outputs[left], outputs[right])):
                row = {'left': left, 'right': right, 'step': step,
                       'left_target_graph': a['target_graph'], 'right_target_graph': b['target_graph']}
                row.update(compare(a, b))
                report.append(row)
        Path(args.result_dir, f'path_check_rank_{rank}_yihou.json').write_text(json.dumps(report, indent=2) + '\n')
        base.log(rank, json.dumps(report))
    dist.barrier()


def main():
    parser = base.make_parser()
    parser.set_defaults(tp_size=4, ep_size=4, batch_size=16, input_len=1024, output_len=64)
    parser.add_argument('--path-steps', type=int, default=2)
    args, extra = parser.parse_known_args()
    base.validate_args(args)
    base.resolve_input_lens(args)
    base.configure_acceptance(args)
    import multiprocessing
    import os
    import time
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs
    from sglang.srt.utils import kill_process_tree, maybe_reindex_device_id
    topology = DecodeTopology(args.tp_size, args.ep_size, args.batch_size, args.enable_dp_attention)
    server_args, cfg, _cli = base.build_server_args(args, extra)
    base.check_server_args(cfg, topology)
    if args.prefill_tokens is None:
        args.prefill_tokens = int(cfg.chunked_prefill_size)
    args.resolved = {name: str(getattr(cfg, name, None)) for name in ("moe_a2a_backend", "moe_runner_backend")}
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
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
