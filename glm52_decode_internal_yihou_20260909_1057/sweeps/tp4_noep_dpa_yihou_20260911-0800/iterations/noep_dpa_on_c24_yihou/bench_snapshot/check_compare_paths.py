#!/usr/bin/env python3
"""Bounded graph/eager differential check using identical fake initial state."""
import argparse
import json
from pathlib import Path

import compare_server as compare
import profile_decode as base
from topology import prepare_dp_metadata


def rank_main(rank, server_args, port_args, args):
    import torch
    import torch.distributed as dist
    base.configure_acceptance(args)
    torch.manual_seed(args.seed)
    target, worker = base.create_workers(server_args, port_args, rank, args)
    owners = [(target.model_runner, 'decode_cuda_graph_runner'),
              (worker.draft_worker, 'cuda_graph_runner'),
              (worker.draft_worker, 'cuda_graph_runner_for_draft_extend')]
    runners = [getattr(owner, key) for owner, key in owners]
    outputs = {}
    with torch.inference_mode():
        for mode in ['graph', 'graph_repeat', 'eager', 'eager_repeat']:
            for (owner, key), runner in zip(owners, runners):
                setattr(owner, key, runner if mode.startswith('graph') else None)
            batch, topology, _ = compare.initialize_state(target, worker, args, args.batch_size, rank)
            torch.manual_seed(args.seed)
            outputs[mode] = []
            for step in range(2):
                batch.prepare_for_decode()
                prepare_dp_metadata(batch, topology, is_extend=False, disable_cuda_graph=mode.startswith('eager'))
                result = worker.forward_batch_generation(batch)
                torch.cuda.synchronize()
                info = result.next_draft_input
                outputs[mode].append({
                    'accept_lens': result.accept_lens.detach().cpu().clone(),
                    'new_seq_lens': result.new_seq_lens.detach().cpu().clone(),
                    'bonus_tokens': info.bonus_tokens.detach().cpu().clone(),
                    'hidden_states': info.hidden_states.detach().float().cpu().clone(),
                    'topk_p': info.topk_p.detach().float().cpu().clone(),
                    'topk_index': info.topk_index.detach().cpu().clone(),
                    'target_graph': bool(result.can_run_cuda_graph),
                })
                base.commit_result(batch, result, result.new_seq_lens.cpu().tolist())
        report = []
        for left, right in [('graph', 'graph_repeat'), ('eager', 'eager_repeat'), ('graph', 'eager')]:
            for step, (graph, eager) in enumerate(zip(outputs[left], outputs[right])):
                row = {'left': left, 'right': right, 'step': step,
                       'left_target_graph': graph['target_graph'], 'right_target_graph': eager['target_graph']}
                for key in ['accept_lens', 'new_seq_lens', 'bonus_tokens', 'topk_index']:
                    row[key + '_equal'] = torch.equal(graph[key], eager[key])
                for key in ['hidden_states', 'topk_p']:
                    a, b = graph[key], eager[key]
                    row[key + '_finite'] = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                    row[key + '_max_abs_diff'] = (a-b).abs().max().item()
                    row[key + '_relative_l2'] = ((a-b).norm() / b.norm().clamp_min(1e-12)).item()
                report.append(row)
        Path(args.result_dir, f'path_check_rank_{rank}_yihou.json').write_text(json.dumps(report, indent=2)+'\n')
        base.log(rank, json.dumps(report))
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = base.make_parser()
    parser.set_defaults(tp_size=4,ep_size=4,batch_size=16,input_len=10000,output_len=500)
    args, extra = parser.parse_known_args()
    args.initial_state = 'fake-server'
    base.configure_acceptance(args)
    import torch.multiprocessing as mp
    from sglang.srt.server_args import ServerArgs, PortArgs
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    p = argparse.ArgumentParser(); ServerArgs.add_cli_args(p)
    sa = ServerArgs.from_cli_args(p.parse_args(base.server_cli(args,extra)))
    Path(args.result_dir).mkdir(parents=True,exist_ok=True)
    _set_envs_and_config(sa)
    mp.spawn(rank_main,args=(sa,PortArgs.init_new(sa),args),nprocs=args.tp_size,join=True)


if __name__ == '__main__':
    main()
