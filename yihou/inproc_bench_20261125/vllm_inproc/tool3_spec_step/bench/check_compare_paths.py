#!/usr/bin/env python3
"""Bounded graph/eager differential check from an identical initial state (vLLM port of GLM-5.2's).

Modes graph / graph_repeat / eager / eager_repeat. Each mode re-fills the batch's KV with the same seed, re-admits
the same requests (same prompts, same per-request sampler seeds) and runs `--compare-steps` verify steps. Eager is
forced by calling the engine's own CUDA-graph dispatcher with need_eager=True (target and DSpark drafter), so the
same runner and kernels-by-shape are used, only graph replay is switched off.
Compared per step: accept lengths and committed context (exact), sampled token ids (exact), drafted token ids
(exact). Execution success is not numerical equivalence; mismatches are reported, not explained.
"""
import json
from pathlib import Path

import profile_decode as base
from batch_state import accept_bounds, unconditional_to_conditional


def make_parser():
    parser = base.make_parser()
    parser.add_argument('--compare-steps', type=int, default=2)
    return parser


def force_eager(enabled):
    import vllm.v1.worker.gpu.model_runner as runner_module
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as draft_module
    for module in (runner_module, draft_module):
        original = getattr(module, '_bench_original_dispatch', None) or module.dispatch_cg_and_sync_dp
        module._bench_original_dispatch = original
        if enabled:
            def eager(*args, _original=original, **kwargs):
                kwargs['need_eager'] = True
                return _original(*args, **kwargs)
            module.dispatch_cg_and_sync_dp = eager
        else:
            module.dispatch_cg_and_sync_dp = original


def run_rank(rank, args, cli, port):
    import os
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(base.topology_of(args).world_size),
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), VLLM_ENABLE_V1_MULTIPROCESSING='0')
    import numpy as np
    import torch
    from vllm.distributed.parallel_state import get_world_group
    engine, worker, runner = base.create_engine(args, rank, cli)
    group = get_world_group().cpu_group
    topology = base.topology_of(args)
    spec = base.resolved_engine_summary(runner, worker)['speculative']
    min_accept, max_accept = accept_bounds(unconditional_to_conditional(spec['synthetic_acceptance_rates']))
    K = runner.num_speculative_steps
    counts = base.count_graph_executions({'target': runner.cudagraph_manager,
                                          'draft': getattr(runner.speculator, 'query_cudagraph_manager', None)})
    outputs, previous_ids = {}, []
    with torch.inference_mode():
        plan = base.BatchPlan(runner, args, topology, rank, min_accept, max_accept)
        vocab = runner.model_config.get_vocab_size()
        rng = np.random.default_rng(args.seed)
        prompts = [rng.integers(0, vocab, isl, dtype=np.int64).tolist() for isl in args.input_lens]
        sampling = [base.default_sampling_params(runner, args, t) for t in plan.reserved_tokens]
        for epoch, mode in enumerate(('graph', 'graph_repeat', 'eager', 'eager_repeat')):
            force_eager(mode.startswith('eager'))
            base.initialize_kv(runner, plan, args.kv_init, args.seed)
            req_ids = base.bootstrap(worker, runner, plan, args, 200 + epoch, prompts, sampling, finished=previous_ids)
            before = dict(counts)
            computed, emitted, steps = list(args.input_lens), [0] * len(req_ids), []
            for _ in range(args.compare_steps):
                tokens = []
                accept, _ = base.run_step(worker, base.decode_output(req_ids, computed, emitted, args), req_ids, tokens)
                states = runner.req_states
                index = [states.req_id_to_index[r] for r in req_ids]
                drafts = states.draft_tokens[index].cpu().tolist()
                computed = base.worker_computed(runner, req_ids)
                emitted = [e + a for e, a in zip(emitted, accept)]
                steps.append({'accept': accept, 'computed': computed, 'sampled': tokens, 'drafts': drafts})
            outputs[mode] = {'steps': steps, 'graph_replays': {k: counts[k] - before[k] for k in counts}}
            previous_ids = req_ids
        force_eager(False)
    result = {}
    reference = outputs['graph']['steps']
    for mode, data in outputs.items():
        rows = []
        for index, (step, ref) in enumerate(zip(data['steps'], reference)):
            rows.append({key: step[key] == ref[key] for key in ('accept', 'computed', 'sampled', 'drafts')}
                        | {'step': index})
        result[mode] = {'equal_to_graph': rows, 'graph_replays': data['graph_replays']}
    reports = [None] * topology.world_size
    torch.distributed.all_gather_object(reports, {'rank': rank, 'result': result}, group=group)
    if rank == 0:
        (Path(args.result_dir) / 'compare_paths_yihou.json').write_text(json.dumps(
            {'by_rank': reports, 'note': 'equality vs the first graph run; eager modes must show 0 graph replays'},
            indent=2) + '\n')
        base.log(rank, json.dumps(result))
    torch.distributed.barrier(group=group)
    os._exit(0)


def main():
    args, extra = make_parser().parse_known_args()
    base.validate_args(args)
    base.resolve_input_lens(args)
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
