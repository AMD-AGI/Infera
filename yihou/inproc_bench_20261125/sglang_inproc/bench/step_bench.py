"""SGLang in-process step bench: drive the engine's own ModelRunner, no Scheduler, no HTTP.

One point = a fixed batch of C requests with uniform ISL / OSL:
  real prefill (optionally in groups that are merged with ScheduleBatch.merge_batch) ->
  OSL-1 plain decode steps, each one timed with a device sync on both sides.
Model loading, memory pools, attention backends and CUDA graphs come from the pinned
`sglang.benchmark.one_batch.load_model` (same objects the server's TpModelWorker builds).

Topology: --tp-size N spawns N rank processes. With --enable-dp-attention, dp = --dp-size and
--batch-size is the GLOBAL concurrency C; every DP rank runs C/dp requests (the same multiset).
Speculative decoding (DSPARK/EAGLE) is NOT driven here; any --speculative-* flag is rejected.

Outputs (in --result-dir): result.json (rank 0, global aggregate), rank_<r>.json, steps.jsonl
(rank 0, per decode step), config.json, profile/ (with --profile). Exit 0 only if every rank finished.

Metric identities: decode_tok_s = C / median_step_s (one token per request per step);
tok_s_per_gpu = decode_tok_s / tp_size; tpot_ms = median_step_s * 1000.
"""

import argparse
import dataclasses
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT_ROOT / "common"))
from redact import redact_argv  # noqa: E402
from stats import summarize  # noqa: E402

OWNED_FLAGS = ("--cuda-graph-max-bs-decode", "--cuda-graph-bs-decode")
REJECTED_PREFIXES = ("--speculative",)


@dataclasses.dataclass
class StepArgs:
    result_dir: str
    batch_size: int = 16
    input_len: int = 1024
    output_len: int = 64
    prefill_group: int = 0  # 0 = whole local batch in one extend
    warmup_output_len: int = 8
    skip_warmup: bool = False
    data_seed: int = 1234
    profile: bool = False
    profile_start_step: int = 4
    profile_num_steps: int = 4
    profile_ranks: str = "0"

    @staticmethod
    def add_cli_args(p: argparse.ArgumentParser):
        g = p.add_argument_group("step bench")
        g.add_argument("--result-dir", required=True)
        g.add_argument("--batch-size", type=int, default=16, help="GLOBAL concurrency C (per DP rank: C/dp)")
        g.add_argument("--input-len", type=int, default=1024)
        g.add_argument("--output-len", type=int, default=64, help="tokens per request; OSL-1 decode steps are timed")
        g.add_argument("--prefill-group", type=int, default=0, help="requests per extend call (0 = all)")
        g.add_argument("--warmup-output-len", type=int, default=8)
        g.add_argument("--skip-warmup", action="store_true")
        g.add_argument("--data-seed", type=int, default=1234, help="seed of the random prompt ids")
        g.add_argument("--profile", action="store_true", help="torch.profiler window over decode steps")
        g.add_argument("--profile-start-step", type=int, default=4)
        g.add_argument("--profile-num-steps", type=int, default=4)
        g.add_argument("--profile-ranks", default="0", help="comma list or 'all'")

    @classmethod
    def from_cli_args(cls, a):
        return cls(**{f.name: getattr(a, f.name) for f in dataclasses.fields(cls)})


def check_argv(argv):
    for a in argv:
        flag = a.split("=", 1)[0]
        if flag in OWNED_FLAGS:
            raise SystemExit(f"{flag} is set by the bench (= local batch); do not pass it")
        if flag.startswith(REJECTED_PREFIXES):
            raise SystemExit(f"{flag}: speculative decoding is not driven by this bench (plain decode only)")


def local_batch(c, dp):
    if c % dp:
        raise SystemExit(f"--batch-size {c} must be divisible by dp={dp}")
    return c // dp


def _sync(dev):
    import torch
    torch.cuda.synchronize(dev)


def _make_reqs(local_bs, isl, osl, seed, rank_offset=0):
    from array import array

    import numpy as np
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 10000, (local_bs, isl), dtype=np.int64)
    sp = SamplingParams(temperature=0, max_new_tokens=osl)
    reqs = []
    for i in range(local_bs):
        r = Req(rid=f"bench-{rank_offset + i}", origin_input_text="", origin_input_ids=array("q", ids[i].tolist()), sampling_params=sp)
        r.full_untruncated_fill_ids = r.origin_input_ids
        r.logprob_start_len = -1
        r.extend_end = len(r.origin_input_ids)
        reqs.append(r)
    return reqs


def _prefill(runner, reqs, group):
    import torch
    group = group or len(reqs)
    batch, toks = None, []
    for s in range(0, len(reqs), group):
        nt, _, b = runner.extend(reqs[s:s + group])
        toks.append(nt)
        if batch is None:
            batch = b
        else:
            batch.merge_batch(b)
    return torch.cat(toks), batch


def _captured_decode_bs(mr):
    """Decode CUDA-graph batch sizes of this ModelRunner, or None if they cannot be found (version-dependent)."""
    for name in ("graph_runner", "decode_cuda_graph_runner", "cuda_graph_runner"):
        gr = getattr(mr, name, None)
        for attr in ("capture_bs", "capture_batch_sizes"):
            v = getattr(gr, attr, None) if gr is not None else None
            if v:
                return sorted(int(x) for x in v)
    return None


def _count_graph_replays(mr):
    """Count decode CUDA-graph executions during the timed loop.

    sglang 6be0d75ae: ModelRunner.forward runs `decode_cuda_graph_runner.execute(...)` only when
    `can_run_graph(forward_batch)` is true; otherwise it falls back to eager. When no graph was captured the
    attribute may point at `mr.eager_runner` -> report 0. Returns None if the layout is not recognised.
    """
    gr = getattr(mr, "decode_cuda_graph_runner", None)
    if gr is None:
        return None
    if gr is getattr(mr, "eager_runner", object()):
        return {"n": 0}
    name = "execute" if hasattr(gr, "execute") else ("replay" if hasattr(gr, "replay") else None)
    if name is None:
        return None
    box = {"n": 0}
    orig = getattr(gr, name)

    def wrapped(*a, **k):
        box["n"] += 1
        return orig(*a, **k)

    setattr(gr, name, wrapped)
    return box


def run_point(runner, dev, a, local_bs, isl, osl, timed, rank, out_dir, profile_this_rank):
    runner.clear()
    reqs = _make_reqs(local_bs, isl, osl, a.data_seed)
    _sync(dev)
    t0 = time.perf_counter()
    next_ids, batch = _prefill(runner, reqs, a.prefill_group)
    _sync(dev)
    prefill_s = time.perf_counter() - t0
    steps, prof = [], None
    for i in range(osl - 1):
        if timed and profile_this_rank and i == a.profile_start_step:
            import torch
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=False)
            prof.start()
        _sync(dev)
        t = time.perf_counter()
        next_ids, _ = runner.decode(next_ids, batch)
        _sync(dev)
        steps.append(time.perf_counter() - t)
        if prof is not None and i == a.profile_start_step + a.profile_num_steps - 1:
            prof.stop()
            pdir = Path(out_dir) / "profile"
            pdir.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(pdir / f"decode_rank{rank}.trace.json.gz"))
            (pdir / f"decode_rank{rank}.top_ops.txt").write_text(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=60))
            prof = None
    runner.cleanup(batch)
    return prefill_s, steps


def rank_main(server_args, port_args, a, gpu_id, tp_rank, dp, local_bs):
    from sglang.benchmark import one_batch as ob
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank
    from sglang.srt.utils import configure_logger

    publish(server_args, role="scheduler",
            ranks=SpawnRanks(world_rank=spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0), gpu_id=gpu_id))
    ob._init_process_global_configs()
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    cfg = resolving_view(server_args)
    # dp / local_bs come from main(), which sized the decode CUDA graph from the raw ServerArgs. Do not
    # recompute them here from the resolved view: on sglang 6be0d75ae that view reported dp=1 under DPA, the ranks
    # then ran C requests each (> captured graph bs) and decode silently went eager (81 ms/step, smoke2).
    t_load = time.perf_counter()
    runner, _tok = ob.load_model(server_args, port_args, gpu_id, tp_rank)
    load_s = time.perf_counter() - t_load
    mr = runner.torch_runner
    dev = mr.device
    graph_bs = _captured_decode_bs(mr)
    need = local_bs * (a.input_len + a.output_len)
    cap = mr.max_total_num_tokens
    out = Path(a.result_dir)
    rec = {"rank": tp_rank, "gpu_id": gpu_id, "local_batch": local_bs, "dp": dp, "captured_decode_bs": graph_bs,
           "decode_graph_covers_batch": (graph_bs is None) or (local_bs <= max(graph_bs)), "max_total_num_tokens": cap,
           "tokens_needed": need, "load_capture_s": load_s}
    if need > cap:
        rec["error"] = f"KV capacity {cap} < needed {need}"
        (out / f"rank_{tp_rank}.json").write_text(json.dumps(rec, indent=2))
        raise SystemExit(rec["error"])
    prof_ranks = None if a.profile_ranks == "all" else {int(x) for x in a.profile_ranks.split(",")}
    profile_this = a.profile and (prof_ranks is None or tp_rank in prof_ranks)
    if not a.skip_warmup:
        t = time.perf_counter()
        run_point(runner, dev, a, local_bs, a.input_len, max(2, a.warmup_output_len), False, tp_rank, out, False)
        rec["warmup_s"] = time.perf_counter() - t
    replays = _count_graph_replays(mr)
    prefill_s, steps = run_point(runner, dev, a, local_bs, a.input_len, a.output_len, True, tp_rank, out, profile_this)
    rec["decode_graph_replays_timed"] = replays["n"] if replays else None
    rec.update({"prefill_s": prefill_s, "decode_steps": len(steps), "decode_s_total": sum(steps),
                "decode_step_s": summarize(steps), "complete": len(steps) == a.output_len - 1})
    (out / f"rank_{tp_rank}.json").write_text(json.dumps(rec, indent=2))
    import torch.distributed as dist
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    gathered = [None] * world
    payload = {"rank": tp_rank, "prefill_s": prefill_s, "steps": steps, "complete": rec["complete"],
               "graph_ok": rec["decode_graph_covers_batch"], "local_batch": local_bs,
               "replays": rec["decode_graph_replays_timed"]}
    if world > 1:
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    if tp_rank == 0:
        n = min(len(g["steps"]) for g in gathered)
        step_max = [max(g["steps"][i] for g in gathered) for i in range(n)]  # a step ends when the slowest rank ends
        with open(out / "steps.jsonl", "w") as f:
            for i, s in enumerate(step_max):
                f.write(json.dumps({"step": i, "step_s": s, "per_rank_s": [g["steps"][i] for g in gathered]}) + "\n")
        st = summarize(step_max)
        prefill_max = max(g["prefill_s"] for g in gathered)
        c = a.batch_size
        result = {
            "engine": "sglang", "tool": "step_bench", "complete": all(g["complete"] for g in gathered),
            "decode_graph_covers_batch": [g["graph_ok"] for g in gathered],
            "local_batch_per_rank": [g["local_batch"] for g in gathered],
            "decode_graph_replays_per_rank": [g["replays"] for g in gathered],
            "batch_size_global": c, "local_batch": local_bs, "dp": dp, "tp_size": cfg.tp_size,
            "ep_size": cfg.ep_size, "num_gpus": cfg.tp_size, "input_len": a.input_len, "output_len": a.output_len,
            "prefill_group": a.prefill_group or local_bs,
            "prefill_s": prefill_max, "prefill_tok_s": c * a.input_len / prefill_max,
            "decode_step_s": st, "tpot_ms": st["median"] * 1000.0,
            "decode_tok_s": c / st["median"], "decode_tok_s_per_gpu": c / st["median"] / cfg.tp_size,
            "load_capture_s": load_s, "speculative": None,
            "synthetic": {"input_ids": "uniform random in [0,10000)", "weights": cfg.load_format,
                          "sampling": "greedy", "acceptance": "n/a (plain decode)"},
        }
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print("[step_bench] " + json.dumps({k: result[k] for k in (
            "complete", "batch_size_global", "input_len", "output_len", "prefill_tok_s", "tpot_ms",
            "decode_tok_s", "decode_tok_s_per_gpu")}), flush=True)
    if cfg.tp_size > 1:
        from sglang.srt.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
        destroy_model_parallel()
        destroy_distributed_environment()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    check_argv(argv)
    from sglang.benchmark import one_batch as ob
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.arg_groups.overrides import declare_resolution
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, Phase
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import kill_process_tree, maybe_reindex_device_id

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ServerArgs.add_cli_args(p)
    StepArgs.add_cli_args(p)
    ns = p.parse_args(argv)
    server_args = ServerArgs.from_cli_args(ns)
    a = StepArgs.from_cli_args(ns)
    Path(a.result_dir).mkdir(parents=True, exist_ok=True)
    dp = server_args.dp_size if server_args.enable_dp_attention else 1
    lbs = local_batch(a.batch_size, dp)
    # Same mechanism as one_batch.main: the decode graph must cover exactly the local batch.
    explicit = server_args.cuda_graph_config
    if isinstance(explicit, CudaGraphConfig):
        explicit = explicit.to_dict()
    gcfg = dict(explicit or {})
    dec = dict(gcfg.get(Phase.DECODE) or {})
    dec["max_bs"] = lbs
    gcfg[Phase.DECODE] = dec
    declare_resolution(server_args, "inproc_bench.sglang.step_bench", cuda_graph_config=gcfg)
    server_args.resolve_once()
    cfg = resolving_view(server_args)
    _set_envs_and_config(server_args)
    (Path(a.result_dir) / "config.json").write_text(json.dumps(
        {"step_args": dataclasses.asdict(a), "argv": redact_argv(argv), "local_batch": lbs, "dp": dp}, indent=2, default=str))
    port_args = PortArgs.init_new(server_args)
    try:
        if cfg.tp_size == 1:
            rank_main(server_args, port_args, a, 0, 0, dp, lbs)
            return 0
        procs = []
        for r in range(cfg.tp_size):
            with maybe_reindex_device_id(r) as gpu_id:
                pr = multiprocessing.Process(target=rank_main, args=(server_args, port_args, a, gpu_id, r, dp, lbs))
                pr.start()
                procs.append(pr)
        for pr in procs:
            pr.join()
        bad = [pr.exitcode for pr in procs if pr.exitcode != 0]
        return 1 if bad else 0
    finally:
        if cfg.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    sys.exit(main())
