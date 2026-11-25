"""vLLM in-process engine step bench: the real V1 EngineCore (scheduler + model executor + sampler) in this
process, stepped by hand, every step timed. No API server, no frontend process, no network.

One point = C requests of random token ids (uniform ISL, OSL, ignore_eos) added at once, then
LLMEngine.step() until all finish. A wrapper around scheduler.update_from_output() records what each step COMPLETED
(scheduled tokens, requests, how many still prefilling; correct under async scheduling / batch queue),
so the decode steady state is identified exactly:
  decode step  = every scheduled request already had its whole prompt computed before this step
  steady window = decode steps while all C requests are still running
Output tokens per step come from DELTA RequestOutputs, so speculative steps (k tokens/step) count right.

Topology: --tensor-parallel-size N runs N GPU workers (multiproc executor) under one in-process EngineCore.
DP > 1 is not driven here (use `vllm bench forward` via forward_bench.py, or the serving stack).

Outputs (in --result-dir): result.json, steps.jsonl, config.json.
Identities: steady_tok_s = steady output tokens / steady seconds; tok_s_per_gpu = steady_tok_s / (tp*pp);
tpot_ms (per request) = steady seconds * 1000 * C / steady output tokens.
"""

import json
import os
import sys
import time
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT_ROOT / "common"))
from redact import redact_env  # noqa: E402
from stats import summarize  # noqa: E402

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # EngineCore in this process (InprocClient)


def add_args(p):
    g = p.add_argument_group("engine step bench")
    g.add_argument("--result-dir", required=True)
    g.add_argument("--batch-size", type=int, default=16, help="requests C, all added at t=0")
    g.add_argument("--input-len", type=int, default=1024)
    g.add_argument("--output-len", type=int, default=64)
    g.add_argument("--warmup-requests", type=int, default=0, help="untimed warmup batch size (0 = same as C)")
    g.add_argument("--warmup-output-len", type=int, default=8)
    g.add_argument("--skip-warmup", action="store_true")
    g.add_argument("--data-seed", type=int, default=1234, help="seed of the random prompt ids (--seed is the engine's)")
    g.add_argument("--max-steps", type=int, default=0, help="stop after N steps (smoke; result marked incomplete)")
    g.add_argument("--profile-steps", default="", help="'A:B' -> torch.profiler over steps [A,B) (engine-core side)")


def classify(sched_out, computed, prompt_len):
    """Return (n_reqs, n_prefill_reqs, prefill_tokens, decode_tokens) and update computed tokens in place."""
    n_pref = pref_tok = dec_tok = 0
    for rid, n in sched_out.num_scheduled_tokens.items():
        before = computed.get(rid, 0)
        if before < prompt_len:
            n_pref += 1
            pref_tok += min(n, prompt_len - before)
            dec_tok += max(0, n - (prompt_len - before))
        else:
            dec_tok += n
        computed[rid] = before + n
    return len(sched_out.num_scheduled_tokens), n_pref, pref_tok, dec_tok


def steady_window(rows, c):
    return [r for r in rows if r["n_prefill_reqs"] == 0 and r["n_reqs"] == c and r["out_tokens"] > 0]


def _requests(n, isl, osl, seed, tag):
    import random

    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    rng = random.Random(seed)
    sp = SamplingParams(temperature=0.0, max_tokens=osl, ignore_eos=True, output_kind=RequestOutputKind.DELTA)
    return [(f"{tag}-{i}", {"prompt_token_ids": [rng.randrange(0, 10000) for _ in range(isl)]}, sp) for i in range(n)]


def run(eng, sched, reqs, isl, timed, max_steps=0, prof_range=None, out_dir=None):
    """Step the engine until the batch finishes. Each row describes the batch(es) a step COMPLETED.

    Attribution is by completion, not by scheduling: with async scheduling the EngineCore uses a batch queue
    (step_with_batch_queue), so a step can schedule batch i+1 and only return batch i (or return nothing).
    scheduler.update_from_output(scheduler_output, ...) is called exactly once per completed batch with that
    batch's SchedulerOutput, so wrapping it classifies the work a step finished. Time of steps that complete
    no batch is carried into the next completing step.
    """
    done = []
    orig_upd = sched.update_from_output

    def upd(scheduler_output, *a, **k):
        done.append(scheduler_output)
        return orig_upd(scheduler_output, *a, **k)

    sched.update_from_output = upd
    for rid, prompt, sp in reqs:
        eng.add_request(rid, prompt, sp)
    rows, computed, step, prof, carry = [], {}, 0, None, 0.0
    finished = set()
    t_start = time.perf_counter()
    try:
        while eng.has_unfinished_requests():
            if timed and prof_range and step == prof_range[0]:
                import torch
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA])
                prof.start()
            done.clear()
            t = time.perf_counter()
            outs = eng.step()
            dt = time.perf_counter() - t
            if prof is not None and step == prof_range[1] - 1:
                prof.stop()
                pd = Path(out_dir) / "profile"
                pd.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(pd / "engine_steps.trace.json.gz"))
                (pd / "top_ops.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=60))
                prof = None
            out_tok = sum(len(o.outputs[0].token_ids) for o in outs if o.outputs)
            for o in outs:
                if o.finished:
                    finished.add(o.request_id)
            step += 1
            if not done:
                carry += dt
                continue
            n_reqs = n_pref = pt = dtk = sched_tok = 0
            for so in done:
                a_, b_, c_, d_ = classify(so, computed, isl)
                n_reqs, n_pref, pt, dtk = n_reqs + a_, n_pref + b_, pt + c_, dtk + d_
                sched_tok += so.total_num_scheduled_tokens
            rows.append({"step": step - 1, "step_s": dt + carry, "carried_s": carry, "batches_done": len(done),
                         "scheduled_tokens": sched_tok, "n_reqs": n_reqs, "n_prefill_reqs": n_pref,
                         "prefill_tokens": pt, "decode_tokens": dtk, "out_tokens": out_tok,
                         "finished_total": len(finished)})
            carry = 0.0
            if max_steps and step >= max_steps:
                break
    finally:
        if max_steps and eng.has_unfinished_requests():
            eng.abort_request([r[0] for r in reqs])
            while eng.has_unfinished_requests():
                eng.step()
        drain(eng)
        sched.update_from_output = orig_upd
    return rows, time.perf_counter() - t_start


def drain(eng, limit=16):
    """Step until the EngineCore batch queue is empty. With async scheduling a batch can still be in flight when
    has_unfinished_requests() turns false; without this, the warmup's last batch completes inside the timed run
    (seen as a 1-request, 6-token 'prefill' row at step 0 of vllm_s4)."""
    core = getattr(getattr(eng, "engine_core", None), "engine_core", None)
    q = getattr(core, "batch_queue", None)
    n = 0
    while q and n < limit:
        eng.step()
        n += 1
    return n


def main(argv=None):
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    p = FlexibleArgumentParser(description=__doc__)
    add_args(p)
    EngineArgs.add_cli_args(p)
    a = p.parse_args(argv)
    out = Path(a.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    ea = EngineArgs.from_cli_args(a)
    if (ea.data_parallel_size or 1) > 1:
        raise SystemExit("data_parallel_size > 1 is not supported by this bench (see module doc)")
    if ea.max_num_seqs is None or ea.max_num_seqs < a.batch_size:
        ea.max_num_seqs = max(a.batch_size, a.warmup_requests or 0)
    (out / "config.json").write_text(json.dumps({"bench": {k: v for k, v in vars(a).items() if k in (
        "batch_size", "input_len", "output_len", "warmup_requests", "warmup_output_len", "skip_warmup", "data_seed",
        "max_steps", "profile_steps")}, "engine_args": redact_env({k: str(v) for k, v in vars(ea).items()})}, indent=2))
    t0 = time.perf_counter()
    llm = LLM.from_engine_args(ea)
    init_s = time.perf_counter() - t0
    eng = llm.llm_engine
    core = eng.engine_core.engine_core
    sched = core.scheduler
    batch_queue = getattr(core, "batch_queue_size", 1)
    if not a.skip_warmup:
        wn = a.warmup_requests or a.batch_size
        run(eng, sched, _requests(wn, a.input_len, max(2, a.warmup_output_len), a.data_seed + 1, "warm"), a.input_len, False)
    prof_range = tuple(int(x) for x in a.profile_steps.split(":")) if a.profile_steps else None
    rows, wall = run(eng, sched, _requests(a.batch_size, a.input_len, a.output_len, a.data_seed, "bench"),
                     a.input_len, True, a.max_steps, prof_range, out)
    with open(out / "steps.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    c = a.batch_size
    steady = steady_window(rows, c)
    st_s = sum(r["step_s"] for r in steady)
    st_tok = sum(r["out_tokens"] for r in steady)
    total_out = sum(r["out_tokens"] for r in rows)
    pc = ea.parallel_config if hasattr(ea, "parallel_config") else None
    ngpu = (ea.tensor_parallel_size or 1) * (ea.pipeline_parallel_size or 1)
    prefill_rows = [r for r in rows if r["n_prefill_reqs"] > 0]
    res = {
        "engine": "vllm", "tool": "engine_step_bench",
        "complete": (not a.max_steps) and total_out == c * a.output_len,
        "batch_size": c, "input_len": a.input_len, "output_len": a.output_len, "num_gpus": ngpu,
        "tp": ea.tensor_parallel_size, "pp": ea.pipeline_parallel_size,
        "speculative_config": str(ea.speculative_config) if ea.speculative_config else None,
        "init_s": init_s, "batch_queue_size": batch_queue, "wall_s": wall, "steps": len(rows), "output_tokens": total_out,
        "prefill_steps": len(prefill_rows), "prefill_s": sum(r["step_s"] for r in prefill_rows),
        "prefill_tok_s": (c * a.input_len / sum(r["step_s"] for r in prefill_rows)) if prefill_rows else None,
        "steady_steps": len(steady), "steady_s": st_s, "steady_output_tokens": st_tok,
        "steady_step_s": summarize([r["step_s"] for r in steady]),
        "tokens_per_req_step": (st_tok / (len(steady) * c)) if steady else None,
        "steady_tok_s": (st_tok / st_s) if st_s else None,
        "steady_tok_s_per_gpu": (st_tok / st_s / ngpu) if st_s else None,
        "tpot_ms": (st_s * 1000.0 * c / st_tok) if st_tok else None,
        "overall_output_tok_s": total_out / wall,
        "synthetic": {"input_ids": "uniform random in [0,10000)", "load_format": ea.load_format,
                      "sampling": "greedy, ignore_eos"},
    }
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
    print("[engine_step] " + json.dumps({k: res[k] for k in ("complete", "batch_size", "input_len", "output_len",
          "prefill_tok_s", "steady_steps", "tokens_per_req_step", "tpot_ms", "steady_tok_s", "steady_tok_s_per_gpu")}),
          flush=True)
    return 0 if res["complete"] or a.max_steps else 1


if __name__ == "__main__":
    sys.exit(main())
