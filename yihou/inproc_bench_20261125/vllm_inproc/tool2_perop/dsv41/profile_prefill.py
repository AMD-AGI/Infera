###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
# Provenance: port of AKLR `aklr/benchmarks/models/glm52/profile_prefill.py` @ 33689332 via the SGLang DSv4.1
# port, to vLLM. One request per DP rank (what chunked prefill produces), every chunk of its TTFT timed: the
# eager full forward (+ last-token logits) as the reference, every op in its own graph. Each chunk is one exact
# batch built by vLLM bench forward's builder: chunk i = `extend@prefix+extend` marked chunked (its prefix
# is the request's earlier chunks); with --prefix-cache-hit the first chunk's prefix is a cache hit, which on
# DSv4.1 recomputes the last SWA window (vLLM's scheduler does the same; bench forward models it).

"""Per-operator prefill timing for DeepSeek-V4.1 on vLLM, rolled up to a TTFT estimate."""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

import model_bench as mb  # noqa: E402
import vllm_bench as vb  # noqa: E402
from vllm_bench import table  # noqa: E402


class Step:
    def __init__(self, prefix, extend):
        self.prefix, self.extend = prefix, extend


class StepResult:
    def __init__(self, step, times, full_ms, counts, skipped, late_rows, gcalls):
        self.step, self.skipped, self.full_ms, self.late_rows = step, skipped, full_ms, late_rows
        mb.add_residuals(times, counts)
        self.times = times
        self.totals = mb.leaf_totals(times, counts, gcalls)
        self.rollup = sum(self.totals.values())


def plan_prefill(args, chunk: int, block: int) -> list[Step]:
    """The chunk timeline the scheduler walks for one request (prefix hit rounded down to the KV block)."""
    cached = int(args.ISL * args.prefix_cache_hit) // block * block
    steps, prefix = [], cached
    while prefix < args.ISL:
        extend = min(chunk, args.ISL - prefix)
        steps.append(Step(prefix, extend))
        prefix += extend
    if not steps:
        raise ValueError(f"--prefix-cache-hit {args.prefix_cache_hit} leaves nothing to prefill at --ISL {args.ISL}")
    return steps


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.speculative_config:
        # Refused rather than ignored (glm52): a drafter runs on a prefill chunk and none of it would be timed.
        raise SystemExit("profile_prefill does not model speculative decoding; drop --speculative-config")
    if args.result_dir:
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    mb.setup_env()
    chunk = args.max_num_batched_tokens or 16384
    args.chunk = chunk
    # +window: a prefix-hit first chunk also recomputes the last SWA window (128 on DSv4.1); the plan keeps `chunk`.
    ea = mb.build_engine_args(args, 1, chunk + 256)
    from vllm.config import set_current_vllm_config

    llm, runner, rank_print = mb.setup(ea)
    with torch.inference_mode(), set_current_vllm_config(runner.vllm_config):
        work(args, runner, rank_print)


def work(args, runner, rank_print):
    layers = mb.Layers(runner)
    rank_print(f"layers: {layers.describe()}")
    routing = mb.install_routing(layers.moe_pairs(), args.routing)
    chunk = args.chunk
    block = min(int(g.block_size or 1) for g in runner.kv_cache_config.kv_cache_groups)
    steps = plan_prefill(args, chunk, block)
    results = []
    for i, st in enumerate(steps):
        rank_print(f"chunk {i + 1}/{len(steps)}: {st.extend} new on {st.prefix}")
        first_hit = i == 0 and st.prefix > 0
        case = mb.prefill_case(f"chunk{i}", 1, st.prefix, st.extend, chunked=not first_hit)
        try:
            batch, case = mb.build(runner, case, args.seed or 0, 0.0, drafter=False)
        except mb.Capacity as e:
            raise SystemExit(f"[BLOCKED_CAPACITY] prefill ISL {args.ISL}: {e}")
        mb.fill_kv(runner, batch, args.kv_fill)
        res = mb.run_inside_forward(runner, batch, lambda f, st=st: measure_step(args, f, layers, st, rank_print))
        results.append(res)
    if routing is not None:
        routing.remove()
    if vb.world_rank() != 0:
        return
    md, headline = report(args, runner, layers, results)
    path = Path(args.out or Path(args.result_dir or ".") / default_out(args, runner, chunk))
    path.write_text(md)
    path.with_suffix(".json").write_text(json.dumps({"ttft_ms": ttft_ms(results), "chunks": [
        {"prefix": r.step.prefix, "extend": r.step.extend, "ms": r.full_ms, "rollup": r.rollup} for r in results]}, indent=1))
    rank_print(headline)
    rank_print(f"report -> {path.resolve()}")


def measure_step(args, f, layers, st, rank_print) -> StepResult:
    runner = f.runner
    ops = mb.collect_ops(layers, runner)
    names = mb.op_names(ops)
    saved, calls = {}, {}
    with mb.record_inputs(ops, saved, calls):
        f.step()
    torch.cuda.synchronize()
    vb.assert_lockstep(saved)
    # Eager, as the server runs a prefill chunk: the reference the roll-up is checked against, and TTFT's sum.
    rank_print("  timing: full forward (eager)")
    full_ms = vb.bench_eager(f.step, args.reps, rounds=args.repeat)
    inner = vb.graph_reps_for(st.extend)
    times, skipped = {}, {}
    recorded = set(saved)
    late_rows = int(f.ctx[0].batch_descriptor.num_tokens) if getattr(f.ctx[0], "batch_descriptor", None) else None
    inline = mb.engram_inline(runner)
    inline.__enter__()
    for name in list(saved):
        rank_print(f"  timing: {name}")
        entry = saved.pop(name)  # popped: a recorded entry pins chunk-sized tensors
        try:
            times[name] = vb.bench(mb.make_call(entry), args.reps, inner=inner, rounds=args.repeat)
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, AssertionError, ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
            first = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
            if "out of memory" in first.lower():
                raise
            skipped[name] = first[:300]
            rank_print(f"    skipped: {first}")
            torch.cuda.synchronize()
        del entry
        if not vb.agreed(name in times):
            times.pop(name, None)
            skipped.setdefault(name, "skipped on another rank")
    inline.__exit__(None, None, None)
    reduce = "max" if mb.moe_spans_ranks(runner) else "min"
    times = vb.world_times({**times, "__full": full_ms}, reduce)
    full_ms = times.pop("__full")
    r = StepResult(st, times, full_ms, layers.counts, skipped, late_rows, {n: calls.get(n, 1) for n in mb.GLOBAL_OPS})
    r.coverage = mb.coverage_note(names, recorded)
    return r


def ttft_ms(results) -> float:
    return sum(r.full_ms for r in results)


def default_out(args, runner, chunk) -> str:
    parts = [mb.model_tag(runner.model_config.model), "prefill", f"isl{args.ISL}", f"chunk{chunk}",
             f"hit{int(args.prefix_cache_hit * 100)}", *mb.parallel_tags(runner), str(runner.cache_config.cache_dtype)]
    parts += filter(None, [vb.load_tag(runner.load_config.load_format)])
    parts.append(time.strftime("%Y%m%d"))
    return "_".join(parts) + ".md"


def summary_table(args, results) -> str:
    ttft = ttft_ms(results)
    new = sum(r.step.extend for r in results)
    rollup = sum(r.rollup for r in results)
    rows = [
        ("TTFT", f"{ttft:.1f} ms", f"{len(results)} chunks, one request per DP rank"),
        ("roll-up", f"{rollup:.1f} ms", "sum of the per-op leaf rows below"),
        ("ratio", f"{rollup / ttft:.3f}", "roll-up / TTFT; forward eager, ops in graphs (launch gaps land as shortfall)"),
        ("computed", f"{new:,} tokens", f"of {args.ISL:,}; the other {args.ISL - new:,} came from cache"),
        ("prompt rate", f"{args.ISL / ttft * 1000:,.0f} tok/s", "whole prompt over TTFT, per request"),
        ("compute rate", f"{new / ttft * 1000:,.0f} tok/s", "only the tokens forwarded"),
    ]
    return table(rows, ("metric", "value", "note"))


def config_table(args, runner, layers) -> str:
    hf = runner.model_config.hf_text_config
    rows = [
        ("model", f"`{type(runner.model).__name__}`  {runner.model_config.model}"),
        ("layers", layers.describe()),
        ("parallel", mb.parallel_line(runner)),
        ("context", f"ISL {args.ISL:,} per request"),
        ("prefix cache", f"{args.prefix_cache_hit:.0%} hit, rounded down to the KV block"),
        ("chunk", f"--max-num-batched-tokens {args.chunk:,} per DP rank (each rank schedules its own; engine budget +256 for a prefix-hit window)"),
        ("indexer", f"top {getattr(hf, 'index_topk', 0)} compressed positions per query"),
        ("kv dtype", f"`{runner.cache_config.cache_dtype}`"),
        ("weights", mb.weights_line(runner)),
        ("routing", "seeded balanced (AKLR draw)" if args.routing == "balanced" else "native"),
        ("measurement", f"{args.reps} graph replays per op; the full forward eager, as the server runs a chunk"
                        + (f", fastest of {args.repeat} passes" if args.repeat > 1 else "")),
    ]
    return table(rows, ("item", "value"))


def timeline_table(results) -> str:
    rows = [(i + 1, f"{r.step.prefix:,}", f"{r.step.extend:,}", r.late_rows if r.late_rows is not None else "",
             r.full_ms, r.rollup, f"{r.rollup / r.full_ms:.3f}") for i, r in enumerate(results)]
    return table(rows, ("chunk", "prefix", "new", "padded tokens", "ms", "roll-up", "ratio"), fmt=".2f") + (
        "\n\n`prefix` is what is already in the KV cache when the chunk starts (a cache hit and the earlier chunks"
        " are the same thing here). Layers past the decoder SWA bounded-replay cut (`+late` kinds) run on each"
        " request's last window only in eager prefill; their rows already carry that shape.")


def growth_table(results) -> str:
    totals = {}
    for r in results:
        for name, ms in r.totals.items():
            totals[name] = totals.get(name, 0.0) + ms
    names = sorted(totals, key=lambda n: -totals[n])
    rollup = sum(totals.values())
    headers = ("op",) + tuple(f"{r.step.prefix:,}" for r in results) + ("total", "share")
    rows = [(mb.tree_name(n, 0),) + tuple(r.totals.get(n, 0.0) for r in results) + (totals[n], f"{totals[n] / rollup * 100:.1f}%")
            for n in names]
    return table(rows, headers, fmt=".2f") + (
        "\n\nms per chunk, columns labelled by the prefix each chunk starts on. Leaf rows only; a column sums to the"
        " chunk's roll-up. Rows that climb across the table read the context; flat rows only see the chunk.")


def kind_sections(result, counts) -> str:
    out = []
    for kind, n in counts.items():
        rows = [(mb.tree_name(name, d), result.times[f"{kind}|{name}"], result.times[f"{kind}|{name}"] * n)
                for name, d in mb.TREE if f"{kind}|{name}" in result.times]
        out.append(f"### {kind} -- x{n} layers\n\n" + table(rows, ("op", "ms/call", "ms/chunk")))
    return "\n\n".join(out)


def report(args, runner, layers, results):
    last = results[-1]
    sections = [
        ("Summary", summary_table(args, results)),
        ("Config", config_table(args, runner, layers)),
        ("Versions", table(vb.versions(), ("component", "version"))),
        ("Memory, per rank", mb.memory_section(runner)),
        ("Chunk timeline", timeline_table(results)),
        ("Hotspots -- one TTFT, all layers", growth_table(results)),
        (f"Per layer kind -- deepest chunk, prefix {last.step.prefix:,}", kind_sections(last, layers.counts)),
        *([("Coverage", last.coverage)] if last.coverage else []),
        *([("Not replayable", note)] if (note := mb.skipped_note(vb.merged_skips(results))) else []),
    ]
    md = "\n\n".join([f"# {mb.model_tag(runner.model_config.model)} prefill profile (vLLM)", time.strftime("%Y-%m-%d %H:%M %Z"),
                      *filter(None, [mb.dummy_banner(runner.load_config.load_format)])]
                     + [f"## {t}\n\n{b}" for t, b in sections])
    ttft = ttft_ms(results)
    new = sum(r.step.extend for r in results)
    headline = (f"TTFT {ttft:.1f} ms over {len(results)} chunks  {args.ISL / ttft * 1000:,.0f} tok/s prompt, "
                f"{new / ttft * 1000:,.0f} tok/s computed  ratio {sum(r.rollup for r in results) / ttft:.3f}")
    return md + "\n", headline


def make_parser():
    parser = mb.make_parser()
    parser.add_argument("--prefix-cache-hit", type=float, default=0.0, help="fraction of ISL already in the KV cache, 0..1")
    return parser


if __name__ == "__main__":
    main()
