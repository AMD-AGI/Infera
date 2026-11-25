"""Per-operator prefill timing for GLM-5.2, rolled up to a TTFT estimate."""

import sys
import time
from pathlib import Path

import torch

# sglang_bench.py and kernel_breakdown.py live one directory up.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import model_bench
import sglang_bench


class StepResult:
    """One chunk's measurements, kept together so the report can walk them."""

    def __init__(self, step, times, full_ms, counts, path, contents, skipped=None):
        self.step = step
        self.skipped = skipped or {}
        self.full_ms = full_ms
        self.path = path
        model_bench.add_residuals(times, counts)
        self.times = times
        self.totals = model_bench.leaf_totals(times, counts)
        self.rollup = sum(self.totals.values())
        self.contents = contents


def measure_step(args, mr, ops, context, step, counts, rank_print) -> StepResult:
    """Time every op for one chunk, plus the full forward it is checked against."""
    batch = model_bench.make_prefill_batch(mr, context, step)
    fb = sglang_bench.forward_batch_of(batch, mr)

    saved = {}
    with model_bench.record_inputs(ops, saved):
        mr.forward(fb)
    torch.cuda.synchronize()
    sglang_bench.assert_lockstep(saved)
    path = model_bench.attn_path(mr)

    # Run eagerly, as the server runs a chunk: this is what the roll-up is checked against and TTFT is summed from.
    rank_print("  timing: full forward")
    full_ms = sglang_bench.bench_eager(
        lambda: mr.model.forward(fb.input_ids, fb.positions, fb),
        args.reps,
        rounds=args.repeat,
    )

    inner = sglang_bench.graph_reps_for(step.extend)
    times = {}
    # Popped, not iterated: a recorded entry pins that op's chunk-sized input tensors.
    skipped = {}
    for name in list(saved):
        rank_print(f"  timing: {name}")
        fn = model_bench.make_call(ops[name], saved.pop(name))
        try:
            times[name] = sglang_bench.bench(fn, args.reps, inner=inner, rounds=args.repeat)
        except torch.cuda.OutOfMemoryError:
            # Not a shape mismatch: every later op would be skipped too, so abort instead of a partial report.
            raise
        except RuntimeError as e:
            # As in profile_decode: only a deterministic failure may be skipped, since every rank must skip the same ops.
            first = str(e).split("\n")[0]
            if "out of memory" in first.lower():
                raise
            skipped[name] = first
            rank_print(f"    skipped: {first}")
        del fn
        if not sglang_bench.agreed(name in times):
            times.pop(name, None)
            skipped.setdefault(name, "skipped on another rank")
    # The page-table row goes back; its KV slots stay, as they are the context the next chunks read.
    model_bench.free_req_rows(batch, mr)
    return StepResult(
        step,
        times,
        full_ms,
        counts,
        path,
        model_bench.residual_contents(mr, path),
        skipped,
    )


def work(server_args, port_args, args, gpu_id, tp_rank):
    # Refused rather than ignored: setup() builds the spec worker anyway, loading draft weights.
    if server_args.speculative_algorithm is not None:
        raise SystemExit(
            "profile_prefill does not model speculative decoding: a draft model"
            " runs on a prefill chunk and none of it would be timed here. Drop"
            " --speculative-algorithm, or use profile_decode.py for the MTP step."
        )
    mr, _, rank_print = model_bench.setup(server_args, port_args, gpu_id, tp_rank)
    layers = model_bench.Layers(mr)
    model_bench.balance_moe_routing(layers)

    with sglang_bench.forward_ctx(mr):
        context = model_bench.make_prefill_context(args, mr)
        steps = model_bench.plan_prefill(args, mr)
        ops = model_bench.collect_ops(layers, mr.model)

        results = []
        for i, step in enumerate(steps):
            rank_print(f"chunk {i + 1}/{len(steps)}: {step.extend} new on {step.prefix}")
            results.append(measure_step(args, mr, ops, context, step, layers.counts, rank_print))

    if tp_rank != 0:
        return
    md, headline = report(args, mr, layers, results, ops)
    path = Path(args.out or default_out(args, mr, layers))
    path.write_text(md)
    rank_print(headline)
    rank_print(f"report -> {path.resolve()}")


def default_out(args, mr, layers) -> str:
    parts = [
        model_bench.MODEL_TAG,
        "prefill",
        f"isl{args.ISL}",
        f"chunk{model_bench.chunk_flag(args, mr)}",
        f"hit{int(args.prefix_cache_hit * 100)}",
        *model_bench.parallel_tags(mr, layers),
    ]
    parts.append(mr.server_args.kv_cache_dtype)
    # Keeps a dummy run from overwriting the real run of the same shape.
    parts += filter(None, [sglang_bench.load_tag(mr.server_args)])
    parts.append(time.strftime("%Y%m%d"))
    return "_".join(parts) + ".md"


def ttft_ms(results) -> float:
    """One request's time to first token: every chunk of its prefill."""
    return sum(r.full_ms for r in results)


def summary_table(args, results) -> str:
    ttft = ttft_ms(results)
    new_tokens = sum(r.step.extend for r in results)
    rollup = sum(r.rollup for r in results)
    rows = [
        ("TTFT", f"{ttft:.1f} ms", f"{len(results)} chunks, one request per rank"),
        ("roll-up", f"{rollup:.1f} ms", "sum of the per-op leaf rows below"),
        (
            "ratio",
            f"{rollup / ttft:.3f}",
            "roll-up / TTFT; 1.0 means nothing was missed",
        ),
        (
            "computed",
            f"{new_tokens:,} tokens",
            f"of {args.ISL:,}; the other {args.ISL - new_tokens:,} came from cache",
        ),
        (
            "prompt rate",
            f"{args.ISL / ttft * 1000:,.0f} tok/s",
            "the whole prompt over TTFT, per request; what a client sees",
        ),
        (
            "compute rate",
            f"{new_tokens / ttft * 1000:,.0f} tok/s",
            "only the tokens forwarded; the same number at a 0% hit",
        ),
    ]
    return sglang_bench.table(rows, ("metric", "value", "note"))


def config_table(args, mr, layers) -> str:
    ps, sa = sglang_bench.parallel_state(mr), mr.server_args
    topk = model_bench.index_topk(mr)
    rows = [
        ("model", f"`{type(mr.model).__name__}`  {sa.model_path}"),
        ("layers", layers.describe()),
        ("parallel", model_bench.parallel_line(mr)),
        *([("moe", line)] if (line := model_bench.moe_line(mr, layers)) else []),
        ("context", f"ISL {args.ISL:,} per request"),
        (
            "prefix cache",
            (
                f"{args.prefix_cache_hit:.0%} hit, rounded down to the"
                f" {model_bench.cfg(sa).page_size}-token page"
            ),
        ),
        (
            "chunk",
            f"--chunked-prefill-size {model_bench.chunk_flag(args, mr):,}"
            + (f" / {ps.attn_dp_size} DP ranks = " if ps.attn_dp_size > 1 else " = ")
            + f"{model_bench.chunk_tokens(args, mr):,} new tokens per forward",
        ),
        ("batch", f"1 request per attention rank, {args.CONC} in total"),
        ("indexer", f"DSA, top {topk:,} tokens per query" if topk else "none, dense"),
        ("kv dtype", f"`{sa.kv_cache_dtype}`"),
        ("weights", f"`{sa.load_format}` load, MXFP4 experts, BF16 elsewhere"),
        (
            "measurement",
            f"{args.reps} graph replays per op;"
            " the full forward eager, the way the server runs a chunk"
            + (f", fastest of {args.repeat} passes" if args.repeat > 1 else ""),
        ),
    ]
    return sglang_bench.table(rows, ("item", "value"))


def timeline_table(results) -> str:
    """One row per chunk, so the shape being measured is on the record."""
    rows = [
        (
            i + 1,
            f"{r.step.prefix:,}",
            f"{r.step.extend:,}",
            r.path,
            r.full_ms,
            r.rollup,
            f"{r.rollup / r.full_ms:.3f}",
        )
        for i, r in enumerate(results)
    ]
    return sglang_bench.table(
        rows, ("chunk", "prefix", "new", "attn", "ms", "roll-up", "ratio"), fmt=".2f"
    ) + (
        "\n\n`prefix` is what is already in the KV pool when the chunk starts --"
        " a cache hit and the chunks before it are the same thing here. `attn`"
        " is the path the DSA backend dispatched to: `mha` is dense and reads"
        " every token, `mla` is the sparse absorbed path decode uses. The test"
        " is on `prefix + new`, the request's length so far, against"
        " SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD, which sglang defaults"
        " to the model's index_topk -- so the switch sits at a fixed depth in"
        " the context and the chunk size only decides which row it lands on."
        " `ratio` is the roll-up over the full forward, the only self-check the"
        " decomposition has -- the forward eager and the ops in graphs, so"
        " anything an eager chunk spends between kernels lands here as a"
        " shortfall."
    )


def growth_table(results, mr) -> str:
    """Ops down the side, chunks across, so what scales with context is visible."""
    totals = {}
    for r in results:
        for name, ms in r.totals.items():
            totals[name] = totals.get(name, 0.0) + ms
    names = sorted(totals, key=lambda n: -totals[n])
    rollup = sum(totals.values())
    # Exact prefixes rather than a rounded `8k`, which would collide for chunks under 1000 tokens.
    headers = ("op",) + tuple(f"{r.step.prefix:,}" for r in results) + ("total", "share")
    rows = [
        (model_bench.tree_name(name, 0, results[-1].contents),)
        + tuple(r.totals.get(name, 0.0) for r in results)
        + (totals[name], f"{totals[name] / rollup * 100:.1f}%")
        for name in names
    ]
    note = (
        "\n\nms per chunk, columns labelled by the prefix each chunk starts on."
        " Leaf rows only, so a column sums to that chunk's roll-up and `share`,"
        " the row's cut of the whole TTFT, adds to 100%. A zero is an op that"
        " never ran in that chunk. Rows that climb across the table are reading"
        " the context; rows that stay flat only ever see the chunk."
    )
    topk = model_bench.index_topk(mr)
    if topk:
        note += (
            " Attention is expected among the flat ones past the indexer's top"
            f" {topk:,}: beyond that it scores a fixed budget per query however"
            " deep the context goes, leaving the indexer's own scan to grow."
        )
    if len({r.path for r in results}) > 1:
        note += (
            " The `attn.<rest>` row spans both attention paths and is named for"
            f" the last chunk's ({results[-1].path}); see the timeline for where"
            " the switch falls."
        )
    return sglang_bench.table(rows, headers, fmt=".2f") + note


def kind_sections(result, counts) -> str:
    out = []
    for kind, n in counts.items():
        rows = [
            (
                model_bench.tree_name(name, depth, result.contents),
                result.times[f"{kind}|{name}"],
                result.times[f"{kind}|{name}"] * n,
            )
            for name, depth in model_bench.TREE
            if f"{kind}|{name}" in result.times
        ]
        table = sglang_bench.table(rows, ("op", "ms/call", "ms/chunk"))
        out.append(f"### {kind} -- x{n} layers\n\n{table}")
    return "\n\n".join(out)


def report(args, mr, layers, results, ops):
    """Returns (markdown, one-line headline for the log)."""
    last = results[-1]
    sections = [
        ("Summary", summary_table(args, results)),
        ("Config", config_table(args, mr, layers)),
        (
            "Versions",
            sglang_bench.table(sglang_bench.versions(), ("component", "version")),
        ),
        ("Memory, per rank", model_bench.memory_section(args, mr, layers, kv_factor=2)),
        ("Chunk timeline", timeline_table(results)),
        ("Hotspots -- one TTFT, all layers", growth_table(results, mr)),
        (
            f"Per layer kind -- deepest chunk, prefix {last.step.prefix:,}",
            kind_sections(last, layers.counts),
        ),
        *([("Coverage", coverage)] if (coverage := model_bench.coverage_note(ops, last.times)) else []),
        *(
            [("Not replayable", note)]
            if (note := model_bench.skipped_note(sglang_bench.merged_skips(results)))
            else []
        ),
    ]
    md = "\n\n".join(
        [
            "# GLM-5.2 prefill profile",
            time.strftime("%Y-%m-%d %H:%M %Z"),
            # Above the tables, not buried in Config.
            *filter(None, [model_bench.dummy_banner(mr.server_args)]),
        ]
        + [f"## {title}\n\n{body}" for title, body in sections]
    )
    ttft = ttft_ms(results)
    new_tokens = sum(r.step.extend for r in results)
    headline = (
        f"TTFT {ttft:.1f} ms over {len(results)} chunks  "
        f"{args.ISL / ttft * 1000:,.0f} tok/s prompt, "
        f"{new_tokens / ttft * 1000:,.0f} tok/s computed"
    )
    return md + "\n", headline


def make_parser():
    parser = model_bench.make_parser()
    parser.add_argument(
        "--prefix-cache-hit",
        type=float,
        default=0.0,
        help="fraction of ISL already in the KV cache, 0..1",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    # Not a flag: chunked prefill gives a long request the forward to itself, and every DP rank schedules its own.
    args.CONC = sglang_bench.prefill_conc(args)
    model_bench.run(work, model_bench.build_server_args(args), args)
