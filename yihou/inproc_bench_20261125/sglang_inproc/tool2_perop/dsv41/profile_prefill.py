###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
# Provenance: port of AKLR `aklr/benchmarks/models/{glm52,deepseekv4}/profile_prefill.py` @ 33689332.

"""Per-operator chunked-prefill timing for DeepSeek-V4.1, rolled up to a synthetic TTFT estimate."""

import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

import model_bench  # noqa: E402
import sglang_bench  # noqa: E402
import shape_export  # noqa: E402

REPLAY_ERRORS = (RuntimeError, AssertionError, ValueError, TypeError, AttributeError, KeyError, IndexError)


class StepResult:
    """One chunk's measurements, kept together so the report can walk them."""

    def __init__(self, step, times, full_ms, counts, contents, tail, skipped=None):
        self.step = step
        self.skipped = skipped or {}
        self.full_ms = full_ms
        self.tail = tail
        model_bench.add_residuals(times, counts)
        self.times = times
        self.totals = model_bench.leaf_totals(times, counts)
        self.rollup = sum(self.totals.values())
        self.contents = contents


def tail_rows(mr):
    """Rows the late-layer SWA tail kept in the last extend forward, or None when the tail was off."""
    backend = getattr(mr, "attn_backend", None)
    meta = getattr(backend, "tail_forward_metadata", None)
    tail = getattr(meta, "late_layer_tail", None)
    idx = getattr(tail, "token_indices", None)
    return int(idx.shape[0]) if isinstance(idx, torch.Tensor) else None


def measure_step(args, mr, layers, ops, context, step, n, rank_print, tp_rank) -> StepResult:
    """Time every op for one chunk, plus the full forward it is checked against."""
    batch = model_bench.make_prefill_batch(mr, context, step, n)
    fb = sglang_bench.forward_batch_of(batch, mr)

    saved = {}
    with shape_export.export_scope(ops, f"prefill_chunk{n}", args.result_dir, tp_rank):
        with model_bench.record_inputs(ops, saved, layers):
            mr.forward(fb)
    torch.cuda.synchronize()
    sglang_bench.assert_lockstep(saved)
    tail = tail_rows(mr)

    # Run eagerly, as the server runs a chunk: this is what the roll-up is checked against and TTFT is summed from.
    rank_print("  timing: full forward")
    full_ms = sglang_bench.bench_eager(
        lambda: mr.model.forward(fb.input_ids, fb.positions, fb), args.reps, rounds=args.repeat
    )

    inner = sglang_bench.graph_reps_for(step.extend)
    times, skipped = {}, {}
    # Popped, not iterated: a recorded entry pins that op's chunk-sized input tensors.
    for name in list(saved):
        rank_print(f"  timing: {name}")
        fn = model_bench.make_call(ops[name], saved.pop(name))
        try:
            times[name] = sglang_bench.bench(fn, args.reps, inner=inner, rounds=args.repeat)
        except torch.cuda.OutOfMemoryError:
            raise
        except REPLAY_ERRORS as e:
            first = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
            if "out of memory" in first.lower():
                raise
            skipped[name] = first[:300]
            rank_print(f"    skipped: {skipped[name]}")
            torch.cuda.synchronize()
        del fn
        if not sglang_bench.agreed(name in times):
            times.pop(name, None)
            skipped.setdefault(name, "skipped on another rank")
    times = model_bench.world_times({**times, "__full": full_ms}, mr)
    full_ms = times.pop("__full")
    # The page-table row goes back; its KV slots stay, as they are the context the next chunks read.
    model_bench.free_req_rows(batch, mr)
    return StepResult(step, times, full_ms, layers.counts, model_bench.residual_contents(mr), tail, skipped)


def work(server_args, port_args, args, gpu_id, tp_rank):
    # Refused rather than ignored: setup() would build the spec worker anyway, loading draft weights.
    if model_bench.cfg(server_args).speculative_algorithm is not None:
        raise SystemExit(
            "profile_prefill does not model speculative decoding: a draft model runs on a prefill chunk and"
            " none of it would be timed here. Drop --speculative-algorithm, or use profile_decode.py for MTP."
        )
    mr, _, rank_print = model_bench.setup(server_args, port_args, gpu_id, tp_rank)
    layers = model_bench.Layers(mr)
    model_bench.balance_moe_routing(layers, args.routing)
    rank_print(f"layers: {layers.describe()}")

    with sglang_bench.forward_ctx(mr):
        context = model_bench.make_prefill_context(args, mr)
        steps = model_bench.plan_prefill(args, mr)
        ops = model_bench.collect_ops(layers, mr.model)
        all_ops = dict(ops)
        results = []
        for i, step in enumerate(steps):
            rank_print(f"chunk {i + 1}/{len(steps)}: {step.extend} new on {step.prefix}")
            results.append(measure_step(args, mr, layers, ops, context, step, i, rank_print, tp_rank))

    if tp_rank != 0:
        return
    md, headline = report(args, mr, layers, results, all_ops)
    path = Path(args.out or Path(args.result_dir or ".") / default_out(args, mr, layers))
    path.write_text(md)
    rank_print(headline)
    rank_print(f"report -> {path.resolve()}")


def default_out(args, mr, layers) -> str:
    parts = [
        model_bench.model_tag(mr.server_args),
        "prefill",
        f"isl{args.ISL}",
        f"chunk{model_bench.chunk_flag(args, mr)}",
        f"hit{int(args.prefix_cache_hit * 100)}",
        *model_bench.parallel_tags(mr, layers),
        str(mr.server_args.kv_cache_dtype),
    ]
    parts += filter(None, [sglang_bench.load_tag(mr.server_args)])
    parts.append(time.strftime("%Y%m%d"))
    return "_".join(parts) + ".md"


def ttft_ms(results) -> float:
    return sum(r.full_ms for r in results)


def summary_table(args, results) -> str:
    ttft = ttft_ms(results)
    new_tokens = sum(r.step.extend for r in results)
    rollup = sum(r.rollup for r in results)
    rows = [
        ("TTFT", f"{ttft:.1f} ms", f"{len(results)} chunks, one request per rank"),
        ("roll-up", f"{rollup:.1f} ms", "sum of the per-op leaf rows below"),
        ("ratio", f"{rollup / ttft:.3f}", "roll-up / synthetic TTFT; agreement is not proof of complete coverage"),
        ("computed", f"{new_tokens:,} tokens", f"of {args.ISL:,}; the other {args.ISL - new_tokens:,} came from cache"),
        ("prompt rate", f"{args.ISL / ttft * 1000:,.0f} tok/s", "the whole prompt over TTFT, per request"),
        ("compute rate", f"{new_tokens / ttft * 1000:,.0f} tok/s", "only the tokens forwarded; the same number at a 0% hit"),
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
        ("prefix cache", f"{args.prefix_cache_hit:.0%} hit, rounded down to the {model_bench.cfg(sa).page_size}-token page"),
        (
            "chunk",
            f"--chunked-prefill-size {model_bench.chunk_flag(args, mr):,}"
            + (f" / {ps.attn_dp_size} DP ranks = " if ps.attn_dp_size > 1 else " = ")
            + f"{model_bench.chunk_tokens(args, mr):,} new tokens per forward",
        ),
        ("batch", f"1 request per attention rank, {args.CONC} in total"),
        ("indexer", f"top {topk:,} per query (C4 / DSv4.1 index-source layers)" if topk else "none"),
        ("late-layer SWA tail", f"layers >= {layers.late_start} run on the tail rows" if layers.late_start is not None else "off"),
        (
            "routing",
            f"seeded balanced (seed {sglang_bench.MOE_ROUTING_SEED}) on every MoE layer"
            if args.routing == "balanced"
            else "**native router on synthetic hidden states (--routing native, diagnostic)**",
        ),
        ("kv dtype", f"`{sa.kv_cache_dtype}`"),
        ("weights", model_bench.weights_line(mr)),
        (
            "measurement",
            f"{args.reps} graph replays per op; the full forward eager, the way the server runs a chunk"
            + (f", fastest of {args.repeat} passes" if args.repeat > 1 else "")
            + (", max over ranks (EP)" if sglang_bench.moe_parallel(mr).ep_size > 1 else ", min over ranks"),
        ),
    ]
    return sglang_bench.table(rows, ("item", "value"))


def timeline_table(results) -> str:
    rows = [
        (
            i + 1,
            f"{r.step.prefix:,}",
            f"{r.step.extend:,}",
            "-" if r.tail is None else f"{r.tail:,}",
            r.full_ms,
            r.rollup,
            f"{r.rollup / r.full_ms:.3f}",
        )
        for i, r in enumerate(results)
    ]
    return sglang_bench.table(rows, ("chunk", "prefix", "new", "tail rows", "ms", "roll-up", "ratio"), fmt=".2f") + (
        "\n\n`prefix` is the synthetic context already in KV. `tail rows` is how many rows the late layers"
        " ran on (DSv4.1 decoder SWA bounded replay); `-` means the tail was off. `ratio` compares graph-timed"
        " operators with an eager full forward, whose inter-kernel launch costs remain."
    )


def growth_table(results, mr) -> str:
    totals = {}
    for r in results:
        for name, ms in r.totals.items():
            totals[name] = totals.get(name, 0.0) + ms
    names = sorted(totals, key=lambda n: -totals[n])
    rollup = sum(totals.values())
    headers = ("op",) + tuple(f"{r.step.prefix:,}" for r in results) + ("total", "share")
    rows = [
        (model_bench.tree_name(name, 0, results[-1].contents),)
        + tuple(r.totals.get(name, 0.0) for r in results)
        + (totals[name], f"{totals[name] / rollup * 100:.1f}%")
        for name in names
    ]
    note = (
        "\n\nms per chunk, columns labelled by the prefix each chunk starts on. Leaf rows only, so a column sums"
        " to that chunk's roll-up and `share` adds to 100%. A zero is an op that never ran in that chunk. Rows"
        " that climb across the table are reading the context; rows that stay flat only ever see the chunk."
    )
    topk = model_bench.index_topk(mr)
    if topk:
        note += (
            f" Attention is expected among the flat ones past the indexer's top {topk:,}: beyond that it scores"
            " a fixed budget per query however deep the context goes, leaving the indexer's own scan to grow."
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
        out.append(f"### {kind} -- x{n} layers\n\n" + sglang_bench.table(rows, ("op", "ms/call", "ms/chunk")))
    return "\n\n".join(out)


def report(args, mr, layers, results, ops):
    last = results[-1]
    sections = [
        ("Summary", summary_table(args, results)),
        ("Config", config_table(args, mr, layers)),
        ("Versions", sglang_bench.table(sglang_bench.versions(), ("component", "version"))),
        ("Memory, per rank", model_bench.memory_section(args, mr, layers, kv_factor=2)),
        ("Chunk timeline", timeline_table(results)),
        ("Hotspots -- one TTFT, all layers", growth_table(results, mr)),
        (f"Per layer kind -- deepest chunk, prefix {last.step.prefix:,}", kind_sections(last, layers.counts)),
        *([("Coverage", c)] if (c := model_bench.coverage_note(ops, last.times)) else []),
        *([("Not replayable", note)] if (note := model_bench.skipped_note(sglang_bench.merged_skips(results))) else []),
    ]
    md = "\n\n".join(
        [
            f"# {model_bench.model_tag(mr.server_args)} prefill profile",
            time.strftime("%Y-%m-%d %H:%M %Z"),
            *filter(None, [model_bench.dummy_banner(mr.server_args)]),
        ]
        + [f"## {title}\n\n{body}" for title, body in sections]
    )
    ttft = ttft_ms(results)
    new_tokens = sum(r.step.extend for r in results)
    headline = (
        f"TTFT {ttft:.1f} ms over {len(results)} chunks  roll-up {sum(r.rollup for r in results):.1f} ms"
        f"  ratio {sum(r.rollup for r in results) / ttft:.3f}  "
        f"{args.ISL / ttft * 1000:,.0f} tok/s prompt, {new_tokens / ttft * 1000:,.0f} tok/s computed"
    )
    return md + "\n", headline


def make_parser():
    parser = model_bench.make_parser()
    parser.add_argument("--prefix-cache-hit", type=float, default=0.0, help="fraction of ISL already in the KV cache, 0..1")
    parser.add_argument("--result-dir", default=None, help="directory for the report/export (default: cwd)")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if not 0.0 <= args.prefix_cache_hit < 1.0:
        raise ValueError("--prefix-cache-hit must be in [0, 1)")
    if args.result_dir:
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    # Not a flag: chunked prefill gives a long request the forward to itself, and every DP rank schedules its own.
    args.CONC = sglang_bench.prefill_conc(args)
    args.concs = [args.CONC]
    model_bench.run(work, model_bench.build_server_args(args), args)


if __name__ == "__main__":
    main()
