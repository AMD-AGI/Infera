"""Per-operator decode timing for GLM-5.2, rolled up to a per-step estimate."""

import contextlib
import sys
import time
from pathlib import Path

import torch

# sglang_bench.py and kernel_breakdown.py live one directory up.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import model_bench
import sglang_bench


def work(server_args, port_args, args, gpu_id, tp_rank):
    mr, spec, rank_print = model_bench.setup(server_args, port_args, gpu_id, tp_rank)
    layers = model_bench.Layers(mr)
    model_bench.balance_moe_routing(layers)
    model_bench.balance_draft_routing(spec)

    with sglang_bench.forward_ctx(mr):

        def new_batch():
            b = model_bench.make_extend_batch(model_bench.make_reqs(args, mr), mr)
            sglang_bench.prepare_decode(b, mr)
            return b

        if args.full_step:
            # A step advances the batch it is given, so each needs its own.
            n = args.full_step
            model_bench.randomize_kv(mr, spec)

            seed = None

            def step(b):
                # Hand on the previous step's DSA top-k seed, which a step leaves on its return value; the server always has one.
                nonlocal seed
                if spec is None:
                    mr.forward(sglang_bench.forward_batch_of(b, mr))
                    return
                if seed is not None:
                    b.spec_info.dsa_topk_indices = seed
                out = spec.forward_batch_generation(b)
                seed = out.next_draft_input.dsa_topk_indices

            def fresh():
                b = new_batch()
                # Seeded outside every clock: in the server the scheduler hands the draft input to the step.
                if spec is not None:
                    model_bench.seed_draft_input(spec, b)
                return b

            # Synced per step: all CPU work sits on the critical path.
            ev_per = []
            phases = model_bench.spec_phases(spec) if spec is not None else []
            ms = []
            for _ in range(n + 2):
                b = fresh()
                ev = {}
                ctx = model_bench.time_calls(phases, ev) if phases else contextlib.nullcontext()
                torch.cuda.synchronize()
                t = time.perf_counter()
                with ctx:
                    step(b)
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t) * 1000.0)
                ev_per.append(ev)

            best = min(range(len(ms)), key=ms.__getitem__)

            rank_print("synced per step ms: " + " ".join(f"{m:.3f}" for m in ms))
            for label, (total, calls) in model_bench.elapsed(ev_per[best]).items():
                rank_print(f"  {label}: {total:.3f} ms  {calls} calls/step")
            rank_print(f"step: synced {ms[best]:.3f} ms")
            return

        batch = new_batch()
        model_bench.randomize_kv(mr, spec)
        fb = sglang_bench.forward_batch_of(batch, mr)

        ops = model_bench.collect_ops(layers, mr.model)
        if spec is not None:
            ops.update(model_bench.draft_ops(spec))
        saved = {}
        if args.trace:
            trace_step(args, spec, new_batch, fb, mr, rank_print, tp_rank)
            return

        with model_bench.record_inputs(ops, saved):
            model_bench.run_step(spec, new_batch, fb, mr)
        torch.cuda.synchronize()
        sglang_bench.assert_lockstep(saved)

        # The reference the roll-up is checked against, graph-captured so it is launch-free like the per-op rows.
        fwd_ms = {}
        rank_print("timing: forwards" if spec else "timing: full forward")
        if spec is not None:
            # A step is several forwards across two models; each is timed in place under its own graph and summed.
            with model_bench.bench_forwards(model_bench.spec_models(spec), args.reps, args.repeat, fwd_ms):
                model_bench.run_step(spec, new_batch, fb, mr)
            full_ms = sum(sum(v) for v in fwd_ms.values())
            for name, each in fwd_ms.items():
                per = " ".join(f"{m:.3f}" for m in each)
                rank_print(f"  {name}: {sum(each):.3f} ms = {per}")
        else:
            full_ms = sglang_bench.bench(
                lambda: mr.model.forward(fb.input_ids, fb.positions, fb),
                args.reps,
                inner=1,
                rounds=args.repeat,
            )

        times = {}
        skipped = {}
        for name, entry in saved.items():
            rank_print(f"timing: {name}")
            try:
                # Rotated so a MALL-sized weight is not read cache-hot.
                with model_bench.rotated_calls(ops[name], entry) as (fns, inner):
                    times[name] = sglang_bench.bench(fns, args.reps, inner=inner, rounds=args.repeat)
            except torch.cuda.OutOfMemoryError:
                # Not a shape mismatch: every later op would be skipped too, so abort instead of a partial report.
                raise
            except RuntimeError as e:
                # Replaying a module outside its own forward can hit a shape check the ambient state no longer satisfies.
                first = str(e).split("\n")[0]
                if "out of memory" in first.lower():
                    raise
                skipped[name] = first
                rank_print(f"  skipped: {skipped[name]}")
            # Agreed across ranks: a rank that kept a row the others dropped would unpair their collectives.
            if not sglang_bench.agreed(name in times):
                times.pop(name, None)
                skipped.setdefault(name, "skipped on another rank")

    # Draft rows sit outside the roll-up on purpose: the nextn layer is neither a kind nor a per-layer row.
    draft_rows = {k: v for k, v in times.items() if k.startswith("draft|")}
    for k in draft_rows:
        times.pop(k)
        ops.pop(k, None)
        saved.pop(k, None)

    if tp_rank != 0:
        return

    if skipped:
        rank_print(f"{len(skipped)} op(s) could not be replayed standalone:")
        for name, err in skipped.items():
            rank_print(f"  {name}: {err}")

    if draft_rows:
        layer_ms = draft_rows.get("draft|layer")
        rank_print("draft model, ms per forward (one draft-decode shape):")
        for k, v in sorted(draft_rows.items(), key=lambda kv: -kv[1]):
            share = f"{v / layer_ms * 100:5.1f}% of layer" if layer_ms else ""
            rank_print(f"  {k.split('|', 1)[1]:26s} {v:8.4f} ms  {share}")

    md, headline = report(
        args,
        mr,
        layers,
        times,
        full_ms,
        model_bench.coverage_note(ops, saved),
        spec,
        draft_ms=sum(fwd_ms.get("draft", [])) if spec else 0.0,
        skipped=model_bench.skipped_note(skipped),
    )
    path = Path(args.out or default_out(args, mr, layers))
    path.write_text(md)
    rank_print(headline)
    rank_print(f"report -> {path.resolve()}")


def trace_step(args, spec, new_batch, fb, mr, rank_print, tp_rank) -> None:
    """Kernel-level breakdown of one step: what the per-op rows are made of."""
    from torch.profiler import ProfilerActivity, profile

    model_bench.run_step(spec, new_batch, fb, mr)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model_bench.run_step(spec, new_batch, fb, mr)
        torch.cuda.synchronize()

    # Self time, not total: device_time_total on an op includes the kernels it launched.
    rows = {}
    for e in prof.key_averages():
        if e.self_device_time_total <= 0:
            continue
        rows[e.key] = (e.self_device_time_total / 1000.0, e.count)
    total = sum(ms for ms, _ in rows.values())
    rank_print(f"kernels: {len(rows)} distinct, {sum(n for _, n in rows.values())} launches, {total:.3f} ms")
    for name, (ms, n) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:35]:
        rank_print(f"  {ms:8.3f} ms  {n:5d} x {ms * 1000 / n:7.2f} us  {name[:78]}")
    # Rank 0 only: export_chrome_trace renames <path>.tmp, so ranks sharing one path race.
    if tp_rank != 0:
        return
    path = Path(args.out or "trace.json")
    prof.export_chrome_trace(str(path))
    rank_print(f"trace -> {path.resolve()}")


def default_out(args, mr, layers) -> str:
    parts = [
        model_bench.MODEL_TAG,
        "mtp" if mr.server_args.speculative_algorithm is not None else "decode",
        f"isl{args.ISL}",
        f"conc{args.CONC}",
        *model_bench.parallel_tags(mr, layers),
    ]
    parts.append(mr.server_args.kv_cache_dtype)
    # Keeps a dummy run from overwriting the real run of the same shape.
    parts += filter(None, [sglang_bench.load_tag(mr.server_args)])
    parts.append(time.strftime("%Y%m%d"))
    return "_".join(parts) + ".md"


def config_table(args, mr, layers) -> str:
    sa = mr.server_args
    rows = [
        ("model", f"`{type(mr.model).__name__}`  {sa.model_path}"),
        ("layers", layers.describe()),
        ("parallel", model_bench.parallel_line(mr)),
        *([("moe", line)] if (line := model_bench.moe_line(mr, layers)) else []),
        (
            "batch",
            f"CONC {args.CONC}, {sglang_bench.local_conc(args, mr)} requests per rank",
        ),
        ("context", f"ISL {args.ISL} per request"),
        ("kv dtype", f"`{sa.kv_cache_dtype}`"),
        *([("spec", model_bench.spec_note(sa))] if sa.speculative_algorithm is not None else []),
        ("weights", f"`{sa.load_format}` load, MXFP4 experts, BF16 elsewhere"),
        (
            "measurement",
            f"{args.reps} graph replays per op; {sglang_bench.GRAPH_REPS} calls per"
            f" graph, more for a weight small enough to rotate over the"
            f" {model_bench.ROTATION_BYTES >> 20} MB buffer"
            + (f", fastest of {args.repeat} passes" if args.repeat > 1 else ""),
        ),
    ]
    return sglang_bench.table(rows, ("item", "value"))


def summary_table(args, mr, full_ms, rollup, spec=None) -> str:
    rows = [
        (
            "forward time" if spec else "step time",
            f"{full_ms:.3f} ms",
            (
                # Counted, not stated: the draft loop breaks before the last step's forward.
                f"{model_bench.spec_forward_count(mr.server_args)} forwards --"
                f" {model_bench.spec_draft_decodes(mr.server_args)} draft decode, 1"
                " draft extend, 1 target verify -- each timed under its own"
                " graph and summed. Not the whole step: the sampling,"
                " accept/reject and bookkeeping between them are only visible"
                " under --full-step"
                if spec
                else "whole model in one graph, the way the server replays decode"
            ),
        ),
        ("roll-up", f"{rollup:.3f} ms", "sum of the per-op leaf rows below"),
        (
            "ratio",
            f"{rollup / full_ms:.3f}",
            (
                "roll-up / forward time; 1.0 means nothing was missed. The"
                " draft is one row in both, so what is short is target ops"
                " with no module to hook"
                if spec
                else "roll-up / step time; 1.0 means nothing was missed"
            ),
        ),
    ]
    acc = model_bench.accept_len() if spec else None
    if spec is None:
        rows.append(
            (
                "throughput",
                f"{args.CONC / full_ms * 1000:.1f} tok/s",
                f"at CONC {args.CONC}",
            )
        )
    elif acc is None:
        rows.append(
            (
                "throughput",
                "not reported",
                (
                    "a speculative step emits as many tokens as were accepted."
                    " Set SGLANG_SIMULATE_ACC_LEN, the way the server runs, for"
                    " a tok/s row; CONC / step time would understate it by"
                    " exactly that factor."
                ),
            )
        )
    else:
        rows.append(
            (
                "throughput",
                f"{args.CONC * acc / full_ms * 1000:.1f} tok/s",
                (
                    f"at CONC {args.CONC}, accept len {acc:.2f} from"
                    " SGLANG_SIMULATE_ACC_LEN -- an assumption the server is"
                    " also run under, not something measured here"
                ),
            )
        )
        rows.append(
            (
                "ITL",
                f"{full_ms / acc:.3f} ms",
                (
                    "forward time / accept len, per request -- a floor, since"
                    " the non-forward part of the step is not counted"
                ),
            )
        )
    return sglang_bench.table(rows, ("metric", "value", "note"))


def hotspot_section(totals, rollup, contents) -> str:
    rows = [
        (model_bench.tree_name(name, 0, contents), ms, f"{ms / rollup * 100:.1f}%")
        for name, ms in sorted(totals.items(), key=lambda kv: -kv[1])
    ]
    note = (
        "\n\nLeaf rows only, so the shares add to 100%. A `<rest>` row is its parent"
        " minus the children listed under it, and the `::` tag names the work that"
        " lands there -- inline code and fused kernels with no module to hook. It is"
        " a difference of two measurements rather than a measurement, so any error"
        " in the parent or in any child lands in it at full size and reversed sign:"
        " a `<rest>` small next to its parent is mostly that error, and can read"
        " negative. "
    )
    if "mlp.experts" in contents:
        note += "`mlp.experts` is labelled too: under an all-to-all the dispatch and combine run inside it."
    else:
        note += (
            "`mlp.<rest>` shrinks on MoE layers under DP attention, because the"
            " post-expert all-reduce it otherwise holds is handed to the"
            " reduce-scatter counted in `layer.<rest>`, leaving only the add that"
            " folds the shared expert's output into the routed one."
        )
    return sglang_bench.table(rows, ("op", "ms/step", "share")) + note


def kind_sections(times, counts, contents) -> str:
    out = []
    for kind, n in counts.items():
        rows = [
            (
                model_bench.tree_name(name, depth, contents),
                times[f"{kind}|{name}"],
                times[f"{kind}|{name}"] * n,
            )
            for name, depth in model_bench.TREE
            if f"{kind}|{name}" in times
        ]
        table = sglang_bench.table(rows, ("op", "ms/call", "ms/step"))
        out.append(f"### {kind} -- x{n} layers\n\n{table}")
    return "\n\n".join(out)


def rollup_table(times, counts) -> str:
    rows = [
        (f"{kind} layer", n, times[f"{kind}|layer"], times[f"{kind}|layer"] * n)
        for kind, n in counts.items()
        if f"{kind}|layer" in times
    ]
    rows += [(name, 1, times[name], times[name]) for name in model_bench.GLOBAL_OPS if name in times]
    return sglang_bench.table(rows, ("op", "x", "ms/call", "ms/step"))


def report(args, mr, layers, times, full_ms, coverage="", spec=None, draft_ms=0.0, skipped=""):
    """Returns (markdown, one-line headline for the log)."""
    counts = layers.counts
    model_bench.add_residuals(times, counts)
    totals = model_bench.leaf_totals(times, counts)
    if draft_ms:
        # One row, not a per-op tree: the draft model is a single nextn layer run several times per step.
        totals["draft model"] = draft_ms
    rollup = sum(totals.values())
    contents = model_bench.residual_contents(mr, "mla")

    sections = [
        ("Summary", summary_table(args, mr, full_ms, rollup, spec)),
        ("Config", config_table(args, mr, layers)),
        (
            "Versions",
            sglang_bench.table(sglang_bench.versions(), ("component", "version")),
        ),
        ("Memory, per rank", model_bench.memory_section(args, mr, layers, spec)),
        (
            "Hotspots -- one step, all layers",
            hotspot_section(totals, rollup, contents),
        ),
        ("Per layer kind -- one step", kind_sections(times, counts, contents)),
        ("Roll-up -- one step", rollup_table(times, counts)),
        *([("Coverage", coverage)] if coverage else []),
        # Beside Coverage: those ops fired but could not be timed, and only they contaminate a `<rest>` row.
        *([("Not replayable", skipped)] if skipped else []),
    ]
    md = "\n\n".join(
        [
            "# GLM-5.2 decode profile",
            time.strftime("%Y-%m-%d %H:%M %Z"),
            # Above the tables, not buried in Config.
            *filter(None, [model_bench.dummy_banner(mr.server_args)]),
        ]
        + [f"## {title}\n\n{body}" for title, body in sections]
    )
    headline = (
        f"{'forwards' if spec else 'step'} {full_ms:.3f} ms  "
        f"roll-up {rollup:.3f} ms  ratio {rollup / full_ms:.3f}  "
        f"{args.CONC / full_ms * 1000:.1f} tok/s"
    )
    return md + "\n", headline


def make_parser():
    parser = model_bench.make_parser()
    parser.add_argument(
        "--trace",
        action="store_true",
        help="profile one step and print its kernels, instead of timing ops; "
        "eager, so kernel durations are real but the gaps between them are not",
    )
    parser.add_argument(
        "--CONC",
        type=int,
        default=1,
        help="concurrent requests, global across DP ranks",
    )
    return parser


# The synced loop takes N+2 samples and reports the fastest; fewer samples do not clear the warm-up.
_IMPLIED_FULL_STEP = 3


if __name__ == "__main__":
    args = make_parser().parse_args()
    model_bench.run(work, model_bench.build_server_args(args), args)
    # The per-op pass cannot see sampling, accept/reject and bookkeeping between forwards, so --full-step is implied.
    if args.speculative_algorithm is not None and not args.full_step:
        args.full_step = _IMPLIED_FULL_STEP
        model_bench.run(work, model_bench.build_server_args(args), args)
