"""DeepSeek-V4 decode: per-module graph timing and full-forward comparison."""

import os
import sys
import time
from pathlib import Path

import torch

# sglang_bench.py and kernel_breakdown.py live one directory up.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import kernel_breakdown
import model_bench
import sglang_bench


def work(server_args, port_args, args, gpu_id, tp_rank):
    mr, rank_print = model_bench.setup(server_args, port_args, gpu_id, tp_rank)
    layers = model_bench.Layers(mr)
    model_bench.balance_moe_routing(layers)
    limit, why = model_bench.decode_capacity(args, mr, layers)
    dp = sglang_bench.parallel_state(mr).attn_dp_size
    for conc in args.concs:
        args.CONC = conc
        if conc % dp:
            rank_print(f"[UNSUPPORTED] CONC {conc} does not split over {dp} DP ranks")
        elif conc > limit:
            rank_print(
                f"[BLOCKED_CAPACITY] CONC {conc} at ISL {args.ISL}: max fitting"
                f" CONC {limit} ({why}); not reduced"
            )
        else:
            measure_conc(args, mr, layers, rank_print, tp_rank)


def measure_conc(args, mr, layers, rank_print, tp_rank) -> None:
    """Time, record and profile one CONC; rank 0 writes its report."""
    with sglang_bench.forward_ctx(mr):
        if args.full_step:
            model_bench.randomize_kv(mr)
            ms = []
            # Fresh native batch per sample, prepared outside the clock.
            for _ in range(args.full_step + 2):
                batch = model_bench.make_decode_batch(args, mr)
                fb = sglang_bench.forward_batch_of(batch, mr)
                torch.cuda.synchronize()
                start = time.perf_counter()
                mr.forward(fb)
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - start) * 1000.0)
            rank_print("synced forward ms: " + " ".join(f"{m:.3f}" for m in ms))
            rank_print(f"CONC {args.CONC} synthetic forward: synced {min(ms):.3f} ms")
            return

        batch = model_bench.make_decode_batch(args, mr)
        model_bench.randomize_kv(mr)
        fb = sglang_bench.forward_batch_of(batch, mr)
        ops = model_bench.collect_ops(layers, mr.model)
        saved = {}
        if args.trace:
            sglang_bench.trace_step(args, fb, mr, rank_print, tp_rank)
            return

        with model_bench.record_inputs(ops, saved):
            mr.forward(fb)
        torch.cuda.synchronize()
        sglang_bench.assert_lockstep(saved)
        if args.export_only:
            from aklr.core.runtime.dispatch import DispatcherRegistry

            # the record must not wait for the recorder's periodic or exit-time save
            recorder = DispatcherRegistry.default().recorder
            if recorder is not None:
                recorder.save()
            rank_print(f"CONC {args.CONC}: one eager forward recorded")
            return

        rank_print("timing: full forward")
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
                kind, _, op_name = name.partition("|")
                if kind in layers.kinds and op_name in ("layer", "attn", "mlp"):
                    # A real forward visits different layer weights; one representative would make parent rows cache-hot.
                    attr = {"layer": "", "attn": "self_attn", "mlp": "mlp"}[op_name]
                    modules = [
                        getattr(layers.layers[i], attr) if attr else layers.layers[i]
                        for i in layers.kinds[kind]
                    ]
                    fns = [model_bench.make_call(mod, entry) for mod in modules]
                    times[name] = sglang_bench.bench(
                        fns,
                        args.reps,
                        inner=max(sglang_bench.GRAPH_REPS, len(fns)),
                        rounds=args.repeat,
                    )
                else:
                    # Rotated so a MALL-sized weight is not read cache-hot.
                    with model_bench.rotated_calls(ops[name], entry) as (fns, inner):
                        times[name] = sglang_bench.bench(fns, args.reps, inner=inner, rounds=args.repeat)
            except torch.cuda.OutOfMemoryError:
                # Not a shape mismatch: every later op would be skipped too, so abort instead of a partial report.
                raise
            except RuntimeError as e:
                # A standalone replay may violate shape or forward-state constraints; only a deterministic failure may be skipped.
                first = str(e).split("\n")[0]
                if "out of memory" in first.lower():
                    raise
                skipped[name] = first
                rank_print(f"  skipped: {skipped[name]}")
            # Agreed across ranks: a rank that kept a row the others dropped would unpair their collectives.
            if not sglang_bench.agreed(name in times):
                times.pop(name, None)
                skipped.setdefault(name, "skipped on another rank")

        times = model_bench.world_times({**times, "full": full_ms}, mr)
        full_ms = times.pop("full")

        # Profile after every timing pass so hooks/Kineto never enter the clocks; all ranks replay to pair collectives.
        profiles, profile_errors = profile_saved_ops(ops, saved, times, rank_print)

    if tp_rank != 0:
        return

    if skipped:
        rank_print(f"{len(skipped)} op(s) could not be replayed standalone:")
        for name, err in skipped.items():
            rank_print(f"  {name}: {err}")

    md, headline = report(
        args,
        mr,
        layers,
        times,
        full_ms,
        model_bench.coverage_note(ops, saved),
        skipped=model_bench.skipped_note(skipped),
        kernels=model_bench.kernel_totals(profiles, layers.counts, times),
        kernel_errors=profile_errors,
    )
    path = Path(args.out or default_out(args, mr, layers))
    if args.out and len(args.concs) > 1:
        path = path.with_name(f"{path.stem}_conc{args.CONC}{path.suffix}")
    path.write_text(md)
    rank_print(headline)
    rank_print(f"report -> {path.resolve()}")


def profile_saved_ops(ops, saved, times, rank_print):
    """Kineto on the same recorded module calls; parent samples are exclusive."""
    profiles, errors = {}, {}
    for label in times:
        kind, _, name = label.rpartition("|")
        children = [
            ops[f"{kind}|{child}"]
            for child in model_bench.RESIDUALS.get(name, ())
            if f"{kind}|{child}" in times
        ]
        rank_print(f"kernels: {label}")
        error = ""
        try:
            profiles[label] = kernel_breakdown.profile_call(
                model_bench.make_call(ops[label], saved[label]), children
            )
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError as exc:
            error = str(exc).split("\n")[0]
            if "out of memory" in error.lower():
                raise
        if not sglang_bench.agreed(label in profiles):
            profiles.pop(label, None)
            error = error or "kernel profiling unavailable on another rank"
        if error:
            row = name + ".<rest>" if name in model_bench.RESIDUALS else name
            errors.setdefault(row, []).append(f"{kind or 'global'}: {error}")
            rank_print(f"  kernel breakdown unavailable: {error}")
    return profiles, errors


def default_out(args, mr, layers) -> str:
    parts = [
        model_bench.model_tag(mr.server_args),
        "decode",
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
    swa_prefix = model_bench.decode_swa_prefix(args, mr)
    moe_bound = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
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
        (
            "KV setup",
            "synthetic performance data; "
            + (
                f"native SWA-tail allocation, prefix [0, {swa_prefix}) "
                "omitted from SWA only; full/compressed context unchanged"
                if swa_prefix
                else "native full-prefix allocation"
            ),
        ),
        ("kv dtype", f"`{sa.kv_cache_dtype}`"),
        ("weights", model_bench.weights_line(mr)),
        *(
            [
                (
                    "AITER activation dispatch",
                    (f"`AITER_BF16_FP8_MOE_BOUND={moe_bound}` (native FP8 activation threshold in tokens)"),
                )
            ]
            if moe_bound is not None
            else []
        ),
        (
            "measurement",
            f"{args.reps} graph replays per op; {sglang_bench.GRAPH_REPS} calls per"
            f" graph, more for a weight small enough to rotate over the"
            f" {model_bench.ROTATION_BYTES >> 20} MB buffer"
            + (f", fastest of {args.repeat} passes" if args.repeat > 1 else "")
            + (", max over ranks (EP)" if sglang_bench.moe_parallel(mr).ep_size > 1 else ", min over ranks"),
        ),
    ]
    return sglang_bench.table(rows, ("item", "value"))


def summary_table(args, mr, full_ms, rollup) -> str:
    rows = [
        (
            "step time",
            f"{full_ms:.3f} ms",
            "whole synthetic model forward in one graph; not serving ITL",
        ),
        ("roll-up", f"{rollup:.3f} ms", "sum of the per-op leaf rows below"),
        (
            "ratio",
            f"{rollup / full_ms:.3f}",
            "roll-up / forward time; agreement does not prove coverage or numerical correctness",
        ),
        (
            "throughput",
            f"{args.CONC / full_ms * 1000:.1f} tok/s",
            f"at CONC {args.CONC}",
        ),
    ]
    return sglang_bench.table(rows, ("metric", "value", "note"))


def hotspot_section(totals, rollup, contents, kernels=None, kernel_errors=None) -> str:
    rows = []
    kernel_errors = kernel_errors or {}
    for name, ms in sorted(totals.items(), key=lambda kv: -kv[1]):
        row = (
            model_bench.tree_name(name, 0, contents),
            ms,
            f"{ms / rollup * 100:.1f}%",
        )
        rows.append(row if kernels is None else (*row, "", "", "", ""))
        if kernels is None:
            continue
        for error in kernel_errors.get(name, ()):
            message = error.replace("|", "\\|").replace("\n", " ")
            rows.append(
                (
                    f"  ↳ **Incomplete kernel profile: {message}**",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                )
            )
        if name not in kernels:
            rows.append(("  ↳ kernel profile unavailable", "", "", "", "", "", ""))
            continue
        items = kernels[name]
        total = sum(v[0] for v in items.values())
        if not items:
            rows.append(
                (
                    "  ↳ no exclusive GPU kernels (all in child rows)",
                    "",
                    "",
                    0,
                    0.0,
                    "n/a",
                    "",
                )
            )
        for key, (gpu_ms, calls, _) in sorted(items.items(), key=lambda kv: (-kv[1][0], str(kv[0]))):
            kernel, api = key if isinstance(key, tuple) else (key, "unavailable")
            symbol = kernel.replace("|", "\\|").replace("\n", " ")
            api = api.replace("|", "\\|").replace("\n", " ")
            share = f"{gpu_ms / total * 100:.2f}%" if total else "n/a"
            rows.append((f"  ↳ `{symbol}`", "", "", calls, gpu_ms, share, f"`{api}`"))
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
    headers = ("op / kernel", "ms/step", "share")
    if kernels is not None:
        headers += (
            "kernel calls/step",
            "profile GPU ms/step",
            "% of module GPU",
            "Python / dispatcher API",
        )
        note += (
            "\n\nIndented rows list **all** Kineto GPU kernels/copies from rank 0's"
            " standalone module replays (one eager sample per layer kind, multiplied"
            " by its layer count; globals once). Kernel percentages sum to 100%"
            " **within each module**, not across this table. Parent `<rest>` samples"
            " exclude separately measured children through module hooks, not by"
            " subtracting independent kernel timings. Fused kernels stay intact."
            " The API column is derived from the kernel's Kineto runtime correlation,"
            " nearest recorded Python frame, and dispatcher external ID. A Python"
            " entry is `module.function:line`, where the line is what the interpreter"
            " records for that function -- its `def`, or its first decorator."
            " Original ms/step and share are unchanged graph timings; the separate"
            " profile columns are diagnostic, not a redistribution of those times."
            " Profiling can alter clocks, launch gaps and TP communication waits."
        )
        # Kineto shortens paths and this image ships two copies of aiter; say which ran.
        roots = kernel_breakdown.package_roots(kernels.values())
        if roots:
            listed = ", ".join(f"`{name}` = `{path}`" for name, path in roots.items())
            note += f"\n\nPackages these APIs were loaded from: {listed}."
        # Full GPU symbols can be very long; don't pad every other row to their width.
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            cells = [f"{value:.3f}" if isinstance(value, float) else str(value) for value in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + note
    return sglang_bench.table(rows, headers) + note


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


def report(
    args,
    mr,
    layers,
    times,
    full_ms,
    coverage="",
    skipped="",
    kernels=None,
    kernel_errors=None,
):
    """Returns (markdown, one-line headline for the log)."""
    from aklr.core.runtime.seam import routes
    from aklr.ops import load_routes

    load_routes()

    counts = layers.counts
    model_bench.add_residuals(times, counts)
    totals = model_bench.leaf_totals(times, counts)
    rollup = sum(totals.values())
    contents = model_bench.residual_contents(mr)
    routed = frozenset(route.api for route in routes())
    ranking = kernel_breakdown.Ranking.of(kernels or {}, totals, routed)

    sections = [
        ("Summary", summary_table(args, mr, full_ms, rollup)),
        ("Top kernels -- one step", ranking.section(sglang_bench.table)),
        ("Config", config_table(args, mr, layers)),
        (
            "Versions",
            sglang_bench.table(sglang_bench.versions(), ("component", "version")),
        ),
        ("Memory, per rank", model_bench.memory_section(args, mr, layers)),
        (
            "Hotspots -- one step, all layers",
            hotspot_section(totals, rollup, contents, kernels, kernel_errors),
        ),
        ("Per layer kind -- one step", kind_sections(times, counts, contents)),
        ("Roll-up -- one step", rollup_table(times, counts)),
        *([("Coverage", coverage)] if coverage else []),
        # Beside Coverage: those ops fired but could not be timed, and only they contaminate a `<rest>` row.
        *([("Not replayable", skipped)] if skipped else []),
    ]
    md = "\n\n".join(
        [
            f"# {model_bench.model_tag(mr.server_args)} decode profile",
            time.strftime("%Y-%m-%d %H:%M %Z"),
            # Above the tables, not buried in Config.
            *filter(None, [model_bench.dummy_banner(mr.server_args)]),
        ]
        + [f"## {title}\n\n{body}" for title, body in sections]
    )
    headline = (
        f"step {full_ms:.3f} ms  "
        f"roll-up {rollup:.3f} ms  ratio {rollup / full_ms:.3f}  "
        f"{args.CONC / full_ms * 1000:.1f} tok/s"
    )
    return md + "\n", headline


def make_parser():
    parser = model_bench.make_parser()
    parser.add_argument(
        "--full-step",
        type=int,
        default=0,
        metavar="N",
        help="time N+2 synthetic forwards under native graphs and report the fastest; "
        "not a serving loop or end-to-end ITL",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="profile one step and print its kernels, instead of timing ops; "
        "eager, so kernel durations are real but the gaps between them are not",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="one eager forward per CONC for AKLR_EXPORT, no timing",
    )
    parser.add_argument(
        "--CONC",
        type=int,
        nargs="+",
        default=[1],
        help="concurrent requests, global across DP ranks; several values sweep in one model load",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    args.concs = sorted(set(args.CONC))
    if args.concs[0] <= 0:
        raise ValueError("--CONC must be positive")
    if args.trace and len(args.concs) > 1:
        raise ValueError("--trace takes a single --CONC")
    if args.export_only and (args.trace or args.full_step):
        raise ValueError("--export-only excludes --trace and --full-step")
    # Any DP split divides this; work() reports each CONC that does not split.
    args.CONC = args.concs[-1] * args.dp_size
    model_bench.run(work, model_bench.build_server_args(args), args)
