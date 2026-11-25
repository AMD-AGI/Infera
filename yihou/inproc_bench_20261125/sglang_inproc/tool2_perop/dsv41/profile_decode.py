###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
# Provenance: port of AKLR `aklr/benchmarks/models/{glm52,deepseekv4}/profile_decode.py` @ 33689332.
# Union of both: glm52's MTP path (spec step, per-forward timing, draft row, simulated-acceptance
# throughput, implied --full-step) and deepseekv4's CONC sweep, capacity refusal, rank-max under EP,
# Kineto kernel breakdown with top-5 targets and --export-only.

"""Per-operator decode timing for DeepSeek-V4.1, rolled up to a per-step estimate."""

import contextlib
import os
import sys
import time
from pathlib import Path

import torch

# sglang_bench.py, kernel_breakdown.py and shape_export.py live one directory up.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import kernel_breakdown  # noqa: E402
import model_bench  # noqa: E402
import sglang_bench  # noqa: E402
import shape_export  # noqa: E402


def work(server_args, port_args, args, gpu_id, tp_rank):
    mr, spec, rank_print = model_bench.setup(server_args, port_args, gpu_id, tp_rank)
    dp = sglang_bench.parallel_state(mr).attn_dp_size
    if dp != args.dp_raw:
        raise RuntimeError(f"runtime attn_dp_size {dp} != launcher dp {args.dp_raw}")
    layers = model_bench.Layers(mr)
    n_bal = model_bench.balance_moe_routing(layers, args.routing)
    n_bal_draft = model_bench.balance_draft_routing(spec, args.routing)
    rank_print(f"layers: {layers.describe()}")
    rank_print(f"balanced routing installed on {n_bal} target + {n_bal_draft} draft MoE layers")
    limit, why = model_bench.decode_capacity(args, mr, layers)
    for conc in args.concs:
        args.CONC = conc
        if conc % dp:
            rank_print(f"[UNSUPPORTED] CONC {conc} does not split over {dp} DP ranks")
        elif conc > limit:
            rank_print(f"[BLOCKED_CAPACITY] CONC {conc} at ISL {args.ISL}: max fitting CONC {limit} ({why}); not reduced")
        else:
            measure_conc(args, mr, spec, layers, rank_print, tp_rank)


def full_step(args, mr, spec, rank_print) -> None:
    """N+2 whole steps under sglang's own graphs, synced per step, fastest wins (the server-comparable number)."""
    import spec_dspark

    replays = _count_graph_replays(mr)
    ms, ev_per = [], []
    phases = spec_dspark.phases(spec) if spec is not None else []
    stepper = spec_dspark.Stepper(args, mr, spec) if spec is not None else None
    if spec is None:
        model_bench.randomize_kv(mr)
    for _ in range(args.full_step + 2):
        if spec is None:
            # Fresh native batch per sample, prepared outside the clock.
            b = model_bench.make_decode_batch(args, mr, args.kv_fill)
            fb = sglang_bench.forward_batch_of(b, mr)
            run = lambda: mr.forward(fb)  # noqa: E731
        else:
            run = stepper.prepare()
        ev = {}
        ctx = time_calls(phases, ev) if phases else contextlib.nullcontext()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with ctx:
            run()
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1000.0)
        ev_per.append(ev)
    best = min(range(len(ms)), key=ms.__getitem__)
    rank_print("synced per step ms: " + " ".join(f"{m:.3f}" for m in ms))
    for label, (total, calls) in elapsed(ev_per[best]).items():
        rank_print(f"  {label}: {total:.3f} ms  {calls} calls/step")
    n = replays["n"] if replays else None
    rank_print(f"decode graph replays during the {len(ms)} samples: {n if n is not None else 'not countable'}")
    acc = model_bench.accept_len() if spec is not None else None
    tok = args.CONC * (acc or 1) / ms[best] * 1000
    rank_print(
        f"CONC {args.CONC} step: synced {ms[best]:.3f} ms"
        + (f"  {tok:.1f} tok/s" if spec is None or acc else "  tok/s not reported (no SGLANG_SIMULATE_ACC_LEN)")
        + (f" at accept len {acc:.2f}" if acc else "")
    )
    if stepper is not None:
        stepper.close()


def measure_conc(args, mr, spec, layers, rank_print, tp_rank) -> None:
    """Time, record and profile one CONC; rank 0 writes its report."""
    import spec_dspark

    with sglang_bench.forward_ctx(mr):
        if args.full_step:
            full_step(args, mr, spec, rank_print)
            return

        stepper = spec_dspark.Stepper(args, mr, spec) if spec is not None else None
        if spec is None:
            batch = model_bench.make_decode_batch(args, mr, args.kv_fill)
            filled = model_bench.randomize_kv(mr) if args.kv_fill == "random" else {}
            fb = sglang_bench.forward_batch_of(batch, mr)
            step = lambda: mr.forward(fb)  # noqa: E731
        else:
            filled = {}
            step = None
        ops = model_bench.collect_ops(layers, mr.model)
        if spec is not None:
            ops.update(spec_dspark.draft_ops(spec))

        if args.trace:
            trace_step(args, (lambda: stepper.prepare()()) if spec else step, rank_print, tp_rank)
            return

        saved = {}
        with shape_export.export_scope(ops, f"conc{args.CONC}", args.result_dir, tp_rank):
            with model_bench.record_inputs(ops, saved, layers):
                (stepper.prepare() if spec else step)()
        torch.cuda.synchronize()
        sglang_bench.assert_lockstep(saved)
        if args.export_only:
            rank_print(f"CONC {args.CONC}: one eager {'step' if spec else 'forward'} recorded for shape export")
            return

        # The reference the roll-up is checked against, graph-captured so it is launch-free like the per-op rows.
        fwd_ms = {}
        if spec is not None:
            rank_print("timing: forwards")
            # A step is several forwards across two models; each is timed in place under its own graph and summed.
            with bench_forwards(spec_dspark.models(spec), args.reps, args.repeat, fwd_ms):
                stepper.prepare()()
            for name, each in fwd_ms.items():
                rank_print(f"  {name}: {sum(each):.3f} ms = " + " ".join(f"{m:.3f}" for m in each))
            full_ms = sum(sum(v) for v in fwd_ms.values())
        else:
            rank_print("timing: full forward")
            full_ms = sglang_bench.bench(
                lambda: mr.model.forward(fb.input_ids, fb.positions, fb), args.reps, inner=1, rounds=args.repeat
            )

        times, skipped = {}, {}
        for name, entry in saved.items():
            rank_print(f"timing: {name}")
            try:
                kind, _, op_name = name.partition("|")
                if kind in layers.kinds and op_name in ("layer", "attn", "mlp"):
                    # A real forward visits different layer weights; one representative would make parent rows cache-hot.
                    attr = {"layer": "", "attn": "self_attn", "mlp": "mlp"}[op_name]
                    ids = layers.kinds[kind]
                    mods = [getattr(layers.layers[i], attr) if attr else layers.layers[i] for i in ids]
                    fns = [
                        model_bench.make_call(m, entry, layer_index=i if op_name == "layer" else None)
                        for i, m in zip(ids, mods)
                    ]
                    times[name] = sglang_bench.bench(
                        fns, args.reps, inner=max(sglang_bench.GRAPH_REPS, len(fns)), rounds=args.repeat
                    )
                else:
                    # Rotated so a MALL-sized weight is not read cache-hot.
                    with model_bench.rotated_calls(ops[name], entry) as (fns, inner):
                        times[name] = sglang_bench.bench(fns, args.reps, inner=inner, rounds=args.repeat)
            except torch.cuda.OutOfMemoryError:
                # Not a shape mismatch: every later op would be skipped too, so abort instead of a partial report.
                raise
            except (RuntimeError, AssertionError, ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
                # A standalone replay may violate shape or forward-state constraints; only a deterministic failure may be skipped.
                first = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
                if "out of memory" in first.lower():
                    raise
                skipped[name] = first[:300]
                rank_print(f"  skipped: {skipped[name]}")
                torch.cuda.synchronize()
            # Agreed across ranks: a rank that kept a row the others dropped would unpair their collectives.
            if not sglang_bench.agreed(name in times):
                times.pop(name, None)
                skipped.setdefault(name, "skipped on another rank")

        times = model_bench.world_times({**times, "__full": full_ms}, mr)
        full_ms = times.pop("__full")

        # Profile after every timing pass so hooks/Kineto never enter the clocks; all ranks replay to pair collectives.
        profiles, profile_errors = ({}, {}) if args.no_kernels else profile_saved_ops(ops, saved, times, rank_print)

    if stepper is not None:
        stepper.close()
    # Draft rows sit outside the roll-up on purpose: the draft model is neither a kind nor a per-layer row.
    draft_rows = {k: v for k, v in times.items() if k.startswith("draft|")}
    for k in draft_rows:
        times.pop(k)
        ops.pop(k, None)
        saved.pop(k, None)
        profiles.pop(k, None)

    if tp_rank != 0:
        return
    if skipped:
        rank_print(f"{len(skipped)} op(s) could not be replayed standalone:")
        for name, err in skipped.items():
            rank_print(f"  {name}: {err}")
    if draft_rows:
        layer_ms = draft_rows.get("draft|layer")
        rank_print("draft model, ms per call (one draft-forward shape):")
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
        spec=spec,
        draft_ms=sum(fwd_ms.get("draft", [])) if spec else 0.0,
        draft_rows=draft_rows,
        skipped=model_bench.skipped_note(skipped),
        kernels=None if args.no_kernels else model_bench.kernel_totals(profiles, layers.counts, times),
        kernel_errors=profile_errors,
        filled=filled,
    )
    path = Path(args.out or Path(args.result_dir or ".") / default_out(args, mr, layers))
    if args.out and len(args.concs) > 1:
        path = path.with_name(f"{path.stem}_conc{args.CONC}{path.suffix}")
    path.write_text(md)
    rank_print(headline)
    rank_print(f"report -> {path.resolve()}")


def _count_graph_replays(mr):
    """Count decode CUDA-graph executions (sglang 6be0d75ae: decode_cuda_graph_runner.execute)."""
    gr = getattr(mr, "decode_cuda_graph_runner", None)
    if gr is None or gr is getattr(mr, "eager_runner", object()):
        return {"n": 0} if gr is not None else None
    name = "execute" if hasattr(gr, "execute") else ("replay" if hasattr(gr, "replay") else None)
    if name is None:
        return None
    box, orig = {"n": 0}, getattr(gr, name)

    def wrapped(*a, **k):
        box["n"] += 1
        return orig(*a, **k)

    setattr(gr, name, wrapped)
    return box


@contextlib.contextmanager
def time_calls(targets, into: dict):
    """CUDA-event timing around chosen methods, one event pair per call (glm52's spec phase split)."""
    saved = []
    for label, obj, name in targets:
        if obj is None or not hasattr(obj, name):
            continue
        orig = getattr(obj, name)
        patched_forward = False
        if isinstance(orig, torch.nn.Module):
            obj, name, orig = orig, "forward", orig.forward
            patched_forward = True

        def wrapped(*a, _l=label, _o=orig, **kw):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            out = _o(*a, **kw)
            end.record()
            into.setdefault(_l, []).append((start, end))
            return out

        setattr(obj, name, wrapped)
        saved.append((obj, name, orig, patched_forward))
    try:
        yield
    finally:
        for obj, name, orig, patched_forward in reversed(saved):
            if patched_forward or name in getattr(obj, "__dict__", {}):
                try:
                    delattr(obj, name)
                except AttributeError:
                    setattr(obj, name, orig)
            else:
                setattr(obj, name, orig)


def elapsed(into: dict) -> dict:
    return {k: (sum(s.elapsed_time(e) for s, e in pairs), len(pairs)) for k, pairs in into.items()}


@contextlib.contextmanager
def bench_forwards(models: dict, reps: int, rounds: int, into: dict):
    """Time every top-level forward of one step, each with its own graph (glm52's MTP reference)."""
    originals = {name: model.forward for name, model in models.items()}
    for name, model in models.items():

        def wrapped(*a, _n=name, _orig=originals[name], **kw):
            into.setdefault(_n, []).append(sglang_bench.bench(lambda: _orig(*a, **kw), reps, inner=1, rounds=rounds))
            return _orig(*a, **kw)

        model.forward = wrapped
    try:
        yield
    finally:
        for name, model in models.items():
            model.__dict__.pop("forward", None)


def profile_saved_ops(ops, saved, times, rank_print):
    """Kineto on the same recorded module calls; parent samples are exclusive."""
    profiles, errors = {}, {}
    for label in times:
        kind, _, name = label.rpartition("|")
        children = [
            ops[f"{kind}|{child}"] for child in model_bench.RESIDUALS.get(name, ()) if f"{kind}|{child}" in times
        ]
        rank_print(f"kernels: {label}")
        error = ""
        fn_child = name == "attn" and f"{kind}|attn.sources" in times
        try:
            with model_bench.fn_child_ranges(fn_child):
                profiles[label] = kernel_breakdown.profile_call(model_bench.make_call(ops[label], saved[label]), children)
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, AssertionError, ValueError, TypeError, AttributeError, KeyError) as exc:
            error = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
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


def trace_step(args, step, rank_print, tp_rank) -> None:
    """Kernel-level breakdown of one step: what the per-op rows are made of."""
    from torch.profiler import ProfilerActivity, profile

    step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    top, rows = sglang_bench.kernel_rows(prof)
    total = sum(ms for ms, _ in rows.values())
    rank_print(f"kernels: {len(rows)} distinct, {sum(n for _, n in rows.values())} launches, {total:.3f} ms")
    for name, (ms, n) in top:
        rank_print(f"  {ms:8.3f} ms  {n:5d} x {ms * 1000 / n:7.2f} us  {name[:78]}")
    # Rank 0 only: export_chrome_trace renames <path>.tmp, so ranks sharing one path race.
    if tp_rank != 0:
        return
    path = Path(args.out or Path(args.result_dir or ".") / "trace.json")
    prof.export_chrome_trace(str(path))
    rank_print(f"trace -> {path.resolve()}")


def default_out(args, mr, layers) -> str:
    parts = [
        model_bench.model_tag(mr.server_args),
        "mtp" if model_bench.cfg(mr.server_args).speculative_algorithm is not None else "decode",
        f"isl{args.ISL}",
        f"conc{args.CONC}",
        *model_bench.parallel_tags(mr, layers),
        str(mr.server_args.kv_cache_dtype),
    ]
    # Keeps a dummy run from overwriting the real run of the same shape.
    parts += filter(None, [sglang_bench.load_tag(mr.server_args)])
    parts.append(time.strftime("%Y%m%d"))
    return "_".join(parts) + ".md"


def config_table(args, mr, layers, filled=None) -> str:
    sa = mr.server_args
    c = model_bench.cfg(sa)
    moe_bound = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    rows = [
        ("model", f"`{type(mr.model).__name__}`  {sa.model_path}"),
        ("layers", layers.describe()),
        ("parallel", model_bench.parallel_line(mr)),
        *([("moe", line)] if (line := model_bench.moe_line(mr, layers)) else []),
        ("batch", f"CONC {args.CONC}, {sglang_bench.local_conc(args, mr)} requests per rank"),
        ("context", f"ISL {args.ISL} per request"),
        (
            "KV setup",
            "synthetic: native full allocation of the ISL context, "
            + ("random KV bytes (no prefill forward)" if args.kv_fill == "random" else "one real prefill forward of random token ids")
            + (", target + draft pools" if c.speculative_algorithm else ""),
        ),
        (
            "routing",
            f"seeded balanced (seed {sglang_bench.MOE_ROUTING_SEED}) on every MoE layer, hash layers included"
            if args.routing == "balanced"
            else "**native router on synthetic hidden states (--routing native, diagnostic)**",
        ),
        ("kv dtype", f"`{sa.kv_cache_dtype}`"),
        *([("spec", model_bench.spec_note(sa))] if c.speculative_algorithm is not None else []),
        ("weights", model_bench.weights_line(mr)),
        *(
            [("AITER activation dispatch", f"`AITER_BF16_FP8_MOE_BOUND={moe_bound}`")]
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


def summary_table(args, mr, full_ms, rollup, spec=None) -> str:
    import spec_dspark

    rows = [
        (
            "forward time" if spec else "step time",
            f"{full_ms:.3f} ms",
            (
                spec_dspark.forward_note(spec)
                + " Not the whole step: sampling, accept/reject and bookkeeping between them are only"
                " visible under --full-step (run automatically after this pass)."
                if spec
                else "whole synthetic model forward in one graph; not serving ITL"
            ),
        ),
        ("roll-up", f"{rollup:.3f} ms", "sum of the per-op leaf rows below" + (" + the draft model row" if spec else "")),
        (
            "ratio",
            f"{rollup / full_ms:.3f}",
            "roll-up / forward time; agreement does not prove coverage or numerical correctness"
            + ("; the draft is one row in both" if spec else ""),
        ),
    ]
    acc = model_bench.accept_len() if spec else None
    if spec is None:
        rows.append(("throughput", f"{args.CONC / full_ms * 1000:.1f} tok/s", f"at CONC {args.CONC}"))
    elif acc is None:
        rows.append(
            (
                "throughput",
                "not reported",
                "a speculative step emits as many tokens as were accepted. Set SGLANG_SIMULATE_ACC_LEN, the way"
                " the server runs, for a tok/s row; CONC / step time would understate it by exactly that factor.",
            )
        )
    else:
        rows.append(
            (
                "throughput",
                f"{args.CONC * acc / full_ms * 1000:.1f} tok/s",
                f"at CONC {args.CONC}, accept len {acc:.2f} from SGLANG_SIMULATE_ACC_LEN -- an assumption the server"
                " is also run under, not something measured here",
            )
        )
        rows.append(
            ("ITL", f"{full_ms / acc:.3f} ms", "forward time / accept len, per request -- a floor, the non-forward part is not counted")
        )
    return sglang_bench.table(rows, ("metric", "value", "note"))


def hotspot_section(totals, rollup, contents, kernels=None, kernel_errors=None) -> str:
    rows = []
    kernel_errors = kernel_errors or {}
    for name, ms in sorted(totals.items(), key=lambda kv: -kv[1]):
        row = (model_bench.tree_name(name, 0, contents), ms, f"{ms / rollup * 100:.1f}%")
        rows.append(row if kernels is None else (*row, "", "", "", ""))
        if kernels is None:
            continue
        for error in kernel_errors.get(name, ()):
            message = error.replace("|", "\\|").replace("\n", " ")
            rows.append((f"  ↳ **Incomplete kernel profile: {message}**", "", "", "", "", "", ""))
        if name not in kernels:
            rows.append(("  ↳ kernel profile unavailable", "", "", "", "", "", ""))
            continue
        items = kernels[name]
        total = sum(v[0] for v in items.values())
        if not items:
            rows.append(("  ↳ no exclusive GPU kernels (all in child rows)", "", "", 0, 0.0, "n/a", ""))
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
        " a `<rest>` small next to its parent is mostly that error, and can read negative. "
    )
    if "mlp.experts" in contents:
        note += "`mlp.experts` is labelled too: under an all-to-all the dispatch and combine run inside it."
    headers = ("op / kernel", "ms/step", "share")
    if kernels is None:
        return sglang_bench.table(rows, headers) + note
    headers += ("kernel calls/step", "profile GPU ms/step", "% of module GPU", "Python / dispatcher API")
    note += (
        "\n\nIndented rows list **all** Kineto GPU kernels/copies from rank 0's standalone module replays (one"
        " eager sample per layer kind, multiplied by its layer count; globals once). Kernel percentages sum to"
        " 100% **within each module**. Parent `<rest>` samples exclude separately measured children through module"
        " hooks. The API column comes from the kernel's Kineto runtime correlation, nearest recorded Python frame,"
        " and dispatcher external ID. ms/step and share are graph timings; the profile columns are diagnostic."
    )
    roots = kernel_breakdown.package_roots(kernels.values())
    if roots:
        note += "\n\nPackages these APIs were loaded from: " + ", ".join(f"`{n}` = `{p}`" for n, p in roots.items()) + "."
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines) + note


def kind_sections(times, counts, contents) -> str:
    out = []
    for kind, n in counts.items():
        rows = [
            (model_bench.tree_name(name, depth, contents), times[f"{kind}|{name}"], times[f"{kind}|{name}"] * n)
            for name, depth in model_bench.TREE
            if f"{kind}|{name}" in times
        ]
        out.append(f"### {kind} -- x{n} layers\n\n" + sglang_bench.table(rows, ("op", "ms/call", "ms/step")))
    return "\n\n".join(out)


def rollup_table(times, counts) -> str:
    rows = []
    for kind, n in counts.items():
        for op in ("layer", *model_bench.OUTSIDE_LAYER):
            if f"{kind}|{op}" in times:
                rows.append((f"{kind} {op}", n, times[f"{kind}|{op}"], times[f"{kind}|{op}"] * n))
    rows += [(name, 1, times[name], times[name]) for name in model_bench.GLOBAL_OPS if name in times]
    return sglang_bench.table(rows, ("op", "x", "ms/call", "ms/step"))


def draft_section(draft_rows: dict, draft_ms: float) -> str:
    rows = [
        (f"`{k.split('|', 1)[1]}`", v, f"{v / draft_ms * 100:.1f}%" if draft_ms else "")
        for k, v in sorted(draft_rows.items(), key=lambda kv: -kv[1])
    ]
    return sglang_bench.table(rows, ("draft op", "ms/call", "% of the draft model row")) + (
        f"\n\nThe `draft model` row in the roll-up is the draft forward(s) of one step timed in place under their"
        f" own graphs ({draft_ms:.3f} ms), not this table: the draft is not decomposed into the roll-up (the"
        " glm52 port keeps it one row too). These per-op rows are one call each, for ranking inside the draft."
    )


def report(args, mr, layers, times, full_ms, coverage="", spec=None, draft_ms=0.0, draft_rows=None, skipped="",
           kernels=None, kernel_errors=None, filled=None):
    """Returns (markdown, one-line headline for the log)."""
    counts = layers.counts
    model_bench.add_residuals(times, counts)
    totals = model_bench.leaf_totals(times, counts)
    if draft_ms:
        # One row, not a per-op tree: the draft model is run several times per step.
        totals["draft model"] = draft_ms
    rollup = sum(totals.values())
    contents = model_bench.residual_contents(mr)
    routed = frozenset()
    try:
        from aklr.core.runtime.seam import routes
        from aklr.ops import load_routes

        load_routes()
        routed = frozenset(route.api for route in routes())
    except Exception:  # noqa: BLE001 -- no aklr in the image: group by the innermost frame
        pass
    ranking = kernel_breakdown.Ranking.of(kernels or {}, totals, routed)

    sections = [
        ("Summary", summary_table(args, mr, full_ms, rollup, spec)),
        *([("Top kernels -- one step", ranking.section(sglang_bench.table))] if kernels is not None else []),
        ("Config", config_table(args, mr, layers, filled)),
        ("Versions", sglang_bench.table(sglang_bench.versions(), ("component", "version"))),
        ("Memory, per rank", model_bench.memory_section(args, mr, layers, spec, filled=filled)),
        ("Hotspots -- one step, all layers", hotspot_section(totals, rollup, contents, kernels, kernel_errors)),
        ("Per layer kind -- one step", kind_sections(times, counts, contents)),
        ("Roll-up -- one step", rollup_table(times, counts)),
        *([("Draft model -- per op", draft_section(draft_rows, draft_ms))] if draft_rows else []),
        *([("Coverage", coverage)] if coverage else []),
        # Beside Coverage: those ops fired but could not be timed, and only they contaminate a `<rest>` row.
        *([("Not replayable", skipped)] if skipped else []),
    ]
    md = "\n\n".join(
        [
            f"# {model_bench.model_tag(mr.server_args)} {'MTP ' if spec else ''}decode profile",
            time.strftime("%Y-%m-%d %H:%M %Z"),
            *filter(None, [model_bench.dummy_banner(mr.server_args)]),
        ]
        + [f"## {title}\n\n{body}" for title, body in sections]
    )
    acc = model_bench.accept_len() if spec else None
    tok = f"{args.CONC * (acc or 1) / full_ms * 1000:.1f} tok/s" if (spec is None or acc) else "tok/s not reported"
    headline = (
        f"{'forwards' if spec else 'step'} {full_ms:.3f} ms  roll-up {rollup:.3f} ms  ratio {rollup / full_ms:.3f}  {tok}"
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
    parser.add_argument("--export-only", action="store_true", help="one eager step per CONC for shape export, no timing")
    parser.add_argument("--no-kernels", action="store_true", help="skip the Kineto per-kernel breakdown")
    parser.add_argument(
        "--CONC",
        type=int,
        nargs="+",
        default=[1],
        help="concurrent requests, global across DP ranks; several values sweep in one model load",
    )
    parser.add_argument("--result-dir", default=None, help="directory for the report/trace/export (default: cwd)")
    return parser


# The synced loop takes N+2 samples and reports the fastest; fewer samples do not clear the warm-up.
_IMPLIED_FULL_STEP = 3


def main(argv=None):
    args = make_parser().parse_args(argv)
    args.concs = sorted(set(args.CONC))
    if args.concs[0] <= 0:
        raise ValueError("--CONC must be positive")
    if args.trace and len(args.concs) > 1:
        raise ValueError("--trace takes a single --CONC")
    if args.export_only and (args.trace or args.full_step):
        raise ValueError("--export-only excludes --trace and --full-step")
    if args.export_only:
        os.environ.setdefault("AKLR_EXPORT", "1")
    if args.result_dir:
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    # Any DP split divides this; work() reports each CONC that does not split.
    dp = args.dp_size if args.enable_dp_attention else 1
    args.CONC = args.concs[-1] if args.concs[-1] % dp == 0 else args.concs[-1] * dp
    model_bench.run(work, model_bench.build_server_args(args), args)
    # The per-op pass cannot see sampling, accept/reject and bookkeeping between forwards, so --full-step is implied.
    if args.speculative_algorithm is not None and not args.full_step and not args.export_only and not args.trace:
        args.full_step = _IMPLIED_FULL_STEP
        args.CONC = args.concs[-1] if args.concs[-1] % dp == 0 else args.concs[-1] * dp
        model_bench.run(work, model_bench.build_server_args(args), args)


if __name__ == "__main__":
    main()
