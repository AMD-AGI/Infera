###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
# Provenance: port of AKLR `aklr/benchmarks/models/{glm52,deepseekv4}/profile_decode.py` @ 33689332 via the
# SGLang DSv4.1 port (sglang_inproc/tool2_perop/dsv41/profile_decode.py), to vLLM. Same union of features:
# glm52's MTP path (draft model one row, simulated-acceptance throughput, implied --full-step) and
# deepseekv4's CONC sweep, capacity refusal, rank-max under cross-rank MoE, Kineto top kernels, export-only.
# Launch: torchrun --nproc-per-node <DP x TP> profile_decode.py <EngineArgs> --ISL .. --CONC ..

"""Per-operator decode timing for DeepSeek-V4.1 on vLLM, rolled up to a per-step estimate."""

import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

import kernel_breakdown  # noqa: E402
import model_bench as mb  # noqa: E402
import shape_export  # noqa: E402
import spec_dspark  # noqa: E402
import vllm_bench as vb  # noqa: E402


def _drafts_from_args(args) -> int:
    sc = args.speculative_config
    if isinstance(sc, str) and sc:
        sc = json.loads(sc)
    if isinstance(sc, dict):
        return int(sc.get("num_speculative_tokens") or 0)
    return int(getattr(sc, "num_speculative_tokens", 0) or 0) if sc else 0


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
    mb.setup_env()
    dp = args.data_parallel_size or 1
    drafts = _drafts_from_args(args)
    local = max(c // dp for c in args.concs) or 1
    ea = mb.build_engine_args(args, local, local * (1 + drafts), cuda_graph=bool(args.full_step))
    from vllm.config import set_current_vllm_config

    llm, runner, rank_print = mb.setup(ea)
    with torch.inference_mode(), set_current_vllm_config(runner.vllm_config):
        work(args, runner, rank_print)
    # The per-op pass cannot see sampling, rejection and bookkeeping between forwards: --full-step is implied
    # under speculation (glm52). It needs a fresh engine with graphs, and a second engine in this process
    # (or an execv under the same torchrun agent store) hung in distributed init on GPU, so scripts/run_t2v.sh
    # launches it as a second torchrun job; run directly, do it by hand.
    if drafts and not (args.full_step or args.export_only or args.trace or args.no_implied_full_step):
        rank_print(f"implied: rerun with --full-step {_IMPLIED_FULL_STEP} as a separate torchrun job (run_t2v.sh does)")


def work(args, runner, rank_print):
    dp = mb.parallel(runner)["dp"]
    layers = mb.Layers(runner)
    rank_print(f"layers: {layers.describe()}")
    routing = mb.install_routing(layers.moe_pairs(), args.routing)
    draft_routing = mb.install_routing(mb.moe_pairs(spec_dspark.draft_layers(runner)), args.routing)
    rank_print(f"balanced routing on {len(routing.pairs) if routing else 0} target + "
               f"{len(draft_routing.pairs) if draft_routing else 0} draft MoE layers ({args.routing})")
    drafts = mb.num_spec(runner)
    spec = mb.spec_config(runner) is not None
    lens = mb.read_isl_file(args.isl_file) if args.isl_file else None
    if lens:
        rank_print(f"per-request contexts from {args.isl_file}: {len(lens)} values, mean {sum(lens) / len(lens):,.0f}, max {max(lens):,}")
    for conc in args.concs:
        args.CONC = conc
        if conc % dp:
            rank_print(f"[UNSUPPORTED] CONC {conc} does not split over {dp} DP ranks")
            continue
        case = mb.decode_case(f"conc{conc}", conc // dp, args.ISL, drafts, lens)
        why = ""
        try:
            batch, case = mb.build(runner, case, args.seed or 0, args.kv_len_jitter, drafter=spec)
        except mb.Capacity as e:
            batch, why = None, str(e)
        if not vb.agreed(batch is not None):
            rank_print(f"[BLOCKED_CAPACITY] CONC {conc} at ISL {args.ISL}: {why or 'another rank'}; not reduced")
            continue
        filled = mb.fill_kv(runner, batch, args.kv_fill)
        if args.full_step:
            res = spec_dspark.full_step(runner, batch, args.full_step, rank_print, conc)
            if vb.world_rank() == 0 and args.result_dir:
                (Path(args.result_dir) / f"full_step_conc{conc}.json").write_text(json.dumps(res, indent=1))
            continue
        res = mb.run_inside_forward(runner, batch, lambda f: measure(args, f, layers, batch, case, filled, rank_print))
        if res is None or vb.world_rank() != 0:
            continue
        write_report(args, runner, layers, batch, case, res, rank_print)
    for r in (routing, draft_routing):
        if r is not None:
            if any(n == 0 for n in r.fired):
                rank_print(f"WARNING: balanced routing never reached {r.fired.count(0)} MoE layer(s): gate fused away")
            r.remove()


def measure(args, f, layers, batch, case, filled, rank_print):
    runner = f.runner
    spec = mb.spec_config(runner) is not None
    ops = mb.collect_ops(layers, runner)
    if spec:
        ops += spec_dspark.draft_ops(runner)
    names = mb.op_names(ops)
    by_name: dict = {}
    for o in ops:
        by_name.setdefault(o.name, []).append(o)
    drafter = spec_dspark.Drafter(runner) if spec else None

    def step_and_draft():
        out = f.step()
        if drafter is not None:
            drafter.prepare(f)
            drafter.propose(out)
        return out

    if args.trace:
        trace_step(args, step_and_draft, rank_print)
        return None

    saved, calls = {}, {}
    mod_ops = {o.name: o.module for o in ops if o.module is not None and o.match is None}
    with shape_export.export_scope(mod_ops, f"conc{args.CONC}", args.result_dir, vb.world_rank(), mb.rows_now):
        with mb.record_inputs(ops, saved, calls):
            out = step_and_draft()
    torch.cuda.synchronize()
    vb.assert_lockstep(saved)
    if args.export_only:
        rank_print(f"CONC {args.CONC}: one eager step recorded for shape export")
        path = export_shapes(args, runner, layers, saved, calls, {}, case, None)
        rank_print(f"operator shapes -> {path}")
        return None

    rank_print("timing: full forward (target forward + logits, one graph)")
    full_ms = vb.bench(f.step, args.reps, inner=1, rounds=args.repeat)
    draft_ms = 0.0
    if drafter is not None:
        rank_print("timing: drafter (DraftBench propose, one graph)")
        draft_ms = vb.bench(lambda: drafter.propose(out), args.reps, inner=1, rounds=args.repeat)

    times, skipped = {}, {}
    inline = mb.engram_inline(runner)
    inline.__enter__()
    for name, entry in saved.items():
        rank_print(f"timing: {name}")
        kind, _, row = name.rpartition("|")
        try:
            cyc = mb.cycled_calls(entry, layers, kind, row)
            if cyc is not None:
                # A real step visits different layer weights; one representative would make parent rows cache-hot.
                times[name] = vb.bench(cyc, args.reps, inner=max(vb.GRAPH_REPS, len(cyc)), rounds=args.repeat)
            else:
                # Rotated so a MALL-sized weight is not read cache-hot.
                with mb.rotated_calls(entry) as (fns, inner):
                    times[name] = vb.bench(fns, args.reps, inner=inner, rounds=args.repeat)
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, AssertionError, ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
            first = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
            if "out of memory" in first.lower():
                raise
            skipped[name] = first[:300]
            rank_print(f"  skipped: {skipped[name]}")
            torch.cuda.synchronize()
        if not vb.agreed(name in times):
            times.pop(name, None)
            skipped.setdefault(name, "skipped on another rank")

    reduce = "max" if mb.moe_spans_ranks(runner) else "min"
    times = vb.world_times({**times, "__full": full_ms, "__draft": draft_ms}, reduce)
    full_ms, draft_ms = times.pop("__full"), times.pop("__draft")

    try:
        profiles, perr = ({}, {}) if args.no_kernels else profile_saved_ops(saved, by_name, times, rank_print)
    finally:
        inline.__exit__(None, None, None)
    draft_rows = {k: v for k, v in times.items() if k.startswith("draft|")}
    for k in draft_rows:
        times.pop(k)
        profiles.pop(k, None)
    target_names = [n for n in names if not n.startswith("draft|")]
    target_saved = {k: v for k, v in saved.items() if not k.startswith("draft|")}
    gcalls = {n: calls.get(n, 1) for n in mb.GLOBAL_OPS}
    shapes = export_shapes(args, runner, layers, target_saved, calls, times, case, profiles)
    return dict(times=times, full_ms=full_ms, draft_ms=draft_ms, draft_rows=draft_rows, skipped=skipped,
                coverage=mb.coverage_note(target_names, target_saved), profiles=profiles, perr=perr, gcalls=gcalls,
                filled=filled, reduce=reduce, shapes=shapes, drafter=spec_dspark.draft_model(runner))


def profile_saved_ops(saved, by_name, times, rank_print):
    """Kineto on the same recorded calls; parent samples exclude children with their own rows."""
    profiles, errors = {}, {}
    for label in list(times):
        if label not in saved:
            continue
        rank_print(f"kernels: {label}")
        error = ""
        try:
            with mb.child_scopes(mb.children_ops(by_name, label, times)):
                profiles[label] = kernel_breakdown.profile_call(mb.make_call(saved[label]), ())
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, AssertionError, ValueError, TypeError, AttributeError, KeyError) as exc:
            error = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
        if not vb.agreed(label in profiles):
            profiles.pop(label, None)
            error = error or "kernel profiling unavailable on another rank"
        if error:
            kind, _, name = label.rpartition("|")
            row = name + ".<rest>" if name in mb.RESIDUALS else name
            errors.setdefault(row, []).append(f"{kind or 'global'}: {error}")
            rank_print(f"  kernel breakdown unavailable: {error}")
    return profiles, errors


def trace_step(args, step, rank_print) -> None:
    """One eager step's kernels (top 35 printed) + chrome trace on rank 0, instead of op timing."""
    from torch.profiler import ProfilerActivity, profile

    step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    top, rows = vb.kernel_rows(prof)
    total = sum(ms for ms, _ in rows.values())
    rank_print(f"kernels: {len(rows)} distinct, {sum(n for _, n in rows.values())} launches, {total:.3f} ms")
    for name, (ms, n) in top:
        rank_print(f"  {ms:8.3f} ms  {n:5d} x {ms * 1000 / n:7.2f} us  {name[:78]}")
    if vb.world_rank() != 0:
        return
    path = Path(args.out or Path(args.result_dir or ".") / "trace.json")
    prof.export_chrome_trace(str(path))
    rank_print(f"trace -> {path.resolve()}")


# ----------------------------------------------------------------------------------------------
# Shape export (operator_shapes.json).


def case_context(runner, case, layers) -> dict:
    reqs = case.requests
    kv = [r.kv_len - r.query_len for r in reqs]
    p = mb.parallel(runner)
    return {
        "model": str(runner.model_config.model), "engine": "vllm", "phase": "decode" if reqs[0].is_decode else "prefill",
        "dp": p["dp"], "tp": p["tp"], "ep": p["ep"], "requests_per_rank": len(reqs),
        "query_tokens_per_request": reqs[0].query_len, "verify_width": reqs[0].query_len if reqs[0].num_draft_tokens else None,
        "draft_tokens": reqs[0].num_draft_tokens, "tokens_per_rank": sum(r.query_len for r in reqs),
        "context_per_request": {"min": min(kv), "mean": sum(kv) / len(kv), "max": max(kv)},
        "kv_cache_dtype": str(runner.cache_config.cache_dtype), "layer_kinds": layers.counts,
    }


def export_shapes(args, runner, layers, saved, calls, times, case, profiles):
    """Per-row shapes for kernel work: inputs, weights, derived GEMM M/N/K, MoE experts/top-k, call counts, the
    row's ms/step and its kernels. Written on every rank's own file (rank 0 = the report's)."""
    rows = []
    counts = layers.counts
    hf = runner.model_config.hf_text_config
    for name, entry in saved.items():
        kind, _, row = name.rpartition("|")
        mult = counts.get(kind, 1) if kind else calls.get(name, 1)
        tens = shape_export._tensors((entry.args, entry.kwargs), [])
        tokens = next((int(t.shape[0]) for t in tens if t.dim() >= 2), None)
        wmod = entry.op.wmod if entry.op.wmod is not None else entry.op.module
        w = getattr(wmod, "weight", None) if isinstance(wmod, torch.nn.Module) else None
        derived = shape_export.gemm_dims(tokens, w if isinstance(w, torch.Tensor) else None)
        if row in ("mlp.experts", "mlp"):
            derived.update(experts=int(hf.n_routed_experts), top_k=int(hf.num_experts_per_tok), hidden=int(hf.hidden_size),
                           intermediate=int(hf.moe_intermediate_size), tokens_local=tokens,
                           tokens_into_experts_note="tokens_local x DP ranks after the naive DP dispatch (no EP)")
        if row.startswith("attn"):
            derived.update(heads=int(hf.num_attention_heads), head_dim=int(hf.head_dim), rope_dim=int(hf.qk_rope_head_dim),
                           index_topk=int(getattr(hf, "index_topk", 0)), index_heads=int(getattr(hf, "index_n_heads", 0)),
                           compress_ratio=int(getattr(_attn_of(layers, kind), "compress_ratio", 0) or 0))
        params = {}
        if isinstance(wmod, torch.nn.Module):
            params = {n: shape_export.spec(p) for n, p in list(wmod.named_parameters(recurse=False))[:8]}
        krn = []
        for (kname, api), (ms, n, _fr) in sorted((profiles or {}).get(name, {}).items(), key=lambda kv: -kv[1][0])[:8]:
            krn.append({"kernel": kname, "api": api, "kineto_ms_per_call": ms, "launches": n})
        rows.append({
            "row": name, "kind": kind or "global", "op": row, "layers": layers.kinds.get(kind) if kind else None,
            "calls_per_step": mult, "ms_per_call": times.get(name), "ms_per_step": (times[name] * mult) if name in times else None,
            "call": getattr(entry.fn, "__qualname__", None) if entry.fn is not None else f"{type(entry.op.module).__module__}.{type(entry.op.module).__qualname__}",
            "args": shape_export.spec(list(entry.args)), "kwargs": shape_export.spec(dict(entry.kwargs)),
            "weights": params, "derived": derived, "kernels": krn,
        })
    out = Path(shape_export.default_dir(args.result_dir)) / f"rank{vb.world_rank()}" / f"operator_shapes.conc{args.CONC}.json"
    try:
        return str(shape_export.write_shapes(out, case_context(runner, case, layers), rows))
    except OSError:
        return None


def _attn_of(layers, kind):
    ids = layers.kinds.get(kind)
    return layers.layers[ids[0]].attn if ids else None


# ----------------------------------------------------------------------------------------------
# Report.


def default_out(args, runner) -> str:
    spec = mb.spec_config(runner) is not None
    isl = f"islfile{Path(args.isl_file).stem[:24]}" if args.isl_file else f"isl{args.ISL}"
    parts = [mb.model_tag(runner.model_config.model), "mtp" if spec else "decode", isl, f"conc{args.CONC}",
             *mb.parallel_tags(runner), str(runner.cache_config.cache_dtype)]
    parts += filter(None, [vb.load_tag(runner.load_config.load_format)])
    parts.append(time.strftime("%Y%m%d"))
    return "_".join(parts) + ".md"


def write_report(args, runner, layers, batch, case, res, rank_print):
    md, headline = report(args, runner, layers, batch, case, res)
    path = Path(args.out or Path(args.result_dir or ".") / default_out(args, runner))
    if args.out and len(args.concs) > 1:
        path = path.with_name(f"{path.stem}_conc{args.CONC}{path.suffix}")
    path.write_text(md)
    summ = {"conc": args.CONC, "isl": args.ISL, "headline": headline, "full_ms": res["full_ms"], "draft_ms": res["draft_ms"],
            "report": str(path), "shapes": res.get("shapes")}
    path.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    rank_print(headline)
    rank_print(f"report -> {path.resolve()}")


def config_table(args, runner, layers, case, res) -> str:
    reqs = case.requests
    kv = [r.kv_len - r.query_len for r in reqs]
    spec = mb.spec_config(runner) is not None
    rows = [
        ("model", f"`{type(runner.model).__name__}`  {runner.model_config.model}"),
        ("layers", layers.describe()),
        ("parallel", mb.parallel_line(runner)),
        ("batch", f"CONC {args.CONC}, {len(reqs)} requests per DP rank x {reqs[0].query_len} query tokens"
                  f" = {sum(r.query_len for r in reqs)} tokens/rank"),
        ("context", (f"per-request list `{args.isl_file}`: mean {sum(kv) / len(kv):,.0f}, {min(kv):,}..{max(kv):,}" if args.isl_file
                     else f"ISL {args.ISL:,} per request") + (f" (jitter +-{args.kv_len_jitter:.0%}: {min(kv):,}..{max(kv):,})" if args.kv_len_jitter else "")),
        ("KV setup", f"synthetic: vLLM bench forward's batch builder (no scheduler), disjoint blocks per request, "
                     f"{'random bytes < 0x7C' if args.kv_fill == 'random' else 'zeros'} in the case's blocks, no prefill forward"),
        ("routing", f"seeded balanced (AKLR draw, seed {vb.MOE_ROUTING_SEED}) injected as router logits, correction bias zeroed"
         if args.routing == "balanced" else "**native router on synthetic hidden states (--routing native, diagnostic)**"),
        ("kv dtype", f"`{runner.cache_config.cache_dtype}`"),
        *([("spec", mb.spec_note(runner))] if spec else []),
        ("weights", mb.weights_line(runner)),
        ("env", ", ".join(f"`{k}={os.environ[k]}`" for k in sorted(os.environ) if k.startswith(("VLLM_ROCM", "VLLM_USE_BREAKABLE", "AITER_")))),
        ("measurement", f"{args.reps} graph replays per op; {vb.GRAPH_REPS} calls per graph, more for a weight small enough to"
                        f" rotate over the {mb.ROTATION_BYTES >> 20} MB buffer"
                        + (f", fastest of {args.repeat} passes" if args.repeat > 1 else "")
                        + f", {res['reduce']} over ranks"),
    ]
    return vb.table(rows, ("item", "value"))


def summary_table(args, runner, full_ms, draft_ms, rollup) -> str:
    spec = mb.spec_config(runner) is not None
    ref = full_ms + draft_ms
    rows = [
        ("forward time" if spec else "step time", f"{ref:.3f} ms",
         spec_dspark.forward_note(runner) + ". Not the whole step: sampling / rejection and the runner's CPU work are"
         " visible only under --full-step (run automatically after this pass)."
         if spec else "target forward + logits in one graph; not serving ITL (see --full-step)"),
        ("roll-up", f"{rollup:.3f} ms", "sum of the per-op leaf rows below" + (" + the draft model row" if spec else "")),
        ("ratio", f"{rollup / ref:.3f}", "roll-up / forward time; agreement does not prove coverage"),
    ]
    acc = mb.accept_len(runner) if spec else None
    if not spec:
        rows.append(("throughput", f"{args.CONC / ref * 1000:.1f} tok/s", f"at CONC {args.CONC}"))
    elif acc is None:
        rows.append(("throughput", "not reported", "a speculative step emits the accepted tokens; run with"
                     " rejection_sample_method=synthetic + synthetic_acceptance_length (as the server) for a tok/s row"))
    else:
        rows.append(("throughput", f"{args.CONC * acc / ref * 1000:.1f} tok/s",
                     f"at CONC {args.CONC}, accept len {acc:.2f} from the speculative config (synthetic, as the server)"))
        rows.append(("ITL", f"{ref / acc:.3f} ms", "forward time / accept len -- a floor, the non-forward part is not counted"))
    return vb.table(rows, ("metric", "value", "note"))


def hotspot_section(totals, rollup, kernels=None, kernel_errors=None) -> str:
    rows = []
    kernel_errors = kernel_errors or {}
    for name, ms in sorted(totals.items(), key=lambda kv: -kv[1]):
        row = (mb.tree_name(name, 0), ms, f"{ms / rollup * 100:.1f}%")
        rows.append(row if kernels is None else (*row, "", "", "", ""))
        if kernels is None:
            continue
        for error in kernel_errors.get(name, ()):
            rows.append((f"  ↳ **Incomplete kernel profile: {error.replace('|', '/')}**", "", "", "", "", "", ""))
        if name not in kernels:
            rows.append(("  ↳ kernel profile unavailable", "", "", "", "", "", ""))
            continue
        items = kernels[name]
        total = sum(v[0] for v in items.values())
        if not items:
            rows.append(("  ↳ no exclusive GPU kernels (all in child rows)", "", "", 0, 0.0, "n/a", ""))
        for key, (gpu_ms, n, _) in sorted(items.items(), key=lambda kv: (-kv[1][0], str(kv[0])))[:12]:
            kernel, api = key if isinstance(key, tuple) else (key, "unavailable")
            share = f"{gpu_ms / total * 100:.2f}%" if total else "n/a"
            rows.append((f"  ↳ `{kernel.replace('|', '/')[:120]}`", "", "", n, gpu_ms, share, f"`{api.replace('|', '/')}`"))
    note = ("\n\nLeaf rows only, so the shares add to 100%. A `<rest>` row is its parent minus the children listed under"
            " it (a difference of two measurements; it can read negative, do not act on small ones). Globals are"
            " multiplied by their calls per step (e.g. `hc_post` also runs for the DSpark aux hidden states).")
    headers = ("op / kernel", "ms/step", "share")
    if kernels is None:
        return vb.table(rows, headers) + note
    headers += ("kernel calls/step", "profile GPU ms/step", "% of module GPU", "Python / dispatcher API")
    note += ("\n\nIndented rows: the top Kineto GPU kernels of rank 0's standalone eager replay of the row (one sample"
             " per layer kind x its layer count), children with their own rows excluded. ms/step and share are graph"
             " timings; the profile columns are diagnostic.")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + note


def kind_sections(times, counts) -> str:
    out = []
    for kind, n in counts.items():
        rows = [(mb.tree_name(name, d), times[f"{kind}|{name}"], times[f"{kind}|{name}"] * n)
                for name, d in mb.TREE if f"{kind}|{name}" in times]
        rows += [(mb.tree_name(name, 0), times[f"{kind}|{name}"], times[f"{kind}|{name}"] * n)
                 for name in mb.OUTSIDE_LAYER if f"{kind}|{name}" in times]
        out.append(f"### {kind} -- x{n} layers\n\n" + vb.table(rows, ("op", "ms/call", "ms/step")))
    return "\n\n".join(out)


def rollup_table(times, counts, gcalls) -> str:
    rows = []
    for kind, n in counts.items():
        for op in ("layer", *mb.OUTSIDE_LAYER):
            if f"{kind}|{op}" in times:
                rows.append((f"{kind} {op}", n, times[f"{kind}|{op}"], times[f"{kind}|{op}"] * n))
    rows += [(name, gcalls.get(name, 1), times[name], times[name] * gcalls.get(name, 1)) for name in mb.GLOBAL_OPS if name in times]
    return vb.table(rows, ("op", "x", "ms/call", "ms/step"))


def draft_section(draft_rows, draft_ms) -> str:
    rows = [(f"`{k.split('|', 1)[1]}`", v, f"{v / draft_ms * 100:.1f}%" if draft_ms else "")
            for k, v in sorted(draft_rows.items(), key=lambda kv: -kv[1])]
    return vb.table(rows, ("draft op", "ms/call", "% of the draft model row")) + (
        f"\n\nThe `draft model` row is the drafter's propose (all draft forwards of one step) timed under one graph"
        f" ({draft_ms:.3f} ms), not this table; these rows are one call each of the first draft layer's ops.")


def report(args, runner, layers, batch, case, res):
    counts = layers.counts
    times = dict(res["times"])
    mb.add_residuals(times, counts)
    totals = mb.leaf_totals(times, counts, res["gcalls"])
    if res["draft_ms"]:
        totals["draft model"] = res["draft_ms"]
    rollup = sum(totals.values())
    spec = mb.spec_config(runner) is not None
    kernels = None if args.no_kernels else mb.kernel_totals(res["profiles"], counts, res["times"], res["gcalls"])
    ranking = kernel_breakdown.Ranking.of(kernels or {}, totals, frozenset())
    ref = res["full_ms"] + res["draft_ms"]
    sections = [
        ("Summary", summary_table(args, runner, res["full_ms"], res["draft_ms"], rollup)),
        *([("Top kernels -- one step", ranking.section(vb.table))] if kernels is not None else []),
        ("Config", config_table(args, runner, layers, case, res)),
        ("Versions", vb.table(vb.versions(), ("component", "version"))),
        ("Memory, per rank", mb.memory_section(runner, batch, res["filled"], res.get("drafter"))),
        ("Hotspots -- one step, all layers", hotspot_section(totals, rollup, kernels, res["perr"])),
        ("Per layer kind -- one step", kind_sections(times, counts)),
        ("Roll-up -- one step", rollup_table(times, counts, res["gcalls"])),
        *([("Draft model -- per op", draft_section(res["draft_rows"], res["draft_ms"]))] if res["draft_rows"] else []),
        *([("Coverage", res["coverage"])] if res["coverage"] else []),
        *([("Not replayable", mb.skipped_note(res["skipped"]))] if res["skipped"] else []),
        *([("Operator shapes", f"`{res['shapes']}` (schema tool2-operator-shapes/1; per row: args, weights, derived"
            " M/N/K / experts / heads, calls and ms per step, top kernels)")] if res.get("shapes") else []),
    ]
    md = "\n\n".join([f"# {mb.model_tag(runner.model_config.model)} {'MTP ' if spec else ''}decode profile (vLLM)",
                      time.strftime("%Y-%m-%d %H:%M %Z"), *filter(None, [mb.dummy_banner(runner.load_config.load_format)])]
                     + [f"## {t}\n\n{b}" for t, b in sections])
    acc = mb.accept_len(runner) if spec else None
    tok = f"{args.CONC * (acc or 1) / ref * 1000:.1f} tok/s" if (not spec or acc) else "tok/s not reported"
    headline = f"{'forwards' if spec else 'step'} {ref:.3f} ms  roll-up {rollup:.3f} ms  ratio {rollup / ref:.3f}  {tok}"
    return md + "\n", headline


def make_parser():
    parser = mb.make_parser()
    parser.add_argument("--CONC", type=int, nargs="+", default=[1], help="global concurrency over DP ranks; several values share one load")
    parser.add_argument("--trace", action="store_true", help="profile one eager step's kernels instead of timing ops")
    parser.add_argument("--export-only", action="store_true", help="one eager step per CONC for shape export, no timing")
    parser.add_argument("--no-kernels", action="store_true", help="skip the Kineto per-kernel breakdown")
    parser.add_argument("--no-implied-full-step", action="store_true", help="do not follow a speculative per-op pass with --full-step 3")
    return parser


_IMPLIED_FULL_STEP = 3

if __name__ == "__main__":
    main()
