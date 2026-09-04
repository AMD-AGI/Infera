#!/usr/bin/env python3
"""Two arms in, one accept-or-reject out, with the numbers that produced it.

The arithmetic lives in `assets/lib/eval_stats.py` and not here, because
`check_no_regression` recomputes every line of this report and the two agreeing
is the whole point of that validator — which cannot happen if each carries its
own copy of the statistics.

**What counts as a regression, and what does not.**

  llm-eval    the Newcombe interval on the difference excludes zero AND the
              difference is negative. Not "the score went down": two runs of the
              same deployment differ by several points at 200 questions, and
              treating that as evidence would make this a coin-flip detector.
  needle      a depth the stock arm retrieved and the patched arm did not. Both
              failing is not a regression — it is a property of the model at that
              length, which is exactly why the frontier run exists and exactly
              why it is not gated.
  smoke       any check that passed on stock and failed on patched.
  performance a relative change worse than the bar, compared round for round.
              Round 1 against round 1 only: cold and warm differ by an order of
              magnitude on this trace.

Nothing here judges an arm on its own. An absolute eval score has no external
baseline, an absolute throughput number has no meaning without the configuration
it was measured on, and both arms were measured in the same session precisely so
that neither has to stand alone.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path

# `parents[0]` and not `parents[1]`: this file sits directly under `assets/`
# rather than in a `<name>.task/` body directory, because the STEPS readme
# calls it by path and it is not a closure body.
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import eval_stats  # noqa: E402

#: Metric -> (label, higher_is_better). Only metrics that describe the SERVICE.
#: `output_token_throughput_tps` is the headline; the latencies are there because
#: a change can trade one for the other and a single metric would hide it.
PERF_METRICS = {
    "output_token_throughput_tps": ("output token throughput", True),
    "request_throughput_rps": ("request throughput", True),
    "ttft_ms": ("time to first token", False),
    "inter_token_latency_ms": ("inter-token latency", False),
    "request_latency_ms": ("request latency", False),
}

#: Which column of the AIPerf summary each metric is read from. A throughput has
#: no percentile; a latency's p95 is the one that moves first under a queue.
PERF_COLUMNS = {"ttft_ms": ("avg", "p90"), "inter_token_latency_ms": ("avg",),
                "request_latency_ms": ("avg", "p90")}


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_eval_index(result: Path) -> dict[str, dict]:
    """`.index` plus each result json: eval name -> {scored, score}.

    `scored` comes from the index because it was counted out of the html report;
    the dataset size is the wrong answer (GSM8K ships 1319 rows, `gsm8k` scores
    1314 and `mixed_prefix_gsm8k` 1299) and the result json does not carry it.
    """
    out: dict[str, dict] = {}
    index = result / "lm_eval" / ".index"
    if not index.is_file():
        return out
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        name, scored, filename = parts[0], parts[1], parts[2]
        payload = read_json(result / "lm_eval" / filename, {}) or {}
        score = payload.get("score", payload.get("mean_score"))
        out[name] = {
            "scored": int(scored or 0),
            "score": float(score) if score is not None else None,
            "seconds": int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None,
        }
    return out


def needle_map(needle: dict) -> dict[str, dict]:
    """(run label, depth) -> the depth record, flattened for comparison."""
    out = {}
    for run in (needle or {}).get("runs", []):
        for depth in run.get("depths", []):
            out[f"{run['label']}@{depth['depth']}"] = depth
    return out


def apply_expect(kopt: str | None) -> dict:
    """m4's own claim about the optimisation, or a stated absence.

    Never `{}`: a missing `expect` and an `expect` nobody filled in read the same
    to a later reader, and the schema requires `source` and `speedup` so that a
    report has to say which of the two it is.
    """
    if kopt:
        found = sorted(Path(kopt).glob("items/codes/*/apply/manifest.json"))
        if found:
            block = json.loads(found[0].read_text(encoding="utf-8")).get("expect")
            if block:
                return block
    return {"source": "unstated", "speedup": None}


def stock_vs_m2_block(stock: Path, evidence: str | None, tolerance: float) -> dict:
    """M5.1.3.1 — does the stock arm reproduce m2's `profiling_mode_off` bench?

    **The blocker of the two new blocks**, because everything downstream assumes
    the two stages measured the same machine. m2's bench and m5's stock arm are
    the same measurement of the same deployment one stage apart; when they
    disagree, this report compares numbers that were never comparable, and no
    amount of care about the patch recovers that.

    Both signs are compared. A stock arm *faster* than m2's is as much evidence
    of a different machine as a slower one — measured on this cluster, one stock
    deployment read 193.59 tok/s beside an idle neighbour and 47 tok/s beside a
    busy one.

    Returns the block the schema requires, filled in as *not measured, and here
    is why* when m2's bench is not reachable. The reason is required: a producer
    that may omit the section leaves a reader unable to tell "not applicable"
    from "not done".
    """
    if not evidence:
        return {"source": None, "tolerance": None, "metrics": [], "ok": None,
                "unavailable_because": "this task received no profiling_evidence input"}
    root = Path(evidence)
    found = sorted(root.glob("items/result/**/profile_export_aiperf.json"))
    found = [p for p in found if "mode_on" not in str(p)] or found
    if not found:
        return {"source": None, "tolerance": None, "metrics": [], "ok": None,
                "unavailable_because": (
                    f"no profile_export_aiperf.json under {root}/items/result — m2's "
                    "profiling_mode_off bench is not in the evidence this task was handed")}

    m2 = json.loads(found[0].read_text(encoding="utf-8"))
    r1 = stock / "items" / "result" / "r1" / "profile_export_aiperf.json"
    if not r1.is_file():
        return {"source": str(found[0].relative_to(root)), "tolerance": None,
                "metrics": [], "ok": None,
                "unavailable_because": "the stock arm recorded no r1/profile_export_aiperf.json"}
    ours = json.loads(r1.read_text(encoding="utf-8"))

    rows, ok = [], True
    for metric, column in (("output_token_throughput", "avg"),
                           ("inter_token_latency", "avg"),
                           ("time_to_first_token", "avg")):
        a = (m2.get(metric) or {}).get(column)
        b = (ours.get(metric) or {}).get(column)
        rel = (b - a) / a if isinstance(a, (int, float)) and a and isinstance(b, (int, float)) else None
        within = None if rel is None else abs(rel) <= tolerance
        if within is False:
            ok = False
        rows.append({"metric": metric, "column": column, "m2": a, "stock": b,
                     "rel_delta": None if rel is None else round(rel, 6),
                     "within_tolerance": within})
    if all(r["rel_delta"] is None for r in rows):
        return {"source": str(found[0].relative_to(root)), "tolerance": tolerance,
                "metrics": rows, "ok": None,
                "unavailable_because": "no metric was present in both records"}
    return {"source": str(found[0].relative_to(root)), "tolerance": tolerance,
            "metrics": rows, "ok": ok, "unavailable_because": None}


def kernel_reconciliation_block(evidence: str | None, kopt: str | None,
                                perf_rows: list[dict], operator_id: str | None) -> dict:
    """M5.1.3.2 — the single-kernel speedup against the end-to-end change.

    **Warning, never a blocker**, exactly as the mission puts it: 作为
    report/warning 报告，不作为 blocker. Amdahl over one kernel: a kernel that is
    `s` of the profile and got `k` times faster caps the end-to-end speedup at
    `1 / ((1-s) + s/k)`.

    The two sides are measured in configurations that differ **on purpose**: `s`
    comes from a profile taken with the profiler attached and CUDA graph off,
    because a graph launch hides the kernels the profiler is there to see, and
    the end-to-end number is taken with neither. So a disagreement is a question
    about the model before it is a question about the run, and gating on it
    would refuse correct work.
    """
    def missing(why: str) -> dict:
        return {"kernel_speedup": None, "kernel_share_of_profile": None,
                "predicted_e2e_speedup": None, "observed_e2e_speedup": None,
                "agrees": None, "unavailable_because": why, "note": None}

    speedup = None
    if kopt:
        for name in ("verification.json", "forge_result.json"):
            for path in sorted(Path(kopt).glob(f"items/codes/*/results/{name}")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                speedup = payload.get("mean_case_speedup") or payload.get("total_speedup")
                if speedup:
                    break
            if speedup:
                break
    if not speedup:
        return missing("m4 recorded no mean_case_speedup for the optimised kernel")

    share = None
    if evidence and operator_id:
        for table in sorted(Path(evidence).glob("items/result/**/*.csv")):
            text = table.read_text(encoding="utf-8", errors="replace")
            if "% Total" not in text:
                continue
            import csv
            import io

            for row in csv.DictReader(io.StringIO(text)):
                if operator_id.lower() in str(row.get("Name", "")).lower():
                    try:
                        share = float(str(row.get("% Total", "")).strip().rstrip("%")) / 100.0
                    except ValueError:
                        share = None
                    break
            if share is not None:
                break
    if share is None:
        return missing(
            f"no row for {operator_id!r} carrying a '% Total' share was found in m2's "
            "kernel table, so the kernel's fraction of the profile is unknown")

    observed = None
    for row in perf_rows:
        if row.get("metric") == "output_token_throughput_tps" and row.get("column") == "avg":
            a, b = row.get("stock"), row.get("patched")
            if a:
                observed = b / a
            break

    predicted = 1.0 / ((1.0 - share) + share / float(speedup))
    agrees = None
    if observed is not None and predicted:
        agrees = abs(observed - predicted) / predicted <= 0.10
    return {"kernel_speedup": round(float(speedup), 6),
            "kernel_share_of_profile": round(share, 6),
            "predicted_e2e_speedup": round(predicted, 6),
            "observed_e2e_speedup": None if observed is None else round(observed, 6),
            "agrees": agrees,
            "unavailable_because": None,
            "note": ("the share is measured with the profiler attached and CUDA graph off and "
                     "the end-to-end number with neither, which is the largest single reason "
                     "the two can legitimately disagree")}


def main() -> int:
    ap = argparse.ArgumentParser()
    # **Two directories, not four.** Six evidence kinds became two (CONTRACT.md
    # §7): one merged `reproducible` handoff per arm, holding that arm's
    # deployment record, its correctness results and its replay rounds.
    ap.add_argument("--stock", required=True)
    ap.add_argument("--patched", required=True)
    ap.add_argument("--overlay", required=True, help="patch_overlay's content dir")
    ap.add_argument("--profiling-evidence", default=None,
                    help="m2's, for M5.1.3.1 and the kernel's share of the profile")
    ap.add_argument("--kernel-optimization", default=None,
                    help="m4's, for the single-kernel speedup (M5.1.3.2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--package", required=True)
    # The same two package variables `check_no_regression` reads out of its own
    # args block. One source, because that validator recomputes this report's
    # verdict and fails when it disagrees — two sources would make them differ
    # over a threshold nobody changed.
    #
    # **These are not to be widened.** The within-arm round-to-round spread on a
    # steady node is ~2%; a previous round raised them to 0.35/0.30 in response
    # to two arms measured fifteen minutes and one co-tenant apart, and the
    # missing control there was a comparability gate at bring-up, not a looser
    # bar (`todo.md` T7). `check_no_regression` refuses a report whose declared
    # bars are looser than its own, so widening here does not even work.
    ap.add_argument("--max-throughput-regression", type=float,
                    default=float(os.environ.get("E2E_MAX_THROUGHPUT_REGRESSION", 0.05)))
    ap.add_argument("--max-latency-regression", type=float,
                    default=float(os.environ.get("E2E_MAX_TTFT_REGRESSION", 0.10)))
    ap.add_argument("--stock-vs-m2-tolerance", type=float,
                    default=float(os.environ.get("E2E_STOCK_VS_M2_TOLERANCE", 0.10)))
    args = ap.parse_args()

    out = Path(args.out)
    # One directory per arm now serves as both `accept` and `bench` root: the
    # merge laid `smoke.json`, `needle.json`, `lm_eval/` and `r<N>/` side by side
    # under one `items/result/`, and nothing collided.
    arms = {
        "stock": {"accept": Path(args.stock), "bench": Path(args.stock)},
        "patched": {"accept": Path(args.patched), "bench": Path(args.patched)},
    }
    # The overlay's plan is the authority on what was mounted; m4's apply block
    # is the authority on what was *claimed*. Keeping them apart is deliberate —
    # `expect` states what somebody expected and has been read as a fact about
    # what happened at least twice on this project.
    plan = read_json(Path(args.overlay) / "items" / "result" / "mounts.json", {}) or {}
    manifest = {
        "operator_id": plan.get("operator_id"),
        "logical_operator": plan.get("logical_operator"),
        "apply_mode": plan.get("apply_mode"),
        "image": plan.get("image"),
        "files": plan.get("mounts") or [],
        "runtime_marker": plan.get("runtime_marker"),
        "expect": apply_expect(args.kernel_optimization),
    }

    loaded = {}
    for arm, paths in arms.items():
        result = paths["accept"] / "items" / "result"
        loaded[arm] = {
            "smoke": read_json(result / "smoke.json", {}) or {},
            "needle": read_json(result / "needle.json", {}) or {},
            "probe": read_json(result / "probe.json", {}) or {},
            "evals": read_eval_index(result),
            "bench": paths["bench"] / "items" / "result",
            "steps": read_json(paths["accept"] / "items" / "env" / "steps.json", {}) or {},
            "context": read_json(paths["accept"] / "items" / "env" / "context.json", {}) or {},
        }

    reasons: list[str] = []

    # ---- smoke ---------------------------------------------------------------
    smoke_rows = []
    stock_checks = {c["name"]: c["ok"] for c in loaded["stock"]["smoke"].get("checks", [])}
    patched_checks = {c["name"]: c["ok"] for c in loaded["patched"]["smoke"].get("checks", [])}
    for name in sorted(set(stock_checks) | set(patched_checks)):
        a, b = stock_checks.get(name), patched_checks.get(name)
        verdict = "REGRESSED" if (a and not b) else ("same" if a == b else "differs")
        smoke_rows.append({"check": name, "stock": a, "patched": b, "verdict": verdict})
        if verdict == "REGRESSED":
            reasons.append(f"smoke check {name!r} passed on stock and failed on patched")

    # ---- needle --------------------------------------------------------------
    needle_rows = []
    a_map, b_map = needle_map(loaded["stock"]["needle"]), needle_map(loaded["patched"]["needle"])
    for key in sorted(set(a_map) | set(b_map)):
        a, b = a_map.get(key, {}), b_map.get(key, {})
        gated = key.startswith("gated@")
        regressed = bool(a.get("ok")) and not bool(b.get("ok"))
        needle_rows.append({
            "run": key,
            "gated": gated,
            "stock_ok": a.get("ok"),
            "patched_ok": b.get("ok"),
            "stock_prompt_tokens": a.get("prompt_tokens"),
            "patched_prompt_tokens": b.get("prompt_tokens"),
            "verdict": "REGRESSED" if regressed else "same",
        })
        if regressed:
            where = "gated" if gated else "frontier"
            reasons.append(f"needle {key} ({where}) retrieved on stock and not on patched")
    # A depth failing on BOTH arms is NOT a regression and is not reported as one.
    # Retrieval at any given depth is not a stable property of this deployment --
    # measured, nine times, in assets/accept/needle.py's table -- so the only
    # thing the needle can say about a patch is that a depth stopped working.
    # What it can say about the deployment is that nothing came back at all, and
    # that floor is check_acceptance's, enforced per arm.

    # ---- llm-eval ------------------------------------------------------------
    # The probe decides whether a score can be read at all. It catches failures
    # that produce a NUMBER rather than an error -- a reasoning parser eating the
    # answer, a shared prefix changing it -- and a score measured through one of
    # those is indistinguishable from a real regression. So a failed probe on
    # either arm makes every eval comparison `uninterpretable`, which is a reason
    # to reject rather than a quiet `same`.
    probe_failed = [arm for arm in arms if not loaded[arm]["probe"].get("ok")]
    eval_rows = []
    for name in sorted(set(loaded["stock"]["evals"]) | set(loaded["patched"]["evals"])):
        a = loaded["stock"]["evals"].get(name)
        b = loaded["patched"]["evals"].get(name)
        if not a or not b or a["score"] is None or b["score"] is None:
            eval_rows.append({"name": name, "verdict": "unmeasured",
                              "stock": a, "patched": b})
            reasons.append(f"eval {name!r} did not produce a score on both arms")
            continue
        row = eval_stats.compare_scores(a["score"], a["scored"], b["score"], b["scored"])
        row["name"] = name
        if probe_failed:
            row["verdict"] = "uninterpretable"
            row["uninterpretable_because"] = f"probe failed on: {', '.join(probe_failed)}"
        eval_rows.append(row)
        if row["verdict"] == "REGRESSED":
            reasons.append(
                f"eval {name!r} regressed: {row['score_a']:.3f} -> {row['score_b']:.3f}, "
                f"95% CI of the difference {row['ci95_delta']} excludes zero"
            )
    if probe_failed:
        for arm in probe_failed:
            for failure in loaded[arm]["probe"].get("failures") or ["(none recorded)"]:
                reasons.append(f"probe failed on the {arm} arm: {failure}")
        reasons.append(
            "every eval comparison is uninterpretable while a probe is failing: the "
            "failures it catches produce a score rather than an error, and that score "
            "cannot be told apart from a real regression"
        )

    # The gap between gsm8k and its shared-prefix variant measures prefix reuse
    # without needing an external baseline: the same questions, one set behind
    # partially shared few-shot prefixes.
    prefix_reuse = {}
    for arm in ("stock", "patched"):
        evals = loaded[arm]["evals"]
        if "gsm8k" in evals and "mixed_prefix_gsm8k" in evals:
            base, mixed = evals["gsm8k"], evals["mixed_prefix_gsm8k"]
            if base["score"] is not None and mixed["score"] is not None:
                prefix_reuse[arm] = round(mixed["score"] - base["score"], 6)

    # ---- performance ---------------------------------------------------------
    perf_rows = []
    #: Per-round breaches. Reported, not judged -- see the append site below.
    round_breaches: list[str] = []
    rounds = sorted(
        {p.name for arm in arms for p in (loaded[arm]["bench"]).glob("r*") if p.is_dir()}
    )
    for round_name in rounds:
        summaries = {
            arm: (read_json(loaded[arm]["bench"] / round_name / "summary.json", {}) or {})
            .get("metrics", {})
            for arm in ("stock", "patched")
        }
        for metric, (label, higher_better) in PERF_METRICS.items():
            for column in PERF_COLUMNS.get(metric, ("avg",)):
                a = summaries["stock"].get(metric, {}).get(column)
                b = summaries["patched"].get(metric, {}).get(column)
                bar = args.max_throughput_regression if higher_better else args.max_latency_regression
                row = eval_stats.perf_verdict(a, b, max_regression=bar, higher_is_better=higher_better)
                row.update(round=round_name, metric=metric, column=column, label=label, bar=bar)
                perf_rows.append(row)
                if row["verdict"] == "REGRESSED":
                    # **A warning now, not a reason.** A single round breaching
                    # is evidence about that round; the verdict is taken on the
                    # reduction below, once per (metric, column). Rejecting here
                    # too would restore the 7R multiple-comparison inflation
                    # that reduction exists to remove — and would do it
                    # invisibly, because the reduced row would say `same`.
                    round_breaches.append(
                        f"{label} ({column}) in {round_name}: {a} -> {b}, "
                        f"{row['rel_delta']:+.1%} against a bar of {bar:.0%} "
                        "(one round; the verdict is on the reduction)"
                    )
        counts = {
            arm: summaries[arm].get("request_count", {}).get("avg") for arm in ("stock", "patched")
        }
        perf_rows.append({
            "round": round_name, "metric": "request_count", "column": "avg",
            "label": "requests replayed", "stock": counts["stock"], "patched": counts["patched"],
            "rel_delta": eval_stats.relative_change(counts["stock"], counts["patched"]),
            "bar": None,
            # Not a pass/fail: the trace decides how many requests there are, and a
            # difference here means the two arms did not replay the same work,
            # which invalidates the comparison rather than failing the patch.
            "verdict": "context",
        })
        if counts["stock"] and counts["patched"] and counts["stock"] != counts["patched"]:
            reasons.append(
                f"{round_name}: the two arms replayed different request counts "
                f"({counts['stock']} vs {counts['patched']}) — they are not comparable"
            )

    # ---- per-request dispersion, for the noise floor ---------------------------
    #
    # AIPerf reports `std` beside `avg` for the per-request latency metrics, and
    # `request_count` for n. Both are already in every round's summary.json, so
    # the floor costs nothing to measure and — being closed form — can be
    # recomputed by the validator from the three numbers the report carries.
    #
    # Averaged across rounds rather than pooled: `std` is a within-round
    # dispersion and the rounds are separate samples of it. n is summed, because
    # pooling requests across rounds is exactly what narrows the floor.
    dispersion: dict[tuple, dict] = {}
    for metric in PERF_METRICS:
        for column in PERF_COLUMNS.get(metric, ("avg",)):
            if column != "avg":
                continue  # a quantile's floor needs a density; see eval_stats.noise_floor
            entry: dict = {}
            for arm in ("stock", "patched"):
                rsds, total_n = [], 0
                for round_name in rounds:
                    summary = (read_json(loaded[arm]["bench"] / round_name / "summary.json", {})
                               or {}).get("metrics", {})
                    block = summary.get(metric) or {}
                    avg, std = block.get("avg"), block.get("std")
                    count = (summary.get("request_count") or {}).get("avg")
                    if isinstance(avg, (int, float)) and avg and isinstance(std, (int, float)):
                        rsds.append(std / avg)
                    if isinstance(count, (int, float)):
                        total_n += int(count)
                if rsds:
                    entry[f"{arm}_rsd"] = round(sum(rsds) / len(rsds), 6)
                    entry[f"{arm}_n"] = total_n
            if entry:
                dispersion[(metric, column)] = entry

    # ---- the judged comparison: ONE row per (metric, column) -------------------
    #
    # **`performance` above is evidence from here on, not the verdict.** It kept
    # one row per round and every row was judged, so R rounds were 7R independent
    # tests at the bar and the gate got *less* trustworthy the more rounds it was
    # given: family-wise false refusal 6.8% at R=1, 29.7% at R=5, 50.5% at R=10,
    # at a 1% per-row rate. Rounds are supposed to average, not multiply.
    #
    # So the rounds are reduced first and the reduced pair is judged once.
    # `eval_stats.reduce_rounds` is the single definition, because
    # `check_no_regression` recomputes this from the same per-round rows and the
    # two agreeing is the whole point of that validator.
    #
    # At R=1 the reduction is the identity, so every artefact this package has
    # produced is judged exactly as it was and no existing verdict moves.
    comparison_rows = []
    for metric, (label, higher_better) in PERF_METRICS.items():
        for column in PERF_COLUMNS.get(metric, ("avg",)):
            per_round = {
                arm: [r[arm] for r in perf_rows
                      if r.get("metric") == metric and r.get("column") == column]
                for arm in ("stock", "patched")
            }
            # Per-round request counts, so the rounds can be pooled by weight.
            counts = {
                arm: [r["stock" if arm == "stock" else "patched"]
                      for r in perf_rows
                      if r.get("metric") == "request_count" and r.get("column") == "avg"]
                for arm in ("stock", "patched")
            }
            # **Two statistics, one verdict, and their disagreement is a third
            # answer.** The pooled mean is judged, because `noise_floor`
            # describes a mean and pooling makes it exact at any R. The median
            # is computed beside it, because it is the one that survives a
            # single bad round — 5 rounds with one 1.40x outlier is absorbed by
            # the median and dragged by the mean.
            #
            # Choosing one and hoping was the alternative and it loses either
            # the floor or the robustness. **Carrying both turns the tension
            # into a signal**: when they reach different verdicts the rounds
            # disagree more than the bar allows, which is a comparability
            # finding rather than a patch finding.
            #
            # Null disagreement rate, both arms drawn from the same arm's own
            # per-request records so there is no patch to find, 3000 trials:
            # worst cell 9.8% (request_latency_ms, stock, R=3), 5.7% at R=5,
            # 1.3% at R=10, and <=0.3% everywhere on the cleaner arm. It falls
            # with more rounds, which is the right direction. R=1 is 0% *by
            # construction* and is not evidence — with one round the two
            # statistics are literally the same number.
            a, n_a = eval_stats.pooled_mean(per_round["stock"], counts["stock"])
            b, n_b = eval_stats.pooled_mean(per_round["patched"], counts["patched"])
            med_a, detail_a = eval_stats.reduce_rounds(per_round["stock"])
            med_b, detail_b = eval_stats.reduce_rounds(per_round["patched"])
            bar = args.max_throughput_regression if higher_better else args.max_latency_regression
            row = eval_stats.perf_verdict(a, b, max_regression=bar, higher_is_better=higher_better)
            by_median = eval_stats.perf_verdict(med_a, med_b, max_regression=bar,
                                                higher_is_better=higher_better)
            row.update(metric=metric, column=column, label=label, bar=bar,
                       reduction="pooled_mean", rounds=detail_a["n"],
                       pooled_n_stock=n_a, pooled_n_patched=n_b,
                       median_stock=med_a, median_patched=med_b,
                       median_verdict=by_median["verdict"],
                       stock_detail=detail_a, patched_detail=detail_b)

            # ---- can this run resolve a difference at that bar at all? -------
            #
            # **The bars do not move; what moves is whether the run is allowed to
            # answer.** A measurement whose own noise floor exceeds the bar
            # cannot tell a real regression from its own scatter, and the sealed
            # 2026-09-02 report is the worked example of what happens then: it
            # blamed a patch for a neighbour's load. `noise_floor` separates a
            # bad patch from a bad measurement.
            #
            # **Noise can only ever buy a refusal, never a pass.** A noisier run
            # gets `uninterpretable`, which is not accepted. There is no input
            # under which more scatter makes a patch easier to keep.
            #
            # Means only — see `eval_stats.noise_floor`. `p90` columns and the
            # two throughput rates carry no usable dispersion, so they keep the
            # fixed bar; p90's fix is pooled samples and throughput's floor
            # awaits the queued round-to-round measurement (`todo.md` T25).
            disp = dispersion.get((metric, column), {})
            # **The R=1 restriction is gone, and `pooled_mean` is why.** It was
            # there because the judged statistic was a median of round averages,
            # whose variance this formula cannot see. The judged statistic is a
            # mean over the pooled requests again, so the floor is exact at any
            # R and pooling narrows it by about sqrt(R) — measured on the sealed
            # arms, ttft p90's floor goes 20.0% -> 5.5% between R=1 and R=5.
            pooled_n = {"stock": n_a, "patched": n_b}
            floors = {arm: eval_stats.noise_floor(disp.get(f"{arm}_rsd"),
                                                  pooled_n[arm] or disp.get(f"{arm}_n"))
                      for arm in ("stock", "patched")}
            usable = [f for f in floors.values() if f is not None]
            # The worse of the two arms: a comparison is only as resolvable as
            # its noisier side.
            floor = max(usable) if usable else None
            row.update(noise_floor=None if floor is None else round(floor, 6),
                       stock_rsd=disp.get("stock_rsd"), patched_rsd=disp.get("patched_rsd"),
                       n_stock=disp.get("stock_n"), n_patched=disp.get("patched_n"))
            # The two statistics disagreeing is itself an uninterpretable run:
            # the rounds differ by more than the bar can see through, so neither
            # answer is the run's answer. Checked before the floor so that the
            # floor's message wins when both apply — a floor above the bar is
            # the more specific finding and names a number.
            if (row["verdict"] not in ("unmeasured",)
                    and by_median["verdict"] not in ("unmeasured",)
                    and by_median["verdict"] != row["verdict"]):
                row["verdict"] = "uninterpretable"
                row["reduction_disagrees"] = True
                reasons.append(
                    f"{label} ({column}) is UNINTERPRETABLE: pooling the rounds says "
                    f"{eval_stats.perf_verdict(a, b, max_regression=bar, higher_is_better=higher_better)['verdict']!r} "
                    f"and their median says {by_median['verdict']!r}. The rounds disagree by more "
                    "than the bar can see through, so this is a statement about the measurement "
                    "and NOT about the patch. More rounds is the fix; a wider bar is not."
                )
            if floor is not None and floor > bar and row["verdict"] != "unmeasured":
                row["verdict"] = "uninterpretable"
                reasons.append(
                    f"{label} ({column}) is UNINTERPRETABLE: this run's own noise floor is "
                    f"{floor:.1%} and the bar is {bar:.0%}, so a difference at the bar cannot "
                    f"be told from scatter. This is a statement about the measurement, NOT "
                    f"about the patch — the arms read {a} and {b}. More rounds narrow the "
                    f"floor; widening the bar would only hide it."
                )

            comparison_rows.append(row)
            if row["verdict"] == "REGRESSED":
                reasons.append(
                    f"{label} ({column}) over {detail_a['n']} round(s): "
                    f"{a} -> {b}, {row['rel_delta']:+.1%} against a bar of {bar:.0%}"
                )

    # ---- did both arms do the same things in the same order? -----------------
    order = {arm: [s["step"] for s in loaded[arm]["steps"].get("steps", [])] for arm in arms}
    if order["stock"] != order["patched"]:
        reasons.append(
            f"the two arms ran different sequences: {order['stock']} vs {order['patched']}"
        )

    stock_vs_m2 = stock_vs_m2_block(
        Path(args.stock), args.profiling_evidence, args.stock_vs_m2_tolerance
    )
    if stock_vs_m2.get("ok") is False:
        breached = [f"{r['metric']} ({r['column']}) {r['rel_delta']:+.1%}"
                    for r in stock_vs_m2["metrics"] if r.get("within_tolerance") is False]
        reasons.append(
            "the stock arm does not reproduce m2's profiling_mode_off bench within "
            f"{args.stock_vs_m2_tolerance:.0%}: " + ", ".join(breached)
            + " — the two stages measured something different, so this comparison is between "
            "numbers that were never comparable.\n"
            # **It used to say "different machines", and this body cannot know
            # that.** Two stages can disagree on the same host: a different
            # engine build, a different serving configuration, or another
            # tenant's load. Measured 2026-09-04 on rung 1 — the deployed engine
            # gave 42.51 ms ITL where m1's floor was calibrated at 32.5 ms, a
            # 31 % gap on one node with nothing moved. Naming the machine would
            # have sent the reader to check the node, which was fine.
            #
            # `image_id` and not `image`: a floating tag is the case where the
            # engine changed and the record does not show it.
            "  Which one is decidable and this body does not decide it: compare "
            "`fixed.image_id` and `fixed.node` in the two `environment.yaml` records. "
            "Different `node` is two machines. Same `node` and different `image_id` is "
            "two engines. Both the same means the same engine served differently on the "
            "same host — load, configuration, or another tenant — and only the run logs "
            "can say which."
        )

    reconciliation = kernel_reconciliation_block(
        args.profiling_evidence, args.kernel_optimization, perf_rows,
        manifest.get("operator_id"),
    )
    warnings: list[str] = []
    # Per-round breaches surface here so a reader sees which rounds moved and by
    # how much, without any of them being able to reject on its own.
    warnings.extend(round_breaches)
    if reconciliation.get("unavailable_because"):
        warnings.append(f"kernel reconciliation not computed: {reconciliation['unavailable_because']}")
    elif reconciliation.get("agrees") is False:
        warnings.append(
            f"the end-to-end change ({reconciliation['observed_e2e_speedup']}x) and what the "
            f"kernel's own speedup predicts ({reconciliation['predicted_e2e_speedup']}x) differ "
            "by more than 10%. " + str(reconciliation.get("note") or "")
        )

    report = {
        # Which contract produced this. Every other structured document in the
        # package carries it; this one is the flow's verdict, so a later reader
        # deciding whether a stored report is still comparable needs it most.
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "patch": {
            "operator_id": manifest.get("operator_id"),
            "logical_operator": manifest.get("logical_operator"),
            "apply_mode": manifest.get("apply_mode"),
            "image": manifest.get("image"),
            "files": len(manifest.get("files") or []),
            "declares_runtime_marker": bool(manifest.get("runtime_marker")),
            "expect": manifest.get("expect"),
        },
        "arms": {
            arm: {
                "node": loaded[arm]["context"].get("node"),
                "endpoint": loaded[arm]["context"].get("endpoint"),
                "sequence": order[arm],
                "probe_ok": loaded[arm]["probe"].get("ok"),
                "smoke_ok": loaded[arm]["smoke"].get("ok"),
                "needle_ok": loaded[arm]["needle"].get("ok"),
            }
            for arm in arms
        },
        "bars": {
            "max_throughput_regression": args.max_throughput_regression,
            "max_latency_regression": args.max_latency_regression,
            "eval_confidence": 0.95,
        },
        "correctness": {
            "smoke": smoke_rows,
            "needle": needle_rows,
            "evals": eval_rows,
            "prefix_reuse_delta": prefix_reuse,
        },
        "performance": perf_rows,
        # The judged rows: one per (metric, column), rounds reduced first.
        # `performance` above is the evidence this stands on.
        "comparison": comparison_rows,
        # M5.1.3.1, a blocker: the stock arm has to reproduce m2's bench, or the
        # two stages measured different machines and nothing above is comparable.
        "stock_vs_m2": stock_vs_m2,
        # M5.1.3.2, a warning: 作为 report/warning 报告，不作为 blocker.
        "kernel_reconciliation": reconciliation,
        "verdict": {"accepted": not reasons, "reasons": reasons, "warnings": warnings},
    }

    items = out / "items"
    items.mkdir(parents=True, exist_ok=True)
    (items / "text.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # **The package's schema, byte for byte** (CONTRACT.md 3.4). A
    # `structured_text` kind copies its schema into `items/schema` at production
    # time, and its validator checks the copy is identical to the package's — so
    # the artefact is self-describing *and* provably not a private fork. It used
    # to be a hand-written summary here, which is a second schema wearing the
    # name of the first.
    shutil.copyfile(
        Path(args.package) / "assets" / "schemas" / "integration_report.schema.json",
        items / "schema",
    )

    # Nothing is written that does not validate. A report that fails its own
    # schema is worse than none: it looks like a record.
    sys.path.insert(0, str(Path(args.package) / "assets" / "lib"))
    import schema as schema_lib  # noqa: E402

    try:
        schema_lib.validate("integration_report", report)
    except schema_lib.SchemaError as exc:
        raise SystemExit(f"compare: the report does not validate:\n{exc}") from exc

    # ---- the human-readable half --------------------------------------------
    lines = [
        f"# Integration verdict: {'ACCEPTED' if not reasons else 'REJECTED'}",
        "",
        f"Patch `{manifest.get('operator_id')}` "
        f"({manifest.get('logical_operator')}), {len(manifest.get('files', []))} file(s), "
        f"applied as `{manifest.get('apply_mode')}` to `{manifest.get('image')}`.",
        "",
        "Both arms were measured in the same session on "
        f"`{loaded['stock']['context'].get('node')}`, in the same order, against the same "
        "trace. Neither arm's numbers mean anything on their own.",
        "",
        "## Correctness",
        "",
        "| check | stock | patched | verdict |",
        "|---|---|---|---|",
    ]
    for row in smoke_rows:
        lines.append(f"| smoke: {row['check']} | {row['stock']} | {row['patched']} | {row['verdict']} |")
    for row in needle_rows:
        tag = "needle" if row["gated"] else "needle (frontier, not gated)"
        lines.append(
            f"| {tag}: {row['run']} | {row['stock_ok']} | {row['patched_ok']} | {row['verdict']} |"
        )
    lines += ["", "| eval | scored | stock | patched | delta | 95% CI of delta | verdict |",
              "|---|---|---|---|---|---|---|"]
    for row in eval_rows:
        if row.get("verdict") == "unmeasured":
            lines.append(f"| {row['name']} | — | — | — | — | — | unmeasured |")
            continue
        lines.append(
            f"| {row['name']} | {row['scored_a']}/{row['scored_b']} | {row['score_a']:.3f} | "
            f"{row['score_b']:.3f} | {row['delta']:+.3f} | "
            f"[{row['ci95_delta'][0]:+.3f}, {row['ci95_delta'][1]:+.3f}] | {row['verdict']} |"
        )
    if prefix_reuse:
        lines += [
            "",
            "Prefix-reuse delta (`mixed_prefix_gsm8k` − `gsm8k`): "
            + ", ".join(f"{arm} {value:+.3f}" for arm, value in prefix_reuse.items())
            + ". The same questions behind partially shared few-shot prefixes; a "
            "difference here is prefix reuse changing answers, which nothing else "
            "in this suite can see.",
        ]

    lines += ["", "## Performance", "",
              "| round | metric | stock | patched | change | bar | verdict |",
              "|---|---|---|---|---|---|---|"]
    for row in perf_rows:
        change = "—" if row["rel_delta"] is None else f"{row['rel_delta']:+.1%}"
        bar = "—" if row.get("bar") is None else f"{row['bar']:.0%}"
        lines.append(
            f"| {row['round']} | {row['label']} ({row['column']}) | {row['stock']} | "
            f"{row['patched']} | {change} | {bar} | {row['verdict']} |"
        )
    lines += [
        "",
        "Round 1 is cold for this trace and round 2 is warm; they are compared only "
        "against their own counterpart. On this trace and this deployment the two "
        "differ by roughly an order of magnitude, because a Mooncake trace carries "
        "`hash_ids` and prefix hit rate decides how much prefill there is to do.",
        "",
        "## Verdict",
        "",
    ]
    if reasons:
        lines.append("Rejected. Reasons:")
        lines += [f"- {reason}" for reason in reasons]
    else:
        lines.append(
            "Accepted: no smoke check, needle depth, eval interval or performance metric "
            "moved past its bar."
        )
    if manifest.get("expect", {}).get("source") == "mock":
        lines += [
            "",
            "**The patch under test is a mock.** It changes no arithmetic, so the correct "
            "answer here is 'no difference' and an accepted verdict is evidence that the "
            "comparison does not invent regressions — not evidence about any real "
            "optimisation.",
        ]
    (items / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out / "README.md").write_text(
        """# integration_report

## Purpose

The two arms of the integration test side by side, and the accept-or-reject
argument built from them. This is what the stage exists to produce.

Nothing here judges an arm on its own. An absolute eval score has no external
baseline and an absolute throughput figure has no meaning without the
configuration behind it; both arms were measured in one session, on one node,
against one trace, in one order, so that neither has to.

`items/report.md` is the same content for a reader. `items/text.json` is what
`check_no_regression` recomputes — it does not read the `verdict` field, it
rebuilds it from the numbers and fails if the two disagree.

## Schema

`items/schema` is the JSON Schema. In prose:

- `patch` — what was under test: operator, apply mode, image, file count, and
  whether it declared runtime markers.
- `arms` — per arm, the node, the endpoint, the step sequence, and the three
  self-reported pass flags.
- `bars` — the thresholds the verdict used. Recorded because a verdict without
  the bar it was measured against cannot be re-read later.
- `correctness.smoke` / `correctness.needle` — per-check arm comparison.
  `verdict` is `REGRESSED` only when stock passed and patched did not; both
  failing is not a regression, and for the frontier needle length it is the
  measured expectation.
- `correctness.evals` — per eval, both arms' scores with Wilson intervals, and
  the difference with a Newcombe interval. `REGRESSED` means the interval
  excludes zero and the difference is negative.
- `correctness.prefix_reuse_delta` — `mixed_prefix_gsm8k` minus `gsm8k` per arm.
- `performance` — one row per round, metric and column. `rel_delta` is
  `(patched - stock) / stock`. Rows with `verdict: context` are reported and not
  judged; the request count is one, because a difference there means the arms
  replayed different work rather than that the patch is slow.
- `verdict` — `accepted` plus every reason it is not.
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(Path(args.package) / "assets" / "lib" / "redact.py"),
            str(out),
            f"TASK_PACKAGE={args.package}",
            f"ZONE={Path.cwd()}",
            "TMPDIR=/tmp",
            f"HOME={Path.home()}",
        ],
        check=True,
    )

    print(f"compare: {'ACCEPTED' if not reasons else 'REJECTED'}")
    for reason in reasons:
        print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
