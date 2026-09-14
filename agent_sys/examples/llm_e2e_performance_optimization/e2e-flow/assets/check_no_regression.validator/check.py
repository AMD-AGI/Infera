#!/usr/bin/env python3
"""`check_no_regression` — trustworthiness, strong.

**This body does not read the report's `verdict` field.** It recomputes every
comparison from the raw numbers the report carries, and fails when its own answer
disagrees with the one the report states.

That is the difference between checking a claim and reading it. `compare` is a
program in this package; a bug in it would otherwise produce a report that
passes its own validation, and the accept-or-reject decision this whole stage
exists to make would rest on nothing. The statistics come from
`assets/lib/eval_stats.py`, which `compare` also uses — sharing the arithmetic is
what makes agreement meaningful, because the thing being checked is the
report's *assembly*, not whether two copies of a Wilson interval match.

`trustworthiness` rather than `usability`, which is what `integration-demo`
called it: what it judges is not whether the report is legible but whether its
conclusion is the one its own numbers support.

**Two rules the carried body did not have**, both from the mission:

`stock_vs_m2` (M5.1.3.1) is a **blocker**. The stock arm and m2's
`profiling_mode_off` bench are the same measurement of the same deployment one
stage apart; if they disagree, the two stages measured different machines and
this report compares numbers that were never comparable.

`kernel_reconciliation` (M5.1.3.2) is a **warning**, exactly as the mission asks
— 作为 report/warning 报告，不作为 blocker. Amdahl over one kernel, with the
kernel's share measured under the profiler with CUDA graph off and the end-to-end
number measured under neither, so the two sides legitimately disagree. The hard
floor stays where it was: the performance bars, at the measured 5% and 10%.

A disagreement fails the handoff even when the recomputation says "accepted".
Two answers that differ mean one of them is wrong, and until that is resolved
neither can be relied on.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import eval_stats  # noqa: E402
import schema as schema_lib  # noqa: E402
import workset_io  # noqa: E402 — the shared report writer; see _report()
import zone  # noqa: E402


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


def _mentioned(needles: list[list[str]], stated: list[str]) -> bool:
    """Does some stated reason name the same *thing* this recomputed one does?

    **Not a prefix match on prose, and that is a fix rather than a preference.**
    The carried body compared `recomputed_reason[:40]` against each stated
    reason, which works only when the two sides phrase a finding identically.
    They do not: `compare` writes for a human — *"output token throughput (avg)
    in r1: 193.59 -> 46.7, -75.9% against a bar of 35%"* — and the recomputation
    writes a metric id. Measured against the real refused report from
    2026-09-02: every one of its seven reasons was reported as "a reason the
    report does not list", while the report listed all seven. The rule had never
    fired before because the only report it had ever graded was an accepted one,
    where both sides carry zero reasons and the comparison is vacuous.

    So the match is structural: each group of alternatives must have one member
    somewhere in the stated reasons. `["ttft_ms", "time to first token"]` is one
    group because the report may name the metric or its label, and `["r1"]` and
    `["(avg)"]` are groups because a regression in a different round or column is
    a different finding.
    """
    lowered = [str(s).lower() for s in stated]
    return all(
        any(alt.lower() in text for alt in group for text in lowered)
        for group in needles
        if group
    )


def recompute(report: dict, args: dict, reasons: list) -> list[str]:
    """Every reason this report should have said "rejected"."""
    found: list[str] = []

    for row in report.get("correctness", {}).get("smoke", []):
        if row.get("stock") and not row.get("patched"):
            found.append(f"smoke check {row.get('check')!r} passed on stock and failed on patched")

    # Only the asymmetric case. A depth both arms miss says something about the
    # deployment at that length and nothing about the patch: retrieval is not a
    # stable property here (the table in assets/accept/needle.py has the nine
    # measurements), so "both failed" is the expected reading rather than a
    # finding. The per-arm floor is check_acceptance's.
    for row in report.get("correctness", {}).get("needle", []):
        if row.get("stock_ok") and not row.get("patched_ok"):
            found.append(f"needle {row.get('run')} retrieved on stock and not on patched")

    # A probe failure on either arm makes every eval score unreadable, so the
    # comparison is `uninterpretable` and that is a reason to reject: the
    # failures the probe catches produce a number rather than an error, and the
    # number is indistinguishable from a real regression.
    probe_failed = [
        name for name, arm in (report.get("arms") or {}).items() if arm.get("probe_ok") is False
    ]
    for name in probe_failed:
        found.append(f"probe failed on the {name} arm, so its eval scores cannot be read")

    for row in report.get("correctness", {}).get("evals", []):
        if row.get("verdict") == "unmeasured":
            found.append(f"eval {row.get('name')!r} did not produce a score on both arms")
            continue
        if row.get("verdict") == "uninterpretable":
            found.append(f"eval {row.get('name')!r} is uninterpretable: {row.get('uninterpretable_because')}")
            continue
        # Recomputed from the scores and counts, not read from `ci95_delta`: the
        # interval is the argument, and copying it would be reading the claim.
        again = eval_stats.compare_scores(
            row["score_a"], row["scored_a"], row["score_b"], row["scored_b"]
        )
        if probe_failed:
            # The arithmetic is still checkable; the conclusion is not, and the
            # report should already have said so.
            continue
        if again["verdict"] != row.get("verdict"):
            found.append(
                f"eval {row.get('name')!r}: the report says {row.get('verdict')!r} and the "
                f"numbers say {again['verdict']!r}"
            )
        if again["verdict"] == "REGRESSED":
            found.append(f"eval {row.get('name')!r} regressed")

    # **The validator's own bars, not the report's.** `args` wins whenever it is
    # set, and it always is here. The report's `bars` are recorded so a reader
    # can see what the producer decided against, and they are checked for being
    # no looser than these (in `check`), because a producer that picks its own
    # bar can pass anything. The sealed 2026-09-02 report is that case: it
    # declares 0.35/0.30, widened in response to two arms measured fifteen
    # minutes and one co-tenant apart, and the right response to that was a
    # comparability gate at bring-up (`todo.md` T7) rather than a looser bar.
    bars = report.get("bars") or {}
    throughput_bar = float(args.get("max_throughput_regression", bars.get("max_throughput_regression", 0.05)))
    latency_bar = float(args.get("max_ttft_regression", bars.get("max_latency_regression", 0.10)))
    higher_better = {"output_token_throughput_tps", "request_throughput_rps"}
    per_round_breaches: list[str] = []
    for row in report.get("performance", []):
        if row.get("verdict") == "context":
            # Reported and not judged -- but a request-count difference means the
            # arms replayed different work, which invalidates the comparison
            # rather than failing the patch, and that has to surface somewhere.
            if row.get("metric") == "request_count" and row.get("stock") != row.get("patched"):
                found.append(
                    f"{row.get('round')}: the arms replayed different request counts "
                    f"({row.get('stock')} vs {row.get('patched')})"
                )
            continue
        metric = row.get("metric")
        up_is_good = metric in higher_better
        again = eval_stats.perf_verdict(
            row.get("stock"),
            row.get("patched"),
            max_regression=throughput_bar if up_is_good else latency_bar,
            higher_is_better=up_is_good,
        )
        if again["verdict"] != row.get("verdict"):
            found.append(
                f"{row.get('round')} {metric} ({row.get('column')}): the report says "
                f"{row.get('verdict')!r} and the numbers say {again['verdict']!r}"
            )
        # **A per-round breach is not a refusal, and this is the fix to a defect
        # in the gate's shape.** Every row used to refuse, so R rounds were 7R
        # independent tests at the bar and the gate got less trustworthy the more
        # evidence it was given: family-wise false refusal 6.8% at R=1, 29.7% at
        # R=5, 50.5% at R=10, at a 1% per-row rate. Rounds must average, not
        # multiply. The verdict is taken once per (metric, column) below, on the
        # reduction; the per-round rows remain and their arithmetic is still
        # checked, because a report whose own rows do not add up is not usable
        # whichever level the verdict is taken at.
        if again["verdict"] == "REGRESSED":
            per_round_breaches.append(f"{row.get('round')} {metric} ({row.get('column')})")

    found.extend(judge_comparison(report, args, throughput_bar, latency_bar,
                                  higher_better, per_round_breaches))

    arms = report.get("arms") or {}
    sequences = {name: tuple(a.get("sequence") or ()) for name, a in arms.items()}
    if len(set(sequences.values())) > 1:
        found.append(f"the arms ran different sequences: {sequences}")

    found.extend(stock_reproduces_m2(report, args))
    return found


def judge_comparison(report: dict, args: dict, throughput_bar: float, latency_bar: float,
                     higher_better: set, per_round_breaches: list) -> list[str]:
    """The verdict, taken once per (metric, column) on the reduced rounds.

    **This is where the gate refuses, and `performance` no longer is.** Judging
    every round separately made R rounds into 7R independent tests at the bar,
    so the gate became less trustworthy the more evidence it was given —
    measured family-wise false refusal 6.8% at R=1, 29.7% at R=5, 50.5% at R=10
    for a 1% per-row rate. Reduction removes that; nothing about the bars
    changes, and at R=1 the reduction is the identity so no existing verdict
    moves.

    **The reduction is recomputed here from `performance`, not read.** Same rule
    as everywhere else in this file: `compare` is a program in this package, and
    a producer that both reduces and reports its own reduction could hide a bug
    in it. `eval_stats.reduce_rounds` is the one definition both sides call.
    """
    rows = report.get("comparison")
    if not isinstance(rows, list) or not rows:
        return [
            "the report carries no `comparison` block. The verdict is taken on the "
            "rounds reduced per (metric, column); a report with only per-round rows "
            "predates that and cannot be judged without reintroducing the "
            "multiple-comparison inflation reduction exists to remove."
        ]

    round_counts: dict[str, dict] = {
        r.get("round"): {"stock": r.get("stock"), "patched": r.get("patched")}
        for r in report.get("performance", [])
        if r.get("metric") == "request_count"
    }
    by_key: dict[tuple, dict[str, list]] = {}
    for row in report.get("performance", []):
        if row.get("verdict") == "context":
            continue
        key = (row.get("metric"), row.get("column"))
        slot = by_key.setdefault(key, {"stock": [], "patched": [],
                                       "n_stock": [], "n_patched": []})
        slot["stock"].append(row.get("stock"))
        slot["patched"].append(row.get("patched"))
        # The per-round request counts live on the `context` rows, keyed by
        # round; pooling needs them as weights.
        rnd = row.get("round")
        counts = round_counts.get(rnd, {})
        slot["n_stock"].append(counts.get("stock"))
        slot["n_patched"].append(counts.get("patched"))

    found: list[str] = []
    judged = set()
    for row in rows:
        metric, column = row.get("metric"), row.get("column")
        judged.add((metric, column))
        up_is_good = metric in higher_better
        bar = throughput_bar if up_is_good else latency_bar
        seen = by_key.get((metric, column))
        if not seen:
            found.append(
                f"{metric} ({column}) is compared but no per-round row supports it — "
                "the reduction stands on nothing this report carries")
            continue
        # **Both statistics, recomputed.** `compare` judges the pooled mean and
        # carries the median beside it; a disagreement between them is itself
        # `uninterpretable`, so a validator that recomputed only one could not
        # check that verdict at all.
        a, _n_a = eval_stats.pooled_mean(seen["stock"], seen["n_stock"])
        b, _n_b = eval_stats.pooled_mean(seen["patched"], seen["n_patched"])
        med_a, _ = eval_stats.reduce_rounds(seen["stock"])
        med_b, _ = eval_stats.reduce_rounds(seen["patched"])
        # The producer's own reduced pair has to match what its rounds reduce to.
        for name, mine, theirs in (("stock", a, row.get("stock")),
                                   ("patched", b, row.get("patched"))):
            if mine is None or theirs is None:
                continue
            if abs(mine - theirs) > max(1e-6, abs(mine) * 1e-6):
                found.append(
                    f"{metric} ({column}): the report's reduced {name} is {theirs} and its own "
                    f"{len(seen[name])} round(s) reduce to {mine}")
        again = eval_stats.perf_verdict(a, b, max_regression=bar, higher_is_better=up_is_good)

        # **The noise floor, recomputed and not read.** Closed form from the
        # three numbers the report carries, so this is the same discipline as
        # every other line here: a producer cannot declare its own measurement
        # resolvable. A row whose floor exceeds its bar is `uninterpretable` —
        # the run cannot tell a difference at the bar from its own scatter —
        # and that is a refusal, never a pass. Noise buys a harder outcome.
        # No R=1 guard any more: the judged statistic is a pooled mean, so the
        # closed form is exact at every R. `n` is the pooled count the producer
        # carried, recomputed above as `_n_a`/`_n_b` and preferred over it.
        pooled_n = {"stock": _n_a, "patched": _n_b}
        floors = [eval_stats.noise_floor(row.get(f"{arm}_rsd"),
                                         pooled_n[arm] or row.get(f"n_{arm}"))
                  for arm in ("stock", "patched")]
        usable = [f for f in floors if f is not None]
        floor = max(usable) if usable else None
        stated = row.get("noise_floor")
        if floor is not None and isinstance(stated, (int, float)):
            if abs(floor - stated) > max(1e-6, abs(floor) * 1e-3):
                found.append(
                    f"{metric} ({column}): the report states a noise floor of {stated:.2%} and "
                    f"its own dispersion gives {floor:.2%}")
        # **Two causes, two messages, and conflating them crashed.** The
        # disagreement between the statistics has no floor attached, so the
        # floor's message cannot serve both — formatting a None as a percentage
        # is what the R=5 fixture caught immediately after this landed.
        by_median = eval_stats.perf_verdict(med_a, med_b, max_regression=bar,
                                            higher_is_better=up_is_good)
        disagrees = (again["verdict"] != "unmeasured"
                     and by_median["verdict"] != "unmeasured"
                     and by_median["verdict"] != again["verdict"])
        floor_exceeds = (floor is not None and floor > bar
                         and again["verdict"] != "unmeasured")
        if disagrees or floor_exceeds:
            again = dict(again, verdict="uninterpretable")

        if again["verdict"] != row.get("verdict"):
            found.append(
                f"{metric} ({column}): the report says {row.get('verdict')!r} and the reduced "
                f"numbers say {again['verdict']!r}")
        if again["verdict"] == "REGRESSED":
            found.append(f"{metric} ({column}) regressed over {len(seen['stock'])} round(s)")
        if floor_exceeds:
            found.append(
                f"{metric} ({column}) is UNINTERPRETABLE: noise floor {floor:.1%} exceeds the "
                f"{bar:.0%} bar, so this run cannot resolve a difference at it. This says the "
                "measurement is unusable, NOT that the patch is bad — do not read it as a pass, "
                "and do not answer it by widening the bar.")
        elif disagrees:
            found.append(
                f"{metric} ({column}) is UNINTERPRETABLE: pooling the rounds says "
                f"{eval_stats.perf_verdict(a, b, max_regression=bar, higher_is_better=up_is_good)['verdict']!r} "
                f"and their median says {by_median['verdict']!r}. The rounds disagree by more than "
                "the bar can see through, so this is about the measurement and NOT the patch.")

    # A per-round row with no reduced row above it would be judged by nothing.
    for key in by_key:
        if key not in judged:
            found.append(f"{key[0]} ({key[1]}) has per-round rows and no comparison row, "
                         "so nothing judged it")

    if per_round_breaches and not found:
        print("  individual rounds breached and the reduction did not: "
              + ", ".join(per_round_breaches))
        print("    Reported, not judged. If this is common, the rounds disagree more than "
              "the bar allows and that is a comparability problem (todo.md T7), not a patch one.")
    return found


def unlisted_findings(report: dict, args: dict) -> list[str]:
    """The findings the recomputation makes that the report's own reasons do not name.

    Agreeing on the boolean is not enough — a verdict that is right by accident is
    not a verdict — so this walks the rows the recomputation judged `REGRESSED`
    and asks whether the report said so anywhere, matching on the round, the
    metric (by id or by label) and the column rather than on wording.
    """
    stated = (report.get("verdict") or {}).get("reasons") or []
    bars = report.get("bars") or {}
    throughput_bar = float(args.get("max_throughput_regression", bars.get("max_throughput_regression", 0.05)))
    latency_bar = float(args.get("max_ttft_regression", bars.get("max_latency_regression", 0.10)))
    higher_better = {"output_token_throughput_tps", "request_throughput_rps"}

    missing: list[str] = []
    # **`comparison` and not `performance`, and getting this wrong reintroduced
    # the whole defect through a side door.** This function used to walk the
    # per-round rows and demand that every REGRESSED one be named in
    # `verdict.reasons`. Once the verdict moved to the reduction, that turned a
    # single bad round into a refusal again — measured: 5 rounds with one 1.40x
    # outlier, the reduced rows all `same`, and this refused anyway on seven
    # "the report's reasons do not name it" findings. The reduction has to be
    # complete or the inflation comes back invisibly, which is worse than never
    # having reduced.
    #
    # Per-round breaches are `warnings` now (`compare.py` writes them there) and
    # `judge_comparison` prints them. A warning is not something the report must
    # justify in `reasons`.
    for row in report.get("comparison", []):
        metric = str(row.get("metric") or "")
        again = eval_stats.perf_verdict(
            row.get("stock"), row.get("patched"),
            max_regression=throughput_bar if metric in higher_better else latency_bar,
            higher_is_better=metric in higher_better,
        )
        if again["verdict"] != "REGRESSED":
            continue
        groups = [
            [g for g in (metric, str(row.get("label") or "")) if g],
            [f"({row.get('column')})"],
        ]
        if not _mentioned(groups, stated):
            missing.append(
                f"{metric} ({row.get('column')}) regressed and the report's "
                "reasons do not name it"
            )

    for row in report.get("correctness", {}).get("evals", []):
        if row.get("score_a") is None or row.get("score_b") is None:
            continue
        again = eval_stats.compare_scores(
            row["score_a"], row["scored_a"], row["score_b"], row["scored_b"]
        )
        if again["verdict"] == "REGRESSED" and not _mentioned([[str(row.get("name") or "")]], stated):
            missing.append(f"eval {row.get('name')!r} regressed and the report's reasons do not name it")
    return missing


def stock_reproduces_m2(report: dict, args: dict) -> list[str]:
    """M5.1.3.1 — 原算子应该复现 2 中 bench 的结果. **A blocker.**

    The stock arm and m2's `profiling_mode_off` bench are the same measurement of
    the same deployment one stage apart. If they disagree, the two stages
    measured different machines — or the same machine in two different states —
    and every number this report compares was taken on a system that moved
    underneath the pipeline. That is not a fault of the patch and it is not
    survivable by the comparison either: the whole chain's arithmetic is void.

    It is a blocker and `kernel_reconciliation` is a warning, and the asymmetry
    is deliberate. This one asks whether the measurements are of the same thing,
    which everything downstream assumes. That one asks whether two models of the
    same effect agree, which is interesting and is not load-bearing.

    The numbers reach this body inside the report, because this validator's only
    input is the report — `compare` reads m2's handoff, which its task already
    holds, and writes the comparison down. The block is REQUIRED by the schema
    precisely so that "we did not do it" has to be written rather than omitted.
    """
    block = report.get("stock_vs_m2")
    if not isinstance(block, dict):
        return [
            "the report carries no `stock_vs_m2` block. M5.1.3.1 requires the stock arm to "
            "reproduce m2's profiling_mode_off bench, and a missing section is indistinguishable "
            "from a failed one to every later reader."
        ]

    if block.get("ok") is None:
        why = (block.get("unavailable_because") or "").strip()
        if not why:
            return [
                "`stock_vs_m2.ok` is null and `unavailable_because` is empty. A comparison that "
                "could not be made has to say why; silence here reads as a pass."
            ]
        # Not a blocker: an honest "m2's bench was not available to this task" is
        # a fact about the run, and refusing it would make the whole flow
        # unrunnable in the modes that legitimately lack an m2 arm. It IS loud.
        print(f"  stock_vs_m2: not measured — {why}")
        return []

    found: list[str] = []
    tolerance = block.get("tolerance")
    if tolerance is None:
        tolerance = float(args.get("stock_vs_m2_tolerance", 0.10))
    tolerance = float(tolerance)
    metrics = block.get("metrics") or []
    if not metrics:
        return ["`stock_vs_m2.ok` is set but no metrics were compared"]

    breached = []
    for row in metrics:
        rel = row.get("rel_delta")
        label = f"{row.get('metric')} ({row.get('column')})"
        if rel is None:
            found.append(f"stock_vs_m2: {label} has no rel_delta, so it was not actually compared")
            continue
        # **Both signs.** A stock arm faster than m2's is as much evidence of a
        # different machine as a slower one, and only checking one direction
        # would pass exactly the case where the later stage got a quieter node.
        if abs(float(rel)) > tolerance:
            breached.append(f"{label} {float(rel):+.1%}")
    if breached:
        found.append(
            "the stock arm does not reproduce m2's profiling_mode_off bench within "
            f"{tolerance:.0%}: " + ", ".join(breached) + ". The two stages measured different "
            "machines, or one machine in two states, so this report's comparison is between "
            "numbers that were never comparable."
        )
    if bool(block.get("ok")) != (not breached):
        found.append(
            f"stock_vs_m2 states ok={block.get('ok')} and the numbers say ok={not breached}"
        )
    if not found:
        print(f"  stock_vs_m2: {len(metrics)} metric(s) within {tolerance:.0%} of m2's bench")
    return found


def reconcile_with_kernel(report: dict, args: dict) -> list[str]:
    """M5.1.3.2 — the single-kernel speedup against the end-to-end change.

    **Warning, never a blocker**, exactly as the mission puts it: 作为
    report/warning 报告，不作为 blocker. Amdahl over one kernel — a kernel that is
    `s` of the profile and got `k` times faster caps the end-to-end speedup at
    `1 / ((1-s) + s/k)` — and the two sides of that equation are measured in
    configurations that differ on purpose: `s` comes from a profile taken with
    the profiler attached and CUDA graph OFF, and the end-to-end number is taken
    with neither. A disagreement is a question about the model, not a fault in
    the run, and treating it as a gate would refuse correct work.

    **Where the hard floor actually lives.** The mission's floor — the patched arm
    must not be slower than stock — is the performance bars in `recompute`, at
    5% and 10%. A literal zero-tolerance floor would be the wrong reading of it:
    the within-arm round-to-round spread on a steady node is ~2%, so "strictly
    not slower" refuses noise about half the time. The bars were measured to be
    right and are not widened here (`todo.md` T7).

    Returned as warnings; the caller prints them and puts them in the report's
    own `verdict.warnings` comparison, and never in `found`.
    """
    block = report.get("kernel_reconciliation")
    if not isinstance(block, dict):
        return [
            "the report carries no `kernel_reconciliation` block (M5.1.3.2). It is a warning "
            "and it is still required to be present: a section that may be omitted is a "
            "section that will be."
        ]
    if block.get("unavailable_because"):
        return [f"kernel reconciliation not computed: {block['unavailable_because']}"]

    k = block.get("kernel_speedup")
    s = block.get("kernel_share_of_profile")
    predicted = block.get("predicted_e2e_speedup")
    observed = block.get("observed_e2e_speedup")
    out: list[str] = []
    if k is None or s is None:
        return ["kernel reconciliation carries no kernel_speedup / kernel_share_of_profile "
                "and no `unavailable_because` to explain it"]

    k, s = float(k), float(s)
    if k <= 0:
        return [f"kernel_speedup is {k}, which is not a ratio"]
    amdahl = 1.0 / ((1.0 - s) + s / k)
    if predicted is not None and abs(float(predicted) - amdahl) > 0.005:
        out.append(
            f"predicted_e2e_speedup is {float(predicted):.4f} and Amdahl over the recorded "
            f"share gives {amdahl:.4f}"
        )
    print(f"  reconciliation: kernel {k:.3f}x at {s:.2%} of the profile predicts {amdahl:.4f}x "
          f"end to end; observed {('%.4f' % float(observed)) + 'x' if observed is not None else 'not recorded'}")
    if observed is None:
        out.append("no observed_e2e_speedup was recorded, so nothing could be reconciled")
        return out
    observed = float(observed)
    # A wide band, because the two sides are measured in different
    # configurations by design. This says "these two do not describe the same
    # world", not "this run is wrong".
    band = float(args.get("reconcile_band", 0.10))
    if amdahl > 0 and abs(observed - amdahl) / amdahl > band:
        out.append(
            f"the end-to-end change ({observed:.3f}x) and what the kernel's own speedup predicts "
            f"({amdahl:.3f}x) differ by more than {band:.0%}. The share was measured with the "
            "profiler attached and CUDA graph off and the end-to-end number with neither, so "
            "this is a question about the model before it is a question about the run."
        )
    if bool(block.get("agrees")) != (not out):
        out.append(f"the block states agrees={block.get('agrees')} and the numbers say {not out}")
    return out


def check(content: Path, args: dict, reasons: list) -> bool:
    text_json = content / "items" / "text.json"
    if not text_json.is_file():
        return _fail(reasons, "items/text.json is missing")
    try:
        report = json.loads(text_json.read_text(encoding="utf-8"))
    except ValueError as exc:
        return _fail(reasons, f"text.json is not readable as JSON: {exc}")

    ok = True

    # The shared schema, in front of the producer and this body alike (G2). It
    # is checked before anything is read out of the document, so a malformed
    # report fails as a malformed report rather than three functions later as a
    # missing key.
    # **An absent `schema` is a refusal, not a skip.** This was `if name:` — so
    # with the arg missing the validator's *first and strongest* check quietly
    # did not run and said nothing, while the other five args degrade to safe
    # defaults that still bite (`0.05`, `0.10`, `0.10` at `:145`, `:146`, `:446`).
    #
    # Surfaced by m2's warning that the probe was passing `args={}` to every
    # validator: mine returned **True** with all six args discarded, which is
    # not evidence it works — it is `items_schema`'s shape, present and checking
    # nothing. The probe has since been fixed to pass the real args, which makes
    # this **invisible precisely when the tooling is working**, so it would not
    # have been found again.
    #
    # The step file always supplies `schema`, so this is unreachable through the
    # package's own wiring. That is the argument for refusing rather than
    # defaulting to a name: if it is ever absent, something is wrong with the
    # caller and a silent pass is the worst available answer.
    name = args.get("schema")
    if not name:
        return _fail(
            reasons,
            "no `schema` arg was supplied, so the report was never validated against one. "
            "This validator's other bars are thresholds with safe defaults; this one is the "
            "document check, and it has no default because a report graded against no schema "
            "has not been graded. Supply `schema: integration_report`.",
        )
    try:
        schema_lib.validate(str(name), report)
    except schema_lib.SchemaError as exc:
        return _fail(reasons, f"the report does not validate against {name!r}:\n{exc}")

    for arm in args.get("require_arms") or ("stock", "patched"):
        if arm not in (report.get("arms") or {}):
            ok = _fail(reasons, f"the report carries no {arm!r} arm")

    if not (content / "items" / "report.md").is_file():
        ok = _fail(reasons, "items/report.md is missing — there is no readable form of the verdict")

    bars = report.get("bars") or {}
    if not bars:
        ok = _fail(
            reasons,
            "the report records no bars. A verdict without the threshold it was decided "
            "against cannot be re-read later.",
        )
    else:
        # **A producer may not widen its own bar.** Recomputation already uses
        # this validator's `args`, so a looser declared bar cannot change the
        # answer — but it can make the report's own rows say `same` where the
        # recomputation says `REGRESSED`, and then the disagreement surfaces as
        # a dozen per-row complaints instead of the one sentence that explains
        # them. Naming it here is the difference between a diagnosis and a list.
        for key, arg_key in (
            ("max_throughput_regression", "max_throughput_regression"),
            ("max_latency_regression", "max_ttft_regression"),
        ):
            declared, mine = bars.get(key), args.get(arg_key)
            if declared is None or mine is None:
                continue
            if float(declared) > float(mine):
                ok = _fail(
                    reasons,
                    f"the report was decided against {key}={float(declared):.0%} and this "
                    f"validator's bar is {float(mine):.0%}. A producer that chooses its own "
                    "threshold can pass anything. If the bar is genuinely wrong for this "
                    "cluster the answer is a comparability gate at bring-up, not a wider bar "
                    "— that is `todo.md` T7, and a previous round got it the other way round.",
                )

    warnings = reconcile_with_kernel(report, args)
    for warning in warnings:
        print(f"  ! {warning}")
    if args.get("reconcile_with_kernel_speedup") == "warn":
        stated_warnings = (report.get("verdict") or {}).get("warnings") or []
        for warning in warnings:
            if not any(warning[:40] in str(s) for s in stated_warnings):
                print(f"    (the report does not carry this warning)")
    elif warnings and args.get("reconcile_with_kernel_speedup") == "block":
        ok = _fail(reasons, "; ".join(warnings))

    recomputed = recompute(report, args, reasons=[])
    stated = report.get("verdict") or {}
    stated_accepted = bool(stated.get("accepted"))
    print(f"  recomputed: {'accepted' if not recomputed else 'REJECTED'} "
          f"({len(recomputed)} reason(s)); report states "
          f"{'accepted' if stated_accepted else 'REJECTED'}")
    for reason in recomputed:
        print(f"    ~ {reason}")

    if args.get("require_verdict_agreement", True):
        if bool(recomputed) == stated_accepted:
            ok = _fail(
                reasons,
                f"the report states accepted={stated_accepted} and recomputation says "
                f"accepted={not recomputed}. One of them is wrong and neither can be used "
                "until that is resolved.",
            )
        else:
            # Same answer; now check they agree on WHY. A verdict that is right by
            # accident is not a verdict.
            missing = unlisted_findings(report, args)
            if missing:
                ok = _fail(
                    reasons,
                    f"recomputation found {len(missing)} reason(s) the report does not list: "
                    + "; ".join(missing[:3]),
                )

    if recomputed:
        # **Split, and this is the requirement rather than a nicety.** "the patch
        # regressed" over a list of `uninterpretable` findings is the exact
        # confusion `uninterpretable` exists to prevent: it blames the change for
        # a measurement that could not answer. That is the mistake the sealed
        # 2026-09-02 report made, and printing it here would reintroduce it in
        # the validator's own summary line while the per-row text said otherwise.
        # Case-insensitive: the word appears shouted in the finding text and
        # lower-cased inside a verdict-disagreement message, and a disagreement
        # about whether a row is uninterpretable is bookkeeping about the
        # report, not evidence about the patch either.
        def _about_measurement(text: str) -> bool:
            low = text.lower()
            return "uninterpretable" in low or "noise floor" in low

        unresolvable = [r for r in recomputed if _about_measurement(r)]
        regressions = [r for r in recomputed if not _about_measurement(r)]
        if regressions:
            ok = _fail(reasons, f"the patch regressed: {'; '.join(regressions[:5])}")
        if unresolvable:
            ok = _fail(
                reasons,
                "this run cannot judge the patch — "
                f"{len(unresolvable)} metric(s) have a noise floor above their bar, so a "
                "difference at the bar is indistinguishable from scatter. NOT a statement "
                f"about the patch: {'; '.join(unresolvable[:3])}",
            )
    return ok


def main() -> int:
    args = zone.args()
    results: dict = {}
    findings: dict = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            # Captured and re-echoed: the lines that explain a PASS go to stdout,
            # and a zone keeps no stdout at all. A person watching the run still
            # sees them; so, now, does anyone reading the zone afterwards.
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    results[hid] = check(content, args, reasons)
            except Exception as exc:  # noqa: BLE001
                # A crash is not a refusal. verdict.json cannot express the
                # difference (todo.md T29); this text is the only place it exists.
                results[hid] = False
                reasons.append(f"THIS VALIDATOR DID NOT RUN: {type(exc).__name__}: {exc}")
            sys.stdout.write(buffer.getvalue())
            notes = [ln.strip() for ln in buffer.getvalue().splitlines() if ln.strip()]
            findings[hid] = ([] if results[hid] else list(reasons),
                             notes + (list(reasons) if results[hid] else []))
        findings.setdefault(hid, (list(reasons), []))
        print(f"check_no_regression: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    # Before write_verdict, deliberately: a crash in the writer must not be able
    # to take the reasons with it, and the verdict is what the phase reads.
    _report(findings, results)
    zone.write_verdict(results)
    return 0


def _report(findings: dict, results: dict) -> None:
    """`workset_io.write_report`, and never a second implementation of it.

    m3 measured that 16 of 21 validators persist nothing, and seven of those are
    this stage's. That matters most here because **stage 5 has never been
    reached**: every other stage has had refusals to learn from, and m5's first
    one would otherwise arrive with the diagnostics switched off.

    `verdicts` is passed rather than letting the heading infer from `problems`
    being non-empty — these bodies keep informational lines in the same
    `reasons` list, which is the case that made the argument exist.

    Wrapped so that a failure to write the report cannot fail the validation:
    the report is evidence *about* a verdict and must never become the reason
    there is not one.
    """
    try:
        workset_io.write_report("check_no_regression", findings, results)
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print("check_no_regression: could not write the validator report: %s" % exc, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
