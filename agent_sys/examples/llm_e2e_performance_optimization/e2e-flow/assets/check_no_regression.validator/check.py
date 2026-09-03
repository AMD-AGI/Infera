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

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import eval_stats  # noqa: E402
import schema as schema_lib  # noqa: E402
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
        if again["verdict"] == "REGRESSED":
            found.append(f"{row.get('round')} {metric} ({row.get('column')}) regressed")

    arms = report.get("arms") or {}
    sequences = {name: tuple(a.get("sequence") or ()) for name, a in arms.items()}
    if len(set(sequences.values())) > 1:
        found.append(f"the arms ran different sequences: {sequences}")

    found.extend(stock_reproduces_m2(report, args))
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
    for row in report.get("performance", []):
        if row.get("verdict") == "context":
            continue
        metric = str(row.get("metric") or "")
        again = eval_stats.perf_verdict(
            row.get("stock"), row.get("patched"),
            max_regression=throughput_bar if metric in higher_better else latency_bar,
            higher_is_better=metric in higher_better,
        )
        if again["verdict"] != "REGRESSED":
            continue
        groups = [
            [str(row.get("round") or "")],
            [g for g in (metric, str(row.get("label") or "")) if g],
            [f"({row.get('column')})"],
        ]
        if not _mentioned(groups, stated):
            missing.append(
                f"{row.get('round')} {metric} ({row.get('column')}) regressed and the report's "
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
    name = args.get("schema")
    if name:
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
        ok = _fail(reasons, f"the patch regressed: {'; '.join(recomputed[:5])}")
    return ok


def main() -> int:
    args = zone.args()
    results = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            results[hid] = check(content, args, reasons)
        print(f"check_no_regression: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    zone.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
