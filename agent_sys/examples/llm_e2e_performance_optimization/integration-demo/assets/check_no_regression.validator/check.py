#!/usr/bin/env python3
"""`check_no_regression` — usability, strong.

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

`usability` rather than `completeness`: what it judges is whether the report can
be used to make the decision it was built for.

A disagreement fails the handoff even when the recomputation says "accepted".
Two answers that differ mean one of them is wrong, and until that is resolved
neither can be relied on.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import eval_stats  # noqa: E402
import store  # noqa: E402


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


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

    return found


def check(content: Path, args: dict, reasons: list) -> bool:
    text_json = content / "items" / "text.json"
    if not text_json.is_file():
        return _fail(reasons, "items/text.json is missing")
    try:
        report = json.loads(text_json.read_text(encoding="utf-8"))
    except ValueError as exc:
        return _fail(reasons, f"text.json is not readable as JSON: {exc}")

    ok = True
    for name in args.get("require_arms") or ():
        if name not in (report.get("arms") or {}):
            ok = _fail(reasons, f"the report carries no {name!r} arm")

    if not (content / "items" / "report.md").is_file():
        ok = _fail(reasons, "items/report.md is missing — there is no readable form of the verdict")

    if not (report.get("bars") or {}):
        ok = _fail(
            reasons,
            "the report records no bars. A verdict without the threshold it was decided "
            "against cannot be re-read later.",
        )

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
            missing = [r for r in recomputed if not any(r[:40] in s for s in stated.get("reasons", []))]
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
    args = store.args()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            results[hid] = check(content, args, reasons)
        print(f"check_no_regression: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
