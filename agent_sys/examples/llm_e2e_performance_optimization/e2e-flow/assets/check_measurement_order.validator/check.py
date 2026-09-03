#!/usr/bin/env python3
"""`check_measurement_order` — trustworthiness, strong.

**This validator is what an ordering guarantee became when it stopped being a
graph edge.**

`integration-demo` had eight leaves, and three of its edges carried an *argument*
rather than a datum. The load-bearing one was `serve_patched ← measure_stock`: it
did not exist because the patched bring-up needed the stock arm's numbers, it
existed because the patched bring-up's *first act is to tear the stock deployment
down*, and connecting it to `serve_stock` instead would let the scheduler run the
teardown while the stock arm was still being measured. That edge is gone. M5.2
says bring-up and use may not be split across agents, so the five leaves are one
task and the ordering is now a numbered list in one readme — **a weaker
guarantee, because a readme is followed by an agent and an edge is enforced by a
scheduler.**

So the guarantee moves here, from `froms` to evidence. The producing task
timestamps every step into each arm's `env/steps.json` as it goes, and this body
reads the two records back and asks the four questions that edge used to answer:

1. **Both arms ran the same steps in the same order.** "Round 1 is cold against
   this trace" is only true of an arm if the same things happened before it. Two
   arms with different sequences are two different experiments, and every
   comparison downstream is between them rather than between stock and patched.
2. **The arms do not overlap in time.** Overlapping arms shared a node, eight
   GPUs and a page cache. Measured on this cluster: the same stock deployment
   read 193.59 tok/s with an idle neighbour and 47 tok/s with a neighbour holding
   all eight cards — a 4× difference produced by nothing the pipeline changed.
3. **Steps do not overlap *within* an arm either.** A saturated trace replay
   running while lm_eval is scoring invalidates both numbers, and neither of them
   looks wrong afterwards.
4. **Every step exited zero.** A step that failed did not measure; a record that
   carries its failure is more honest than one that omits it, and both are
   equally unusable as a comparison.

**It needs both arms in one phase, and that is a feature.** The two kinds are
produced by one task (M5.2), so the output phase stages both together. If only
one arm reaches this body, the two arms were produced by two tasks — which is
precisely the split M5.2 forbids — and saying so is more useful than grading half
a comparison.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402

ARMS = ("stock", "patched")


def parse_time(value) -> dt.datetime | None:
    """ISO 8601 to an aware datetime, or `None`.

    Aware, always: a naive timestamp compared against an aware one raises, and a
    record written on a node in one timezone and read on a login node in another
    would otherwise be silently off by hours. A record with no offset is read as
    UTC, which is what every producer in this package writes.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.timezone.utc)


def load_steps(content: Path) -> tuple[dict | None, str]:
    path = content / "items" / "env" / "steps.json"
    if not path.is_file():
        return None, "items/env/steps.json is missing — the step order was never recorded"
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except ValueError as exc:
        return None, f"items/env/steps.json is not readable as JSON: {exc}"


def spans(record: dict, reasons: list) -> list[tuple[str, dt.datetime, dt.datetime]]:
    """`[(name, start, end)]` in the recorded order, or `[]` with reasons filed."""
    out: list[tuple[str, dt.datetime, dt.datetime]] = []
    for i, step in enumerate(record.get("steps") or []):
        name = step.get("step")
        if not name:
            reasons.append(f"steps[{i}] has no `step` name")
            continue
        started = parse_time(step.get("started"))
        if started is None:
            reasons.append(f"step {name!r} has no readable `started` timestamp ({step.get('started')!r})")
            continue
        seconds = step.get("seconds")
        try:
            duration = float(seconds)
        except (TypeError, ValueError):
            reasons.append(f"step {name!r} has no readable `seconds` ({seconds!r})")
            continue
        if duration < 0:
            reasons.append(f"step {name!r} records a negative duration ({duration})")
            continue
        # `rc` is a string in the sealed records ("0") and an int in some
        # producers. Both are the shell's exit status; neither is a reason to
        # care which type it arrived as.
        rc = step.get("rc")
        if rc is not None and str(rc) != "0":
            reasons.append(
                f"step {name!r} exited {rc}. A step that failed did not measure, so this arm's "
                "record is not a measurement whatever else it contains."
            )
        out.append((str(name), started, started + dt.timedelta(seconds=duration)))
    return out


def satisfies(expected: str, seen: list[str]) -> bool:
    """Did `expected` run?

    Exact match, or a round-suffixed form: `bench` is satisfied by `bench_r1`
    and `bench_r2`, because the number of replay rounds is a `--var` and the
    expectation is that the replay happened, not that it happened once. Nothing
    else is fuzzy — a step called `smoke_v2` does not satisfy `smoke`, because a
    renamed step is a changed method.
    """
    for name in seen:
        if name == expected:
            return True
        tail = name[len(expected):]
        if name.startswith(expected + "_r") and tail[2:].isdigit():
            return True
    return False


def check_pair(records: dict[str, dict], args: dict, reasons: list) -> bool:
    ok = True
    orders: dict[str, list[str]] = {}
    windows: dict[str, tuple[dt.datetime, dt.datetime]] = {}

    for arm in ARMS:
        record = records[arm]
        declared = record.get("arm")
        if declared != arm:
            ok = False
            reasons.append(
                f"the {arm} handoff's steps.json declares arm={declared!r}. The two arms are "
                "told apart by what they recorded, not by which slot they were written to."
            )
        arm_reasons: list = []
        timeline = spans(record, arm_reasons)
        for reason in arm_reasons:
            reasons.append(f"{arm}: {reason}")
        if arm_reasons:
            ok = False
        if not timeline:
            reasons.append(f"{arm}: no usable steps were recorded")
            return False
        orders[arm] = [name for name, _, _ in timeline]

        # 3. no overlap inside the arm.
        for (a_name, _, a_end), (b_name, b_start, _) in zip(timeline, timeline[1:]):
            if b_start < a_end:
                ok = False
                reasons.append(
                    f"{arm}: {b_name!r} started {(a_end - b_start).total_seconds():.0f}s before "
                    f"{a_name!r} finished. Two measurements sharing a deployment measure each "
                    "other; neither number is usable and neither looks wrong afterwards."
                )
        windows[arm] = (min(s for _, s, _ in timeline), max(e for _, _, e in timeline))
        print(f"  {arm}: {' -> '.join(orders[arm])}")
        print(f"  {arm}: {windows[arm][0].isoformat()} .. {windows[arm][1].isoformat()}")

    # 1. same steps, same order.
    if args.get("require_same_step_order", True) and orders["stock"] != orders["patched"]:
        ok = False
        reasons.append(
            f"the arms ran different sequences: stock {orders['stock']} vs patched "
            f"{orders['patched']}. 'Round 1 is cold against this trace' is only true if the "
            "same thing happened before it on both arms, so this invalidates every comparison "
            "downstream rather than only the steps that differ."
        )

    for expected in args.get("expected_steps") or ():
        for arm in ARMS:
            if not satisfies(str(expected), orders[arm]):
                ok = False
                reasons.append(
                    f"{arm}: the expected step {expected!r} does not appear in {orders[arm]}"
                )

    # 2. arms disjoint in time, and stock first.
    if args.get("require_arms_disjoint_in_time", True):
        stock_end, patched_start = windows["stock"][1], windows["patched"][0]
        if patched_start < stock_end:
            ok = False
            reasons.append(
                f"the patched arm started at {patched_start.isoformat()}, "
                f"{(stock_end - patched_start).total_seconds():.0f}s before the stock arm "
                f"finished at {stock_end.isoformat()}. Two deployments on one node share eight "
                "GPUs and a page cache: measured here, one stock deployment read 193.59 tok/s "
                "beside an idle neighbour and 47 tok/s beside a busy one, which is a 4x swing "
                "produced by nothing the pipeline changed."
            )
        else:
            gap = (patched_start - stock_end).total_seconds()
            print(f"  arms are disjoint; the patched arm began {gap:.0f}s after the stock arm ended")
            # Not a failure, and worth saying: a long gap is where the node's
            # load can change underneath the comparison. That is `todo.md` T7 —
            # the missing control is a comparability gate at bring-up, and this
            # is the number a reader needs to judge whether it mattered.
            if gap > float(args.get("warn_gap_seconds", 3600)):
                print(
                    f"  note: {gap / 60:.0f} minutes separate the two arms. The design controls "
                    "for session, node, trace, order and image and NOT for node load at "
                    "measurement time; the wider that gap, the less that control is worth."
                )
    return ok


def main() -> int:
    args = zone.args()
    ids = zone.inputs()
    records: dict[str, dict] = {}
    by_arm_id: dict[str, str] = {}
    problems: list[str] = []

    for hid in ids:
        content = zone.content_of(hid)
        if content is None:
            problems.append(f"{hid}: no staged content")
            continue
        record, why = load_steps(content)
        if record is None:
            problems.append(f"{hid}: {why}")
            continue
        arm = str(record.get("arm") or "")
        if arm not in ARMS:
            problems.append(f"{hid}: steps.json declares arm={arm!r}, which is not one of {list(ARMS)}")
            continue
        if arm in records:
            problems.append(f"{hid}: a second {arm!r} arm was staged; there is exactly one of each")
            continue
        records[arm] = record
        by_arm_id[arm] = hid

    reasons: list = list(problems)
    if set(records) == set(ARMS):
        verdict = check_pair(records, args, reasons) and not problems
    else:
        verdict = False
        reasons.append(
            f"this phase staged {sorted(records) or 'no'} arm(s) and both are needed. The two "
            "kinds are produced by ONE task (M5.2), so they arrive together; only one reaching "
            "here means the arms were produced by two tasks, which is the split that made this "
            "validator necessary in the first place."
        )

    results = {hid: verdict for hid in ids}
    for hid in ids:
        print(f"check_measurement_order: {hid} {'PASS' if verdict else 'FAIL'}")
    for reason in reasons:
        print(f"  - {reason}")
    zone.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
