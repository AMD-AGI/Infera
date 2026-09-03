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

## The weak link, and what corroborates it

`env/steps.json` is written **by the same body whose ordering it attests to**.
Read on its own it is a file the producer could have written at the end from
memory, and "the arms did not overlap" is the one claim nothing else in the graph
can recover afterwards — the deployments are gone.

So the timestamps are cross-checked against a record **the producer did not
write**: AIPerf's own `start_time` and `end_time` in each round's
`profile_export_aiperf.json`. Two rules come out of it, and they answer different
questions:

**Disjointness, independently.** The stock arm's last AIPerf window must end
before the patched arm's first begins. This is the strong one and it is
offset-free: both timestamps come from the same clock read the same way, so a
node in any timezone gives the same answer.

**Containment, with a skew allowance.** Each `bench_r<N>` AIPerf window must lie
inside the step window `steps.json` claims for it. AIPerf writes naive local time
and the steps record writes UTC with an offset, so a node that is not on UTC
shows a constant skew — which is a timezone, not a fabrication, and is reported
rather than failed. **What is failed is the two arms disagreeing about that
offset**: one clock cannot be in two timezones twenty minutes apart, so a
per-arm difference is evidence that at least one arm's record was not written
while it ran.
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


def aiperf_windows(content: Path, reasons: list, arm: str) -> dict[str, tuple[dt.datetime, dt.datetime]]:
    """`{"bench_r1": (start, end)}` from each round's AIPerf export.

    **The corroborating record, and the reason it corroborates: AIPerf wrote it,
    not the body under audit.** A producer that assembled `steps.json` at the end
    from memory would have to have guessed these to the second.

    Naive timestamps, deliberately left naive. AIPerf writes local node time with
    no offset; converting here would mean assuming a timezone, and the two rules
    that use these either cancel the offset (AIPerf against AIPerf) or measure it
    (AIPerf against the steps record).
    """
    out: dict[str, tuple[dt.datetime, dt.datetime]] = {}
    for export in sorted(content.glob("items/result/r*/profile_export_aiperf.json")):
        tag = f"bench_{export.parent.name}"
        try:
            payload = json.loads(export.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            reasons.append(f"{arm}: {export.parent.name}/profile_export_aiperf.json is unreadable: {exc}")
            continue
        start, end = payload.get("start_time"), payload.get("end_time")
        try:
            out[tag] = (dt.datetime.fromisoformat(str(start)), dt.datetime.fromisoformat(str(end)))
        except (TypeError, ValueError):
            reasons.append(
                f"{arm}: {export.parent.name} records start_time={start!r} end_time={end!r}, "
                "which cannot be read — the step record has nothing to be checked against"
            )
    return out


def corroborate(arm: str, timeline: list[tuple[str, dt.datetime, dt.datetime]],
                windows: dict[str, tuple[dt.datetime, dt.datetime]],
                skew: float, reasons: list) -> float | None:
    """Each replay's own timestamps against the step the record claims for it.

    Returns the arm's measured clock offset in seconds (steps minus AIPerf), or
    `None` when there was nothing to compare. The caller compares the two arms'
    offsets, which is where a fabricated record actually shows.
    """
    by_name = {name: (start, end) for name, start, end in timeline}
    offsets: list[float] = []
    for tag, (a_start, a_end) in sorted(windows.items()):
        if tag not in by_name:
            reasons.append(
                f"{arm}: AIPerf recorded a replay for {tag} and steps.json has no such step. "
                "A round that ran and was not recorded is a round the other arm cannot be "
                "compared against."
            )
            continue
        s_start, s_end = by_name[tag]
        # The offset that would make AIPerf's window sit inside the step's. Taken
        # at the start rather than averaged: the two ends can differ legitimately
        # because the step includes the container plumbing around the replay.
        offset = (s_start.replace(tzinfo=None) - a_start).total_seconds()
        offsets.append(offset)
        span = (a_end - a_start).total_seconds()
        step_span = (s_end - s_start).total_seconds()
        if span > step_span + skew:
            reasons.append(
                f"{arm}: {tag} ran for {span:.0f}s by AIPerf's own record and steps.json "
                f"claims a {step_span:.0f}s step. The step cannot be shorter than the replay "
                "inside it."
            )
    if not offsets:
        return None
    spread = max(offsets) - min(offsets)
    if spread > skew:
        reasons.append(
            f"{arm}: the replay rounds disagree about the clock by {spread:.0f}s "
            f"(offsets {[round(o) for o in offsets]}). One arm has one clock."
        )
    return offsets[0]


def check_pair(records: dict[str, dict], contents: dict[str, Path],
               args: dict, reasons: list) -> bool:
    ok = True
    orders: dict[str, list[str]] = {}
    windows: dict[str, tuple[dt.datetime, dt.datetime]] = {}
    timelines: dict[str, list[tuple[str, dt.datetime, dt.datetime]]] = {}
    aiperf: dict[str, dict[str, tuple[dt.datetime, dt.datetime]]] = {}
    offsets: dict[str, float | None] = {}

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
        timelines[arm] = timeline
        print(f"  {arm}: {' -> '.join(orders[arm])}")
        print(f"  {arm}: {windows[arm][0].isoformat()} .. {windows[arm][1].isoformat()}")

        # The corroborating half. Collected per arm here; compared across arms
        # below, because that is where a record written from memory shows.
        skew = float(args.get("max_clock_skew_seconds", 90))
        arm_reasons = []
        aiperf[arm] = aiperf_windows(contents[arm], arm_reasons, arm)
        offsets[arm] = corroborate(arm, timeline, aiperf[arm], skew, arm_reasons)
        reasons.extend(arm_reasons)
        if arm_reasons:
            ok = False
        if not aiperf[arm] and args.get("require_corroboration", True):
            ok = False
            reasons.append(
                f"{arm}: no replay carries AIPerf's own start_time/end_time, so steps.json "
                "is uncorroborated. It is written by the same body whose ordering it "
                "attests to, and 'the arms did not overlap' is the one claim nothing else "
                "in the graph can recover once the deployments are gone."
            )

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

    # ---- the corroboration, across the arms ---------------------------------
    skew = float(args.get("max_clock_skew_seconds", 90))

    # **The strong rule, and it is offset-free.** Both timestamps come from the
    # same clock read the same way, so this answers "did the arms overlap?"
    # without trusting either steps.json or a timezone.
    if aiperf.get("stock") and aiperf.get("patched"):
        stock_end = max(e for _, e in aiperf["stock"].values())
        patched_start = min(s for s, _ in aiperf["patched"].values())
        if patched_start < stock_end:
            ok = False
            reasons.append(
                f"AIPerf's own records overlap: the patched arm's replay began "
                f"{patched_start.isoformat()} and the stock arm's ended {stock_end.isoformat()}. "
                "This is measured by the load generator rather than by the task under audit, "
                "so it stands whatever steps.json says."
            )
        else:
            print(
                f"  corroborated: AIPerf's own windows are disjoint too "
                f"({(patched_start - stock_end).total_seconds():.0f}s apart)"
            )

    a, b = offsets.get("stock"), offsets.get("patched")
    if a is not None and b is not None:
        if abs(a - b) > skew:
            ok = False
            reasons.append(
                f"the two arms disagree about the clock by {abs(a - b):.0f}s (stock {a:.0f}s, "
                f"patched {b:.0f}s between AIPerf's timestamps and steps.json's). One node has "
                "one clock, so at least one arm's step record was not written while it ran."
            )
        elif max(abs(a), abs(b)) > skew:
            # Consistent on both arms: a timezone, not a fabrication. Reported.
            print(
                f"  note: steps.json runs {a:.0f}s ahead of AIPerf's own timestamps on both "
                "arms. AIPerf writes naive local time and the step record writes UTC, so a "
                "node off UTC shows exactly this; it is consistent, so it is a clock and not "
                "a record written after the fact."
            )
    return ok


def main() -> int:
    args = zone.args()
    ids = zone.inputs()
    records: dict[str, dict] = {}
    contents: dict[str, Path] = {}
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
        contents[arm] = content
        by_arm_id[arm] = hid

    reasons: list = list(problems)
    if set(records) == set(ARMS):
        verdict = check_pair(records, contents, args, reasons) and not problems
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
