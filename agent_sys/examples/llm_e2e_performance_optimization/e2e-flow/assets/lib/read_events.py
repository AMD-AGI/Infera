#!/usr/bin/env python3
"""Read a run's event store.

Written 2026-09-04 by the checkpoint writer, at the leader's request, because
this query had answered three questions no log could and exactly one person
knew how to ask it.

**What the store is, since the shape is not guessable.** A run keeps its events
as *one JSON object per file* under `<run>/store/event/`, NOT as a `.jsonl`
stream — a `.jsonl` scan matches zero lines and reads as "no events", which is
the wrong answer rather than an error. Filenames are
`<task_id>%23<attempt>%23<event_id>.json` (`%23` is an escaped `#`), so they do
not sort chronologically: **sort by the `at` field, never by filename or mtime.**

Each object carries at least:

    at                  RFC3339 with microseconds, e.g. 2026-09-03T17:31:21.321310Z
    kind                phase_done | subgraph_done | output_absent | escalated |
                        handling_failed | monitor_gave_up
    task_id             the task this is about
    attributes          free-form; the useful keys are `phase`, `message`, `why`
    severity, fingerprint, handoff_id, exception_*

**Why it is worth reading.** The runner's stdout log says what a phase did; the
event store says what the graph *concluded*. Three findings came only from here:
an escalation that reached the top and found nobody to instruct; a
`build_workset` that was declared absent 13.8 s after clearing input validation
rather than being cut by a 20 s stall detector; and a retry that deadlocked on
its own half-open handoff version. None of the three appear in any log.

**Two traps measured the hard way.**

1. A run's *duration* is not a closure's duration. Sort per task and diff the
   phase boundaries; do not reason about a leaf from the run's elapsed time.
2. Empty `logs/`, `playground/` and `tmp/` in a task zone are **not** evidence
   the body did not run — they are empty for every task in every run examined,
   including ones that produced valid handoffs. Do not use them as a liveness
   signal.

Usage:

    python3 read_events.py <run_dir>              # whole timeline
    python3 read_events.py <run_dir> --kind escalated,output_absent
    python3 read_events.py <run_dir> --task 356505d8
    python3 read_events.py <run_dir> --phases     # per-task phase durations

`<run_dir>` is a directory under a run root. There are two run roots as of
2026-09-04 and a tally that reads only one will under-report:

    /shared_nfs/yihou/agent_sys/ws_handoff_refine/runroot/runs   (frozen, ro)
    /home/yihou/agent_sys_runroot/runs                           (live)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def load(run_dir: str) -> list[dict]:
    """Every event in the run, sorted by `at`. Unreadable files are skipped loudly."""
    out = []
    pattern = os.path.join(run_dir, "store", "event", "*.json")
    for path in glob.glob(pattern):
        try:
            out.append(json.load(open(path)))
        except (OSError, ValueError) as exc:
            print(f"read_events: skipping {os.path.basename(path)}: {exc}", file=sys.stderr)
    if not out:
        print(
            f"read_events: no events under {pattern}\n"
            "  If you expected some: the store is one JSON object per file, not .jsonl.",
            file=sys.stderr,
        )
    return sorted(out, key=lambda e: e.get("at", ""))


def _text(event: dict) -> str:
    attrs = event.get("attributes") or {}
    return str(attrs.get("message") or attrs.get("why") or attrs.get("detail") or "")


def timeline(events: list[dict], kinds: set[str] | None, task: str | None) -> None:
    for e in events:
        if kinds and e.get("kind") not in kinds:
            continue
        tid = e.get("task_id") or ""
        if task and not tid.startswith(task):
            continue
        print(f"{e.get('at','')}  {e.get('kind',''):16s} {tid[:8]}  {_text(e)[:100]}")


def phases(events: list[dict]) -> None:
    """Per-task phase boundaries — the trap-1 answer: which closure spent the time."""
    per = defaultdict(list)
    for e in events:
        if e.get("kind") != "phase_done":
            continue
        attrs = e.get("attributes") or {}
        per[e.get("task_id") or ""].append((e.get("at", ""), attrs.get("phase", "")))
    for tid, rows in sorted(per.items(), key=lambda kv: kv[1][0][0]):
        rows.sort()
        print(f"{tid[:8]}")
        prev = None
        for at, phase in rows:
            gap = ""
            if prev:
                gap = f"  (+{_seconds(prev, at):.1f}s)"
            print(f"    {at}  {phase}{gap}")
            prev = at


def _seconds(a: str, b: str) -> float:
    from datetime import datetime

    fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
    pa = datetime.strptime(a.replace("Z", "+0000"), fmt)
    pb = datetime.strptime(b.replace("Z", "+0000"), fmt)
    return (pb - pa).total_seconds()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--kind", default="", help="comma-separated kinds to keep")
    ap.add_argument("--task", default="", help="task id prefix")
    ap.add_argument("--phases", action="store_true", help="per-task phase durations")
    args = ap.parse_args()

    events = load(args.run_dir)
    if not events:
        return 1
    if args.phases:
        phases(events)
    else:
        timeline(events, set(k for k in args.kind.split(",") if k) or None, args.task or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
