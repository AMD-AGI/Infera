#!/usr/bin/env python3
"""Has anything escalated to the user in this run — what, when, and why.

    python3 assets/lib/runprobe.py                     # newest run under the default root
    python3 assets/lib/runprobe.py <run dir>           # one run
    python3 assets/lib/runprobe.py --root <runs dir>   # newest under another root

Read-only, and safe on a run that is still going. Answers three questions and
stops.

## Why this exists, and why the run log cannot answer it

`cli/main.py:1015` ends a run when `(not holding or blocked)` and nothing has
changed for 20 s. **`blocked` is the whole question**, because a non-empty
`blocked` satisfies that condition *regardless of `holding`* — so a leaf that is
genuinely executing gets torn down. And `blocked` is
`[t for t in live if _awaiting_a_decision(t, registry)]`, which reads an
escalation **record**.

**The log never prints the transition.** It prints the *consequence*, once, in
the `done` line at the cut — by which point the run is over. So "is this run
exposed?" was, until this file, a question you could only answer after losing
the run, or with a grep composed on the spot.

That cost a full day. On 2026-09-04 the question *"will a KernelForge campaign
at rung 4 be cut 20 s in?"* was answered **twice, confidently, in opposite
directions**, both times by reading run logs — first "no, only a failed run is
cut", then "yes, the escalation is structural". Both wrong. What settled it was
one query against the store: **no escalation record at all, 26 minutes into a
live rung 1.** The log could not have told anyone that; the store answers it in
under a second.

## The three questions, and what each one is for

1. **Are there escalations that reached the user, and how many.** This is
   exactly `blocked`'s input: `monitor/base.py:145 reached_the_user` is
   `kind == ESCALATED and attributes["target"] == "user"`. None of these, and
   the stall guard reduces to `not holding` — a working leaf is safe.
2. **What triggered each one.** `_escalate` (`base.py:783-789`) writes an
   `ESCALATED` record carrying `why` at **every hop** of the walk, and only at
   the root does `_to_user` (`:799-808`) add `target: user`. So the *first*
   record of a chain names the task that started it and the reason — the fact
   that makes a cut attributable instead of mysterious.
3. **When, against which task and phase.** Whether the escalation predates the
   long quiet stretch or arrives during it is the difference between "this run
   was always exposed" and "something happened".

## What it deliberately does not do

It does not decide whether the run *should* end — that is `monitor` spec §11 and
still open — and it does not touch `agent_sys/cli/`. It reports the input to a
decision somebody else makes.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from datetime import datetime

#: `monitor/protocols.py:93` and `monitor/base.py:141`. Named here rather than
#: matched loosely, so a rename upstream makes this file wrong out loud instead
#: of quietly reporting "no escalations" — which is the reassuring answer and
#: therefore the dangerous one to get by accident.
ESCALATED = "escalated"
TARGET = "target"
TARGET_USER = "user"

#: `task_graph/models.py:105`, plus the two the CLI treats as done-for-now. A
#: task in any of these is not `live`, so its escalation no longer contributes
#: to `blocked`.
NOT_LIVE = {"succeeded", "cancelled", "failed"}

DEFAULT_ROOT = "/home/yihou/agent_sys_runroot/runs"


def _load(path: pathlib.Path):
    """A record, or `None` if it is not readable *yet*.

    **Mid-run is the case that matters**, and a store being written under you
    means a file can be caught half-flushed. That is not corruption and it is
    not an error to abort on; it is one record this pass could not see. Counted
    and reported, because a probe that silently skipped them could report "no
    escalations" while looking away from the one that mattered.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _when(record: dict) -> str:
    return str(record.get("at") or "")


def _elapsed(a: str, b: str) -> str:
    try:
        fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
        t0 = datetime.strptime(a.replace("Z", "+0000"), fmt)
        t1 = datetime.strptime(b.replace("Z", "+0000"), fmt)
        return f"+{(t1 - t0).total_seconds():.0f}s"
    except (ValueError, TypeError):
        return "?"


def probe(run: pathlib.Path) -> int:
    store = run / "store"
    if not (store / "event").is_dir():
        print(f"runprobe: no store/event under {run}", file=sys.stderr)
        return 2

    tasks, unreadable_tasks = {}, 0
    for path in (store / "task").glob("*.json"):
        record = _load(path)
        if record is None:
            unreadable_tasks += 1
            continue
        tasks[record.get("id")] = record

    events, unreadable = [], 0
    for path in (store / "event").glob("*.json"):
        record = _load(path)
        if record is None:
            unreadable += 1
            continue
        events.append(record)
    events.sort(key=_when)

    if not events:
        print(f"runprobe: {run.name}: no readable events yet")
        return 0
    started = _when(events[0])

    def name(task_id) -> str:
        task = tasks.get(task_id) or {}
        return f"{task.get('closure') or '?'}[{task.get('status') or '?'}]"

    escalations = [e for e in events if e.get("kind") == ESCALATED]
    to_user = [e for e in escalations if (e.get("attributes") or {}).get(TARGET) == TARGET_USER]

    print(f"runprobe: {run.name}   started {started}   {len(events)} events, {len(tasks)} tasks")
    if unreadable or unreadable_tasks:
        print(f"  note: {unreadable} event and {unreadable_tasks} task file(s) not readable this "
              f"pass — expected mid-run, re-run to see them")

    # **Is anything still happening?** The store cannot answer this and must not
    # be read as if it could: a run that dies abruptly leaves its last state
    # written as `running`, and **nothing will ever overwrite it**. So every
    # status below is a *last known* state, and without a liveness signal the
    # tool reports a corpse in the present tense.
    #
    # Measured the hard way, 2026-09-04: I read this tool's own output — "latest
    # event +1s into the run, 2357s ago" — and reported "39 minutes in,
    # deploy_and_prove still running". The run had been dead for 39 minutes and
    # 2357 was the evidence. I had written that wall-clock line specifically so
    # a *live* run would not be misread as stale, and then misread a *dead* run
    # as live off the same number.
    #
    # The newest mtime anywhere in the run tree is the signal the store lacks —
    # a live run is writing zone logs and handoff content constantly, and this
    # is what the leader used to establish the death. Cheap, and it is part of
    # answering question 3 honestly rather than a fourth question.
    newest = 0.0
    for path in run.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    write_age = time.time() - newest if newest else -1.0

    # **mtime alone cannot tell a quiet phase from a corpse, and saying it can is
    # the same error in the other direction.** A 60 s threshold called this very
    # run "NOT WRITING" while it was healthy — m1's agent polls on a `sleep 115`
    # cycle, so it writes nothing for two minutes at a time by design. A tool
    # that cries dead on a live run gets ignored, and then it is useless on the
    # day the run really is dead.
    #
    # **Two signals, neither of them a verdict**, and the restraint is the point.
    #
    # A first cut printed `ALIVE`/`NOT WRITING` and was wrong both ways within a
    # minute: it called a healthy run dead (m1's agent polls on `sleep 115`, so
    # it writes nothing for two minutes by design), and it called a two-hour-old
    # corpse alive by matching **this probe's own command line**. A liveness
    # verdict that flips on how you invoked the tool is worse than no verdict.
    #
    # ## Look for children by cwd, not for a parent by name
    #
    # m1's rule, reached independently, and it is the only check that has not
    # been wrong: **a cwd is a property of the process; a command line is a
    # property of how it was invoked.** An argument can spoof the second and
    # cannot spoof the first.
    #
    # The cmdline scan this replaces produced a false ALIVE on a run that had
    # been dead ninety seconds — a transient shell carrying the path matched,
    # including ones this probe spawns. The cwd walk also *finds* what a name
    # match misses: `grep agent_sys.cli.main` cannot see the `claude` binaries a
    # run spawns, which is why two owners reported "no agent alive" in good
    # faith and both were wrong about an orphan that was still running.
    holder = None
    me = {str(os.getpid()), str(os.getppid())}
    for entry in pathlib.Path("/proc").glob("[0-9]*"):
        if entry.name in me:
            continue
        try:
            cwd = os.readlink(entry / "cwd")
        except OSError:
            continue  # not ours to read, or gone between glob and readlink
        if cwd == str(run) or cwd.startswith(str(run) + "/"):
            holder = f"{entry.name} (cwd {cwd[len(str(run)):] or '/'})"
            break

    age = "never written" if write_age < 0 else f"{write_age:.0f}s ago"
    seen = f"process {holder} is working inside this run tree" if holder \
        else "no process has its cwd in this run tree (it may be driven from another host)"
    print(f"  observed: last write {age}; {seen}")
    print("  Neither is proof. runprobe cannot tell a long quiet phase from a stopped run,")
    print("  and **a recent write does not mean alive** — a terminating run flushes on its")
    print("  way out, so the last thing a dying run does looks exactly like a live one.")
    print("  Every status below is LAST KNOWN: a killed run leaves its tasks reading")
    print("  `running` for ever and nothing overwrites them.")

    # ---- 1. is `blocked` non-empty ------------------------------------------
    print(f"\n1. escalations reaching the user: {len(to_user)}   "
          f"(all escalation hops: {len(escalations)})")
    if not to_user:
        print("   -> `blocked` was EMPTY as of the last record. The stall guard reduces to")
        print("      `not holding`, so a leaf holding a thread is not cut.")
        if not holder:
            print("      NOTE: no process was found for this run — so this says the run was not")
            print("      exposed up to its last record, NOT that it is still running unexposed.")
    else:
        live = [e for e in to_user
                if (tasks.get(e.get("task_id")) or {}).get("status") not in NOT_LIVE]
        # Present tense only when the tree says something is still being written.
        # "on a still-live task" about a run that died two hours ago is the same
        # defect as the status list below, one line up.
        on = "on a task still non-terminal" if holder else "on a task not terminal when last written"
        print(f"   -> `blocked` was NON-EMPTY ({len(live)} {on}).")
        print("      Any 20 s without a task-table change ends the run, even with a leaf working.")

    # ---- 2. what triggered each chain ---------------------------------------
    print("\n2. what triggered them")
    if not escalations:
        print("   nothing has escalated.")
    for entry in escalations:
        attrs = entry.get("attributes") or {}
        mark = "-> USER" if attrs.get(TARGET) == TARGET_USER else "  hop  "
        print(f"   {mark}  {_when(entry)}  {_elapsed(started, _when(entry))}  "
              f"{name(entry.get('task_id'))}")
        print(f"            why: {attrs.get('why') or '(none recorded)'}")

    # ---- 3. when, against the task table ------------------------------------
    print("\n3. when, against what the graph was doing")
    if to_user:
        first = _when(to_user[0])
        print(f"   first reached the user at {first} ({_elapsed(started, first)} into the run)")
        before = [e for e in events if _when(e) < first]
        print(f"   {len(before)} events preceded it; the run was exposed from that point on")
    else:
        # **Say how long ago the last event was, not only when it was.** The
        # store records *transitions*, not progress, so a run 40 minutes into
        # one phase shows a latest event at +1 s — which reads as a stale probe
        # unless the gap is spelled out. It is the opposite: a long gap with no
        # escalation is the reassuring case, and the reader has to be able to
        # tell it from "this tool only looked at the first second".
        latest = _when(events[-1])
        now = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        print(f"   latest event {latest} ({_elapsed(started, latest)} into the run), "
              f"{_elapsed(latest, now).lstrip('+')} ago")
        print("   nothing has escalated in that whole window. The store records transitions,")
        print("   not progress, so a long gap here means one phase has been running throughout.")
    running = sorted(name(t) for t, r in tasks.items() if r.get("status") not in NOT_LIVE)
    label = "live tasks now" if holder else "tasks NOT in a terminal state when the store was last written"
    print(f"   {label}: {', '.join(running) if running else '(none)'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="a run directory; default is the newest under --root")
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"runs directory (default {DEFAULT_ROOT})")
    a = ap.parse_args()

    if a.run:
        run = pathlib.Path(a.run)
    else:
        root = pathlib.Path(a.root)
        runs = sorted((p for p in root.glob("*") if p.is_dir()), key=lambda p: p.name)
        if not runs:
            print(f"runprobe: no runs under {root}", file=sys.stderr)
            return 2
        run = runs[-1]
    if not run.is_dir():
        print(f"runprobe: {run} is not a directory", file=sys.stderr)
        return 2
    return probe(run)


if __name__ == "__main__":
    raise SystemExit(main())
