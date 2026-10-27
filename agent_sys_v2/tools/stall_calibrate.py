"""Calibrate 疑似卡住 (stall_minutes) on recorded runs: replay events.jsonl files through events.Follow and report the
quiet spells by what the run was doing, how the foreground tools with a timeout ended, and the reminders that
Follow.stall would have given at a few thresholds.

  python tools/stall_calibrate.py <events.jsonl or a directory to search> ... [--list paths.txt] [--minutes 5 10 15 20 30]
                                  [--json out]

Any events.jsonl of the same SDK frame format will do. Played back runs (frames marked "replay") and copies of one run
(the same first line) are counted once or not at all. A quiet spell is the time from the last event that shows the
foreground going on (Follow.moves) to the next one, put in a class by the state that event left:
  model        the session runs, nothing in the foreground: the model call
  fg:<tool>    a foreground tool in flight that has no timeout (a subagent, Read, an MCP tool)
  timed        a foreground tool with a timeout in flight (Bash, TaskOutput): it is judged by its timeout instead
  background   the session is idle and background commands run
  turn over    the turn has ended, the session still runs: background subagents keep it running
  idle, check  the round is over; the acceptance checks run
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_sys_v2.events import DROP, Follow  # noqa: E402


def files(args):
    for a in map(Path, args):
        if a.is_dir():
            yield from sorted(a.rglob("events.jsonl"))
        elif a.is_file():
            yield a


def frames(path):
    out = []
    with open(path, "rb") as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if (
                isinstance(e, dict)
                and e.get("subtype") not in DROP
                and isinstance(e.get("ts"), (int, float))
            ):
                out.append(e)
    return out


def klass(f):
    if f.checking:
        return "check"
    if f.state == "idle":
        return "background" if f.bg else "idle"
    if f.ended:
        return "turn over"
    if not f.fg:
        return "model"
    if any(f.timeout(x) is not None for x in f.fg.values()):
        return "timed"
    return "fg:" + str(next(iter(f.fg.values()))["tool"])


def replay(path, minutes):
    """One run: its quiet spells, its timed tools and the reminders per threshold."""
    f = Follow(path.parent)
    quiet, timed, alarms = defaultdict(list), [], {m: [] for m in minutes}
    starts = {}  # tool id -> (tool, ts, timeout) of the timed tools in flight
    on = {m: None for m in minutes}  # the reminder in force at threshold m
    spell = None  # the class of the quiet spell going on
    for e in frames(path):
        ts = e["ts"]
        moves = f.moves(e)
        if spell and moves:
            quiet[spell].append(ts - f.last_ts)
        for m in minutes:  # just before this event: the longest the spell got
            s = f.stall(ts - 1e-6, m * 60)
            if s and not on[m]:
                alarms[m].append({**s, "at": ts, "file": str(path)})
            on[m] = s
        f.feed(e)
        if moves:
            spell = klass(f)
        for k, x in f.fg.items():
            if k not in starts and (s := f.timeout(x)) is not None and x["ts"] is not None:
                starts[k] = (x["tool"], x["ts"], s, x["input"].get("timeout"))
        for k in [k for k in starts if k not in f.fg]:
            tool, t0, s, declared = starts.pop(k)
            timed.append({"tool": tool, "seconds": ts - t0, "timeout": s, "declared": declared})
    return quiet, timed, alarms


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--list", type=Path, help="a file with one path a line, besides the arguments")
    ap.add_argument("--minutes", type=float, nargs="+", default=[5, 10, 15, 20, 30])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    seen, runs, skipped = set(), 0, defaultdict(int)
    quiet, timed, alarms, hours = defaultdict(list), [], defaultdict(list), 0.0
    listed = args.list.read_text().split("\n") if args.list else []
    for path in files([*args.paths, *filter(None, listed)]):
        try:
            with open(path, "rb") as fh:
                head = [fh.readline() for _ in range(20)]
        except OSError:
            continue
        if any(b'"replay":' in line for line in head):
            skipped["played back"] += 1
            continue
        if not head[0] or head[0] in seen:
            skipped["copy or empty"] += 1
            continue
        seen.add(head[0])
        q, t, a = replay(path, args.minutes)
        fr = frames(path)
        hours += (fr[-1]["ts"] - fr[0]["ts"]) / 3600 if fr else 0
        runs += 1
        for k, v in q.items():
            quiet[k] += v
        timed += t
        for m, v in a.items():
            alarms[m] += v
    print(f"runs {runs} ({hours:.1f} h of events); left out: {dict(skipped)}")
    print("\nquiet spells (s), by what the run was doing")
    print(f"  {'class':24s} {'n':>7s} {'p50':>7s} {'p99':>7s} {'p99.9':>7s} {'max':>8s}")
    for k, v in sorted(quiet.items(), key=lambda kv: -len(kv[1])):
        print(
            f"  {k:24s} {len(v):7d} {pct(v, 0.5):7.1f} {pct(v, 0.99):7.1f} {pct(v, 0.999):7.1f} {max(v):8.1f}"
        )
    print("\nforeground tools with a timeout: ran (s), and past their timeout (s)")
    by = defaultdict(list)
    for x in timed:
        by[x["tool"]].append(x)
    for tool, v in sorted(by.items()):
        over = [x["seconds"] - x["timeout"] for x in v]
        reached = [o for o in over if o > -1]
        print(
            f"  {tool:12s} n={len(v)} longest={max(x['seconds'] for x in v):.1f} reached the timeout={len(reached)}"
            f" most past it={max(over):.1f} declared above 600 s={sum(1 for x in v if (x['declared'] or 0) > 600000)}"
        )
    print("\nreminders Follow.stall would have given")
    for m in args.minutes:
        v = alarms[m]
        kinds = defaultdict(int)
        for x in v:
            kinds[f"{x['kind']}:{x['tool'] or '-'}"] += 1
        print(f"  {m:g} min: {len(v)} {dict(kinds)}")
    if args.json:
        args.json.write_text(
            json.dumps({"quiet": quiet, "timed": timed, "alarms": alarms}, indent=1)
        )


if __name__ == "__main__":
    main()
