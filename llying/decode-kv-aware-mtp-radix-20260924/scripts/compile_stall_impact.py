#!/usr/bin/env python3
"""Estimate how much the prefill tilelang JIT compiles cost an AgentX run.

Each compile event shows up in the prefill log as a pair of C++ warnings
(`tmpXXXX.cpp:NNN: warning: equality comparison ...`). From the on-demand
reproduction (PREFILL-JIT-STALL.zh-CN.md) the whole prefill is stalled from
about 6 s before the first warning of an event to about 7 s after it.
For every profiling-phase request in flight before its first token during such
a window, the overlap is counted as TTFT added by the stall.

Usage: compile_stall_impact.py PROFILE_EXPORT_JSONL PREFILL_LOG
Prints one line per compile event and a summary. The AgentX client and the
prefill container run on the same host, so their clocks agree.
"""

import datetime as dt
import json
import re
import sys

BEFORE_S, AFTER_S, GROUP_S = 6.0, 7.0, 10.0
WARN = re.compile(r"^(\S+)Z? .*\.cpp:\d+:\d+: warning: equality comparison")


def ts(s: str) -> float:
    s = s.rstrip("Z")
    head, _, frac = s.partition(".")
    base = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return base.timestamp() + float("0." + (frac or "0")[:6])


def events(log: str) -> list:
    times = sorted(ts(m.group(1)) for line in open(log) if (m := WARN.match(line)))
    out = []
    for t in times:
        if out and t - out[-1][-1] <= GROUP_S:
            out[-1].append(t)
        else:
            out.append([t])
    return [(e[0] - BEFORE_S, e[0] + AFTER_S) for e in out]


def main() -> int:
    reqs = []
    for line in open(sys.argv[1]):
        d = json.loads(line)
        md, m = d.get("metadata", {}), d.get("metrics", {})
        if md.get("benchmark_phase") != "profiling" or "time_to_first_token" not in m:
            continue
        start = md["request_start_ns"] / 1e9
        reqs.append((start, start + m["time_to_first_token"]["value"] / 1000.0))
    if not reqs:
        print("no profiling records")
        return 1
    p0, p1 = min(r[0] for r in reqs), max(r[1] for r in reqs)
    total_ttft = sum(e - s for s, e in reqs)
    added, touched = 0.0, set()
    for w0, w1 in events(sys.argv[2]):
        where = "profiling" if p0 <= w1 and w0 <= p1 else "outside profiling"
        hit = [(i, min(e, w1) - max(s, w0)) for i, (s, e) in enumerate(reqs) if s < w1 and e > w0]
        if where == "profiling":
            added += sum(o for _, o in hit)
            touched.update(i for i, _ in hit)
        print(f"compile event {dt.datetime.fromtimestamp(w0 + BEFORE_S, dt.timezone.utc):%H:%M:%S} "
              f"({where}): {len(hit)} requests waiting, {sum(o for _, o in hit):.1f} request-seconds of TTFT")
    n = len(reqs)
    print(f"profiling requests: {n}; touched by a compile: {len(touched)} ({len(touched) / n:.2%}); "
          f"TTFT added: {added:.1f} s total = {added / n:.3f} s per request "
          f"= {added / total_ttft:.2%} of summed TTFT (mean TTFT {total_ttft / n:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
