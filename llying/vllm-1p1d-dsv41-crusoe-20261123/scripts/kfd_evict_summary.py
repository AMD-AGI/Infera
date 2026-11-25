#!/usr/bin/env python3
"""KFD queue evictions and compaction in a run, from its host-telemetry.txt (scripts/host_telemetry.sh).

Every telemetry line carries each GPU process's cumulative KFD evicted_ms and the kernel's compaction / migration / THP
counters. This prints, for the whole run and for an optional window (e.g. AgentX's scored window), the evictions of at
least MIN_MS per process (count, total, longest) and the counter deltas. EXPERIMENT 4.11: the acceptance bar for
throughput comparisons is engine evictions under 1% of the scored window and none longer than 5 s.
usage: kfd_evict_summary.py LOG_DIR [START_HHMMSS END_HHMMSS] [MIN_MS=500]
"""
import re
import sys

COUNTERS = ("compact_stall", "thp_fault_alloc", "pgmigrate_success")


def secs(t):
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s


def load(path):
    rows = []
    for line in open(path, errors="replace"):
        if not line.startswith("== ") or "evicted_ms=" not in line:
            continue
        ev = dict((p, int(v)) for p, v in re.findall(r"(\d+):(\d+)", re.search(r"evicted_ms=(\S*)", line).group(1)))
        ctr = {k: int(m.group(1)) for k in COUNTERS if (m := re.search(k + r" (\d+)", line))}
        rows.append((line.split()[1], ev, ctr))
    return rows


def report(rows, label, min_ms):
    if len(rows) < 2:
        print(f"{label}: no telemetry")
        return
    per = {}
    for (_, a, _), (t, b, _) in zip(rows, rows[1:]):
        for p, v in b.items():
            d = v - a.get(p, v)
            if d >= min_ms:
                per.setdefault(p, []).append((t, d / 1000))
    span = secs(rows[-1][0]) - secs(rows[0][0])
    deltas = ", ".join(f"{k} {rows[-1][2].get(k, 0) - rows[0][2].get(k, 0)}" for k in COUNTERS)
    total = sum(x for v in per.values() for _, x in v)
    print(f"{label} ({rows[0][0]}-{rows[-1][0]}, {span} s): {sum(len(v) for v in per.values())} evictions >= "
          f"{min_ms} ms, {total:.1f} s in all; {deltas}")
    for p, v in sorted(per.items(), key=lambda kv: -sum(x for _, x in kv[1])):
        tot = sum(x for _, x in v)
        print(f"  pid {p}: {len(v)} evictions, {tot:.1f} s ({100 * tot / max(span, 1):.2f}% of the window), "
              f"longest {max(x for _, x in v):.1f} s, first {v[0][0]}")


def main():
    log_dir = sys.argv[1]
    window = sys.argv[2:4] if len(sys.argv) >= 4 else None
    min_ms = int(sys.argv[4]) if len(sys.argv) >= 5 else 500
    rows = load(f"{log_dir}/host-telemetry.txt")
    report(rows, "whole run", min_ms)
    if window:
        lo, hi = (secs(w if ":" in w else ":".join((w[0:2], w[2:4], w[4:6]))) for w in window)
        report([r for r in rows if lo <= secs(r[0]) <= hi], "window", min_ms)


if __name__ == "__main__":
    main()
