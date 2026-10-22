#!/usr/bin/env python3
"""Time-weighted decode context load of an AgentX/aiperf run.

From aiperf's per-request export (profile_export.jsonl[.gz]), each request is
in decode from first token (request_start + TTFT) to request_end; during that
interval its context is modeled as ISL + OSL * fraction-of-decode-elapsed
(tokens arrive at a constant rate). Over the window it reports:
  rows      time-average number of requests in decode (decode concurrency)
  sum_ctx   time-average sum of their contexts (what the decode indexer reads)
  ctx/row   sum_ctx / rows
plus the same per --bin seconds, and an instantaneous snapshot at --at.

Usage:
  agentx_context_load.yihou.py PATH [--phase profiling] [--bin 300]
      [--start EPOCH_S --end EPOCH_S] [--at EPOCH_S | --at "YYYY-mm-ddTHH:MM:SSZ"]
PATH is a profile_export.jsonl(.gz) or a directory searched for one (for
example agentx-c40-A2b/). Window default: span of the selected phase's
requests. Cancelled and failed requests (no first token) count only while
they actually decoded, if TTFT is present. Read-only; stdlib only.
"""
import argparse
import datetime as dt
import gzip
import json
import os
import sys


def find_export(path):
    if os.path.isfile(path):
        return path
    hits = []
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.startswith("profile_export") and (
                f.endswith(".jsonl") or f.endswith(".jsonl.gz")
            ):
                hits.append(os.path.join(root, f))
    if not hits:
        sys.exit(f"no profile_export*.jsonl[.gz] under {path}")
    if len(hits) > 1:
        print(f"[note] several exports, using the first of: {hits}", file=sys.stderr)
    return sorted(hits)[0]


def metric(rec, name):
    m = rec.get("metrics", {}).get(name)
    return None if m is None else m.get("value")


def load(path, phase):
    opener = gzip.open if path.endswith(".gz") else open
    reqs, skipped = [], 0
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            md = rec.get("metadata", {})
            if phase and md.get("benchmark_phase") != phase:
                continue
            start, end = md.get("request_start_ns"), md.get("request_end_ns")
            ttft = metric(rec, "time_to_first_token")
            isl = metric(rec, "input_sequence_length")
            osl = metric(rec, "output_sequence_length") or 0
            if None in (start, end, ttft, isl):
                skipped += 1
                continue
            a = start / 1e9 + ttft / 1e3
            b = end / 1e9
            if b <= a:
                skipped += 1
                continue
            reqs.append((a, b, float(isl), float(osl), start / 1e9))
    return reqs, skipped


def integrate(reqs, c0, c1):
    """(integral of rows dt, integral of sum context dt) over [c0, c1]."""
    rows_int = ctx_int = 0.0
    for a, b, isl, osl, _ in reqs:
        c, d = max(a, c0), min(b, c1)
        if d <= c:
            continue
        rows_int += d - c
        ctx_int += isl * (d - c) + osl / (b - a) * ((d - a) ** 2 - (c - a) ** 2) / 2
    return rows_int, ctx_int


def snapshot(reqs, t):
    ctx = [
        isl + osl * (t - a) / (b - a) for a, b, isl, osl, _ in reqs if a <= t < b
    ]
    return ctx


def parse_time(s):
    try:
        return float(s)
    except ValueError:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc
        ).timestamp()


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--phase", default="profiling", help="'' for all phases")
    ap.add_argument("--bin", type=float, default=300.0)
    ap.add_argument("--start", type=parse_time)
    ap.add_argument("--end", type=parse_time)
    ap.add_argument("--at", type=parse_time)
    args = ap.parse_args()

    path = find_export(args.path)
    reqs, skipped = load(path, args.phase)
    if not reqs:
        sys.exit(f"no usable requests in {path} (phase={args.phase!r})")
    t0 = args.start if args.start is not None else min(r[4] for r in reqs)
    t1 = args.end if args.end is not None else max(r[1] for r in reqs)
    T = t1 - t0
    rows_int, ctx_int = integrate(reqs, t0, t1)
    print(f"export={path} phase={args.phase!r} requests={len(reqs)} skipped={skipped}")
    print(f"window {iso(t0)} .. {iso(t1)} ({T:.0f} s)")
    rows, sum_ctx = rows_int / T, ctx_int / T
    print(
        f"ALL  rows {rows:6.2f}  sum_ctx {sum_ctx/1e6:7.3f} M  "
        f"ctx/row {sum_ctx/max(rows, 1e-9)/1e3:7.1f} K"
    )
    t = t0
    while t < t1:
        e = min(t + args.bin, t1)
        ri, ci = integrate(reqs, t, e)
        span = e - t
        print(
            f"  {iso(t)}  rows {ri/span:6.2f}  sum_ctx {ci/span/1e6:7.3f} M  "
            f"ctx/row {ci/max(ri, 1e-9)/1e3:7.1f} K"
        )
        t = e
    if args.at is not None:
        ctx = snapshot(reqs, args.at)
        if ctx:
            print(
                f"AT {iso(args.at)}  rows {len(ctx)}  sum_ctx {sum(ctx)/1e6:.3f} M  "
                f"max {max(ctx)/1e3:.1f} K  mean {sum(ctx)/len(ctx)/1e3:.1f} K"
            )
        else:
            print(f"AT {iso(args.at)}  no request in decode")


if __name__ == "__main__":
    main()
