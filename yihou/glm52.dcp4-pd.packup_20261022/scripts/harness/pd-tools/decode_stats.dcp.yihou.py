#!/usr/bin/env python3
"""Decode/prefill server-log stats for PD runs, DP-attention or DCP decode.

Generalizes the 22k kit's tools/decode_stats.yihou.py: the log prefix may
carry any of DP/PP/ATTN_CP/MOE_DP/TP/EP ranks (a DCP decode without DP
attention logs "[ts TP0 EP0]", no DP), logs may be .gz, and the window can
come from an AgentX benchmark.log or explicit --start/--end.

Usage:
  decode_stats.dcp.yihou.py [--label L] [--interval 40]
      [--bench BENCHMARK_LOG | --start "YYYY-mm-dd HH:MM:SS" --end "..."]
      DECODE_LOG [DECODE_LOG ...] [--prefill PREFILL_LOG ...]

Only one TP rank per DP group is counted (the lowest TP seen for that DP),
so a TP4 decode is one stream of lines. "ms/step" = window / (lines *
interval / nDP), wall time per decode step per DP, idle gaps included.
Columns:
  lines, ms/step, run (mean #running-req per DP), tok (mean KV tokens in use),
  usage (mean token usage), acc (mean accept len), gen/DP (mean gen tok/s per
  DP line) and gen_total (sum over DP of the per-DP mean), per-req
  (mean gen/running), graph% (lines with cuda graph: True), xfer>0%, q>0%,
  prealloc>0%, retracted>0%, imb (per 10 s bin: max/mean of per-DP running),
  pf_batches/pf_new/pf_cached (prefill lines in the window).
"""
import argparse
import datetime as dt
import gzip
import re
import statistics as st
from collections import defaultdict

TS = "%Y-%m-%d %H:%M:%S"
HEAD = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)((?: [A-Z_]+\d+)*)\] ")
RANK = re.compile(r" ([A-Z_]+)(\d+)")
FIELD = {
    "run": re.compile(r"#running-req: (\d+)"),
    "tok": re.compile(r"#token: (\d+)"),
    "usage": re.compile(r"token usage: ([\d.]+)"),
    "acc": re.compile(r"accept len: ([\d.]+)"),
    "xfer": re.compile(r"#transfer-req: (\d+)"),
    "prealloc": re.compile(r"#prealloc-req: (\d+)"),
    "retract": re.compile(r"#retracted-req: (\d+)"),
    "graph": re.compile(r"cuda graph: (\w+)"),
    "gen": re.compile(r"gen throughput \(token/s\): ([\d.]+)"),
    "q": re.compile(r"#queue-req: (\d+)"),
}
PRE = re.compile(r"Prefill batch.*?#new-token: (\d+), #cached-token: (\d+)")
BIN_S = 10


def read_lines(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="ignore") as f:
        for line in f:
            yield line


def parse_head(line):
    m = HEAD.search(line)
    if not m:
        return None
    ranks = {k: int(v) for k, v in RANK.findall(m.group(2))}
    return dt.datetime.strptime(m.group(1), TS), ranks, line[m.end() :]


def bench_window(path):
    s = e = None
    for line in read_lines(path):
        if "Phase profiling (profiling) started" in line:
            s = line[:8]
        elif "Phase profiling (profiling) complete" in line:
            e = line[:8]
    return s, e


def decode_rows(paths):
    rows = []
    for p in paths:
        for line in read_lines(p):
            if "Decode batch" not in line:
                continue
            h = parse_head(line)
            if not h:
                continue
            t, ranks, rest = h
            row = {"t": t, "dp": ranks.get("DP", 0), "tp": ranks.get("TP", 0)}
            for k, rx in FIELD.items():
                m = rx.search(rest)
                row[k] = m.group(1) if m else None
            if row["run"] is None or row["gen"] is None:
                continue
            rows.append(row)
    # Keep one TP rank per DP: the lowest TP rank seen for that DP.
    lead = {}
    for r in rows:
        lead[r["dp"]] = min(lead.get(r["dp"], r["tp"]), r["tp"])
    return [r for r in rows if r["tp"] == lead[r["dp"]]]


def resolve_window(args, rows):
    if args.start and args.end:
        return dt.datetime.strptime(args.start, TS), dt.datetime.strptime(args.end, TS)
    if args.bench:
        s, e = bench_window(args.bench)
        if not (s and e):
            raise SystemExit(f"no profiling window in {args.bench}")
        best = None
        for d in sorted({r["t"].date() for r in rows}):
            a = dt.datetime.combine(d, dt.datetime.strptime(s, "%H:%M:%S").time())
            b = dt.datetime.combine(d, dt.datetime.strptime(e, "%H:%M:%S").time())
            if b < a:
                b += dt.timedelta(days=1)
            n = sum(a <= r["t"] <= b for r in rows)
            if best is None or n > best[2]:
                best = (a, b, n)
        return best[0], best[1]
    return min(r["t"] for r in rows), max(r["t"] for r in rows)


def pct(xs):
    xs = [x for x in xs if x is not None]
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decode_logs", nargs="+")
    ap.add_argument("--prefill", nargs="*", default=[])
    ap.add_argument("--bench")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--label", default="run")
    ap.add_argument("--interval", type=int, default=40, help="--decode-log-interval")
    args = ap.parse_args()

    rows = decode_rows(args.decode_logs)
    if not rows:
        raise SystemExit("no decode lines")
    a, b = resolve_window(args, rows)
    rows = [r for r in rows if a <= r["t"] <= b]
    if not rows:
        raise SystemExit(f"no decode lines in window {a}..{b}")
    dps = sorted({r["dp"] for r in rows})
    ndp = len(dps)
    win_ms = max((b - a).total_seconds() * 1000, 1.0)

    def col(k, cast):
        return [cast(r[k]) if r[k] is not None else None for r in rows]

    run, gen = col("run", int), col("gen", float)
    per_dp_gen = defaultdict(list)
    for r in rows:
        per_dp_gen[r["dp"]].append(float(r["gen"]))
    bins = defaultdict(lambda: defaultdict(list))
    for r in rows:
        bins[int((r["t"] - a).total_seconds()) // BIN_S][r["dp"]].append(int(r["run"]))
    imb = []
    for bd in bins.values():
        if len(bd) == ndp and ndp > 1:
            m = [st.mean(v) for v in bd.values()]
            if st.mean(m) > 0:
                imb.append(max(m) / st.mean(m))
    pf = []
    for p in args.prefill:
        for line in read_lines(p):
            if "Prefill batch" not in line:
                continue
            h = parse_head(line)
            m = PRE.search(line)
            if h and m and a <= h[0] <= b:
                pf.append((h[1].get("DP", 0), h[1].get("TP", 0), int(m.group(1)), int(m.group(2))))
    pf_lead = {}
    for dp_, tp_, _, _ in pf:
        pf_lead[dp_] = min(pf_lead.get(dp_, tp_), tp_)
    pf = [x for x in pf if x[1] == pf_lead[x[0]]]
    pf_n, pf_new, pf_cached = len(pf), sum(x[2] for x in pf), sum(x[3] for x in pf)
    per_req = [g / r for g, r in zip(gen, run) if r]
    lines_per_dp = "/".join(str(sum(r["dp"] == d for r in rows)) for d in dps)
    print(
        f"{args.label} {a:%m-%d %H:%M:%S}+{win_ms / 1000:.0f}s nDP={ndp} lines={len(rows)} "
        f"(per DP {lines_per_dp}) "
        f"ms/step={win_ms / (len(rows) * args.interval / ndp):.1f} "
        f"run={mean(run):.2f} tok={mean(col('tok', int)) / 1e3:.0f}k "
        f"usage={mean(col('usage', float)):.3f} acc={mean(col('acc', float)):.3f} "
        f"gen/DP={mean(gen):.1f} gen_total={sum(st.mean(v) for v in per_dp_gen.values()):.1f} "
        f"per-req={mean(per_req):.1f} "
        f"graph%={pct([g == 'True' if g else None for g in col('graph', str)]):.1f} "
        f"xfer>0%={pct([x > 0 if x is not None else None for x in col('xfer', int)]):.1f} "
        f"q>0%={pct([x > 0 if x is not None else None for x in col('q', int)]):.1f} "
        f"prealloc>0%={pct([x > 0 if x is not None else None for x in col('prealloc', int)]):.1f} "
        f"retracted>0%={pct([x > 0 if x is not None else None for x in col('retract', int)]):.1f} "
        f"imb={mean(imb):.3f}(n={len(imb)})"
        + (
            f" | pf_batches={pf_n} pf_new={pf_new / 1e6:.1f}M pf_cached={pf_cached / 1e6:.1f}M"
            if args.prefill
            else ""
        )
    )


if __name__ == "__main__":
    main()
