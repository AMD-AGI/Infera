#!/usr/bin/env python3
"""Decode/prefill server-log stats over an AgentX profiling window.

Usage: decode_stats.yihou.py [--label L] SERVER_LOGS_DIR BENCHMARK_LOG [...more pairs]
       decode_stats.yihou.py --runs   (built-in table of the v0521dbg rounds)

Window = benchmark.log "Phase profiling (profiling) started" .. "Phase profiling (profiling) complete"
(HH:MM:SS, same clock as the server logs; the date is resolved from the server logs, midnight-safe).
Decode lines are logged every 40 decode steps per DP (decode_log_interval default), one line per DP.

Columns:
  lines        decode-batch lines in the window (all DPs)
  ms/step/DP   window_ms / (lines * 40 / nDP)  -- wall per decode step per DP (idle gaps NOT excluded)
  run          mean #running-req per line (= per DP)
  tok          mean #token (KV tokens in use per DP);  usage = mean token usage
  acc          mean accept len
  gen/DP       mean gen throughput (token/s) per line;  per-req = mean(gen/running) over running>0
  xfer>0%      % lines with #transfer-req > 0;  q>0% = % lines with #queue-req > 0
  imb          DP imbalance: per 10 s bin with all DPs present, max/mean of per-DP mean running-req; averaged
  pf_batches / pf_new / pf_cached   prefill batch lines in the window, sums of #new-token / #cached-token
"""
import datetime as dt
import glob
import os
import re
import statistics as st
import subprocess
import sys
from collections import defaultdict

STEPS_PER_LINE = 40
BIN_S = 10
R = "/shared_nfs/yihou_glm52/v0521dbg_20261010/runs"
RUNS = [  # label, server-logs dir, benchmark.log
    ("e1c40-r2", "e1c/server-logs", "e1c/agentx-c40-r2/benchmark.log"),
    ("e2b4-C44", "e2b4/server-logs", "e2b4/agentx-c44/benchmark.log"),
    ("e2c-C44", "e2c/server-logs", "e2c/agentx-c44/benchmark.log"),
    ("e2c-C40stack", "e2c/server-logs", "e2c/agentx-c40-e2cstack/benchmark.log"),
    ("e2c-C40flushed", "e2c/server-logs", "e2c/agentx-c40-e2cstack-flushed/benchmark.log"),
    ("e2d", "e2d/server-logs", "e2d/agentx-c40/benchmark.log"),
    ("e2e", "e2e/server-logs", "e2e/agentx-c40/benchmark.log"),
    ("e2f", "e2f/server-logs", "e2f/agentx-c40/benchmark.log"),
]

DEC = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) DP(\d+) TP\d+\] Decode batch, #running-req: (\d+), #token: (\d+), "
    r"token usage: ([\d.]+), accept len: ([\d.]+).*?#transfer-req: (\d+).*?cuda graph: (\w+), "
    r"gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
PRE = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) DP(\d+) TP\d+\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), "
    r"#cached-token: (\d+)")
TS = "%Y-%m-%d %H:%M:%S"


def window(bench_log):
    s = e = None
    for line in open(bench_log, errors="ignore"):
        if "Phase profiling (profiling) started" in line:
            s = line[:8]
        elif "Phase profiling (profiling) complete" in line:
            e = line[:8]
    return s, e


def strings(path):
    return subprocess.run(["strings", path], capture_output=True, text=True).stdout


def resolve(s, e, dates):
    """Pick the server-log date so the HH:MM:SS window lands inside the log span."""
    t0 = dt.datetime.strptime(s, "%H:%M:%S").time()
    t1 = dt.datetime.strptime(e, "%H:%M:%S").time()
    for d in sorted(dates):
        a = dt.datetime.combine(d, t0)
        b = dt.datetime.combine(d, t1)
        if b < a:
            b += dt.timedelta(days=1)
        yield a, b


def stats(label, sdir, blog):
    s, e = window(blog)
    if not (s and e):
        return f"{label}: no profiling window in {blog}"
    dec_txt = "".join(strings(p) for p in sorted(glob.glob(os.path.join(sdir, "decode-*.log"))))
    pre_txt = "".join(strings(p) for p in sorted(glob.glob(os.path.join(sdir, "prefill-*.log"))))
    dec_all = [m.groups() for m in DEC.finditer(dec_txt)]
    dates = {dt.datetime.strptime(g[0], TS).date() for g in dec_all}
    best = None
    for a, b in resolve(s, e, dates):
        rows = [g for g in dec_all if a <= dt.datetime.strptime(g[0], TS) <= b]
        if best is None or len(rows) > len(best[2]):
            best = (a, b, rows)
    if not best or not best[2]:
        return f"{label}: window {s}-{e} no decode rows"
    a, b, rows = best
    dps = sorted({int(g[1]) for g in rows})
    ndp = len(dps)
    win_ms = (b - a).total_seconds() * 1000
    run = [int(g[2]) for g in rows]
    tok = [int(g[3]) for g in rows]
    usage = [float(g[4]) for g in rows]
    acc = [float(g[5]) for g in rows]
    xfer = [int(g[6]) for g in rows]
    gen = [float(g[8]) for g in rows]
    q = [int(g[9]) for g in rows]
    per_req = [g / r for g, r in zip(gen, run) if r]
    bins = defaultdict(lambda: defaultdict(list))
    for g in rows:
        t = dt.datetime.strptime(g[0], TS)
        bins[int((t - a).total_seconds()) // BIN_S][int(g[1])].append(int(g[2]))
    imb = []
    for bd in bins.values():
        if len(bd) == ndp:
            m = [st.mean(v) for v in bd.values()]
            if st.mean(m) > 0:
                imb.append(max(m) / st.mean(m))
    per_dp_lines = {d: sum(1 for g in rows if int(g[1]) == d) for d in dps}
    pre = [m.groups() for m in PRE.finditer(pre_txt)
           if a <= dt.datetime.strptime(m.group(1), TS) <= b]
    return (f"{label:15s} {a:%m-%d %H:%M:%S}+{win_ms/1000:.0f}s lines={len(rows)} "
            f"(per DP {'/'.join(str(per_dp_lines[d]) for d in dps)}) "
            f"ms/step/DP={win_ms / (len(rows) * STEPS_PER_LINE / ndp):.1f} run={st.mean(run):.2f} "
            f"tok={st.mean(tok)/1e3:.0f}k usage={st.mean(usage):.3f} acc={st.mean(acc):.3f} "
            f"gen/DP={st.mean(gen):.1f} per-req={st.mean(per_req):.1f} "
            f"xfer>0%={100*sum(x > 0 for x in xfer)/len(xfer):.1f} q>0%={100*sum(x > 0 for x in q)/len(q):.1f} "
            f"imb={st.mean(imb) if imb else float('nan'):.3f}(n={len(imb)}) | "
            f"pf_batches={len(pre)} pf_new={sum(int(p[3]) for p in pre)/1e6:.1f}M "
            f"pf_cached={sum(int(p[4]) for p in pre)/1e6:.1f}M")


def main(argv):
    if argv == ["--runs"]:
        for label, sd, bl in RUNS:
            print(stats(label, f"{R}/{sd}", f"{R}/{bl}"), flush=True)
        return
    label = None
    args = list(argv)
    while args:
        if args[0] == "--label":
            label = args[1]
            args = args[2:]
            continue
        sd, bl = args[0], args[1]
        args = args[2:]
        print(stats(label or os.path.basename(os.path.dirname(os.path.abspath(bl))), sd, bl), flush=True)
        label = None


if __name__ == "__main__":
    main(sys.argv[1:] or ["--runs"])
