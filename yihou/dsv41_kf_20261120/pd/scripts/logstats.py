#!/usr/bin/env python3
"""Summarise vLLM Rust-frontend log_stats lines of one engine log over a time window.

  logstats.py LOG [--start 'MM-DD HH:MM:SS'] [--end 'MM-DD HH:MM:SS'] [--label NAME]

Prints p10 / p50 / p90 / mean of prompt and generation throughput, running / waiting / deferred requests, KV fetches
in progress, GPU KV use and prefix-cache hit rate, plus the mean acceptance length. Lines without a field are skipped
for that field. One line per log stats interval (the engine's VLLM_LOG_STATS_INTERVAL).
"""
import argparse
import re
import statistics

TS = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d)\S*\s.*?\[log_stats\.rs:\d+\]")
FIELDS = {
    "prompt_tps": r"Avg prompt tput: ([\d.]+)",
    "gen_tps": r"Avg generation tput: ([\d.]+)",
    "running": r"Reqs Running: (\d+)",
    "waiting": r"Waiting: (\d+)",
    "deferred": r"Deferred: (\d+)",
    "kv_fetch": r"KV fetch: \d+ waiting to start, (\d+) in progress",
    "kv_used_pct": r"GPU KV cache used: ([\d.]+)%",
    "prefix_hit_pct": r"Prefix cache hit rate: ([\d.]+)%",
    "ext_hit_pct": r"External prefix cache hit rate: ([\d.]+)%",
}
AL = re.compile(r"Mean acceptance length: ([\d.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    vals = {k: [] for k in FIELDS}
    al = []
    n = 0
    with open(a.log, errors="replace") as f:
        for line in f:
            if "log_stats" not in line:
                continue
            line = ANSI.sub("", line)
            m = TS.search(line)
            if not m:
                continue
            t = m.group(1)
            if (a.start and t < a.start) or (a.end and t > a.end):
                continue
            m2 = AL.search(line)
            if m2:
                al.append(float(m2.group(1)))
                continue
            if "Avg prompt tput" not in line:
                continue
            n += 1
            for k, rx in FIELDS.items():
                m3 = re.search(rx, line)
                if m3:
                    vals[k].append(float(m3.group(1)))
    print(f"{a.label or a.log}: {n} stats lines" + (f" [{a.start} .. {a.end}]" if a.start or a.end else ""))
    for k, xs in vals.items():
        if xs:
            print(f"  {k:15s} p10 {q(xs, .1):12.1f}  p50 {q(xs, .5):12.1f}  p90 {q(xs, .9):12.1f}  mean {statistics.fmean(xs):12.1f}")
    if al:
        print(f"  {'accept_len':15s} p50 {q(al, .5):.2f} mean {statistics.fmean(al):.2f} (n={len(al)})")


if __name__ == "__main__":
    main()
