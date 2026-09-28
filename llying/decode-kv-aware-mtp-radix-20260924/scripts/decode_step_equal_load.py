#!/usr/bin/env python3
"""Decode step time per DP rank at equal load, from `Decode batch` log lines.

Each rank logs every decode_log_interval steps; ranks log at different times and
an idle rank's counter does not advance, so consecutive-line gaps are not usable.
Instead: step_ms ~= running * accept_len / gen_throughput * 1000 for each line,
binned by the rank's running requests and its context tokens.

Usage: decode_step_equal_load.py START END DECODE_LOG [START END DECODE_LOG ...]
  START/END: HH:MM:SS bounds of the AgentX profiling phase for that log.
"""
import re
import statistics as st
import sys

PAT = re.compile(
    r"^\S+T(\d\d:\d\d:\d\d)\S* \[[^\]]*DP(\d+) TP\d+\] Decode batch, #running-req: (\d+), #token: (\d+), "
    r"token usage: [0-9.]+, accept len: ([0-9.]+).*?gen throughput \(token/s\): ([0-9.]+)")


def load(log, a, b):
    out = []
    for line in open(log, errors="replace"):
        m = PAT.match(line)
        if not m or not (a <= m.group(1) <= b):
            continue
        run, tok, acc, gen = int(m.group(3)), int(m.group(4)), float(m.group(5)), float(m.group(6))
        if run and gen > 0:
            out.append((run, tok, 1000 * run * acc / gen))
    return out


def main():
    args = sys.argv[1:]
    arms = []
    for i in range(0, len(args), 3):
        name = args[i + 2].split("/")[-3].split("-launch")[0]
        arms.append((name, load(args[i + 2], args[i], args[i + 1])))
    for n, d in arms:
        print(f"{n}: lines={len(d)} mean running/rank={st.mean(x[0] for x in d):.2f} "
              f"mean ctx/rank={st.mean(x[1] for x in d) / 1e6:.2f}M")
    run_bins = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 12)]
    ctx_bins = [(0, 0.5e6), (0.5e6, 1e6), (1e6, 1.5e6), (1.5e6, 9e9)]
    print(f"{'running':>8s} {'ctx/rank':>10s} | " + " | ".join(f"{n:>8s} n  p50ms" for n, _ in arms))
    for rlo, rhi in run_bins:
        for clo, chi in ctx_bins:
            cells = []
            for _, d in arms:
                v = sorted(x[2] for x in d if rlo <= x[0] <= rhi and clo <= x[1] < chi)
                cells.append((len(v), v[len(v) // 2] if v else float("nan")))
            if min(c[0] for c in cells) < 30:
                continue
            print(f"{rlo:3d}-{rhi:<4d} {clo / 1e6:4.1f}-{min(chi, 9e6) / 1e6:<4.1f}M | " +
                  " | ".join(f"{c[0]:8d} {c[1]:6.1f}" for c in cells))


if __name__ == "__main__":
    main()
