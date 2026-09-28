#!/usr/bin/env python3
"""KV-aware routing hits per role during the AgentX profiling window.

Usage: router_hits.py EVIDENCE_PERF_DIR ARM [ARM ...] [--conc N]
Reads <arm>-router.log (kv-aware `pick` lines) and takes the profiling window
from the first/last `profiling]` line of <arm>-agentx-c<N>.log (client clock,
same host as the router).
"""
import re
import sys


def window(log):
    ts = [l[:12] for l in open(log, errors="replace") if "profiling]" in l]
    return ts[0], ts[-1]


def main():
    args = sys.argv[1:]
    conc = "40"
    if "--conc" in args:
        i = args.index("--conc")
        conc = args[i + 1]
        del args[i:i + 2]
    base, arms = args[0], args[1:]
    for arm in arms:
        a, b = window(f"{base}/{arm}-agentx-c{conc}.log")
        rows = []
        for l in open(f"{base}/{arm}-router.log", errors="replace"):
            if 'policy="kv-aware"' not in l or " pick" not in l or not a <= l[11:23] <= b:
                continue
            rows.append((re.search(r"role=(\w+)", l).group(1), int(re.search(r"cache_hits=(\d+)", l).group(1)),
                         int(re.search(r"request_blocks=(\d+)", l).group(1))))
        for role in ("Prefill", "Decode"):
            r = [x for x in rows if x[0] == role]
            hit_reqs = sum(1 for x in r if x[1] > 0)
            hits, blocks = sum(x[1] for x in r), sum(x[2] for x in r)
            print(f"{arm} {role:7s} picks={len(r)} requests_with_hit={hit_reqs / max(len(r), 1):.1%} "
                  f"hit_blocks/request_blocks={hits / max(blocks, 1):.3f}")


if __name__ == "__main__":
    main()
