"""Bottleneck numbers of one 2P1D run from its engine logs, over the AgentX measurement window.
  python3 analyze_pd_logs.py RUN_DIR WINDOW_START WINDOW_END      (times as 2026-10-20T17:17)
Prefill (per instance): median time of a full chunk (>= 8000 tokens) taken back to back (queue > 0),
uncached tokens/s, mean waiting requests, share of 10 s bins with a queue.
Decode: requests running in total and on the busiest of the 8 DP ranks (10 s bins), step time
by the busiest rank's running count, mean requests waiting for preallocation / KV transfer."""

import collections
import datetime
import re
import statistics as st
import sys
from pathlib import Path

run, t0, t1 = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
# DP-attention ranks log as "DP<r> TP<t>"; a pipeline-parallel prefill logs as "PP<s>", every stage
# repeating the same batch, and a DCP decode (one batch for all its GPUs) as "TP<t>", so only stage 0
# and TP 0 are counted (as rank 0).
TS = re.compile(r"^(\S+)Z \[\S+ \S+ (?:DP(\d+) TP\d+|PP(\d+)|TP(\d+))\] ")
PRE = re.compile(r"Prefill batch, #new-seq: \d+, #new-token: (\d+), #cached-token: (\d+).*#queue-req: (\d+)")
DEC = re.compile(r"Decode batch, #running-req: (\d+).*#prealloc-req: (\d+), #transfer-req: (\d+)")


def lines(name):
    for line in open(run / "logs" / f"glm52_2p1d-{name}.log", errors="replace"):
        m = TS.match(line)
        if m and t0 <= m.group(1) <= t1 and m.group(3) in (None, "0") and m.group(4) in (None, "0"):
            yield datetime.datetime.fromisoformat(m.group(1)[:26]), int(m.group(2) or 0), line


def bin10(ts):
    return ts.replace(second=ts.second // 10 * 10, microsecond=0)


for inst in ("prefill0", "prefill1"):
    if not (run / "logs" / f"glm52_2p1d-{inst}.log").exists():
        continue
    last, chunk_ms, new_tok, queue = {}, [], 0, collections.defaultdict(dict)
    for ts, dp, line in lines(inst):
        m = PRE.search(line)
        if not m:
            continue
        new, q = int(m.group(1)), int(m.group(3))
        new_tok += new
        queue[bin10(ts)][dp] = q
        if dp in last and last[dp][1] >= 8000 and last[dp][2] > 0:
            chunk_ms.append((ts - last[dp][0]).total_seconds() * 1000)
        last[dp] = (ts, new, q)
    secs = (datetime.datetime.fromisoformat(t1) - datetime.datetime.fromisoformat(t0)).total_seconds()
    qsum = [sum(v.values()) for v in queue.values()]
    chunk = f"{st.median(chunk_ms):.0f} ms" if chunk_ms else "n/a"
    print(
        f"{inst}: full-chunk median {chunk} (n={len(chunk_ms)}), "
        f"uncached {new_tok / secs:,.0f} tok/s, waiting mean {st.mean(qsum):.2f}, "
        f"bins with a queue {100 * sum(1 for q in qsum if q) / len(qsum):.0f}%"
    )

decode = [(ts, dp, m) for ts, dp, line in lines("decode") if (m := DEC.search(line))]
# A 10 s bin counts once three quarters of the decode's DP ranks have logged in it.
need = max(1, 3 * (max(dp for _, dp, _ in decode) + 1) // 4)
state, steps, last = collections.defaultdict(dict), collections.defaultdict(list), {}
for ts, dp, m in decode:
    b = bin10(ts)
    state[b][dp] = tuple(map(int, m.groups()))
    if dp in last and (ts - last[dp]).total_seconds() < 20 and len(state[b]) >= need:
        busiest = max(v[0] for v in state[b].values())
        steps[busiest].append((ts - last[dp]).total_seconds() * 1000 / 40)
    last[dp] = ts
full = [v for v in state.values() if len(v) >= need]
tot = [sum(x[0] for x in v.values()) for v in full]
top = sorted(max(x[0] for x in v.values()) for v in full)
pre = [sum(x[1] for x in v.values()) for v in full]
xfer = [sum(x[2] for x in v.values()) for v in full]
print(
    f"decode: running total median {st.median(tot):.0f}, busiest rank median {st.median(top):.0f} "
    f"p90 {top[int(len(top) * 0.9)]}, prealloc-wait {st.mean(pre):.1f}, transfer-wait {st.mean(xfer):.1f}"
)
print("decode step ms by busiest-rank running: " + ", ".join(
    f"@{r} {st.median(steps[r]):.0f}" for r in sorted(steps) if len(steps[r]) >= 30))
