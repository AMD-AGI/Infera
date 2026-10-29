"""Validate the whole-run eviction timing (aiperf server-metrics slices) against the evictions seen in a 4-rank prefill
trace of the same run, and align trace steps with the "Prefill batch" log lines.
  1. per rank: trace steps <-> log rows matched by batch composition (bs, tokens); time of the log line minus the step
     annotation start (the log line is printed inside get_new_batch_prefill, before run_batch's annotation)
  2. per rank: every evict_for_alloc frame in the trace (start, duration) next to the metrics slice that counted it
  python3 trace_evict_check.py COMPACT_DIR PREFILL_LOG SERVER_METRICS_EXPORT.json [--endpoint 27201]"""

import argparse
import gzip
import json
import re
import statistics as st
import sys
from datetime import datetime, timezone

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from beat_usage_window import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("compact")
ap.add_argument("log")
ap.add_argument("metrics")
ap.add_argument("--endpoint", default="27201")
a = ap.parse_args()

hms = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S.%f")[:-3]
R = [json.load(gzip.open(f"{a.compact}/pp{r}.json.gz", "rt")) for r in range(4)]
rows = load(a.log, None)
M = json.load(open(a.metrics))["metrics"]


def slices(name, rank):
    for s in M[name]["series"]:
        if a.endpoint in s["endpoint_url"] and s["labels"].get("pp_rank") == str(rank):
            return s["timeslices"]


comp = lambda name: tuple(int(x) for x in re.search(r"bs=(\d+) toks=(\d+)", name).groups())
for r in range(4):
    steps = R[r]["steps"]
    lr = [x for x in rows if x["rank"] == r and steps[0]["t"] - 30 <= x["t"] <= steps[-1]["t"] + 30]
    lc = [(x["seqs"], x["new"]) for x in lr]
    sc = [comp(s["name"]) for s in steps]
    best = max(range(-10, len(lc) - 10), key=lambda k: sum(1 for i, c in enumerate(sc) if 0 <= i + k < len(lc) and lc[i + k] == c))
    match = sum(1 for i, c in enumerate(sc) if 0 <= i + best < len(lc) and lc[i + best] == c)
    dt = [lr[i + best]["t"] - s["t"] for i, s in enumerate(steps) if 0 <= i + best < len(lr) and lc[i + best] == sc[i]]
    print(f"PP{r}: {len(steps)} steps, best log offset {best:+d} matches {match}/{len(steps)}; "
          f"log line minus step annotation start: p10 / p50 / p90 {sorted(dt)[len(dt) // 10] * 1e3:.0f} / "
          f"{st.median(dt) * 1e3:.0f} / {sorted(dt)[9 * len(dt) // 10] * 1e3:.0f} ms")
    for s, i in zip(steps, range(len(steps))):
        s["log"] = lr[i + best] if 0 <= i + best < len(lr) and lc[i + best] == sc[i] else None

print("\nevict_for_alloc frames in the trace vs metrics slices (count, sum) ending within [start, end + 1.5 s]")
for r in range(4):
    ev = [x for x in R[r]["sched"] if x[0] == "p" and re.search(r"unified_radix_cache\.py\(\d+\): evict_for_alloc$", x[3])]
    sl = [x for x in slices("sglang:eviction_duration_seconds", r) if x["count"]]
    t0, t1 = R[r]["steps"][0]["t"], R[r]["steps"][-1]["t"]
    sl_in = [x for x in sl if t0 <= x["end_ns"] / 1e9 <= t1 + 1]
    print(f"PP{r}: trace {len(ev)} evictions ({sum(x[2] for x in ev):.2f} s); metrics slices in the trace window: "
          f"{sum(x['count'] for x in sl_in)} evictions ({sum(x['sum'] for x in sl_in):.2f} s)")
    for x in ev:
        s0, d = x[1], x[2]
        near = [y for y in sl if s0 <= y["end_ns"] / 1e9 <= s0 + d + 1.5]
        txt = ", ".join(f"slice end {hms(y['end_ns'] / 1e9)} n{y['count']} {y['sum']:.3f}s" for y in near)
        print(f"   {hms(s0)} dur {d:.3f}  ->  {txt or 'NO SLICE'}")
