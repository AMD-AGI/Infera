"""Per-stage times of a pipeline-parallel prefill from its log: every stage logs "Prefill batch" when it starts a
batch, in the same order, so the k-th line of PP<s+1> minus the k-th line of PP<s> is stage s's time on that batch
plus any wait in front of stage s+1, and the gap between consecutive lines of one stage is the pipeline's beat.
Over a measurement window, for batches with at least MIN_NEW new tokens (default: all), prints the median and p90
of each stage offset and each stage's beat, by batch size band.
  python3 pp_stage_times.py PREFILL_LOG WINDOW_START WINDOW_END [MIN_NEW]      (times as 2026-10-23T09:16:47)"""

import collections
import datetime
import re
import statistics as st
import sys

log, t0, t1 = sys.argv[1], sys.argv[2], sys.argv[3]
min_new = int(sys.argv[4]) if len(sys.argv) > 4 else 0
LINE = re.compile(r"^(\S+)Z \[\S+ \S+ PP(\d+)\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)"
                  r".*#queue-req: (\d+)")

stages = collections.defaultdict(list)
for line in open(log, errors="replace"):
    m = LINE.match(line)
    if m and t0 <= m.group(1) <= t1:
        ts = datetime.datetime.fromisoformat(m.group(1)[:26])
        stages[int(m.group(2))].append((ts, int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))))
n_stage = max(stages) + 1
# The window can cut the pipeline mid-flight, so stage s+1's first line need not be stage s's first batch:
# align each stage to PP0 by the shift that best matches (seqs, new tokens, cached tokens).
for s in range(1, n_stage):
    sig0 = [b[1:4] for b in stages[0]]
    best = max(range(-4, 5), key=lambda d: sum(
        1 for k in range(max(0, -d), min(len(sig0), len(stages[s]) - d)) if sig0[k] == stages[s][k + d][1:4]))
    stages[s] = stages[s][best:] if best >= 0 else [stages[s][0]] * 0 + stages[s]
    if best < 0:
        stages[0] = stages[0][-best:]
        for t in range(1, s):
            stages[t] = stages[t][-best:]
n = min(len(v) for v in stages.values())
print(f"{n_stage} stages, {n} batches in window (per stage: {[len(stages[s]) for s in range(n_stage)]})")


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


bands = [(0, 2048), (2048, 8000), (8000, 10**9)]
for lo, hi in bands:
    idx = [k for k in range(1, n) if lo <= stages[0][k][2] < hi and stages[0][k][2] >= min_new]
    if not idx:
        continue
    print(f"\nbatches with {lo}-{hi if hi < 10**9 else 'max'} new tokens: {len(idx)}, "
          f"median new {st.median(stages[0][k][2] for k in idx):.0f}, "
          f"median cached {st.median(stages[0][k][3] for k in idx):.0f}, "
          f"median seqs {st.median(stages[0][k][1] for k in idx):.0f}")
    for s in range(n_stage - 1):
        off = [(stages[s + 1][k][0] - stages[s][k][0]).total_seconds() * 1000 for k in idx]
        print(f"  PP{s} -> PP{s + 1} offset ms: median {q(off, .5):7.1f}  p90 {q(off, .9):7.1f}")
    for s in range(n_stage):
        beat = [(stages[s][k][0] - stages[s][k - 1][0]).total_seconds() * 1000 for k in idx if stages[s][k][4] > 0]
        print(f"  PP{s} beat ms (queue > 0, n={len(beat)}): median {q(beat, .5):7.1f}  p90 {q(beat, .9):7.1f}")
