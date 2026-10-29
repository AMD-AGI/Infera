"""Who waits for whom between prefill and decode, per second of the scored window (prefill optimistic_prefill_attempts = 0:
a request can be prefilled only after decode has preallocated its KV, i.e. decode's prealloc queue = prefill's bootstrap wait).
Decode log ("Decode batch", every step, per DP rank): #prealloc-req (waiting for decode KV), #transfer-req (KV allocated,
request at prefill or in transfer), token usage, pre-allocated usage. Prefill PP0 log ("Prefill batch", per batch):
#queue-req (bootstrapped, ready to compute). A second with no PP0 batch counts as prefill idle.
  python3 pd_queue_cause.py PREFILL_LOG DECODE_LOG START_HH:MM:SS END_HH:MM:SS"""

import collections
import re
import statistics as st
import sys

plog, dlog, t0, t1 = sys.argv[1:5]
pp = re.compile(r"\[\S+ (\d\d:\d\d:\d\d) PP0\] Prefill batch.*?#queue-req: (\d+).*?#bootstrap-req: (\d+)")
dp = re.compile(r"\[\S+ (\d\d:\d\d:\d\d) DP(\d) TP\d+\] Decode batch, #running-req: (\d+), #token: \d+, token usage: ([\d.]+)"
                r".*?pre-allocated usage: ([\d.]+), #prealloc-req: (\d+), #transfer-req: (\d+)")
pre = {}
for line in open(plog, errors="replace"):
    m = pp.search(line)
    if m and t0 <= m.group(1) <= t1:
        q, b = int(m.group(2)), int(m.group(3))
        old = pre.get(m.group(1), (0, 0, 0))
        pre[m.group(1)] = (max(old[0], q), max(old[1], b), old[2] + 1)
dec = collections.defaultdict(dict)
for line in open(dlog, errors="replace"):
    m = dp.search(line)
    if m and t0 <= m.group(1) <= t1:
        dec[m.group(1)][int(m.group(2))] = (int(m.group(3)), float(m.group(4)), float(m.group(5)), int(m.group(6)), int(m.group(7)))
from datetime import datetime, timedelta
last, rows = {}, []
cur = datetime.strptime(t0, "%H:%M:%S")
end = datetime.strptime(t1, "%H:%M:%S")
if end < cur:
    end += timedelta(days=1)
while cur <= end:
    s = cur.strftime("%H:%M:%S")
    cur += timedelta(seconds=1)
    last.update(dec.get(s, {}))  # carry each rank's latest state forward (ranks log every few seconds)
    if len(last) < 4:
        continue
    d = last.values()
    p = pre.get(s)
    rows.append(dict(
        busy=p is not None, ready=(p[0] if p else 0),
        prealloc=sum(x[3] for x in d), transfer=sum(x[4] for x in d), running=sum(x[0] for x in d),
        usage=st.mean(x[1] for x in d), preuse=st.mean(x[2] for x in d)))
n = len(rows)
f = lambda c: sum(1 for r in rows if c(r)) / n
print(f"{n} seconds (decode ranks carried forward to their latest log line)")
print(f"prefill idle (no PP0 batch that second): {f(lambda r: not r['busy']):.0%}; "
      f"prefill busy with a ready queue: {f(lambda r: r['busy'] and r['ready'] > 0):.0%}; busy, queue empty: {f(lambda r: r['busy'] and r['ready'] == 0):.0%}")
print(f"decode prealloc queue > 0: {f(lambda r: r['prealloc'] > 0):.0%} (mean {st.mean(r['prealloc'] for r in rows):.1f} requests)")
print("2x2 (share of seconds):")
for pa in (True, False):
    for pb in (True, False):
        print(f"  decode prealloc {'>0' if pa else '=0'} & prefill {'idle' if pb else 'busy'}: "
              f"{f(lambda r: (r['prealloc'] > 0) == pa and (not r['busy']) == pb):.0%}")
idle = [r for r in rows if not r["busy"]]
if idle:
    print(f"while prefill is idle: decode prealloc > 0 in {sum(1 for r in idle if r['prealloc'] > 0) / len(idle):.0%} of the seconds, "
          f"mean prealloc {st.mean(r['prealloc'] for r in idle):.1f}, decode token usage {st.mean(r['usage'] for r in idle):.2f}")
busyq = [r for r in rows if r["busy"] and r["ready"] > 0]
if busyq:
    print(f"while prefill has a ready queue: decode prealloc > 0 in {sum(1 for r in busyq if r['prealloc'] > 0) / len(busyq):.0%}, "
          f"mean prealloc {st.mean(r['prealloc'] for r in busyq):.1f}")
print(f"decode KV: token usage mean {st.mean(r['usage'] for r in rows):.2f}, of which preallocated (requests still at prefill / in "
      f"transfer) {st.mean(r['preuse'] for r in rows):.2f}; running {st.mean(r['running'] for r in rows):.0f}, "
      f"transfer-req {st.mean(r['transfer'] for r in rows):.1f}, prealloc-req {st.mean(r['prealloc'] for r in rows):.1f}")
