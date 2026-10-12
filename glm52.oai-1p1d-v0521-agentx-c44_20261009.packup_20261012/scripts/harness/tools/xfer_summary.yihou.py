#!/usr/bin/env python3
"""Summarise YIHOU-XFER (+ YIHOU-NONFINAL) lines of a prefill log in a time window.
Usage: xfer_summary.yihou.py PREFILL_LOG START_ISO [END_ISO]
XFER fields per shard line (conn.yihou-xferlog-par / conn.yihou-xferfix): calls, GiB, wall, wait_*,
par_*, and (xferfix only) sync_n / sync_mean / sync_max / sync_sum / idle. Absent fields count as 0.
NONFINAL (prefill.yihou-nonfinal-event): nonfinal_chunks, fallback, event_done_at_send per minute."""
import re, subprocess, sys
from datetime import datetime
t0 = datetime.fromisoformat(sys.argv[2]); t1 = datetime.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else None
def num(k, l):
    m = re.search(rf'(?<![\w/]){re.escape(k)}=([\d.]+)', l)
    return float(m.group(1)) if m else 0.0
xf, nf, modes = [], [], set()
for l in subprocess.run(['strings', sys.argv[1]], capture_output=True, text=True).stdout.splitlines():
    if 'YIHOU-XFER shard=' not in l and 'YIHOU-NONFINAL mode=' not in l: continue
    try: t = datetime.fromisoformat(l[:26])
    except ValueError: continue
    if t < t0 or (t1 and t >= t1): continue
    if 'YIHOU-XFER shard=' in l:
        xf.append({k: num(k, l) for k in ('calls', 'GiB', 'wall', 'wait_mean', 'wait_max', 'wall_max', 'deq',
                                         'par_group_wall_mean', 'par_group_wall_max',
                                         'sync_n', 'sync_max', 'sync_sum', 'idle', 'span')})
    elif 'span=' in l:
        modes.add(re.search(r'mode=(\S+)', l).group(1))
        nf.append({k: num(k, l) for k in ('nonfinal_chunks', 'fallback', 'event_done_at_send')})
if xf:
    S = lambda k: sum(r[k] for r in xf)
    gib, wall, calls, deq = S('GiB'), S('wall'), S('calls'), S('deq')
    print(f"XFER lines={len(xf)} calls={calls:.0f} GiB={gib:.1f} wall_sum={wall:.1f}s -> GiB/s={gib/wall if wall else 0:.2f}"
          f" | wait_mean(deq-weighted)={sum(r['wait_mean']*r['deq'] for r in xf)/max(deq,1):.3f}s wait_max={max(r['wait_max'] for r in xf):.3f}s"
          f" | wall/call={wall/max(calls,1):.2f}s wall_max={max(r['wall_max'] for r in xf):.2f}s"
          f" | par_group_wall_max={max(r['par_group_wall_max'] for r in xf):.3f}s")
    sn = S('sync_n')
    print(f"     sync_n={sn:.0f} sync_sum={S('sync_sum'):.1f}s sync_mean={S('sync_sum')/max(sn,1):.3f}s sync_max={max(r['sync_max'] for r in xf):.3f}s"
          f" | idle_sum={S('idle'):.1f}s over span_sum={S('span'):.0f}s ({100*S('idle')/max(S('span'),1):.1f}% idle)")
else:
    print("XFER: no lines in window")
if nf:
    S = lambda k: sum(r[k] for r in nf)
    print(f"NONFINAL lines={len(nf)} mode={','.join(sorted(modes))} nonfinal_chunks={S('nonfinal_chunks'):.0f} "
          f"fallback={S('fallback'):.0f} event_done_at_send={S('event_done_at_send'):.0f}")
else:
    print("NONFINAL: no lines in window")
