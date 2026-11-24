"""Stalls from the per-step recorder, aligned with host telemetry.

A stall is a gap of more than THRESH_S between two execute_model calls of one worker when the earlier step ran real
requests (dummy=0, reqs>0): with requests running or queued the engine steps continuously, so such a gap means the
previous step's GPU work did not complete (idle periods follow steps without real requests and are not counted).
For each stall it sums, over the stall window plus the preceding telemetry sample, the kernel's page migrations,
direct-compaction stalls, khugepaged collapses and KFD queue-eviction time (host_telemetry.sh), and the GPU busy %.

usage: slow_steps.py LOG_DIR [THRESH_S=5]   (reads LOG_DIR/steplog-{p,d}/ and hang*/steplog-*/, host-telemetry.txt)
"""
import datetime
import glob
import os
import re
import sys

KEYS = ("pgmigrate_success", "compact_stall", "compact_isolated", "thp_collapse_alloc", "thp_fault_alloc")


def utc(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S")


def telemetry(path, day):
    rows = []
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.startswith("== "):
            continue
        hms = line.split()[1]
        t = datetime.datetime.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=datetime.timezone.utc).timestamp()
        busy = re.search(r"busy=([0-9,]*)", line).group(1).strip(",")
        ev = re.search(r"evicted_ms=([0-9:,]*)", line)
        ev = sum(int(x.split(":")[1]) for x in ev.group(1).strip(",").split(",") if ":" in x) if ev else None
        c = {k: int(v) for k, v in re.findall(r"(\w+) (\d+)", line) if k in KEYS}
        rows.append((t, busy, ev, c))
    return rows


def window(tel, t0, t1):
    pre = [r for r in tel if r[0] <= t0]
    inside = [r for r in tel if t0 < r[0] <= t1 + 2]
    if not pre or not inside:
        return None
    a, b = pre[-1], inside[-1]
    d = {k: b[3].get(k, 0) - a[3].get(k, 0) for k in KEYS}
    ev = b[2] - a[2] if a[2] is not None and b[2] is not None else None
    busy_max = max(max((int(x) for x in r[1].split(",") if x), default=0) for r in inside)
    return d, ev, busy_max


def stalls(csv, thresh):
    hdr, prev, out = None, None, []
    for line in open(csv):
        p = line.rstrip().split(",")
        if p[0] == "#S":
            hdr = p
            continue
        if p[0] != "S" or hdr is None:
            continue
        r = dict(zip(hdr, p))
        t = float(r["wall"])
        if prev is not None and t - float(prev["wall"]) > thresh and prev["dummy"] == "0" and int(prev["reqs"]) > 0:
            out.append((float(prev["wall"]), t, prev, r))
        prev = r
    return out


def main():
    d = sys.argv[1]
    thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 5
    csvs = sorted(glob.glob(f"{d}/steplog-*/steps-*.csv") + glob.glob(f"{d}/hang*/steplog-*/steps-*.csv"))
    if not csvs:
        print("no steplog CSVs")
        return
    first = next(float(l.split(",")[1]) for l in open(csvs[0]) if l.startswith("S,"))
    day = datetime.datetime.fromtimestamp(first, datetime.timezone.utc).strftime("%Y-%m-%d")
    tel = telemetry(f"{d}/host-telemetry.txt", day)
    print(f"stalls > {thresh:g} s after a real step; telemetry samples: {len(tel)}")
    print("leg/rank            start     end        gap_s  prev_step(reqs,toks,mode,ctx_max)   "
          "pgmigrate  compact_stall  thp_collapse  kfd_evicted_ms  gpu_busy_max")
    total = {}
    for csv in csvs:
        leg = os.path.relpath(csv, d).replace("steplog-", "").replace("/steps-", ":").replace(".csv", "")
        for t0, t1, p, _ in stalls(csv, thresh):
            w = window(tel, t0, t1)
            tw = (f"{w[0]['pgmigrate_success']:>9}  {w[0]['compact_stall']:>13}  {w[0]['thp_collapse_alloc']:>12}  "
                  f"{w[1] if w[1] is not None else '-':>14}  {w[2]:>12}") if w else "(no telemetry)"
            print(f"{leg:<19} {utc(t0)}  {utc(t1)}  {t1 - t0:>6.1f}  "
                  f"{p['reqs']:>4},{p['toks']:>6},{p['mode']:<9},{p['ctx_max']:>7}          {tw}")
            total[leg] = total.get(leg, 0) + t1 - t0
    for leg, s in total.items():
        print(f"total stalled {leg}: {s:.0f} s")


if __name__ == "__main__":
    main()
