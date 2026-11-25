"""How long the UMBP server was unavailable: its KFD queue evictions, from host_telemetry.sh.

The UMBP server is taken to be the first KFD process that appears after telemetry starts (up.sh starts it before
either leg; processes already present belong to other jobs). KFD adds an eviction's duration to evicted_ms when the
queues come back, so each jump is one eviction that just ended. Prefill's "UMBP standalone lookup ... failed" and
"transfers are refused" warnings are counted alongside. With an AIPerf log, the total is also given for the scored
window (the 3600 s after "Phase profiling (profiling) started").

usage: umbp_avail.py LOG_DIR [MIN_S=3]
"""
import datetime
import glob
import re
import sys


def main():
    d = sys.argv[1]
    min_s = float(sys.argv[2]) if len(sys.argv) > 2 else 3
    t0 = None
    for a in glob.glob(f"{d}/agentx-*/result/conc_*/aiperf_artifacts/logs/aiperf.log"):
        for line in open(a, errors="replace"):
            if "Phase profiling (profiling) started" in line:
                t0 = datetime.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                break
    first, umbp, prev, ev, day, last = None, None, {}, [], None, None
    for line in open(f"{d}/host-telemetry.txt", errors="replace"):
        if not line.startswith("== "):
            continue
        hms = datetime.time.fromisoformat(line.split()[1])
        if day is None:
            # Telemetry starts before AgentX: a first sample later in the day than t0 is from the day before.
            day = datetime.date(2000, 1, 1) if t0 is None else (
                t0.date() - datetime.timedelta(days=1) if hms > t0.time() else t0.date())
        t = datetime.datetime.combine(day, hms)
        if last is not None and t < last - datetime.timedelta(hours=12):
            day += datetime.timedelta(days=1)
            t += datetime.timedelta(days=1)
        last = t
        m = re.search(r"evicted_ms=(\S*)", line)
        cur = {p: int(v) for p, v in re.findall(r"(\d+):(\d+)", m.group(1))} if m else {}
        if first is None:
            first = set(cur)
        elif umbp is None:
            new = [p for p in cur if p not in first]
            umbp = new[0] if new else None
        if umbp and umbp in prev and cur.get(umbp, 0) > prev[umbp]:
            dur = (cur[umbp] - prev[umbp]) / 1000
            ev.append((t - datetime.timedelta(seconds=dur), t, dur))
        prev = cur
    warn = refused = 0
    try:
        for line in open(f"{d}/prefill.log", errors="replace"):
            warn += "UMBP standalone lookup" in line and "failed" in line
            refused += "transfers are refused while" in line
    except OSError:
        pass
    big = [e for e in ev if e[2] >= min_s]
    print(f"UMBP server pid {umbp}: {len(ev)} evictions, {sum(e[2] for e in ev):.0f} s in total, longest "
          f"{max([e[2] for e in ev] or [0]):.0f} s; {len(big)} of at least {min_s:g} s")
    for a, b, dur in big:
        print(f"  {a:%H:%M:%S} - {b:%H:%M:%S}  {dur:.1f} s")
    if t0 is not None:
        w1 = t0 + datetime.timedelta(seconds=3600)
        tot = sum(max(0.0, (min(b, w1) - max(a, t0)).total_seconds()) for a, b, _ in ev)
        print(f"unavailable in the scored window ({t0:%H:%M:%S} + 3600 s): {tot:.0f} s ({100 * tot / 3600:.1f}%)")
    print(f"prefill UMBP warnings: {warn} unresponsive, {refused} refused transfers")


if __name__ == "__main__":
    main()
