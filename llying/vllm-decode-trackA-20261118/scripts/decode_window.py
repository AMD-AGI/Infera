#!/usr/bin/env python3
"""Decode-phase step time from a `vllm bench serve --save-detailed` result.

Every request's chunk arrival times are start + ttft + cumsum(itl). The window starts when the last request got its
first token (all prefills done, every request decoding) and ends when the first request finished, so the batch is
constant inside it. With MTP each streamed chunk is one engine step, so the ITLs that end inside the window are decode
step times at full concurrency.

With --trace TRACE_LOG (trace_when_decoding.sh output), chunks that end between start_profile - 1 s and
stop_profile + 2 s are dropped (profiling slows the steps and stop_profile stalls the engine while it writes): the
bench's relative clock is mapped to wall time through the result's end timestamp ("date") and duration (about 1 s).
Also prints the per-5 s medians inside the window.
  decode_window.py RESULT.json [--trace TRACE_LOG] [--bins]
"""
import argparse
import datetime as dt
import json
import re
import statistics


def trace_interval(path):
    t = {}
    for line in open(path):
        m = re.match(r"(start|stop)_profile (\d\d:\d\d:\d\d)", line)
        if m:
            t[m.group(1)] = m.group(2)
    return (t.get("start"), t.get("stop")) if len(t) == 2 else None


def window(path, trace=None, bins=False):
    d = json.load(open(path))
    starts, ttfts, itls = d["start_times"], d["ttfts"], d["itls"]
    t0 = min(starts)
    first = [s + t for s, t in zip(starts, ttfts)]
    ends = [f + sum(i) for f, i in zip(first, itls)]
    w0, w1 = max(first), min(ends)
    excl = None
    if trace:
        iv = trace_interval(trace)
        if iv:
            end_wall = dt.datetime.strptime(d["date"], "%Y%m%d-%H%M%S")
            start_wall = end_wall - dt.timedelta(seconds=d["duration"])
            day = start_wall.date()

            def rel(hms):
                w = dt.datetime.combine(day, dt.datetime.strptime(hms, "%H:%M:%S").time())
                return (w - start_wall).total_seconds() + t0

            excl = (rel(iv[0]) - 1, rel(iv[1]) + 2)
    steps, binned = [], {}
    for f, il in zip(first, itls):
        t = f
        for x in il:
            t += x
            if w0 < t <= w1 and not (excl and excl[0] <= t <= excl[1]):
                steps.append(x)
                binned.setdefault(int((t - w0) // 5), []).append(x)
    q = sorted(steps)

    def pct(p):
        return round(q[min(len(q) - 1, int(p * len(q)))] * 1e3, 2) if q else None

    r = {
        "file": path,
        "n_req": len(starts),
        "window_s": round(w1 - w0, 1),
        "window_start_s": round(w0 - t0, 1),
        "excluded_s": [round(excl[0] - w0, 1), round(excl[1] - w0, 1)] if excl else None,
        "n_itl": len(steps),
        "step_p50_ms": pct(0.5),
        "step_mean_ms": round(statistics.fmean(steps) * 1e3, 2) if steps else None,
        "step_p90_ms": pct(0.9),
        "step_p99_ms": pct(0.99),
        "tpot_p50_ms": round(d.get("median_tpot_ms", 0), 1),
        "itl_p50_ms": round(d.get("median_itl_ms", 0), 1),
        "ttft_max_s": round(max(ttfts), 1),
        "accept_len": round(d.get("spec_decode_acceptance_length", 0) or 0, 2),
        "duration_s": round(d.get("duration", 0), 1),
    }
    if bins:
        r["bins_p50_ms"] = {
            b * 5: round(sorted(v)[len(v) // 2] * 1e3, 2) for b, v in sorted(binned.items())
        }
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--trace")
    ap.add_argument("--bins", action="store_true")
    a = ap.parse_args()
    print(json.dumps(window(a.result, a.trace, a.bins)))
