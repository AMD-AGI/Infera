#!/usr/bin/env python3
"""Per-call view of the DSA top-k kernels in a decode trace: how many k_scatter calls a step makes, how their
durations are distributed, and where in the step the slow ones sit (index within the step's sequence of calls).
Usage: kscatter_calls.py TRACE.json.gz [TRACE...]
"""

import gzip
import json
import statistics
import sys


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def main():
    for path in sys.argv[1:]:
        ev = [e for e in load(path)["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_op", "Kernel")]
        ev.sort(key=lambda e: e["ts"])
        ks = [e for e in ev if "k_scatter" in e["name"]]
        hist = [e for e in ev if "logits_hist" in e["name"]]
        if not ks:
            print(path, "no k_scatter")
            continue
        d = [e["dur"] for e in ks]
        print(f"== {path.rsplit('/', 1)[-1]}")
        print(f"   k_scatter calls {len(ks)}  logits_hist calls {len(hist)}  total {sum(d)/1000:.1f} ms  "
              f"median {statistics.median(d):.1f} us  p90 {sorted(d)[int(len(d)*0.9)]:.1f} us  max {max(d):.1f} us")
        # Group calls into steps by gaps: a gap larger than 5 ms between consecutive k_scatter calls starts a new step.
        steps, cur = [], [ks[0]]
        for a, b in zip(ks, ks[1:]):
            if b["ts"] - (a["ts"] + a["dur"]) > 5000:
                steps.append(cur)
                cur = []
            cur.append(b)
        steps.append(cur)
        per = [len(s) for s in steps]
        print(f"   steps {len(steps)}  calls/step min {min(per)} median {statistics.median(per)} max {max(per)}")
        # Duration by position within the step (positions with the same index across steps).
        n = int(statistics.median(per))
        by_pos = {}
        for s in steps:
            if len(s) != n:
                continue
            for i, e in enumerate(s):
                by_pos.setdefault(i, []).append(e["dur"])
        if by_pos:
            meds = [(i, statistics.median(v)) for i, v in sorted(by_pos.items())]
            top = sorted(meds, key=lambda x: -x[1])[:8]
            print("   slowest positions (index: median us): " + ", ".join(f"{i}: {m:.0f}" for i, m in top))
            print("   position medians: " + " ".join(f"{m:.0f}" for _, m in meds))
        args = ks[0].get("args", {})
        print("   first call args:", {k: args[k] for k in list(args)[:12]})
        big = sorted(ks, key=lambda e: -e["dur"])[:5]
        print("   largest calls (us, grid):", [(round(e["dur"]), e.get("args", {}).get("grid")) for e in big])


if __name__ == "__main__":
    main()
