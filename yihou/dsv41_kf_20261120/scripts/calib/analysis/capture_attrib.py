#!/usr/bin/env python3
"""Capture-site attribution coverage of graph-replayed dispatches in a calibrated-profiler window (mode capture).

Usage: capture_attrib.py TRACE_DIR PID [PID ...]
For window dispatches launched by hipGraphLaunch: share with graph_exec_id, with a non-zero capture_correlation_id,
and with that id resolved to a HIP API record traced during capture (op name of the captured call).
"""
import collections
import json
import sys
from bisect import bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_trace  # noqa: E402

d = Path(sys.argv[1])
for pid in sys.argv[2:]:
    sd = json.loads((d / f"steps-{pid}.json").read_text())
    steps = sorted(sd["steps"], key=lambda s: s["t0"])
    st = [s["t0"] for s in steps]
    tr = calib_trace.load(d / f"trace-{pid}.json")
    c = collections.Counter()
    ops = collections.Counter()
    for x in tr["dispatch"]:
        a = tr["api"].get(x["corr"])
        if not a or a["op"] != "hipGraphLaunch":
            continue
        i = bisect_right(st, a["start"]) - 1
        if i < 0 or not (steps[i]["t0"] <= a["start"] <= steps[i]["t1"]):
            continue
        stg = steps[i]["stage"]
        c[(stg, "graph")] += 1
        c[(stg, "graph_exec")] += x["graph_exec"] != 0
        c[(stg, "capture_corr")] += x["capture_corr"] != 0
        cap = tr["api"].get(x["capture_corr"]) if x["capture_corr"] else None
        if cap:
            c[(stg, "resolved")] += 1
            ops[(stg, cap["op"])] += 1
    out = {"pid": pid}
    for stg in ("VERIFY", "DRAFT"):
        n = c[(stg, "graph")]
        out[stg] = {"graph_dispatches": n, "graph_exec_frac": c[(stg, "graph_exec")] / n if n else None,
                    "capture_corr_frac": c[(stg, "capture_corr")] / n if n else None,
                    "resolved_frac": c[(stg, "resolved")] / n if n else None,
                    "captured_ops": {k[1]: v for k, v in ops.items() if k[0] == stg}}
    print(json.dumps(out))
