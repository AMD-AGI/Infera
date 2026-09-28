#!/usr/bin/env python3
"""Per-second, per-DP-rank decode series from an aiperf server_metrics_export.json.

The export is ~1.4 GB and json.load needs several times that in RAM, so run this
on a compute node, not the login node.

Usage: server_metrics_extract.py SERVER_METRICS_EXPORT_JSON[.gz] ARM OUT_CSV [DECODE_PORT]
Artifacts: OUT_CSV with one row per (profiling second, DP rank): gauges as the
slice average, counters as the slice increment.
"""
import collections
import csv
import gzip
import json
import sys

WANT = {
    "sglang:num_running_reqs": "avg", "sglang:decode_sum_seq_lens": "avg",
    "sglang:gen_throughput": "avg", "sglang:spec_accept_length": "avg",
    "sglang:num_queue_reqs": "avg", "sglang:hicache_host_used_tokens": "avg",
    "sglang:hicache_backup_tokens": "total", "sglang:load_back_tokens": "total",
    "sglang:evicted_tokens": "total",
}


def main():
    path, arm, out = sys.argv[1:4]
    port = sys.argv[4] if len(sys.argv) > 4 else "29002"
    d = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    prof = d["summary"]["phase_time_ranges"]["profiling"]
    rows = collections.defaultdict(dict)
    for name, kind in WANT.items():
        for s in d["metrics"].get(name, {}).get("series", []):
            dp = (s.get("labels") or {}).get("dp_rank")
            if port not in s.get("endpoint_url", "") or dp is None:
                continue
            for t in s.get("timeslices") or []:
                if not prof["start_ns"] <= t["start_ns"] < prof["end_ns"]:
                    continue
                key = ((t["start_ns"] - prof["start_ns"]) // 10**9, dp)
                if kind == "total":
                    if t.get("total") is not None:
                        rows[key][name] = rows[key].get(name, 0) + t["total"]
                else:
                    rows[key][name] = t.get("avg")
    cols = sorted(WANT)
    w = csv.writer(open(out, "w"))
    w.writerow(["arm", "sec", "dp"] + [c.split(":")[1] for c in cols])
    for (sec, dp), r in sorted(rows.items()):
        w.writerow([arm, sec, dp] + [r.get(c, "") for c in cols])
    print(f"{arm}: {len(rows)} rank-seconds -> {out}")


if __name__ == "__main__":
    main()
