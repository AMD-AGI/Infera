#!/usr/bin/env python3
"""Summarize AgentX perf arms next to yihou's t2f reference, in t2-summary.csv columns.

Usage: perf_summary.py EVIDENCE_PERF_DIR [T2_SUMMARY_CSV]
Reads <dir>/<arm>-agentx-c<conc>/agentx_conc<conc>.json for every arm present,
prints CSV rows (same columns as yihou's make_summary.yihou.py) and, when the
t2 summary is given, appends its t2f row for the same concurrency.
"""

import csv
import glob
import json
import os
import re
import sys

COLS = ["arm", "conc", "req_profiled", "err_dropped", "tput_total_tps", "tput_in_tps",
        "tput_out_tps", "per_gpu_total_tps", "ttft_mean_s", "ttft_p50_s", "ttft_p90_s",
        "ttft_p95_s", "e2e_mean_s", "e2e_p50_s", "itl_mean_ms", "itl_p95_ms",
        "srv_overall_cache_hit", "srv_gpu_cache_hit", "kv_gpu_usage_pct",
        "osl_actual_mean", "osl_expected_mean"]


def g(x, n=2):
    return None if x is None else round(x, n)


def row(arm: str, conc: int, path: str) -> list:
    d = json.load(open(path))
    rm = d["request_metrics"]
    lat, th, tok = rm["latency"], rm["throughput"], rm["tokens"]
    ra, sm = d["request_accounting"], d.get("server_metrics", {})
    cache, kv = sm.get("cache", {}), sm.get("kv_cache", {})
    return [arm, conc, ra["records_profiled"], ra["records_error_dropped"],
            g(th["total"]["tokens_per_second"]), g(th["input"]["tokens_per_second"]),
            g(th["output"]["tokens_per_second"]), g(th["per_gpu"]["total_tput_tps"]),
            g(lat["ttft"]["mean"]), g(lat["ttft"]["p50"]), g(lat["ttft"]["p90"]),
            g(lat["ttft"]["p95"]), g(lat["e2el"]["mean"]), g(lat["e2el"]["p50"]),
            g(lat["itl"]["mean"] * 1000), g(lat["itl"]["p95"] * 1000),
            g(cache.get("overall_cache_hit_rate"), 4), g(cache.get("gpu_cache_hit_rate"), 4),
            kv.get("gpu_usage_pct"), g(tok["output_actual"]["mean"]),
            g(tok["output_expected"]["mean"])]


def main() -> int:
    base = sys.argv[1]
    out = csv.writer(sys.stdout)
    out.writerow(COLS)
    concs = set()
    for path in sorted(glob.glob(os.path.join(base, "*-agentx-c*", "agentx_conc*.json"))):
        m = re.search(r"/([^/]+)-agentx-c(\d+)/agentx_conc\d+\.json$", path)
        arm, conc = m.group(1), int(m.group(2))
        concs.add(str(conc))
        out.writerow(row(arm, conc, path))
    if len(sys.argv) > 2:
        for r in csv.DictReader(open(sys.argv[2])):
            if r["arm"] == "t2f" and r["conc"] in concs:
                out.writerow([f"t2f(yihou,base image)"] + [r[c] for c in COLS[1:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
