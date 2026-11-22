#!/usr/bin/env python3
"""Per-run engine and client metrics from AIPerf's server_metrics_export.csv + profile_export_aiperf.json.

Histogram rows in the CSV carry one more field than the header: count, count_rate, sum, sum_rate, mean,
p1, p5, p10, p25, p50, p75, p90, p95, p99, ..., bucket counts ("le=cum;...") last. Percentiles here are
recomputed from the merged buckets of all ranks of a role (Prometheus-style linear interpolation), means as
sum/count. Gauge rows follow the header: avg, min, max, std, p1, p5, p10, p25, p50, p75, p90, p95, p99.

usage: compare_runs.py LABEL=ARTIFACT_DIR[:P_PORT,D_PORT] ...   (agg runs: LABEL=DIR:PORT)
"""
import csv
import json
import re
import sys

GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc")
HISTS = ("vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds", "vllm:time_to_first_token_seconds",
         "vllm:request_decode_time_seconds", "vllm:e2e_request_latency_seconds", "vllm:inter_token_latency_seconds",
         "vllm:request_render_duration_seconds", "vllm:mooncake_xfer_time_seconds",
         "infera_time_to_first_token_seconds", "infera_request_duration_seconds")


def buckets(field):
    out = []
    for kv in field.split(";"):
        le, c = kv.split("=")
        out.append((float("inf") if le == "+Inf" else float(le), float(c)))
    return out


def quantile(q, merged):
    total = merged[-1][1]
    if total <= 0:
        return None
    rank, prev_le, prev_c = q * total, 0.0, 0.0
    for le, c in merged:
        if c >= rank:
            if le == float("inf"):
                return prev_le
            return prev_le + (le - prev_le) * ((rank - prev_c) / (c - prev_c) if c > prev_c else 0)
        prev_le, prev_c = le, c
    return prev_le


def load(path, ports):
    rows = list(csv.reader(line for line in open(path) if not line.startswith("#")))
    gauges, hists = {}, {}
    for r in rows[1:]:
        if len(r) < 5:
            continue
        port = r[0].rsplit(":", 1)[-1]
        typ, metric = r[1], r[2]
        if port not in ports and not metric.startswith("infera_"):
            continue
        role = ports.get(port, "R")
        if typ == "gauge" and metric in GAUGES:
            eng = next((r[i - 2] for i, v in enumerate(r) if v == "deepseek-ai/DeepSeek-V4.1-Flash"), "")
            gauges[(role, eng, metric)] = dict(avg=float(r[4]), p50=float(r[12]), p90=float(r[14]))
        elif typ == "histogram" and metric in HISTS:
            bfield = next((v for v in reversed(r) if re.match(r"^[0-9.e+-]+=\d", v)), None)
            if not bfield:
                continue
            h = hists.setdefault((role, metric), dict(count=0.0, sum=0.0, b={}))
            h["count"] += float(r[4] or 0)
            h["sum"] += float(r[6] or 0)
            for le, c in buckets(bfield):
                h["b"][le] = h["b"].get(le, 0.0) + c
    out = {}
    for k, h in hists.items():
        merged = sorted(h["b"].items())
        out[k] = dict(n=h["count"], mean=h["sum"] / h["count"] if h["count"] else None,
                      p50=quantile(0.5, merged), p90=quantile(0.9, merged))
    return gauges, out


def client(path):
    d = json.load(open(path))
    rec = d.get("records", d)
    def g(name, stat):
        m = rec.get(name) or {}
        return m.get(stat)
    return {k: (g(k, "avg"), g(k, "p50"), g(k, "p90")) for k in
            ("time_to_first_token", "inter_token_latency", "request_latency", "output_token_throughput_per_user")}


def main():
    for spec in sys.argv[1:]:
        label, rest = spec.split("=", 1)
        d, _, ports = rest.partition(":")
        p = ports.split(",")
        portmap = {p[0]: "P", p[1]: "D"} if len(p) == 2 else {p[0]: "A"}
        gauges, hists = load(f"{d}/server_metrics_export.csv", portmap)
        print(f"=== {label}")
        for (role, eng, m), v in sorted(gauges.items()):
            print(f"  gauge {role} rank{eng} {m:<28} avg={v['avg']:.3f} p50={v['p50']:.3f} p90={v['p90']:.3f}")
        for (role, m), v in sorted(hists.items()):
            f = lambda x: "-" if x is None else f"{x:.3f}"
            print(f"  hist  {role} {m:<36} n={v['n']:.0f} mean={f(v['mean'])} p50={f(v['p50'])} p90={f(v['p90'])}")
        try:
            for k, v in client(f"{d}/profile_export_aiperf.json").items():
                print(f"  client {k:<34} avg/p50/p90 = {v}")
        except Exception as e:  # noqa: BLE001
            print(f"  client: {e}")


if __name__ == "__main__":
    main()
