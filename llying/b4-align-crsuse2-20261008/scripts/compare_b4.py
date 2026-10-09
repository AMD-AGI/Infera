#!/usr/bin/env python3
"""Compare a crsuse2 run with B4 using experiment-summary-20261008's extraction.

Both sides go through the same steps as build_report.py: summary.json from the
aus analyze.py (points["80"]) and decode-local-prefix.json from
analyze_decode_prefix.py. Writes RUN/analysis/compare-b4.json and prints a table.
Usage: compare_b4.py RUN_DIR
"""

import json, sys
from pathlib import Path

LLYING = Path(__file__).resolve().parents[2]
B4 = LLYING / "router-capacity-campaign-20260928/b4"


def metrics(summary, prefix):
    a = json.loads(summary.read_text())["points"]["80"]
    phase, q = a["phases"]["profiling"], a["headline"]["request_metrics"]
    lat, t, cache = q["latency"], q["throughput"], phase["cache"]
    m = {}
    for kind in ["ttft", "e2el", "itl"]:
        for stat in ["mean", "p50", "p90", "p95"]:
            m[f"{kind}_{stat}" + ("_ms" if kind == "itl" else "_s")] = lat[kind][stat] * (1000 if kind == "itl" else 1)
    for stat in ["p50", "p90"]:
        m[f"intvty_{stat}_tps"] = lat["intvty"][stat]
    m.update(total_tps_per_gpu=t["per_gpu"]["total_tput_tps"], output_tps_per_gpu=t["per_gpu"]["output_tput_tps"],
             mean_input_tokens=q["tokens"]["input"]["mean"], mean_output_tokens=q["tokens"]["output_actual"]["mean"])
    m.update({k: a["runner_accounting"]["profiling"][k] for k in ["sent", "completed", "cancelled", "errors"]})
    for stage in ["prefill/queue_ms", "prefill/forward_envelope_ms", "decode/transfer_wait_ms", "decode/generation_ms"]:
        for stat in ["mean", "p50", "p90", "p99"]:
            m[stage.replace("/", "_") + "_" + stat] = phase["stages_ms"][stage][stat]
    for label, key in [("p_miss_pct", "miss_tokens"), ("p_device_hit_pct", "cached_device"), ("p_host_hit_pct", "cached_host")]:
        m[label] = cache[key] / cache["input_tokens"] * 100
    d = json.loads(prefix.read_text())
    m["d_local_prefix_pct"] = d["prefix_tokens_last_allocation"] / d["input_tokens_last_allocation"] * 100
    return m


def main():
    run = Path(sys.argv[1])
    b4 = metrics(B4 / "summary.json", B4 / "decode-local-prefix.json")
    ours = metrics(run / "analysis/summary.json", run / "analysis/decode-local-prefix.json")
    rows = {k: {"b4": b4[k], "crsuse2": ours[k], "delta_pct": (ours[k] / b4[k] - 1) * 100 if b4[k] else None} for k in b4}
    (run / "analysis/compare-b4.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{'metric':34s} {'B4 (aus)':>14s} {'crsuse2':>14s} {'delta':>9s}")
    for k, r in rows.items():
        delta = f"{r['delta_pct']:+8.2f}%" if r["delta_pct"] is not None else "        -"
        print(f"{k:34s} {r['b4']:14.3f} {r['crsuse2']:14.3f} {delta}")


if __name__ == "__main__":
    main()
