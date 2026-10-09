#!/usr/bin/env python3
"""Compare a crsuse2 run with a reference using experiment-summary-20261008's extraction.

Both sides go through the same steps as build_report.py: summary.json from the
aus analyze.py (points["80"]) and decode-local-prefix.json from
analyze_decode_prefix.py, read from DIR or DIR/analysis. The reference defaults
to B4. Per-GPU throughput divides by each run's own GPU count.
Writes RUN/analysis/compare-<label>.json and prints a table.
Usage: compare_b4.py RUN_DIR [REFERENCE_DIR LABEL]
"""

import json, sys
from pathlib import Path

LLYING = Path(__file__).resolve().parents[2]
B4 = LLYING / "router-capacity-campaign-20260928/b4"


def find(directory, name):
    return next(p for p in (directory / name, directory / "analysis" / name) if p.exists())


def metrics(directory):
    a = json.loads(find(directory, "summary.json").read_text())["points"]["80"]
    phase, q = a["phases"]["profiling"], a["headline"]["request_metrics"]
    lat, t, cache = q["latency"], q["throughput"], phase["cache"]
    m = {}
    for kind in ["ttft", "e2el", "itl"]:
        for stat in ["mean", "p50", "p90", "p95"]:
            m[f"{kind}_{stat}" + ("_ms" if kind == "itl" else "_s")] = lat[kind][stat] * (1000 if kind == "itl" else 1)
    for stat in ["p50", "p90"]:
        m[f"intvty_{stat}_tps"] = lat["intvty"][stat]
    m.update(total_tps_per_gpu=t["per_gpu"]["total_tput_tps"], output_tps_per_gpu=t["per_gpu"]["output_tput_tps"],
             output_tps=t["output"]["tokens_per_second"], input_tps=t["input"]["tokens_per_second"],
             mean_input_tokens=q["tokens"]["input"]["mean"], mean_output_tokens=q["tokens"]["output_actual"]["mean"])
    m.update({k: a["runner_accounting"]["profiling"][k] for k in ["sent", "completed", "cancelled", "errors"]})
    for stage in ["prefill/queue_ms", "prefill/forward_envelope_ms", "prefill/bootstrap_ms", "prefill/transfer_tail_ms",
                  "decode/alloc_wait_ms", "decode/transfer_wait_ms", "decode/generation_ms"]:
        for stat in ["mean", "p50", "p90", "p99"]:
            m[stage.replace("/", "_") + "_" + stat] = phase["stages_ms"][stage][stat]
    m["decode_kv_budget_blocked"] = phase.get("blocked_request_counts", {}).get("kv_budget", 0)
    for label, key in [("p_miss_pct", "miss_tokens"), ("p_device_hit_pct", "cached_device"), ("p_host_hit_pct", "cached_host")]:
        m[label] = cache[key] / cache["input_tokens"] * 100
    d = json.loads(find(directory, "decode-local-prefix.json").read_text())
    m["d_local_prefix_pct"] = d["prefix_tokens_last_allocation"] / d["input_tokens_last_allocation"] * 100
    return m


def main():
    run = Path(sys.argv[1])
    reference, label = (Path(sys.argv[2]), sys.argv[3]) if len(sys.argv) > 3 else (B4, "b4")
    ref, ours = metrics(reference), metrics(run)
    rows = {k: {label: ref[k], "run": ours[k], "delta_pct": (ours[k] / ref[k] - 1) * 100 if ref[k] else None}
            for k in ref}
    (run / "analysis" / f"compare-{label}.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{'metric':34s} {label:>14s} {'run':>14s} {'delta':>9s}")
    for k, r in rows.items():
        delta = f"{r['delta_pct']:+8.2f}%" if r["delta_pct"] is not None else "        -"
        print(f"{k:34s} {r[label]:14.3f} {r['run']:14.3f} {delta}")


if __name__ == "__main__":
    main()
