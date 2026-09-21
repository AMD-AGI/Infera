#!/usr/bin/env python3
# Regenerate results/t2-summary.csv from the per-point agentx_conc<N>.json files.
# Usage: python3 make_summary.yihou.py <results_dir> > t2-summary.csv
#   <results_dir> holds t2f/agentx_conc{40,56,72,96}.json and
#   t2e/agentx_conc{40,56}.json (the custom-all-reduce-ON A/B arm).
import json, sys, os

ROWS = [("t2f", 40), ("t2f", 56), ("t2f", 72), ("t2f", 96),
        ("t2e", 40), ("t2e", 56)]
COLS = ["arm", "conc", "req_profiled", "err_dropped",
        "tput_total_tps", "tput_in_tps", "tput_out_tps", "per_gpu_total_tps",
        "ttft_mean_s", "ttft_p50_s", "ttft_p90_s", "ttft_p95_s",
        "e2e_mean_s", "e2e_p50_s",
        "itl_mean_ms", "itl_p95_ms",
        "srv_overall_cache_hit", "srv_gpu_cache_hit", "kv_gpu_usage_pct",
        "osl_actual_mean", "osl_expected_mean"]

def main(base):
    print(",".join(COLS))
    for arm, c in ROWS:
        p = os.path.join(base, arm, f"agentx_conc{c}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rm = d["request_metrics"]; lat = rm["latency"]; th = rm["throughput"]
        tok = rm["tokens"]; ra = d["request_accounting"]; sm = d.get("server_metrics", {})
        cache = sm.get("cache", {}); kv = sm.get("kv_cache", {})
        def g(x, n=2):
            return round(x, n) if isinstance(x, (int, float)) else x
        row = [arm, c, ra["records_profiled"], ra["records_error_dropped"],
               g(th["total"]["tokens_per_second"]), g(th["input"]["tokens_per_second"]),
               g(th["output"]["tokens_per_second"]), g(th["per_gpu"]["total_tput_tps"]),
               g(lat["ttft"]["mean"]), g(lat["ttft"]["p50"]), g(lat["ttft"]["p90"]), g(lat["ttft"]["p95"]),
               g(lat["e2el"]["mean"]), g(lat["e2el"]["p50"]),
               g(lat["itl"]["mean"]*1000), g(lat["itl"]["p95"]*1000),
               g(cache.get("overall_cache_hit_rate"), 4), g(cache.get("gpu_cache_hit_rate"), 4),
               kv.get("gpu_usage_pct"),
               g(tok["output_actual"]["mean"]), g(tok["output_expected"]["mean"])]
        print(",".join(str(x) for x in row))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
