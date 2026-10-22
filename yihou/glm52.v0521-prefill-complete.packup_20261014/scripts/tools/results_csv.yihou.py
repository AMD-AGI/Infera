# Packup C: build results/results.csv from AgentX result JSONs (score = request_metrics.throughput.per_gpu.total_tput_tps).
# Usage: python3 -I -B results_csv.yihou.py > results.csv     (reads only small agentx_conc*.json files)
import csv, json, sys
R = "/shared_nfs/yihou_glm52/v0521dbg_20261010/runs"
runs = [  # (label, node, policy, hicache_gb, write_policy, decode_mem, role, json)
    ("q1 A3 (ref)", "crsuse2-m2m-137", "early", 450, "write_through", 0.95, "reference best C48 (3rd run on stack)", "q1/agentx-c48-A3/agentx_conc48.json"),
    ("pc48b A1", "crsuse2-m2m-274", "prefill_complete", 370, "write_back", 0.96, "fresh, unscored", "pc48b/agentx-c48-A1/agentx_conc48.json"),
    ("pc48b A2", "crsuse2-m2m-274", "prefill_complete", 370, "write_back", 0.96, "warm, scored", "pc48b/agentx-c48-A2/agentx_conc48.json"),
    ("pc48e A1", "crsuse2-m2m-274", "early", 370, "write_back", 0.96, "fresh, unscored (control)", "pc48e/agentx-c48-A1/agentx_conc48.json"),
    ("pc48e A2", "crsuse2-m2m-274", "early", 370, "write_back", 0.96, "warm, scored (control)", "pc48e/agentx-c48-A2/agentx_conc48.json"),
]
w = csv.writer(sys.stdout)
w.writerow(["run", "node", "decode_alloc_policy", "prefill_hicache_gb", "prefill_hicache_write_policy", "decode_mem_fraction", "role", "conc",
            "score_tok_s_per_gpu", "qps_mean", "input_tok_mean", "output_tok_mean", "ttft_mean_s", "ttft_p50_s", "ttft_p95_s",
            "tpot_mean_ms", "e2el_mean_s", "e2el_p95_s", "gpu_cache_hit", "cpu_cache_hit", "overall_cache_hit",
            "records_profiled", "records_error_dropped", "cpu_pool_tokens", "json"])
for label, node, pol, hc, wp, dm, role, p in runs:
    d = json.load(open(R + "/" + p)); r = d["request_metrics"]; L = r["latency"]
    sm = d.get("server_metrics") or {}; c = sm.get("cache", {}); kv = sm.get("kv_cache", {}); a = d.get("request_accounting", {})
    w.writerow([label, node, pol, hc, wp, dm, role, d.get("conc"), round(r["throughput"]["per_gpu"]["total_tput_tps"], 1),
                r["qps"]["mean"], round(r["tokens"]["input"]["mean"], 1), round(r["tokens"]["output_actual"]["mean"], 1),
                L["ttft"]["mean"], L["ttft"]["p50"], L["ttft"]["p95"], round(L["tpot"]["mean"] * 1000, 2), L["e2el"]["mean"], L["e2el"]["p95"],
                c.get("gpu_cache_hit_rate"), c.get("cpu_cache_hit_rate"), c.get("overall_cache_hit_rate"),
                a.get("records_profiled"), a.get("records_error_dropped"), kv.get("cpu_total_tokens"), R + "/" + p])
