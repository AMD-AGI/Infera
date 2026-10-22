#!/usr/bin/env python3
"""Extract the AgentX metrics of the HiCache-sensitivity runs into one CSV (stdout).
Usage: python3 -B extract_results.yihou.py > results.csv   (reads the original JSONs on /shared_nfs and /home)."""
import json, csv, sys
R = "/shared_nfs/yihou_glm52/v0521dbg_20261010/runs"
B = "/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-hicache-ceiling-sweep/glm52.hicache-ceiling.packup_20260924/results"
# label, json, node, image, model, stack, prefill_hicache_gb_per_rank, prefill_mem, decode_mem, like_for_like_vs_hc50
RUNS = [
 ("hc50 A1", R+"/hc50/agentx-c40-A1/agentx_conc40.json", "136", "9872bbda", "GLM-5.2-MXFP4 256e", "fresh", 50, 0.85, 0.95, "self"),
 ("hc50 A2", R+"/hc50/agentx-c40-A2/agentx_conc40.json", "136", "9872bbda", "GLM-5.2-MXFP4 256e", "warm (flushed)", 50, 0.85, 0.95, "self"),
 ("e2g A1", R+"/e2g/agentx-c40/agentx_conc40.json", "102", "9d755dd6", "GLM-5.2-MXFP4 256e", "fresh", 450, 0.85, 0.95, "yes except node+image build"),
 ("e2g A2", R+"/e2g/agentx-c40-A2/agentx_conc40.json", "102", "9d755dd6", "GLM-5.2-MXFP4 256e", "warm (flushed)", 450, 0.85, 0.95, "yes except node+image build"),
 ("k1 A1", R+"/k1/agentx-c44/agentx_conc44.json", "102", "9d755dd6", "GLM-5.2-MXFP4 256e", "fresh", 360, 0.85, 0.95, "no: C44"),
 ("k1 A2", R+"/k1/agentx-c44-A2/agentx_conc44.json", "102", "9d755dd6", "GLM-5.2-MXFP4 256e", "warm (flushed)", 360, 0.85, 0.95, "no: C44"),
 ("e1c r2", R+"/e1c/agentx-c40-r2/agentx_conc40.json", "102", "9d755dd6", "GLM-5.2-MXFP4 256e", "fresh (r1 died at start)", 153.81, 0.85, 0.85, "no: decode mem 0.85, E1 config"),
 ("8e armA S154 C56", B+"/armA-reference-S154/armA-136-agentx_conc56.json", "136", "v0519 nextnfix-hicache", "synthetic 8-expert", "separate run", 154, 0.85, 0.85, "no: v0.5.19, 8-expert, C56"),
 ("8e r001-hc274 C56", B+"/r001-hc274/agentx_conc56.json", "see ceiling packup", "v0519 nextnfix-hicache", "synthetic 8-expert", "separate run", 274, 0.85, 0.85, "no: v0.5.19, 8-expert, C56"),
 ("8e r001-hc450chunk512 C56", B+"/r001-hc450chunk512/agentx_conc56.json", "see ceiling packup", "v0519 nextnfix-hicache", "synthetic 8-expert", "separate run", 450, 0.85, 0.85, "no: v0.5.19, 8-expert, C56"),
]
F = ["run","conc","node","image","model","stack","prefill_hicache_gb_per_rank","prefill_mem","decode_mem","like_for_like_vs_hc50",
     "score_tok_s_chip","qps","isl_mean","osl_actual_mean","ttft_mean_s","ttft_p50_s","ttft_p95_s","tpot_mean_ms","e2el_mean_s",
     "gpu_hit","cpu_hit","overall_hit","theoretical_hit","prompt_total","prompt_gpu_hit_tok","prompt_cpu_hit_tok","prompt_computed",
     "cpu_total_tokens","gpu_total_tokens","records_profiled","records_error","duration_s","json"]


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


w = csv.writer(sys.stdout)
w.writerow(F)
for lab, p, node, img, model, stack, hc, pm, dm, lfl in RUNS:
    try:
        d = json.load(open(p))
    except Exception as e:
        w.writerow([lab, "ERR %s" % e])
        continue
    met = d.get("request_metrics", {})
    sm = d.get("server_metrics") or {}
    tp = g(met, "latency", "tpot", "mean")
    w.writerow([lab, d.get("conc"), node, img, model, stack, hc, pm, dm, lfl,
        round(g(met, "throughput", "per_gpu", "total_tput_tps"), 1), g(met, "qps", "mean"),
        round(g(met, "tokens", "input", "mean") or 0), round(g(met, "tokens", "output_actual", "mean") or 0),
        g(met, "latency", "ttft", "mean"), g(met, "latency", "ttft", "p50"), g(met, "latency", "ttft", "p95"),
        None if tp is None else round(tp * 1000, 2), g(met, "latency", "e2el", "mean"),
        g(sm, "cache", "gpu_cache_hit_rate"), g(sm, "cache", "cpu_cache_hit_rate"), g(sm, "cache", "overall_cache_hit_rate"),
        g(met, "cache", "theoretical_cache_hit_rate"),
        g(sm, "tokens", "prompt_total"), g(sm, "tokens", "prompt_by_source", "gpu_cache_hit"),
        g(sm, "tokens", "prompt_by_source", "cpu_or_external_cache_hit"), g(sm, "tokens", "prompt_by_source", "computed"),
        g(sm, "kv_cache", "cpu_total_tokens"), g(sm, "kv_cache", "gpu_total_tokens"),
        g(d, "request_accounting", "records_profiled"), g(d, "request_accounting", "records_error_dropped"),
        g(met, "throughput", "duration_seconds"), p])
