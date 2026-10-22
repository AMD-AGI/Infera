# Build results/results.csv from the AgentX JSONs inside this packup (run: python3 -I make_results_csv.yihou.py <packup>)
import json, sys, os, csv
P = sys.argv[1]
rows = [
 ("tb56","A1",56,"runshare v2 0.5/5s","fresh (unscored)","results/tb56/agentx-c56-A1/agentx_conc56.json"),
 ("tb56","A2",56,"runshare v2 0.5/5s","warm (scored)","results/tb56/agentx-c56-A2/agentx_conc56.json"),
 ("tb56","A3",64,"runshare v2 0.5/5s","warm 3rd run","results/tb56/agentx-c64-A3/agentx_conc64.json"),
 ("tb56n","A1",56,"none","fresh (unscored)","results/tb56n/agentx-c56-A1/agentx_conc56.json"),
 ("tb56n","A2",56,"none","warm (scored)","results/tb56n/agentx-c56-A2/agentx_conc56.json"),
 ("tb56n","A3",64,"none","warm 3rd run","results/tb56n/agentx-c64-A3/agentx_conc64.json"),
 ("ref:q1-137","A2",44,"none (HiCache 450)","warm","results/reference/q1-137/q1-A2-C44.agentx_conc44.json"),
 ("ref:q1-137","A3",48,"none (HiCache 450)","warm 3rd run","results/reference/q1-137/q1-A3-C48.agentx_conc48.json"),
 ("ref:q1-137","A4",52,"none (HiCache 450)","warm 4th run","results/reference/q1-137/q1-A4-C52.agentx_conc52.json"),
 ("ref:pc40d-102","A1",40,"runshare v2 0.5/5s (HiCache 450)","fresh","results/reference/pc40d-A1-C40-runshare2/agentx_conc40.json"),
]+[("ref:t2f-v0519-crossnode","sweep",c,"none (v0.5.19 cross-node)","sweep point","results/reference/t2f-v0519-crossnode/agentx_conc%d.json"%c) for c in (40,56,72,96)]
cols = ["round","run","conc","patch","protocol","score_tok_s_chip","qps","isl_mean","ttft_mean_s","ttft_p50_s","ttft_p95_s","tpot_mean_ms","gpu_hit","cpu_hit","overall_hit","records_ok","records_err","cpu_pool_tok","json"]
out = csv.writer(open(os.path.join(P,"results","results.csv"),"w",newline=""))
out.writerow(cols)
for rnd,run,c,patch,prot,p in rows:
    d = json.load(open(os.path.join(P,p)))
    r = d["request_metrics"]; L = r["latency"]; sm = d.get("server_metrics") or {}
    ca = sm.get("cache") or {}; kv = sm.get("kv_cache") or {}; a = d.get("request_accounting") or {}
    def f(v, k=3): return "" if v is None else round(v, k)
    tp = L.get("tpot") or {}
    out.writerow([rnd,run,c,patch,prot, f(r["throughput"]["per_gpu"]["total_tput_tps"],1), f(r["qps"]["mean"]),
        f(r["tokens"]["input"]["mean"],0),
        f(L["ttft"]["mean"],2), f(L["ttft"].get("p50"),2), f(L["ttft"].get("p95"),2),
        f(None if tp.get("mean") is None else tp["mean"]*1000,2),
        f(ca.get("gpu_cache_hit_rate")), f(ca.get("cpu_cache_hit_rate")), f(ca.get("overall_cache_hit_rate")),
        a.get("records_profiled",""), a.get("records_error_dropped",""), kv.get("cpu_total_tokens",""), p])

