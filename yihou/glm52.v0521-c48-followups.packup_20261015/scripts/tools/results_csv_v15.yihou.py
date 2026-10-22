# Build results.csv from AgentX JSONs (read-only). Usage: results_csv_v15.yihou.py offload|followups
import json, sys, os
R = "/shared_nfs/yihou_glm52/v0521dbg_20261010/runs"
SETS = {
 "offload": [
  ("pf90 A1 (ref)", "089", "pf90", "none", "fresh", "pf90/agentx-c48-A1/agentx_conc48.json"),
  ("pf90 A2 (ref)", "089", "pf90", "none", "warm(scored)", "pf90/agentx-c48-A2/agentx_conc48.json"),
  ("ofs0 A1", "089", "ofs0", "G8K1 (1/8, 9 layers)", "fresh", "ofs0/agentx-c48-A1/agentx_conc48.json"),
  ("ofs0 A2", "089", "ofs0", "G8K1 (1/8, 9 layers)", "warm(scored)", "ofs0/agentx-c48-A2/agentx_conc48.json"),
  ("ofg4 A1", "252", "ofg4", "G4K1 (1/4, 19 layers)", "fresh", "ofg4/agentx-c48-A1/agentx_conc48.json"),
  ("ofg4 A2", "252", "ofg4", "G4K1 (1/4, 19 layers)", "warm(scored)", "ofg4/agentx-c48-A2/agentx_conc48.json"),
  ("ofg4 A3 C56", "252", "ofg4", "G4K1 (1/4, 19 layers)", "warm", "ofg4/agentx-c56-A3/agentx_conc56.json"),
  ("ofg2 A1", "288", "ofg2", "G2K1 (1/2, 39 layers)", "fresh", "ofg2/agentx-c48-A1/agentx_conc48.json"),
  ("ofg2 A2", "288", "ofg2", "G2K1 (1/2, 39 layers)", "warm(scored)", "ofg2/agentx-c48-A2/agentx_conc48.json"),
  ("ofg2 A3 C56", "288", "ofg2", "G2K1 (1/2, 39 layers)", "warm", "ofg2/agentx-c56-A3/agentx_conc56.json"),
 ],
 "followups": [
  ("pf90 A1 (ref)", "089", "pf90", "pf90", "fresh", "pf90/agentx-c48-A1/agentx_conc48.json"),
  ("pf90 A2 (ref)", "089", "pf90", "pf90", "warm(scored)", "pf90/agentx-c48-A2/agentx_conc48.json"),
  ("tw90 A1", "089", "tw90", "pf90 + decode --tokenizer-worker-num 4", "fresh", "tw90/agentx-c48-A1/agentx_conc48.json"),
  ("tw90 A2", "089", "tw90", "pf90 + decode --tokenizer-worker-num 4", "warm(scored)", "tw90/agentx-c48-A2/agentx_conc48.json"),
  ("wts48 A2 (ref)", "136", "wts48", "write_through_selective C48", "warm(scored)", "wts48/agentx-c48-A2/agentx_conc48.json"),
  ("wts64 (C64)", "136", "wts48", "write_through_selective C64", "warm(3rd stack use)", "wts48/agentx-c64-A2/agentx_conc64.json"),
 ]}
print("run,node,round,variant,stack,conc,score_tok_s_chip,qps,isl_mean,ttft_mean_s,ttft_p95_s,tpot_ms,gpu_hit,cpu_hit,overall_hit,ok,err,json")
for n, node, rd, var, st, p in SETS[sys.argv[1]]:
    fp = os.path.join(R, p)
    if not os.path.exists(fp):
        print(",".join([n, node, rd, '"%s"' % var, st] + [""] * 12 + ["(not available at packing) " + p])); continue
    d = json.load(open(fp)); r = d["request_metrics"]; L = r["latency"]
    c = (d.get("server_metrics") or {}).get("cache", {}); a = d.get("request_accounting", {})
    f = lambda x: "" if x is None else "%.3f" % x
    ttft = L.get("ttft", {}); tpot = L.get("tpot", {})
    print(",".join(map(str, [n, node, rd, '"%s"' % var, st, d.get("conc"),
        "%.2f" % r["throughput"]["per_gpu"]["total_tput_tps"], "%.3f" % r["qps"]["mean"], "%.0f" % r["tokens"]["input"]["mean"],
        f(ttft.get("mean")), f(ttft.get("p95")), f((tpot.get("mean") or 0) * 1000 if (tpot.get("mean") or 0) < 1 else tpot.get("mean")),
        f(c.get("gpu_cache_hit_rate")), f(c.get("cpu_cache_hit_rate")), f(c.get("overall_cache_hit_rate", c.get("cache_hit_rate"))),
        a.get("records_profiled", ""), a.get("records_error_dropped", ""), p])))
