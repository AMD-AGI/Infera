# Build results.csv from AgentX JSONs (read-only). Usage: results_csv_v15.yihou.py offload|followups|highconc
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

SETS["highconc"] = [
  ("pf90 A1 (ref)", "089", "pf90", "none; HiCache 450; mem 0.90", "fresh", "pf90/agentx-c48-A1/agentx_conc48.json"),
  ("pf90 A2 (ref)", "089", "pf90", "none; HiCache 450; mem 0.90", "warm(scored)", "pf90/agentx-c48-A2/agentx_conc48.json"),
  ("tb56n A2 C56 (ref; NOT like-for-like)", "085", "tb56n", "none; HiCache ratio 1.5 (~154 GB); mem 0.88", "warm", "tb56n/agentx-c56-A2/agentx_conc56.json"),
  ("tb56 A2 C56 (ref; NOT like-for-like)", "085", "tb56", "runshare v2; HiCache ~154 GB; mem 0.88", "warm", "tb56/agentx-c56-A2/agentx_conc56.json"),
  ("ofg2 A1", "288", "ofg2", "G2K1 1/2", "fresh", "ofg2/agentx-c48-A1/agentx_conc48.json"),
  ("ofg2 A2", "288", "ofg2", "G2K1 1/2", "ABORTED (oai released)", "ofg2/agentx-c48-A2/agentx_conc48.json"),
  ("ofg4 A1", "252", "ofg4", "G4K1 1/4", "ABORTED (oai released)", "ofg4/agentx-c48-A1/agentx_conc48.json"),
  ("ofg4b A1", "136", "ofg4b", "G4K1 1/4", "fresh", "ofg4b/agentx-c48-A1/agentx_conc48.json"),
  ("ofg4b A2", "136", "ofg4b", "G4K1 1/4", "warm(scored)", "ofg4b/agentx-c48-A2/agentx_conc48.json"),
  ("ofg4b A3 C56", "136", "ofg4b", "G4K1 1/4", "warm(3rd)", "ofg4b/agentx-c56-A3/agentx_conc56.json"),
  ("ofg3b A1", "v3-020", "ofg3b", "G3K1 1/3", "fresh", "ofg3b/agentx-c48-A1/agentx_conc48.json"),
  ("ofg3b A2", "v3-020", "ofg3b", "G3K1 1/3", "warm(scored)", "ofg3b/agentx-c48-A2/agentx_conc48.json"),
  ("ofg3b A3 C56", "v3-020", "ofg3b", "G3K1 1/3", "warm(3rd)", "ofg3b/agentx-c56-A3/agentx_conc56.json"),
  ("ofg2h3 A1 C56", "v3-026", "ofg2h3", "G2K1 1/2 + decode tok4/detok4", "fresh", "ofg2h3/agentx-c56-A1/agentx_conc56.json"),
  ("ofg2h3 A2 C56", "v3-026", "ofg2h3", "G2K1 1/2 + decode tok4/detok4", "warm(scored)", "ofg2h3/agentx-c56-A2/agentx_conc56.json"),
  ("ofg2h3 C64", "v3-026", "ofg2h3", "G2K1 1/2 + decode tok4/detok4", "warm(3rd)", "ofg2h3/agentx-c64-A2/agentx_conc64.json"),
  ("ofg4h A1 C56", "136", "ofg4h", "G4K1 1/4 + decode tok4/detok4", "fresh", "ofg4h/agentx-c56-A1/agentx_conc56.json"),
  ("ofg4h A2 C56", "136", "ofg4h", "G4K1 1/4 + decode tok4/detok4", "warm(scored)", "ofg4h/agentx-c56-A2/agentx_conc56.json"),
  ("ofg4h A3 C64", "136", "ofg4h", "G4K1 1/4 + decode tok4/detok4", "warm(3rd)", "ofg4h/agentx-c64-A3/agentx_conc64.json"),
]
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
