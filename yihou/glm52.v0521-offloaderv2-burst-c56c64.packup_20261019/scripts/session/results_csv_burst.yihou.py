# Build results.csv for glm52.v0521-offloaderv2-burst-c56c64.packup_20261019 from AgentX JSONs (read-only).
# Usage: python3 results_csv_burst.yihou.py [unofficial_root]
import json, sys, os
R = "/shared_nfs/yihou_glm52/v0521dbg_20261010/runs"
U = sys.argv[1] if len(sys.argv) > 1 else "/shared_nfs/yihou_glm52/v0521cap2_20261013/unofficial"
T4 = " + decode tok4/detok4"
ROWS = [
 # (run, node, round, variant, stack, json path relative to R, or absolute)
 ("pf90 A1 (ref; no offload)", "089", "pf90", "none; HiCache 450; mem 0.90", "fresh", "pf90/agentx-c48-A1/agentx_conc48.json"),
 ("pf90 A2 (ref; no offload)", "089", "pf90", "none; HiCache 450; mem 0.90", "warm(scored)", "pf90/agentx-c48-A2/agentx_conc48.json"),
 ("ofs0 A2 (earlier kit)", "089", "ofs0", "G8K1 1/8", "warm(scored)", "ofs0/agentx-c48-A2/agentx_conc48.json"),
 ("ofg2 A1 (earlier kit)", "288", "ofg2", "G2K1 1/2", "fresh", "ofg2/agentx-c48-A1/agentx_conc48.json"),
 ("ofg3b A2 (earlier kit)", "v3-020", "ofg3b", "G3K1 1/3", "warm(scored)", "ofg3b/agentx-c48-A2/agentx_conc48.json"),
 ("ofg4b A2 (earlier kit)", "136", "ofg4b", "G4K1 1/4", "warm(scored)", "ofg4b/agentx-c48-A2/agentx_conc48.json"),
 ("ofg4b A3 C56 (earlier kit)", "136", "ofg4b", "G4K1 1/4", "warm(3rd)", "ofg4b/agentx-c56-A3/agentx_conc56.json"),
 ("tb56n A2 C56 (ref; NOT like-for-like)", "085", "tb56n", "none; HiCache ~154 GB; mem 0.88", "warm", "tb56n/agentx-c56-A2/agentx_conc56.json"),
 ("tb56n A3 C64 (ref; NOT like-for-like)", "085", "tb56n", "none; HiCache ~154 GB; mem 0.88", "warm(3rd)", "tb56n/agentx-c64-A3/agentx_conc64.json"),
 ("ofg2h5 A1 C56", "v3-008", "ofg2h5", "G2K1 1/2" + T4, "fresh", "ofg2h5/agentx-c56-A1/agentx_conc56.json"),
 ("ofg3h2 A1 C56", "v3-010", "ofg3h2", "G3K1 1/3" + T4, "fresh", "ofg3h2/agentx-c56-A1/agentx_conc56.json"),
 ("ofg3h2 A2 C56", "v3-010", "ofg3h2", "G3K1 1/3" + T4, "warm(scored)", "ofg3h2/agentx-c56-A2/agentx_conc56.json"),
 ("ofg2h17 A1 C56", "v3-006", "ofg2h17", "G2K1 1/2" + T4, "fresh", "ofg2h17/agentx-c56-A1/agentx_conc56.json"),
 ("ofg2h21 A1 C56", "v3-006", "ofg2h21", "G2K1 1/2" + T4, "fresh", "ofg2h21/agentx-c56-A1/agentx_conc56.json"),
 ("ofg2h21 A2 C56", "v3-006", "ofg2h21", "G2K1 1/2" + T4, "warm(scored)", "ofg2h21/agentx-c56-A2/agentx_conc56.json"),
 ("ofg3h14 A1 C64", "v3-006", "ofg3h14", "G3K1 1/3" + T4, "fresh", "ofg3h14/agentx-c64-A1/agentx_conc64.json"),
 ("ofg3h14 A2 C64", "v3-006", "ofg3h14", "G3K1 1/3" + T4, "warm(scored)", "ofg3h14/agentx-c64-A2/agentx_conc64.json"),
 ("ofg2h22 A1 C64", "v3-006", "ofg2h22", "G2K1 1/2" + T4, "fresh", "ofg2h22/agentx-c64-A1/agentx_conc64.json"),
 ("ofg2h22 A2 C64", "v3-006", "ofg2h22", "G2K1 1/2" + T4, "warm(scored)", "ofg2h22/agentx-c64-A2/agentx_conc64.json"),
 ("ofg4h4 A1 C64 UNOFFICIAL (1 NUL line dropped)", "v3-006", "ofg4h4", "G4K1 1/4" + T4, "fresh", U + "/ofg4h4-A1-nulline-dropped/agentx_conc64.json"),
 ("ofg4h4 A2 C64 UNOFFICIAL (1 NUL line dropped; cont_bench)", "v3-006", "ofg4h4", "G4K1 1/4" + T4, "warm", U + "/ofg4h4-A2-nulline-dropped/agentx_conc64.json"),
 ("ofg4h5 A1 C64", "v3-006", "ofg4h5", "G4K1 1/4" + T4, "fresh", "ofg4h5/agentx-c64-A1/agentx_conc64.json"),
 ("ofg4h5 A2 C64", "v3-006", "ofg4h5", "G4K1 1/4" + T4, "warm(scored)", "ofg4h5/agentx-c64-A2/agentx_conc64.json"),
]
print("run,node,round,variant,stack,conc,score_tok_s_chip,qps,isl_mean,ttft_mean_s,ttft_p95_s,tpot_ms,gpu_hit,cpu_hit,overall_hit,ok,err,json")
for n, node, rd, var, st, p in ROWS:
    fp = p if p.startswith("/") else os.path.join(R, p)
    if not os.path.exists(fp):
        print(",".join([n, node, rd, '"%s"' % var, st] + [""] * 12 + ["(missing) " + p])); continue
    d = json.load(open(fp)); r = d["request_metrics"]; L = r["latency"]
    c = (d.get("server_metrics") or {}).get("cache", {}); a = d.get("request_accounting", {})
    f = lambda x: "" if x is None else "%.3f" % x
    ttft = L.get("ttft", {}); tpot = L.get("tpot", {})
    print(",".join(map(str, [n, node, rd, '"%s"' % var, st, d.get("conc"),
        "%.2f" % r["throughput"]["per_gpu"]["total_tput_tps"], "%.3f" % r["qps"]["mean"], "%.0f" % r["tokens"]["input"]["mean"],
        f(ttft.get("mean")), f(ttft.get("p95")), f((tpot.get("mean") or 0) * 1000 if (tpot.get("mean") or 0) < 1 else tpot.get("mean")),
        f(c.get("gpu_cache_hit_rate")), f(c.get("cpu_cache_hit_rate")), f(c.get("overall_cache_hit_rate", c.get("cache_hit_rate"))),
        a.get("records_profiled", ""), a.get("records_error_dropped", ""), p])))
