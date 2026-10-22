# Build results.csv for the prefill-mem / hicache-policy packup from the AgentX JSONs (read-only).
import json, sys, os
R = "/shared_nfs/yihou_glm52/v0521dbg_20261010/runs"
runs = [
 ("q1 A3 (ref)", "137", "q1", "0.88", "write_through", "warm(3rd)", "q1/agentx-c48-A3/agentx_conc48.json"),
 ("pf90 A1", "089", "pf90", "0.90", "write_through", "fresh", "pf90/agentx-c48-A1/agentx_conc48.json"),
 ("pf90 A2", "089", "pf90", "0.90", "write_through", "warm(scored)", "pf90/agentx-c48-A2/agentx_conc48.json"),
 ("pf92 A1", "085", "pf92", "0.92", "write_through", "fresh", "pf92/agentx-c48-A1/agentx_conc48.json"),
 ("pf92 A2", "085", "pf92", "0.92", "write_through", "LOST(NODE_FAIL)", "pf92/agentx-c48-A2/agentx_conc48.json"),
 ("pf88 A1", "274", "pf88", "0.88", "write_through", "LOST(NODE_FAIL)", "pf88/agentx-c48-A1/agentx_conc48.json"),
 ("wts48 A1", "136", "wts48", "0.88", "write_through_selective", "killed@~50min", "wts48/agentx-c48-A1/agentx_conc48.json"),
 ("wts48 A2", "136", "wts48", "0.88", "write_through_selective", "warm(scored;after partial A1)", "wts48/agentx-c48-A2/agentx_conc48.json"),
 ("wts64 (C64)", "136", "wts48", "0.88", "write_through_selective", "warm(3rd stack use)", "wts48/agentx-c64-A2/agentx_conc64.json"),
 ("pc48b A2 (ref)", "274", "pc48b", "0.88", "write_back(prefill_complete)", "warm(scored)", "pc48b/agentx-c48-A2/agentx_conc48.json"),
 ("pc64 A1", "089", "pc64", "0.88", "write_back(prefill_complete)", "fresh", "pc64/agentx-c64-A1/agentx_conc64.json"),
 ("pc64 A2", "089", "pc64", "0.88", "write_back(prefill_complete)", "ABORTED(user)", "pc64/agentx-c64-A2/agentx_conc64.json"),
]
print("run,node,round,prefill_mem,prefill_hicache_policy,stack,conc,score_tok_s_chip,qps,isl_mean,ttft_mean_s,ttft_p95_s,tpot_ms,gpu_hit,cpu_hit,overall_hit,ok,err,json")
for n, node, rd, pm, pol, st, p in runs:
    fp = os.path.join(R, p)
    if not os.path.exists(fp):
        print(",".join([n, node, rd, pm, pol, st] + [""] * 12 + ["(no JSON) " + fp])); continue
    d = json.load(open(fp)); r = d["request_metrics"]; L = r["latency"]
    if not r.get("throughput"):
        a = d.get("request_accounting", {})
        print(",".join(map(str, [n, node, rd, pm, pol, st, d.get("conc")] + [""] * 9 + [a.get("records_profiled"), a.get("records_error_dropped"), "(JSON has no metrics) " + fp]))); continue
    c = (d.get("server_metrics") or {}).get("cache", {}); a = d.get("request_accounting", {})
    f = lambda x: "" if x is None else "%.3f" % x
    print(",".join(map(str, [n, node, rd, pm, pol, st, d.get("conc"),
        "%.2f" % r["throughput"]["per_gpu"]["total_tput_tps"], "%.3f" % r["qps"]["mean"], "%.0f" % r["tokens"]["input"]["mean"],
        "%.2f" % L["ttft"]["mean"], "%.2f" % L["ttft"]["p95"], "%.2f" % (L["tpot"]["mean"] * 1000),
        f(c.get("gpu_cache_hit_rate")), f(c.get("cpu_cache_hit_rate")), f(c.get("overall_cache_hit_rate")),
        a.get("records_profiled"), a.get("records_error_dropped"), fp])))
