import json
R="/shared_nfs/yihou_glm52/v0521dbg_20261010/runs"
runs=[("q1 A3 (ref)","q1/agentx-c48-A3/agentx_conc48.json"),("e2g A2 (ref)","e2g/agentx-c40-A2/agentx_conc40.json"),
("x8d A1","x8d/agentx-c48-A1/agentx_conc48.json"),
("x8p A1","x8p/agentx-c48-A1/agentx_conc48.json"),("x8p A2","x8p/agentx-c48-A2/agentx_conc48.json"),
("x8d2 A1","x8d2/agentx-c48-A1/agentx_conc48.json"),("x8d2 A2","x8d2/agentx-c48-A2/agentx_conc48.json"),("x8d2 A3","x8d2/agentx-c52-A3/agentx_conc52.json"),
("x8pc A1","x8pc/agentx-c48-A1/agentx_conc48.json"),("x8pc A2","x8pc/agentx-c48-A2/agentx_conc48.json"),
("x128p A1","x128p/agentx-c48-A1/agentx_conc48.json"),("x128p A2","x128p/agentx-c48-A2/agentx_conc48.json"),
("tb56 A1","tb56/agentx-c56-A1/agentx_conc56.json"),("tb56 A2","tb56/agentx-c56-A2/agentx_conc56.json"),("tb56 A3","tb56/agentx-c64-A3/agentx_conc64.json"),
("tb56n A1","tb56n/agentx-c56-A1/agentx_conc56.json"),("tb56n A2","tb56n/agentx-c56-A2/agentx_conc56.json"),("tb56n A3","tb56n/agentx-c64-A3/agentx_conc64.json"),
("pc48b A1","pc48b/agentx-c48-A1/agentx_conc48.json"),("pc48b A2","pc48b/agentx-c48-A2/agentx_conc48.json"),
("pc48e A1","pc48e/agentx-c48-A1/agentx_conc48.json"),("pc48e A2","pc48e/agentx-c48-A2/agentx_conc48.json"),
("hc50 A1","hc50/agentx-c40-A1/agentx_conc40.json"),("hc50 A2","hc50/agentx-c40-A2/agentx_conc40.json")]
print("run|C|score|qps|ISL|TTFTmean|TTFTp95|TPOTms|gpu_hit|cpu_hit|overall_hit|ok|err|cpu_pool_tok")
for n,p in runs:
    try: d=json.load(open(R+"/"+p))
    except Exception as e: print(n,"|ERR",e); continue
    r=d["request_metrics"]; L=r["latency"]; c=(d.get("server_metrics") or {}).get("cache",{}); kv=(d.get("server_metrics") or {}).get("kv_cache",{})
    a=d.get("request_accounting",{})
    f=lambda x: "-" if x is None else "%.3f"%x
    print("%s|%s|%.0f|%.3f|%.0f|%.2f|%.2f|%.1f|%s|%s|%s|%s|%s|%s"%(n,d.get("conc"),r["throughput"]["per_gpu"]["total_tput_tps"],r["qps"]["mean"],r["tokens"]["input"]["mean"],L["ttft"]["mean"],L["ttft"]["p95"],L["tpot"]["mean"]*1000,f(c.get("gpu_cache_hit_rate")),f(c.get("cpu_cache_hit_rate")),f(c.get("overall_cache_hit_rate")),a.get("records_profiled"),a.get("records_error_dropped"),kv.get("cpu_total_tokens")))
