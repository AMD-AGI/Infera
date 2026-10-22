import json,sys
R="/shared_nfs/yihou_glm52/v0521dbg_20261010/runs"
rows=[("x8d A1 C48 fresh","x8d/agentx-c48-A1/agentx_conc48.json"),("x8p A1 C48 fresh","x8p/agentx-c48-A1/agentx_conc48.json"),("tb56 A1 C56 fresh","tb56/agentx-c56-A1/agentx_conc56.json"),
("q1 A1 C44 fresh","q1/agentx-c44/agentx_conc44.json"),("q1 A2 C44 warm","q1/agentx-c44-A2/agentx_conc44.json"),("q1 A3 C48 warm","q1/agentx-c48-A3/agentx_conc48.json"),("q1 A4 C52 warm","q1/agentx-c52-A4/agentx_conc52.json")]+[tuple(a.split("=")) for a in sys.argv[1:]]
for n,p in rows:
    try: d=json.load(open(R+"/"+p))
    except Exception as e: print(n,"ERR",e); continue
    r=d["request_metrics"]; t=r["throughput"]; L=r["latency"]
    print("%-20s score %7.0f  qps %.3f  ISL %7.0f  TTFT %.2f/%.2f  TPOT %.1fms" % (n,t["per_gpu"]["total_tput_tps"],r["qps"]["mean"],r["tokens"]["input"]["mean"],L["ttft"]["mean"],L["ttft"]["p95"],L["tpot"]["mean"]*1000))
