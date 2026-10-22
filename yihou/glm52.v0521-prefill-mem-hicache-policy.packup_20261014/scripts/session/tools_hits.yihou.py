import json,sys
R="/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/"
for a in sys.argv[1:]:
    n,p=a.split("=")
    try:
        d=json.load(open(R+p)); r=d["request_metrics"]; L=r["latency"]; c=d["server_metrics"]["cache"]
        print("%-16s score %7.0f qps %.3f ISL %7.0f TTFT %.2f/%.2f TPOT %.1f gpu %.3f cpu %.3f err %s"%(n,r["throughput"]["per_gpu"]["total_tput_tps"],r["qps"]["mean"],r["tokens"]["input"]["mean"],L["ttft"]["mean"],L["ttft"]["p95"],L["tpot"]["mean"]*1000,c["gpu_cache_hit_rate"] or 0,c["cpu_cache_hit_rate"] or 0,d["request_accounting"]["records_error_dropped"]))
    except Exception as e: print(n,"n/a")
