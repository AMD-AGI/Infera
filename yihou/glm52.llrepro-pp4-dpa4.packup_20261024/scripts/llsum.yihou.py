# llsum (task llrepro 2026-10-23): one summary row (same columns as summary.dcpx) for a llying-kit run dir.
import json, sys, glob, subprocess
rd = sys.argv[1]; hist = "/shared_nfs/yihou_glm52/v0521cap2_20261013/hist.yihou.py"
d = json.load(open(glob.glob(f"{rd}/agentx/agentx_conc*.json")[0]))
reqm = d["request_metrics"]; cache = d["server_metrics"]["cache"]; lat = reqm["latency"]
row = [rd.rstrip("/").split("/")[-1], "%.0f" % reqm["throughput"]["per_gpu"]["total_tput_tps"], "%.3f" % reqm["qps"]["mean"],
       "%.0f" % reqm["tokens"]["input"]["mean"], "%.2f" % lat["ttft"]["p50"], "%.2f" % lat["ttft"]["p90"], "%.1f" % (lat["tpot"]["mean"] * 1e3),
       "%.3f" % cache["gpu_cache_hit_rate"], "%.3f" % cache["cpu_cache_hit_rate"], str(d["request_accounting"]["records_error_dropped"])]
csvs = glob.glob(f"{rd}/agentx/**/server_metrics_export.csv", recursive=True)
h = {}
if csvs:
    try: h = json.loads(subprocess.run(["python3", "-I", hist, "llr", csvs[0]], capture_output=True, text=True).stdout)
    except ValueError: pass
for k, s in [("prefill|kv_transfer_latency_ms|", 1e-3), ("prefill|per_stage_req_latency_seconds|prefill_transfer_kv_cache", 1),
             ("prefill|queue_time_seconds|", 1), ("decode|per_stage_req_latency_seconds|decode_transferred", 1)]:
    v = h.get(k); row += ["%.2f/%.2f" % (v["p50"] * s, v["p90"] * s) if v else "-"]
print("round score qps isl ttft_p50 ttft_p90 tpot gpu_hit cpu_hit err kvlat xfer_stage pqueue dtransferred")
print(" ".join(row))
