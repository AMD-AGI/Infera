#!/usr/bin/env bash
# summary (task dcpx packup 2026-10-22): rebuild results/summary.tsv. Usage: summary.yihou.sh <stage>
S=$1; R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; N=/shared_nfs/yihou_glm52/v0521cap2_20261013
ROUNDS="ppda1 ppda2 ppdc4 ppdc2 dpax1"
python3 -I - "$R" "$N/hist.yihou.py" $ROUNDS > "$S/results/summary.tsv" <<'PY'
import json, sys, glob, os, subprocess
R, hist = sys.argv[1], sys.argv[2]
keys = [("kvlat", "prefill|kv_transfer_latency_ms|", 1e-3), ("kvspeed", "prefill|kv_transfer_speed_gb_s|", 1),
        ("kvmb", "prefill|kv_transfer_total_mb|", 1), ("xfer_stage", "prefill|per_stage_req_latency_seconds|prefill_transfer_kv_cache", 1),
        ("pqueue", "prefill|queue_time_seconds|", 1), ("dtransferred", "decode|per_stage_req_latency_seconds|decode_transferred", 1)]
hdr = ["round", "bench", "conc", "score", "qps", "isl_mean", "ttft_p50", "ttft_p90", "tpot_ms", "gpu_hit", "cpu_hit", "err"]
for k, _, _ in keys: hdr += [f"{k}_p50", f"{k}_p90", f"{k}_avg"]
print("\t".join(hdr))
for r in sys.argv[3:]:
    for b in sorted(glob.glob(f"{R}/{r}/agentx-c*-A*")):
        js = glob.glob(f"{b}/agentx_conc*.json")
        if not js: continue
        d = json.load(open(js[0]))
        try:
            rm = d["request_metrics"]; c = d["server_metrics"]["cache"]; L = rm["latency"]; _ = rm["throughput"]["per_gpu"]["total_tput_tps"]
        except (KeyError, TypeError):
            print("\t".join([r, os.path.basename(b), str(d.get("conc")), "INCOMPLETE"])); continue
        row = [r, os.path.basename(b), str(d.get("conc")), "%.0f" % rm["throughput"]["per_gpu"]["total_tput_tps"],
               "%.3f" % rm["qps"]["mean"], "%.0f" % rm["tokens"]["input"]["mean"], "%.2f" % L["ttft"]["p50"],
               "%.2f" % L["ttft"]["p90"], "%.1f" % (L["tpot"]["mean"] * 1e3), "%.3f" % c["gpu_cache_hit_rate"],
               "%.3f" % c["cpu_cache_hit_rate"], str(d["request_accounting"]["records_error_dropped"])]
        csvs = glob.glob(f"{b}/**/server_metrics_export.csv", recursive=True)
        h = {}
        if csvs:
            out = subprocess.run(["python3", "-I", hist, r, csvs[0]], capture_output=True, text=True).stdout
            try: h = json.loads(out)
            except ValueError: h = {}
        for _, k, s in keys:
            v = h.get(k)
            row += ["%.2f" % (v[x] * s) if v else "" for x in ("p50", "p90", "avg")] if v else ["", "", ""]
        print("\t".join(row))
PY

