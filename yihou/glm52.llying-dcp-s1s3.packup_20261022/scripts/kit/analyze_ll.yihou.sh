#!/usr/bin/env bash
# Post-run report for one llkit stage run. Runs ON THE COMPUTE NODE (spur exec), never on a login node:
# dumps the live containers' logs into RUN_DIR/logs-live/ (docker logs; containers untouched), then parses.
# Usage: bash analyze_ll.yihou.sh RUN_DIR CTR [CONC=80]  -> RUN_DIR/analysis.txt
set -uo pipefail
RD=$1; CTR=$2; CONC=${3:-80}
T=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/tools
A=$RD/agentx; L=$RD/logs-live; mkdir -p "$L"
for c in prefill0 decode; do docker logs --timestamps "$CTR-$c" > "$L/$c.log" 2>&1; done
{
echo "## $RD"
python3 - "$A/agentx_conc$CONC.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1])); m = j["request_metrics"]; l = m["latency"]
print("ok", j.get("num_requests_successful"), "accounting", j.get("request_accounting", {}).get("error_categories"),
      "err_dropped", j.get("request_accounting", {}).get("records_error_dropped"))
print("tok/s/GPU", round(m["throughput"]["per_gpu"]["total_tput_tps"]), "total", m["throughput"]["total"])
def q(k, *ps): return " ".join(f"{p}={round(l[k][p]*(1000 if k in ('tpot','itl') else 1), 2)}" for p in ps if p in l[k])
print("TTFT s:", q("ttft", "mean", "p50", "p90", "p95"))
print("TPOT ms:", q("tpot", "mean", "p50", "p90", "p95"))
if "itl" in l: print("ITL ms:", q("itl", "mean", "p50", "p90"))
PY
python3 $T/errphase.yihou.py $(ls $A/aiperf_artifacts/profile_export.jsonl 2>/dev/null || ls $A/*/profile_export.jsonl 2>/dev/null | head -1)
start=$(grep -m1 "Phase profiling (profiling) started" $A/runner.log | cut -c1-8)
day=$(date -u -r $A/runner.log +%F); s="${day}T$start"; e=$(date -u -d "$s Z + 1800 sec" +%FT%T)
echo "profiling window $s .. $e"
for c in prefill0 decode; do awk -v a="$s" -v b="$e" '$1>=a && $1<b' $L/$c.log | grep -aE "Prefill batch,|Decode batch," > $L/$c.win.log; done
echo "-- decode lines per rank:"; grep -oE "\[[0-9-]+ [0-9:]+ ([A-Z]+[0-9]+ ?)+\]" $L/decode.win.log | sed -E 's/\[[0-9-]+ [0-9:]+ //' | sort | uniq -c
echo "-- queue readout prefill (all PP ranks pooled)"; python3 $T/extract_pd_stats.py --prefill $L/prefill0.win.log --rank "" --run-min 1 2>&1 | sed -n '/\[prefill\]/,/^$/p'
echo "-- queue readout decode (all ranks pooled; per-rank for DP)"; python3 $T/extract_pd_stats.py --decode $L/decode.win.log --rank "" --run-min 1 2>&1 | sed -n '/\[decode\]/,/retracted/p'
for r in "DP0 TP0" "DP1 TP1" "DP2 TP2" "DP3 TP3" "TP0 EP0"; do
  grep -q " $r[] ]" $L/decode.win.log || continue
  echo "-- step p50/p90 by running, rank '$r'"; python3 $T/step_quantiles.yihou.py $L/decode.win.log --rank "$r"
done
echo "-- faults"; for c in prefill0 decode; do echo "$c $(grep -acE 'Memory access fault|Traceback|out of memory|Fatal Python' $L/$c.log)"; done
} 2>&1 | tee $RD/analysis.txt
