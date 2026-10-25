#!/usr/bin/env bash
# gather8 (task llrepro packup 2026-10-25, ON 135): stage llr7r..llr17 (trace replicate, ATOM/Dynamo-ranked experiments,
# replicate, chunk test) into a NEW yihou stage dir. Read-only on sources. Full container logs and the torch traces are
# NOT copied (paths in README); analysis outputs > 1 MB are gzipped.
set -uo pipefail
S=$1; D=/shared_nfs/yihou_glm52/llrepro_20261023
[[ "$S" == *yihou* && ! -e "$S" ]] || { echo "stage must be new and contain yihou"; exit 1; }
mkdir -p $S/results/llr $S/scripts $S/analysis/llr7r $S/analysis/probe_llr7r
RUNS="llr7r llr9c llr10 llr11 llr12 llr13 llr14 llr14r llr15 llr16 llr17"
for r in $RUNS; do d=$D/work/runs/$r-c80; o=$S/results/llr/$r-c80; mkdir -p $o
  cp -p $d/engine-config.env $d/workers.json $o/ 2>/dev/null; cp -p $d/agentx/agentx_conc80.json $d/agentx/runtime.env $o/ 2>/dev/null
  f=$(find $d/agentx -name server_metrics_export.csv | head -1)
  if [ -n "$f" ]; then gzip -c $f > $o/server_metrics_export.csv.gz
    python3 -I /shared_nfs/yihou_glm52/v0521cap2_20261013/hist.yihou.py $r $f > $o/hist.json 2>/dev/null; fi
  grep -a -E "KV Cache is allocated|Capture .* end|max_total_num_tokens|Memory access fault|HSA_STATUS|Traceback|disable_custom_all_reduce|EXPERIMENTAL|yihou-kf.*(installed|RAISED)" $d/logs/yihou-llr-decode.log \
    | grep -a -v "Decode batch" | cut -c1-400 | head -200 > $o/decode.keylines.txt
  grep -a -E "KV Cache is allocated|server_args|HSA_STATUS|Memory access fault" $d/logs/yihou-llr-prefill0.log | cut -c1-400 | head -40 > $o/prefill.keylines.txt
  grep -a -m1 "starting infera-router" $d/logs/yihou-llr-router.log | cut -c1-3000 > $o/router.startup.txt
  python3 -I $D/llsum.yihou.py $d 2>/dev/null | tail -1 >> $S/results/llr/summary.txt
  python3 -I $D/bneck.yihou.py $d > $o/bneck.txt 2>&1
  s=$(grep "Phase profiling (profiling) started" $D/run_llr.$r.yihou.log | head -1 | cut -c1-8)
  [ -n "$s" ] && python3 -I $D/qseries.yihou.py $d $s $(date -u -d "$s 60 min" +%T) 10 > $o/qseries.txt 2>&1
  cp -p $D/run_llr.$r.yihou.log $o/run_llr.log 2>/dev/null; cp -p $D/guard.$r.yihou.log $o/guard.log 2>/dev/null
done
sed -i '1i round score qps isl ttft_p50 ttft_p90 tpot gpu_hit cpu_hit err kvlat(p50/p90) xfer_stage pqueue dtransferred' $S/results/llr/summary.txt
python3 -I $D/qseries.yihou.py $D/work/runs/llr7-c80 03:13:05 04:13:05 5 > $S/results/llr/llr7-c80.qseries.txt
cp -p $D/llr7r.site.env $D/llr9*.site.env $D/llr1[0-7]*.site.env $D/run_llr.yihou.sh $D/guard_llr.yihou.sh $D/prof_llr7r.yihou.sh \
  $D/node_probe1s.yihou.sh $D/qseries.yihou.py $D/bneck.yihou.py $D/llsum.yihou.py $D/argdiff.yihou.py $D/gather8.yihou.sh $S/scripts/ 2>/dev/null
(cd $D/analysis_llr7r.yihou && find . -type f | while read -r f; do mkdir -p "$S/analysis/llr7r/$(dirname "$f")"
  if [ $(stat -c %s "$f") -gt 1000000 ]; then gzip -c "$f" > "$S/analysis/llr7r/$f.gz"; else cp -p "$f" "$S/analysis/llr7r/$f"; fi; done)
cp -p $D/trace_llr7r.yihou/probe.* $D/trace_llr7r.yihou/start_profile.*.json $S/analysis/probe_llr7r/
for k in 1 2 3; do gzip -c $D/trace_llr7r.yihou/pyspy.decode.$k.txt > $S/analysis/probe_llr7r/pyspy.decode.$k.txt.gz; done
mkdir -p $S/analysis/ec_port; cp -p $D/ec_port.yihou/*.diff $D/ec_port.yihou/README.md $S/analysis/ec_port/ 2>/dev/null
du -sh $S; find $S -type f -size +4M -exec ls -la {} \;
echo GATHER8_DONE
