#!/usr/bin/env bash
# gather10 (task llrepro packup 2026-10-26, ON 135): stage llr24 (C96, current best) + C96 resource probes + analysis
# llr21 config replicates, C88) into a NEW yihou stage dir. Read-only on sources. Full container logs are NOT copied.
set -uo pipefail
S=$1; D=/shared_nfs/yihou_glm52/llrepro_20261023
[[ "$S" == *yihou* && ! -e "$S" ]] || { echo "stage must be new and contain yihou"; exit 1; }
mkdir -p $S/results/llr $S/scripts
RUNS="llr24"
for r in $RUNS; do d=$(ls -d $D/work/runs/$r-c* | head -1); b=$(basename $d); o=$S/results/llr/$b; mkdir -p $o
  cp -p $d/engine-config.env $d/workers.json $o/ 2>/dev/null; cp -p $d/agentx/agentx_conc*.json $d/agentx/runtime.env $o/ 2>/dev/null
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
mkdir -p $S/analysis; cp -p $D/probe_c96.yihou/c96.* $S/analysis/ 2>/dev/null; cp -p $D/OPT_LEDGER.md $D/probe_mem.yihou.sh $S/ 2>/dev/null; for r in llr23-c88 llr24-c96; do python3 -I /home/yihou/yihou_tmp/isl_series.yihou.py /mnt/m2m_nobackup/yihou_llrepro/work/runs/$r/agentx/aiperf_artifacts/profile_export.jsonl > $S/analysis/isl_series.$r.txt 2>&1; done
cp -p $D/llr24.site.env $D/llr25.site.env /home/yihou/yihou_tmp/isl_series.yihou.py /home/yihou/jsonl_score.yihou.py $D/run_llr.yihou.sh $D/guard_llr.yihou.sh $D/prof_llr7r.yihou.sh \
  $D/node_probe1s.yihou.sh $D/qseries.yihou.py $D/bneck.yihou.py $D/llsum.yihou.py $D/argdiff.yihou.py $D/gather9.yihou.sh $D/kit/sglang_2p1d_glm5.2/engine/worker.sh $D/kit/sglang_2p1d_glm5.2/engine/common.sh $S/scripts/ 2>/dev/null
du -sh $S; find $S -type f -size +4M -exec ls -la {} \;
echo GATHER10_DONE
