#!/usr/bin/env bash
# gather13 (task llrepro packup 2026-10-27 07Z, ON 135): stage llr37-llr39 (concurrency C104/C112, prefill chunk 16384)
# into a NEW yihou stage dir. Read-only on sources.
set -uo pipefail
S=$1; D=/shared_nfs/yihou_glm52/llrepro_20261023; P=$D/prep_next.yihou
[[ "$S" == *yihou* && ! -e "$S" ]] || { echo "stage must be new and contain yihou"; exit 1; }
mkdir -p $S/results/llr $S/scripts $S/sites
RUNS="llr37 llr38 llr39"
for r in $RUNS; do d=$(ls -d $D/work/runs/$r-c* | head -1); b=$(basename $d); o=$S/results/llr/$b; mkdir -p $o
  cp -p $d/engine-config.env $d/workers.json $o/ 2>/dev/null; cp -p $d/agentx/agentx_conc*.json $d/agentx/runtime.env $o/ 2>/dev/null
  f=$(find $d/agentx -name server_metrics_export.csv | head -1)
  if [ -n "$f" ]; then gzip -c $f > $o/server_metrics_export.csv.gz
    python3 -I /shared_nfs/yihou_glm52/v0521cap2_20261013/hist.yihou.py $r $f > $o/hist.json 2>/dev/null; fi
  grep -a -E "KV Cache is allocated|Capture .* end|max_total_num_tokens|Memory access fault|HSA_STATUS|Traceback|disable_custom_all_reduce|EXPERIMENTAL|PTPC|yihou-kf.*(installed|RAISED)|accept len" $d/logs/yihou-llr-decode.log \
    | grep -a -v "Decode batch" | cut -c1-400 | head -200 > $o/decode.keylines.txt
  grep -a "Decode batch" $d/logs/yihou-llr-decode.log | tail -20 | cut -c1-400 > $o/decode.lastbatches.txt
  grep -a -E "KV Cache is allocated|server_args|HSA_STATUS|Memory access fault" $d/logs/yihou-llr-prefill0.log | cut -c1-400 | head -40 > $o/prefill.keylines.txt
  python3 -I $D/llsum.yihou.py $d 2>/dev/null | tail -1 >> $S/results/llr/summary.txt
  python3 -I $D/bneck.yihou.py $d > $o/bneck.txt 2>&1
  s=$(grep "Phase profiling (profiling) started" $D/run_llr.$r.yihou.log | head -1 | cut -c1-8)
  [ -n "$s" ] && python3 -I $D/qseries.yihou.py $d $s $(date -u -d "$s 60 min" +%T) 10 > $o/qseries.txt 2>&1
  j=$(find $d/agentx -name profile_export.jsonl | head -1); [ -n "$j" ] && python3 -I /home/yihou/yihou_tmp/isl_series.yihou.py $j > $o/isl_series.txt 2>&1
  cp -p $D/run_llr.$r.yihou.log $o/run_llr.log 2>/dev/null; cp -p $D/guard.$r.yihou.log $o/guard.log 2>/dev/null
done
sed -i '1i round score qps isl ttft_p50 ttft_p90 tpot gpu_hit cpu_hit err kvlat(p50/p90) xfer_stage pqueue dtransferred' $S/results/llr/summary.txt
cp -p $D/llr3[7-9].site.env $D/llr4[01].site.env $D/llr37r.site.env $S/sites/ 2>/dev/null
cp -p $D/OPT_LEDGER.md $S/; mkdir -p $S/nfs_cleanup; cp -p /shared_nfs/yihou_glm52/du_audit.yihou/{REPORT.md,cleanup_run.log,cleanup_run.yihou.sh,gen_cmds.yihou.py,validate.yihou.sh,du_after.txt,genlists.py} $S/nfs_cleanup/ 2>/dev/null; for f in del_step1 del_step2 gzip_step3 del_step4_root; do gzip -c /shared_nfs/yihou_glm52/du_audit.yihou/$f.yihou.list > $S/nfs_cleanup/$f.yihou.list.gz; done
cp -p $D/queue_runs.yihou.sh $D/run_llr.yihou.sh $D/guard_llr.yihou.sh $D/llsum.yihou.py $D/bneck.yihou.py $D/qseries.yihou.py $D/gather12.yihou.sh $D/gather13.yihou.sh \
  /home/yihou/yihou_tmp/isl_series.yihou.py $D/kit/sglang_2p1d_glm5.2/engine/worker.sh $D/kit/sglang_2p1d_glm5.2/engine/common.sh $S/scripts/ 2>/dev/null
du -sh $S; find $S -type f -size +4M -exec ls -la {} \;
echo GATHER13_DONE
