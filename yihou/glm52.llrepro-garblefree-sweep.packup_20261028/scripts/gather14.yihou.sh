#!/usr/bin/env bash
# gather14 (task llrepro packup 2026-10-28, ON 135): stage llr37r, llr40-llr48, llr45r (decode flags, KF fused indexer, C108, garble-free settings sweep)
# into a NEW yihou stage dir. Read-only on sources.
set -uo pipefail
S=$1; D=/shared_nfs/yihou_glm52/llrepro_20261023; P=$D/prep_next.yihou
[[ "$S" == *yihou* && ! -e "$S" ]] || { echo "stage must be new and contain yihou"; exit 1; }
mkdir -p $S/results/llr $S/scripts $S/sites
RUNS="llr37r llr40 llr41 llr42 llr43 llr44 llr45 llr45r llr46 llr47 llr48"
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
cp -p $D/llr37r*.site.env $D/llr4[0-9].site.env $D/llr45r.site.env $D/llr48r.site.env $S/sites/ 2>/dev/null; mkdir -p $S/kfov_fused; cp -p $D/kfov_fused.yihou/overlay/yihou_kf/__init__.py $D/kfov_fused.yihou/overlay/yihou_kf/__init__.py.before-guard $S/kfov_fused/ 2>/dev/null; diff -u /shared_nfs/yihou_glm52/kf_e2e_20261023/kf/e2e/overlay/yihou_kf/__init__.py $D/kfov_fused.yihou/overlay/yihou_kf/__init__.py > $S/kfov_fused/module_path_alias.diff; mkdir -p $S/analysis; for r in llr37 llr37r llr44 llr45 llr45r llr46 llr47 llr48; do d=$(ls -d $D/work/runs/$r-c* | head -1); echo "$r $(grep -a -c \"Decode transfer failed\" $d/logs/yihou-llr-decode.log)"; done > $S/analysis/transfer_timeouts.txt; grep -a -m3 \"Decode transfer failed\" $D/work/runs/llr45r-c104/logs/yihou-llr-decode.log | cut -c1-600 > $S/analysis/transfer_timeout.sample.txt; cp -p $P/aiter_retune/prefill/decode_bf16_untuned_gemm.csv $S/analysis/prefill_untuned_gemm.llr45.csv 2>/dev/null
cp -p $D/OPT_LEDGER.md $S/
cp -p $D/queue_runs.yihou.sh $D/run_llr.yihou.sh $D/guard_llr.yihou.sh $D/llsum.yihou.py $D/bneck.yihou.py $D/qseries.yihou.py $D/gather13.yihou.sh $D/gather14.yihou.sh \
  /home/yihou/yihou_tmp/isl_series.yihou.py $D/kit/sglang_2p1d_glm5.2/engine/worker.sh $D/kit/sglang_2p1d_glm5.2/engine/common.sh $S/scripts/ 2>/dev/null
du -sh $S; find $S -type f -size +4M -exec ls -la {} \;
echo GATHER14_DONE
