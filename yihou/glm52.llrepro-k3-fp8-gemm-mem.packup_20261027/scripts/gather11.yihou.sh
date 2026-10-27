#!/usr/bin/env bash
# gather11 (task llrepro packup 2026-10-26 PM, ON 135): stage llr25-llr28 (graph cap 80 OOM, draft FP8, MTP K3 = new best,
# MLA bmm configs) + side work (real MTP acceptance on pool node 190, MXFP4 draft patch, MLA bmm microbench, aiter GEMM
# retune) into a NEW yihou stage dir. Read-only on sources. Full container logs are NOT copied.
set -uo pipefail
S=$1; D=/shared_nfs/yihou_glm52/llrepro_20261023; P=$D/prep_next.yihou
[[ "$S" == *yihou* && ! -e "$S" ]] || { echo "stage must be new and contain yihou"; exit 1; }
mkdir -p $S/results/llr $S/scripts $S/sites
RUNS="llr25 llr26 llr27 llr28"
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
cp -p $D/llr2[4-9].site.env $D/llr31.site.env $S/sites/ 2>/dev/null
# real MTP acceptance (pool node 190)
mkdir -p $S/acceptance/results; cp -p $D/acc.yihou/{analyze.py,bench.py,build.py,build_real.py,client.sh,runall.sh,start.sh,wait_run.sh,ipctest.py} $S/acceptance/ 2>/dev/null
cp -p $D/acc.yihou/results/summary.md $S/acceptance/results/; for f in $D/acc.yihou/results/*.jsonl; do gzip -c $f > $S/acceptance/results/$(basename $f).gz; done
# MXFP4 draft patch
mkdir -p $S/mxfp4_draft; cp -pr $D/mxdraft.yihou/{mxfp4_draft.diff,patched,start.sh,bench.py} $S/mxfp4_draft/ 2>/dev/null
# MLA bmm configs (PR #209 table) + microbench
mkdir -p $S/mla_bmm; cp -pr $D/mlacfg.yihou/{src,out} $S/mla_bmm/; cp -p $P/mla_bmm/bench_mla_bmm_configs.py $P/mla_bmm/out/r1.log $P/mla_bmm/out/r1.json $S/mla_bmm/
# aiter GEMM retune
mkdir -p $S/aiter_retune/c96 $S/aiter_retune/k3; cp -p $P/aiter_retune/*.py $S/aiter_retune/
cp -p $P/aiter_retune/c96/{decode_bf16_untuned_gemm.csv,decode_bf16_untuned_fmoe.csv,yihou_glm52_dpa_bf16_tuned_gemm.csv} $S/aiter_retune/c96/
cp -p $P/aiter_retune/k3/*.csv $P/aiter_retune/k3/*.sh $S/aiter_retune/k3/ 2>/dev/null
cp -p $D/OPT_LEDGER.md $S/
cp -p $D/queue_runs.yihou.sh $D/run_llr.yihou.sh $D/guard_llr.yihou.sh $D/llsum.yihou.py $D/bneck.yihou.py $D/qseries.yihou.py $D/gather10.yihou.sh $D/gather11.yihou.sh \
  /home/yihou/yihou_tmp/isl_series.yihou.py $D/kit/sglang_2p1d_glm5.2/engine/worker.sh $D/kit/sglang_2p1d_glm5.2/engine/common.sh $S/scripts/ 2>/dev/null
du -sh $S; find $S -type f -size +4M -exec ls -la {} \;
echo GATHER11_DONE
