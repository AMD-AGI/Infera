#!/usr/bin/env bash
# gather6 (task llrepro packup 2026-10-24, ON 135): stage llr5..llr7 (KF kernels, decode custom AR, KF top-k) on the
# reproduced llying 4+4 C80 base into a NEW yihou stage dir. Read-only on sources; full container logs are NOT copied.
set -uo pipefail
S=$1; D=/shared_nfs/yihou_glm52/llrepro_20261023; O=/shared_nfs/yihou_glm52/kf_e2e_20261023/kf/e2e/overlay
[[ "$S" == *yihou* && ! -e "$S" ]] || { echo "stage must be new and contain yihou"; exit 1; }
mkdir -p $S/results/llr $S/results/probe $S/kit $S/scripts $S/kf_overlay
for r in llr3-c80 llr5-c80 llr6-c80 llr7-c80; do d=$D/work/runs/$r
  python3 -I $D/llsum.yihou.py $d | tail -1 >> $S/results/llr/summary.txt
  python3 -I $D/bneck.yihou.py $d > $S/results/llr/bneck.$r.txt
  [ $r = llr3-c80 ] && continue
  o=$S/results/llr/$r; mkdir -p $o
  cp -p $d/engine-config.env $d/workers.json $o/ 2>/dev/null; cp -p $d/agentx/agentx_conc80.json $d/agentx/runtime.env $o/ 2>/dev/null
  f=$(find $d/agentx -name server_metrics_export.csv | head -1); [ -n "$f" ] && gzip -c $f > $o/server_metrics_export.csv.gz
  grep -a -E "KV Cache is allocated|available_gpu_mem|max_total_num_tokens|Memory access fault|Traceback|disable_custom_all_reduce|yihou-kf" $d/logs/yihou-llr-decode.log \
    | grep -a -v "Decode batch" | cut -c1-400 | head -400 > $o/decode.keylines.txt
  grep -a -E "KV Cache is allocated|available_gpu_mem|Traceback" $d/logs/yihou-llr-prefill0.log | head -40 > $o/prefill.keylines.txt
done
sed -i '1i round score qps isl ttft_p50 ttft_p90 tpot gpu_hit cpu_hit err kvlat(p50/p90) xfer_stage pqueue dtransferred' $S/results/llr/summary.txt
cp -p $D/probe/llr6.* $S/results/probe/
cp -a $D/kit/sglang_2p1d_glm5.2 $S/kit/
cp -p $D/llr5.site.env $D/llr6.site.env $D/llr7.site.env $D/llr8.site.env $D/run_llr.yihou.sh $D/kit_kfhook.yihou.sh \
  $D/kit_arhook.yihou.sh $D/queue_llr6.yihou.sh $D/llsum.yihou.py $D/bneck.yihou.py $D/node_probe.yihou.sh \
  $D/run_llr.llr5.yihou.log $D/run_llr.llr6.yihou.log $D/run_llr.llr7.yihou.log $S/scripts/ 2>/dev/null
cp -p $O/sitecustomize.py $S/kf_overlay/; mkdir -p $S/kf_overlay/yihou_kf; cp -p $O/yihou_kf/*.py $S/kf_overlay/yihou_kf/
(cd $O && md5sum sitecustomize.py yihou_kf/*.py) > $S/kf_overlay/MD5SUMS.source
du -sh $S; find $S -type f -size +4M -exec ls -la {} \;
echo GATHER6_DONE
