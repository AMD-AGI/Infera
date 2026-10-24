#!/usr/bin/env bash
# gather7 (task llrepro packup addendum 2026-10-24, ON 135): stage the voided llr8 (decode mem 0.95 abort) and the llr7
# decode row-bucket stats into a NEW yihou stage dir. Read-only on sources.
set -uo pipefail
S=$1; D=/shared_nfs/yihou_glm52/llrepro_20261023; d=$D/work/runs/llr8-c80
[[ "$S" == *yihou* && ! -e "$S" ]] || { echo "stage must be new and contain yihou"; exit 1; }
o=$S/results/llr/llr8-c80; mkdir -p $o $S/scripts
cp -p $d/engine-config.env $d/workers.json $o/ 2>/dev/null; cp -p $d/agentx/runtime.env $o/ 2>/dev/null
grep -a -E "KV Cache is allocated|avail mem|max_total_num_tokens|Memory access fault|Traceback|disable_custom_all_reduce|yihou-kf.*(installed|RAISED)" $d/logs/yihou-llr-decode.log \
  | grep -a -v "Decode batch" | cut -c1-400 | head -200 > $o/decode.keylines.txt
grep -a -n -B3 -A6 "HSA_STATUS_ERROR" $d/logs/yihou-llr-decode.log | cut -c1-400 | head -60 > $o/decode.abort.txt
grep -a "Decode batch" $d/logs/yihou-llr-decode.log | tail -8 | cut -c1-400 > $o/decode.last_batches.txt
for r in llr7-c80; do
  grep -a "Decode batch" $D/work/runs/$r/logs/yihou-llr-decode.log | awk '$2>="03:27:48" && $2<="04:10:13"' \
   | awk -F'#running-req: ' '{split($2,a,","); r=a[1]*6; n++; if(r<=48)s++; else if(r<=128)m++; else l++}
       END{printf "llr7 steady 03:27:48-04:10:13 decode log lines n=%d by TARGET_VERIFY rows (6 x running/rank): <=48 %.3f  49-128 %.3f  >128 %.3f\n", n, s/n, m/n, l/n}' \
   > $S/results/llr/llr7-c80.rowbuckets.txt
done
cp -p $D/llr8.site.env $D/run_llr.llr8.yihou.log $D/gather6.yihou.sh $D/gather7.yihou.sh $S/scripts/ 2>/dev/null
du -sh $S; echo GATHER7_DONE
