#!/usr/bin/env bash
# Collect small gmu artifacts into one tarball on NFS (run on a compute node). Creates files only; deletes nothing.
set -uo pipefail
G=/shared_nfs/yihou_glm52/dcp_pd_20261021/gmu; S=/shared_nfs/yihou_glm52/dcp_pd_20261021/src
W=/shared_nfs/yihou_glm52/llying_dcp_20261022/work; K=/shared_nfs/yihou_glm52/llying_dcp_20261022/kit
O=$G/packup_stage2.yihou; mkdir -p $O/patches $O/memlog $O/runs $O/stages $O/tools $O/startup $O/vram
B=$S/sglang-ab7a898b77/python/sglang
diff -u $B/srt/managers/scheduler_components/metrics_reporter.py $S/sglang-ab7a898b77-memlog.yihou/python/sglang/srt/managers/scheduler_components/metrics_reporter.py > $O/patches/01-memlog.metrics_reporter.diff
diff -u $S/sglang-ab7a898b77-memlog.yihou/python/sglang/srt/managers/scheduler_components/metrics_reporter.py $S/sglang-ab7a898b77-memlog-ec.yihou/python/sglang/srt/managers/scheduler_components/metrics_reporter.py > $O/patches/02-ec-tp0only.metrics_reporter.diff
diff -u $B/srt/disaggregation/decode.py $S/sglang-ab7a898b77-memlog-ec2.yihou/python/sglang/srt/disaggregation/decode.py > $O/patches/03-ec2-allrank.decode.diff
cp $K/stages/S2*.env $O/stages/
cp $G/tools/*.sh $G/tools/*.txt $G/tools/*.py $O/tools/ 2>/dev/null
cp $W/launch_stage2.yihou.sh $W/launch_s2g092.yihou.sh $W/mk_stage_*.yihou.sh $O/tools/ 2>/dev/null
cp $G/*.mem.txt $O/startup/ 2>/dev/null
cp $G/vram.*.txt $O/vram/ 2>/dev/null
for d in $W/S2-227/runs/S2-20261022T172214Z $W/S2[a-o]-*/runs/* $W/S2r-227/runs/*; do
  [ -d "$d" ] || continue; n=$(basename $d); mkdir -p $O/runs/$n
  for f in analysis.txt engine-config.env workers.json agentx/agentx_conc80.json; do [ -f $d/$f ] && cp $d/$f $O/runs/$n/; done
  L=$d/logs-live/decode.log; [ -f $L ] || continue
  grep -E "TP0.*(Load weight|KV Cache is allocated|Memory pool end|Capture .* (begin|end)|max_total_num_tokens)" $L | cut -c1-300 > $O/runs/$n/decode.startup_mem.txt
  m=$(grep -E "TP0.*mem: dev_free" $L | sed -E 's/^(.{30}).*#running-req: ([0-9]+).*token usage: ([0-9.]+).*(mem: dev_free.*peak_rsv [0-9.]+).*/\1 run=\2 tok=\3 \4/')
  [ -n "$m" ] && printf '%s\n' "$m" > $O/memlog/$n.TP0.memlog.txt
  a=$(grep -m3 -B2 "OUT_OF_RESOURCES" $L | cut -c1-300)
  [ -n "$a" ] && printf '%s\n' "$a" > $O/runs/$n/decode.abort_excerpt.txt
done
tar czf $G/packup_small2.yihou.tgz -C $G packup_stage2.yihou && ls -la $G/packup_small2.yihou.tgz && du -sh $O
