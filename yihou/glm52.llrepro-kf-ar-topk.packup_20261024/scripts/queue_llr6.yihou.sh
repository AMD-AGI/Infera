#!/usr/bin/env bash
# queue_llr6 (task llrepro 2026-10-24, ON 135): after llr5's run_llr is done, add the AR hook and run llr6.
D=/shared_nfs/yihou_glm52/llrepro_20261023
for i in $(seq 1 240); do grep -q RUN_LLR_DONE $D/run_llr.llr5.yihou.log && break; sleep 30; done
grep -q RUN_LLR_DONE $D/run_llr.llr5.yihou.log || { echo "llr5 not done"; exit 1; }
bash $D/kit_arhook.yihou.sh && bash $D/run_llr.yihou.sh $D/llr6.site.env llr6-c80
