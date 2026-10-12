#!/usr/bin/env bash
# Wait for hold job 207495 to be RUNNING, then start the e2g pipeline (gated on GO.e2g + no microbench
# containers + GPU-idle gate inside run.v2 up).
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
for i in $(seq 1 240); do
  st=$(squeue -h -j 207495 -o %T 2>/dev/null)
  echo "$(date -u +%FT%TZ) job 207495 state=${st:-?}"
  [[ "$st" == RUNNING ]] && break
  sleep 60
done
[[ "$st" == RUNNING ]] || { echo "$(date -u +%FT%TZ) gave up waiting for 207495"; exit 1; }
sleep 30
cd $H && source env.e2g-102.yihou.sh && \
  GO_FILE=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/GO.e2g.yihou \
  WAIT_PREFIXES="yihou- e2-yihou e0-h1-yihou dbgagent-yihou" \
  exec bash pipeline.yihou.sh
