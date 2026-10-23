#!/usr/bin/env bash
# Node-side: ensure image, arm the job-bound guard, run a gmu series detached.
# Usage: spur exec <JOB> bash -c 'cd /tmp && bash node_go.yihou.sh <JOB> <NK> <series-file> [GLOBAL_ENVX]'
set -uo pipefail
JOB=$1; NK=$2; SER=$3; GENV=${4:-}
G=/shared_nfs/yihou_glm52/dcp_pd_20261021/gmu; T=$G/tools
TAR=/shared_nfs/yihou_glm52/llying_dcp_20261022/images/infera-sglang_v0522-d395c3b3-1020-mc4340.tar
IMG=infera-sglang:v0522-d395c3b3-1020-mc4340
LOG=$G/node_go.$NK.$(basename "$SER" .txt).log
[[ -e $LOG ]] && { echo "exists: $LOG"; exit 1; }
setsid nohup bash -c "
  echo \$(date -u +%FT%TZ) start
  docker image inspect --format '{{.Id}}' $IMG 2>/dev/null | grep -q d3d6a9c20ce0 || docker load -i $TAR
  docker image inspect --format '{{.Id}}' $IMG | grep -q d3d6a9c20ce0 || { echo 'image ID mismatch'; exit 1; }
  echo \$(date -u +%FT%TZ) image ok
  bash /shared_nfs/yihou_glm52/dcp_pd_20261021/pd/tools/job_container_guard.yihou.sh status $JOB $NK 2>&1 | head -3
  setsid nohup bash /shared_nfs/yihou_glm52/dcp_pd_20261021/pd/tools/job_container_guard.yihou.sh start $JOB $NK > /dev/null 2>&1 < /dev/null &
  sleep 5
  NK=$NK GLOBAL_ENVX='$GENV' bash $T/run_series.v2.yihou.sh $SER
" > "$LOG" 2>&1 < /dev/null &
echo "launched $NK $SER log=$LOG"
