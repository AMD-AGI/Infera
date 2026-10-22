#!/usr/bin/env bash
# Node-side: load the S1-S3 image tar, verify its ID, run stage S<s> detached. Usage: launch_s3.yihou.sh <s> <NK> [EXTRA_ENV]
set -uo pipefail
s=$1; NK=$2; EXTRA=${3:-}
L=/shared_nfs/yihou_glm52/llying_dcp_20261022
LOG=$L/work/run_stage.S$s-$NK.log
[[ -e $LOG ]] && { echo "log exists, not relaunching: $LOG"; exit 1; }
IP=$(ip -4 -o addr show ens3 | awk '{print $4}' | cut -d/ -f1)
setsid nohup bash -c "
  docker image inspect --format '{{.Id}}' infera-sglang:v0522-d395c3b3-1020-mc4340 2>/dev/null | grep -q d3d6a9c20ce0 ||
    docker load -i $L/images/infera-sglang_v0522-d395c3b3-1020-mc4340.tar
  docker image inspect --format '{{.Id}}' infera-sglang:v0522-d395c3b3-1020-mc4340 | grep -q d3d6a9c20ce0 || { echo 'image ID mismatch'; exit 1; }
  NK=$NK NODE_IP=$IP PREFILL_ENV='$EXTRA' DECODE_ENV='$EXTRA' bash $L/kit/run_stage.yihou.sh $s
" > $LOG 2>&1 < /dev/null &
echo "launched S$s NK=$NK IP=$IP extra='$EXTRA' log=$LOG"
