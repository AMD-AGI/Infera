#!/usr/bin/env bash
# S2 + decode --mem-fraction-static 0.92 (gmu load run), one 30-min AgentX C80, same-node P4D4. Node-side.
set -uo pipefail
NK=$1
L=/shared_nfs/yihou_glm52/llying_dcp_20261022
LOG=$L/work/run_stage.S2g092-$NK.log
[[ -e $LOG ]] && { echo "log exists: $LOG"; exit 1; }
IP=$(ip -4 -o addr show ens3 | awk '{print $4}' | cut -d/ -f1)
setsid nohup bash -c "
  docker image inspect --format '{{.Id}}' infera-sglang:v0522-d395c3b3-1020-mc4340 | grep -q d3d6a9c20ce0 || { echo 'image missing'; exit 1; }
  NK=$NK NODE_IP=$IP DECODE_EXTRA_ARGS='--mem-fraction-static 0.92' bash $L/kit/run_stage.yihou.sh 2
" > $LOG 2>&1 < /dev/null &
echo "launched S2g092 NK=$NK IP=$IP log=$LOG"
