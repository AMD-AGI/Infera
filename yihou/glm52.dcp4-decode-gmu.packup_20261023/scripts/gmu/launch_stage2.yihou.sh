#!/usr/bin/env bash
# launch_stage with /opt/rocm/bin on PATH (rocm-smi left the default PATH on these nodes ~18:29Z 10-22) and log suffix.
# Usage: launch_stage2.yihou.sh <s> <NK> <suffix>   (kills a still-waiting run_stage for the same stage/NK first)
set -uo pipefail
s=$1; NK=$2; SUF=$3
L=/shared_nfs/yihou_glm52/llying_dcp_20261022
LOG=$L/work/run_stage.S$s-$NK.$SUF.log
[[ -e $LOG ]] && { echo "log exists: $LOG"; exit 1; }
for p in $(pgrep -f "run_stage.yihou.sh $s\$"); do [[ $p != $$ ]] && kill "$p" && echo "killed waiting run_stage pid $p"; done
IP=$(ip -4 -o addr show ens3 | awk '{print $4}' | cut -d/ -f1)
setsid nohup bash -c "export PATH=/opt/rocm/bin:\$PATH; NK=$NK NODE_IP=$IP bash $L/kit/run_stage.yihou.sh $s" > $LOG 2>&1 < /dev/null &
echo "launched S$s NK=$NK IP=$IP log=$LOG"
