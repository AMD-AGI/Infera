#!/bin/bash
# Usage (on the crusoe node): cru_launch_point.sh <cfg> <gpus> <port-off> <conc> <tag-prefix>
# Launches serve.sh + run_point.sh (official) in background with the crusoe env; prints the tag.
set -u
R=/shared_nfs/yihou/dsv41_vllm_pd_20261119
CFG=$1; GPUS=$2; OFF=$3; CONC=$4; PFX=$5
T="$PFX-$(date -u +%m%d%H%M)"; echo "NEW $T"
source "$R/ws/scripts/kit/cru_env.sh"
PORT=$((8000 + OFF))
nohup bash -c "bash $R/ws/scripts/kit/serve.sh $CFG $GPUS $T --port-offset $OFF --wait > $R/launch/$T.serve.log 2>&1 && bash $R/ws/scripts/kit/run_point.sh $T http://127.0.0.1:$PORT $CONC 2 official > $R/launch/$T.point.log 2>&1; echo CHAIN_DONE rc=\$? >> $R/launch/$T.point.log" > /dev/null 2>&1 &
sleep 15; tail -n1 "$R/launch/$T.serve.log"
