#!/bin/bash
# Usage (inside spur exec on the crusoe node): cru_swap_after.sh <old-tag> <cfg> <gpus> <port-offset> <conc> <new-tag-prefix>
# Waits for CHAIN_DONE of <old-tag>, prints its ledger row, stops it, then launches <cfg> at <conc> on <gpus>.
set -u
R=/shared_nfs/yihou/dsv41_vllm_pd_20261119
OLD=$1; CFG=$2; GPUS=$3; OFF=$4; CONC=$5; PFX=$6
for i in $(seq 1 1080); do grep -q CHAIN_DONE "$R/launch/$OLD.point.log" && break; sleep 10; done
grep -q CHAIN_DONE "$R/launch/$OLD.point.log" || { echo "TIMEOUT waiting $OLD"; exit 3; }
tail -n1 "$R/launch/$OLD.point.log"
grep "$OLD" "$R/results/ledger.csv" | cut -d, -f1-12,18,26,27
bash "$R/ws/scripts/kit/stop.sh" "$OLD" > "$R/launch/$OLD.stop.log" 2>&1
T="$PFX-$(date -u +%m%d%H%M)"; echo "NEW $T"
source "$R/ws/scripts/kit/cru_env.sh"
PORT=$((8000 + OFF))
nohup bash -c "bash $R/ws/scripts/kit/serve.sh $CFG $GPUS $T --port-offset $OFF --wait > $R/launch/$T.serve.log 2>&1 && bash $R/ws/scripts/kit/run_point.sh $T http://127.0.0.1:$PORT $CONC 2 official > $R/launch/$T.point.log 2>&1; echo CHAIN_DONE rc=\$? >> $R/launch/$T.point.log" > /dev/null 2>&1 &
sleep 20; tail -n1 "$R/launch/$T.serve.log"
