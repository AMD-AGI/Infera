#!/bin/bash
# launch_remote.sh <node> <config-id> <gpus> <port-offset> <conc> <tag-prefix> [bench-gpus=2]
# Launch serve.sh --wait then run_point.sh (official) on <node> detached; verifies 'launched' and the runs dir.
set -euo pipefail
N=$1; CFG=$2; G=$3; OFF=$4; C=$5; PFX=$6; BG=${7:-2}
K=/home/yihou/dev/git.16-19/infera.dsv41/yihou/dsv41_vllm_pd_20261119/scripts/kit
L=/perf_apps/yihou/dsv41_vllm_pd_20261119/launch; mkdir -p "$L"
T=$PFX-$(date -u +%m%d%H%M); P=$((8000+OFF))
bash "$K/stage.sh" >/dev/null
bash "$K/serve.sh" "$CFG" "$G" "dry-$T" --port-offset "$OFF" --dry-run >/dev/null
CMD="setsid nohup bash -c 'bash $K/serve.sh $CFG $G $T --port-offset $OFF --wait > $L/$T.serve.log 2>&1 && bash $K/run_point.sh $T http://127.0.0.1:$P $C $BG official > $L/$T.point.log 2>&1; echo CHAIN_DONE rc=\$? >> $L/$T.point.log' >/dev/null 2>&1 < /dev/null & echo launched"
out=$(timeout 60 ssh -n -o BatchMode=yes "$N" "$CMD")
[ "$out" = launched ] || { echo "LAUNCH FAILED on $N: '$out'"; exit 1; }
for i in $(seq 1 30); do [ -d /perf_apps/yihou/dsv41_vllm_pd_20261119/runs/$T ] && break; sleep 2; done
[ -d /perf_apps/yihou/dsv41_vllm_pd_20261119/runs/$T ] && echo "OK $T" || { echo "NO RUN DIR for $T"; exit 1; }
