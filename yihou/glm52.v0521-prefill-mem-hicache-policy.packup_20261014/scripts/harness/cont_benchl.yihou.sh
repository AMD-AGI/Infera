#!/usr/bin/env bash
# Task v0521cap2: continue a chain_cap2 round on its LIVE stack after a pipeline stop that was a known false negative
# (cont_benchl: LOGIN-HOST variant for plain-ssh nodes, e.g. 136; no local shim, no hostname check.)
# (pc48b: audit flagged the same hook mount on both legs as a role-only violation; the 93 other checks passed and
# both legs report allocation_policy=prefill_complete). Runs: [A1 fresh @C1 ->] flush -> A2 @C2. Runs ON the node.
# Usage: bash cont_bench.yihou.sh <env-file> <C1|-> <C2>
set -uo pipefail
ENVF=$1; C1=$2; C2=$3
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
ts(){ date -u +%FT%TZ; }
cd $H && source "$ENVF"
[[ -z "$YIHOU_JOB" ]] || { echo "$(ts) STOP: cont_benchl is for plain-ssh nodes"; exit 1; }
MD=$N/runs/$YIHOU_ROUND; mkdir -p "$MD"; BASE_OVR="$YIHOU_OVERRIDES"; D="bash $H/run.v3.yihou.sh"
msampler(){ setsid nohup python3 $CAP/tools/metrics_sampler.yihou.py "$MD/metrics.$1.tsv" 5400 5 \
    decode=http://$YIHOU_IP:29357 prefill=http://$YIHOU_IP:29101 > /dev/null 2>&1 < /dev/null & echo "$(ts) sampler $1 started"; }
flush(){ local port=$1 i code
  for i in $(seq 1 60); do code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code"; [[ "$code" == 200 ]] && return 0; sleep 10; done; return 1; }
run(){ local tag=$1 c=$2; msampler $tag; echo "$(ts) stage bench $tag C$c"
  YIHOU_OVERRIDES="$BASE_OVR CONC=$c" BENCH_TAG=$tag $D bench || { echo "$(ts) STOP at bench $tag"; return 1; }
  echo "$(ts) $tag C$c DONE"; }
echo "$(ts) cont start round=$YIHOU_ROUND C1=$C1 C2=$C2"
if [[ "$C1" != - ]]; then run A1 "$C1" || exit 1; fi
flush 29357 || exit 1; flush 29101 || exit 1
run A2 "$C2" || exit 1
echo "$(ts) chain DONE (stack left running)"
