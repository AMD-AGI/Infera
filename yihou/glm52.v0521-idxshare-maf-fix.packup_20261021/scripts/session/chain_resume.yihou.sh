#!/usr/bin/env bash
# Task g4tune 2026-10-20: resume a chain_cap2 round whose pipeline stopped at the audit stage, on the SAME live stack
# (no engine restart): audit -> A1 @C1 (fresh stack, no flush, as pipeline.v3) -> flush -> A2 @C2, like chain_cap2.
# Usage (ON the node): chain_resume.yihou.sh <env> <C1> <C2>
set -uo pipefail
ENVF=$1; C1=$2; C2=$3
N=/shared_nfs/yihou_glm52/v0521cap2_20261013; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
export PATH=$N/shimbin:$PATH
ts(){ date -u +%FT%TZ; }
cd $H && source "$ENVF"
[[ "$(hostname -s)" == "$YIHOU_NODE" ]] || { echo "$(ts) STOP: wrong node"; exit 1; }
RD=$R/$YIHOU_ROUND; MD=$N/runs/$YIHOU_ROUND; mkdir -p "$MD"
BASE_OVR="$YIHOU_OVERRIDES"; D="bash $H/run.v3.yihou.sh"
msampler(){ setsid nohup python3 $CAP/tools/metrics_sampler.yihou.py "$MD/metrics.$1.tsv" 5400 5 \
    decode=http://$YIHOU_IP:29357 prefill=http://$YIHOU_IP:29101 > /dev/null 2>&1 < /dev/null &
  echo "$(ts) sampler $1 started"; }
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code"; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
echo "$(ts) resume round=$YIHOU_ROUND C1=$C1 C2=$C2"
YIHOU_OVERRIDES="$BASE_OVR CONC=$C1" $D audit > "$RD/audit.resume.txt" 2>&1 || { echo "$(ts) STOP: audit $(tail -1 $RD/audit.resume.txt)"; exit 1; }
echo "$(ts) audit $(tail -1 $RD/audit.resume.txt)"
msampler A1
echo "$(ts) stage bench A1 C$C1"
YIHOU_OVERRIDES="$BASE_OVR CONC=$C1" BENCH_TAG=A1 $D bench || { echo "$(ts) STOP at bench A1"; exit 1; }
echo "$(ts) A1 C$C1 DONE"
flush 29357 || { echo "$(ts) STOP: decode flush failed"; exit 1; }
flush 29101 || { echo "$(ts) STOP: prefill flush failed"; exit 1; }
msampler A2
echo "$(ts) stage bench A2 C$C2"
YIHOU_OVERRIDES="$BASE_OVR CONC=$C2" BENCH_TAG=A2 $D bench || { echo "$(ts) STOP at bench A2"; exit 1; }
echo "$(ts) A2 C$C2 DONE"
echo "$(ts) chain DONE (stack left running)"
