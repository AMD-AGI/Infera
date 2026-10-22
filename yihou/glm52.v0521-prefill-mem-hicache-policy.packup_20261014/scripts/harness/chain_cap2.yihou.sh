#!/usr/bin/env bash
# Task v0521cap2: generic warm-stack chain that runs ON the held node (local ssh shim, no login-host/spur dependency):
#   pipeline.v3 (up -> census -> smoke -> audit v3 -> A1 fresh @C1) -> flush -> A2 @C2 (scored) [-> flush -> A3 @C3].
# One /metrics sampler (host python3, 5 s) per bench run into $N/runs/<round>/metrics.<tag>.tsv.
# Usage (from a login host):
#   spur exec <job> bash -c "cd /; setsid nohup bash $H/chain_cap2.yihou.sh $H/env.<round>.yihou.sh C1 C2 [C3] \
#       > $R/<round>.chain.log 2>&1 < /dev/null &"
set -uo pipefail
ENVF=$1; C1=$2; C2=$3; C3=${4:-}
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
export PATH=$N/shimbin:$PATH
ts(){ date -u +%FT%TZ; }
cd $H && source "$ENVF"
[[ "$(hostname -s)" == "$YIHOU_NODE" ]] || { echo "$(ts) STOP: chain must run on $YIHOU_NODE (here $(hostname -s))"; exit 1; }
[[ "$(command -v ssh)" == "$N/shimbin/ssh" ]] || { echo "$(ts) STOP: local ssh shim not first on PATH"; exit 1; }
RD=$R/$YIHOU_ROUND; MD=$N/runs/$YIHOU_ROUND; mkdir -p "$RD" "$MD"
BASE_OVR="$YIHOU_OVERRIDES"
D="bash $H/run.v3.yihou.sh"
msampler(){ setsid nohup python3 $CAP/tools/metrics_sampler.yihou.py "$MD/metrics.$1.tsv" 5400 5 \
    decode=http://$YIHOU_IP:29357 prefill=http://$YIHOU_IP:29101 > /dev/null 2>&1 < /dev/null &
  echo "$(ts) sampler $1 started"; }
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code"; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
score(){ local j; j=$(ls "$RD"/agentx-c$1-$2/agentx_conc$1.json 2>/dev/null | head -1)
  [[ -n "$j" ]] && python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print('score', d['request_metrics']['throughput']['per_gpu']['total_tput_tps'])" "$j" 2>/dev/null; }
echo "$(ts) chain start round=$YIHOU_ROUND node=$YIHOU_NODE C1=$C1 C2=$C2 C3=${C3:-none} CONFIG=$CONFIG"
echo "$(ts) overrides: $BASE_OVR"
# Foreign GPU users (e.g. orphan containers of preempted jobs) must not hold the GPUs we are about to take.
kfd=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+ ')
[[ "$kfd" == 0 ]] || { echo "$(ts) STOP: $kfd KFD processes already on the node"; rocm-smi --showpids; exit 1; }
# A1: fresh stack. BENCH_TAG=A1 so the out dir is agentx-c<C1>-A1.
export YIHOU_OVERRIDES="$BASE_OVR CONC=$C1" BENCH_TAG=A1
( until grep -q "stage bench" "$R/$YIHOU_ROUND.pipeline.log" 2>/dev/null; do
    grep -qE "STOP|gave up" "$R/$YIHOU_ROUND.pipeline.log" 2>/dev/null && exit 0; sleep 20; done; msampler A1 ) &
bash $H/pipeline.v3.yihou.sh > "$R/$YIHOU_ROUND.pipeline.log" 2>&1
grep -q "PIPELINE DONE" "$R/$YIHOU_ROUND.pipeline.log" || { echo "$(ts) STOP: A1 pipeline did not finish (see $YIHOU_ROUND.pipeline.log)"; exit 1; }
echo "$(ts) A1 C$C1 done $(score $C1 A1)"
run_warm(){ local tag=$1 c=$2
  flush 29357 || { echo "$(ts) STOP: decode flush failed"; return 1; }
  flush 29101 || { echo "$(ts) STOP: prefill flush failed"; return 1; }
  msampler $tag
  echo "$(ts) stage bench $tag C$c"
  YIHOU_OVERRIDES="$BASE_OVR CONC=$c" BENCH_TAG=$tag $D bench || { echo "$(ts) STOP at bench $tag"; return 1; }
  echo "$(ts) $tag C$c DONE $(score $c $tag)"; }
run_warm A2 "$C2" || exit 1
[[ -n "$C3" ]] && { run_warm A3 "$C3" || exit 1; }
echo "$(ts) chain DONE (stack left running)"
