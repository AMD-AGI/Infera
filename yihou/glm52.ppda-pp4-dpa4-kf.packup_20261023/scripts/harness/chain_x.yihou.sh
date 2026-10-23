#!/usr/bin/env bash
# chain_x (task g4tune xnode, 2026-10-21): CROSS-NODE copy of chain_cap2.yihou.sh. Runs ON the prefill/control
# node (YIHOU_NODE) with plain ssh (no spur shim) to itself and to the decode node:
#   pipeline.v3x (up -> census -> smoke -> audit v3x -> A1 fresh @C1) -> flush -> A2 @C2 (scored) [-> A3 @C3].
# Differences vs chain_cap2: decode URLs on XNODE_DECODE_IP; KFD + disk guards on BOTH nodes; a disk watcher
# (10 min) that stops the round if / or /mnt/m2m_nobackup reaches 95 % on either node (136 was lost for days to
# a full disk, 10-15..10-19).
# The env file must export XNODE_DECODE_NODE / XNODE_DECODE_IP AND carry the same values in YIHOU_OVERRIDES.
# Usage (ON YIHOU_NODE): setsid nohup bash chain_x.yihou.sh <env> C1 C2 [C3] > $R/<round>.chain.log 2>&1 < /dev/null &
set -uo pipefail
ENVF=$1; C1=$2; C2=$3; C3=${4:-}
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
ts(){ date -u +%FT%TZ; }
cd $H && source "$ENVF"
[[ "$(hostname -s)" == "$YIHOU_NODE" ]] || { echo "$(ts) STOP: chain must run on $YIHOU_NODE (here $(hostname -s))"; exit 1; }
[[ -z "$YIHOU_JOB" ]] || { echo "$(ts) STOP: xnode chain is for plain-ssh nodes (YIHOU_JOB must be empty)"; exit 1; }
DNODE=${XNODE_DECODE_NODE:?}; DIP=${XNODE_DECODE_IP:?}
[[ " $YIHOU_OVERRIDES " == *" XNODE_DECODE_NODE=$DNODE "* && " $YIHOU_OVERRIDES " == *" XNODE_DECODE_IP=$DIP "* &&
   " $YIHOU_OVERRIDES " == *" NODE_RW_ROOT= "* ]] ||
    { echo "$(ts) STOP: YIHOU_OVERRIDES must carry XNODE_DECODE_NODE=$DNODE XNODE_DECODE_IP=$DIP NODE_RW_ROOT="; exit 1; }
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
RD=$R/$YIHOU_ROUND; MD=$N/runs/$YIHOU_ROUND; mkdir -p "$RD" "$MD"
BASE_OVR="$YIHOU_OVERRIDES"
D="bash $H/run.v3x.yihou.sh"
disk_ok(){  # node -> 0 if the docker store is on the nvme LV and / and /mnt are below $2 %
  "${SSH[@]}" -n "$1" "findmnt -n -o SOURCE /mnt/m2m_nobackup | grep -q nvme && \
    docker info -f '{{.DockerRootDir}}' 2>/dev/null | grep -q '^/mnt/m2m_nobackup/' && \
    r=\$(df --output=pcent / | tail -1 | tr -dc 0-9) && m=\$(df --output=pcent /mnt/m2m_nobackup | tail -1 | tr -dc 0-9) && \
    echo \"$1 /=\$r% /mnt=\$m%\" && [ \$r -lt $2 ] && [ \$m -lt $2 ]"; }
msampler(){ setsid nohup python3 $CAP/tools/metrics_sampler.yihou.py "$MD/metrics.$1.tsv" 5400 5 \
    decode=http://$DIP:29357 prefill=http://$YIHOU_IP:29101 > /dev/null 2>&1 < /dev/null &
  echo "$(ts) sampler $1 started"; }
flush(){ local ip=$1 port=$2 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$ip:$port/flush_cache")
    echo "$(ts) flush $ip:$port try $i -> $code"; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
score(){ local j; j=$(ls "$RD"/agentx-c$1-$2/agentx_conc$1.json 2>/dev/null | head -1)
  [[ -n "$j" ]] && python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print('score', d['request_metrics']['throughput']['per_gpu']['total_tput_tps'])" "$j" 2>/dev/null; }
echo "$(ts) chain start round=$YIHOU_ROUND prefill=$YIHOU_NODE decode=$DNODE C1=$C1 C2=$C2 C3=${C3:-none} CONFIG=$CONFIG"
echo "$(ts) overrides: $BASE_OVR"
for n in "$YIHOU_NODE" "$DNODE"; do
  disk_ok "$n" 92 || { echo "$(ts) STOP: disk guard failed on $n (docker store not on nvme LV, or / or /mnt >= 92 %)"; exit 1; }
  kfd=$("${SSH[@]}" -n "$n" "rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+ '")
  [[ "$kfd" == 0 ]] || { echo "$(ts) STOP: ${kfd:-?} KFD processes already on $n"; exit 1; }
done
# Disk watcher: every 10 min; at >= 95 % on either node stop the round (docker stop via run.v3x stop) and exit.
( while sleep 600; do
    for n in "$YIHOU_NODE" "$DNODE"; do
      out=$(disk_ok "$n" 95) && { echo "$(ts) diskwatch $out"; continue; }
      echo "$(ts) diskwatch ALARM on $n: ${out:-no answer}"
      [[ -n "$out" ]] || continue   # ssh hiccup: no action
      echo "$(ts) diskwatch: stopping round $YIHOU_ROUND"; $D stop; exit 0
    done
    grep -q "chain DONE\|STOP" "$R/$YIHOU_ROUND.chain.log" 2>/dev/null && exit 0
  done ) >> "$R/$YIHOU_ROUND.diskwatch.log" 2>&1 &
export YIHOU_OVERRIDES="$BASE_OVR CONC=$C1" BENCH_TAG=A1
( until grep -q "stage bench" "$R/$YIHOU_ROUND.pipeline.log" 2>/dev/null; do
    grep -qE "STOP|gave up" "$R/$YIHOU_ROUND.pipeline.log" 2>/dev/null && exit 0; sleep 20; done; msampler A1 ) &
bash $H/pipeline.v3x.yihou.sh > "$R/$YIHOU_ROUND.pipeline.log" 2>&1
grep -q "PIPELINE DONE" "$R/$YIHOU_ROUND.pipeline.log" || { echo "$(ts) STOP: A1 pipeline did not finish (see $YIHOU_ROUND.pipeline.log)"; exit 1; }
echo "$(ts) A1 C$C1 done $(score $C1 A1)"
run_warm(){ local tag=$1 c=$2
  flush "$DIP" 29357 || { echo "$(ts) STOP: decode flush failed"; return 1; }
  flush "$YIHOU_IP" 29101 || { echo "$(ts) STOP: prefill flush failed"; return 1; }
  msampler $tag
  echo "$(ts) stage bench $tag C$c"
  YIHOU_OVERRIDES="$BASE_OVR CONC=$c" BENCH_TAG=$tag $D bench || { echo "$(ts) STOP at bench $tag"; return 1; }
  echo "$(ts) $tag C$c DONE $(score $c $tag)"; }
run_warm A2 "$C2" || exit 1
[[ -n "$C3" ]] && { run_warm A3 "$C3" || exit 1; }
echo "$(ts) chain DONE (stack left running)"
