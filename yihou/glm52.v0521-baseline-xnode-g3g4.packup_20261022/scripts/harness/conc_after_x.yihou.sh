#!/usr/bin/env bash
# conc_after_x (task g4tune xnode, 2026-10-21): ON the prefill node, wait for <round>'s chain_x to finish
# ("chain DONE", stack left running), then run more warm benches on the SAME live stack, one per concurrency:
# flush decode + prefill -> metrics sampler -> run.v3x bench (BENCH_TAG=A<n>), in the given order. Stops at the
# first failure or when the disk guard (/ or /mnt >= 95 % on either node) trips. Never stops the stack itself.
# Usage: setsid nohup bash conc_after_x.yihou.sh <env> <first-tag-number> C [C ...] > $R/<round>.conc.log 2>&1 &
#   e.g. conc_after_x.yihou.sh env.g4x2-135x136.yihou.sh 3 60 64 68 72   -> A3 C60, A4 C64, A5 C68, A6 C72
set -uo pipefail
ENVF=$1; T=$2; shift 2
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
ts(){ date -u +%FT%TZ; }
cd $H && source "$ENVF"
[[ "$(hostname -s)" == "$YIHOU_NODE" ]] || { echo "$(ts) STOP: run on $YIHOU_NODE"; exit 1; }
DNODE=${XNODE_DECODE_NODE:?}; DIP=${XNODE_DECODE_IP:?}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
RD=$R/$YIHOU_ROUND; MD=$N/runs/$YIHOU_ROUND; BASE_OVR="$YIHOU_OVERRIDES"
D="bash $H/run.v3x.yihou.sh"
disk_ok(){ "${SSH[@]}" -n "$1" "findmnt -n -o SOURCE /mnt/m2m_nobackup | grep -q nvme && \
    r=\$(df --output=pcent / | tail -1 | tr -dc 0-9) && m=\$(df --output=pcent /mnt/m2m_nobackup | tail -1 | tr -dc 0-9) && \
    echo \"$1 /=\$r% /mnt=\$m%\" && [ \$r -lt 95 ] && [ \$m -lt 95 ]"; }
flush(){ local ip=$1 port=$2 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$ip:$port/flush_cache")
    echo "$(ts) flush $ip:$port try $i -> $code"; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
score(){ local j; j=$(ls "$RD"/agentx-c$1-$2/agentx_conc$1.json 2>/dev/null | head -1)
  [[ -n "$j" ]] && python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print('score', d['request_metrics']['throughput']['per_gpu']['total_tput_tps'])" "$j" 2>/dev/null; }
echo "$(ts) waiting for $YIHOU_ROUND chain DONE; then concs: $*"
until grep -q "chain DONE" "$R/$YIHOU_ROUND.chain.log" 2>/dev/null; do
  grep -q "STOP" "$R/$YIHOU_ROUND.chain.log" 2>/dev/null && { echo "$(ts) STOP: chain stopped early"; exit 1; }
  sleep 60; done
for c in "$@"; do
  tag=A$T; T=$((T + 1))
  for n in "$YIHOU_NODE" "$DNODE"; do disk_ok "$n" || { echo "$(ts) STOP: disk guard on $n"; exit 1; }; done
  flush "$DIP" 29357 || { echo "$(ts) STOP: decode flush failed"; exit 1; }
  flush "$YIHOU_IP" 29101 || { echo "$(ts) STOP: prefill flush failed"; exit 1; }
  setsid nohup python3 $CAP/tools/metrics_sampler.yihou.py "$MD/metrics.$tag.tsv" 5400 5 \
    decode=http://$DIP:29357 prefill=http://$YIHOU_IP:29101 > /dev/null 2>&1 < /dev/null &
  echo "$(ts) stage bench $tag C$c"
  YIHOU_OVERRIDES="$BASE_OVR CONC=$c" BENCH_TAG=$tag $D bench || { echo "$(ts) STOP at bench $tag C$c"; exit 1; }
  echo "$(ts) $tag C$c DONE $(score $c $tag)"
done
echo "$(ts) conc DONE (stack left running)"
