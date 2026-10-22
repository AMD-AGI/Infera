#!/usr/bin/env bash
# Task v0521cap Track B round pc40 (102, after k1): up -> census -> smoke -> audit -> long-prompt burst smoke
# (64 x ~200k-word prompts; must trigger YIHOU-RUNSHARE blocks, all complete) -> AgentX C40 A1 (fresh) -> flush ->
# A2 (warm, scored vs e2g A2 22,114). /metrics samplers (5 s) for burst, A1, A2 into $CAP/runs/pc40d/.
# Launch: setsid nohup bash chain_pc40.yihou.sh > runs/pc40d.chain.log 2>&1 < /dev/null &
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
D="bash $H/run.v2.yihou.sh"
ts(){ date -u +%FT%TZ; }
cd $H && source $H/env.pc40d-102.yihou.sh
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS=$YIHOU_NODE:$YIHOU_JOB
mkdir -p $CAP/runs/pc40d $R/pc40d
msampler(){ local tag=$1 dur=$2
  ssh $YIHOU_NODE "docker rm -f yihou-msampler-pc40d$tag >/dev/null 2>&1; docker run -d --rm --name yihou-msampler-pc40d$tag --network host \
    -v $CAP:$CAP --entrypoint python3 infera-sglang:v0521-yihou-1010-mc4340 $CAP/tools/metrics_sampler.yihou.py \
    $CAP/runs/pc40d/metrics.$tag.tsv $dur 5 decode=http://$YIHOU_IP:29357 prefill=http://$YIHOU_IP:29101" >/dev/null
  echo "$(ts) sampler $tag started"; }
echo "$(ts) pc40d chain start: waiting for pc40c + smoke2"
until grep -qE "faults:|no smoke2" $R/pc40c.smoke2.log 2>/dev/null || grep -q STOP $R/pc40c.chain.log; do sleep 60; done
echo "$(ts) pc40c finished -> stop pc40c stack"
( source $H/env.pc40c-102.yihou.sh && $D stop ) || echo "$(ts) k1 stop rc!=0 (checking)"
left=$(ssh $YIHOU_NODE "docker ps --format '{{.Names}}' | grep -c '^glm52-v0521-yihou-102-pc40c-'")
[[ "$left" == 0 ]] || { echo "$(ts) STOP: $left pc40c containers still running"; exit 1; }
echo "$(ts) stage up";     $D up || { echo "$(ts) STOP at up"; exit 1; }
c="$R/pc40d/census.txt"; $D census > "$c" 2>&1; cat "$c"
if grep -qE '"Memory access fault"=[1-9]|kernel_code_entry=[1-9]|OOM/Traceback  : [1-9]|OOM/Traceback  : [0-9]+/[1-9]' "$c"; then
  echo "$(ts) STOP at census"; exit 1; fi
grep -q 'rank-PID overlap prefill/decode: none' "$c" || { echo "$(ts) STOP at census: rank PID overlap"; exit 1; }
echo "$(ts) stage smoke";  $D smoke || { echo "$(ts) STOP at smoke"; exit 1; }
echo "$(ts) stage audit";  $D audit > "$R/pc40d/audit.txt" 2>&1 || { echo "$(ts) STOP at audit"; exit 1; }; tail -1 "$R/pc40d/audit.txt"
DL=$R/pc40d/server-logs/decode-0.a1.log
echo "$(ts) patch active check: $(strings $DL 2>/dev/null | grep -ac YIHOU_DECODE_RUNNING_KV_SHARE) env lines"
msampler burst 900
echo "$(ts) stage burst (64 x 200000 words, max_tokens 16)"
python3 $CAP/tools/longprompt_burst.yihou.py http://$YIHOU_IP:28100 64 200000 16 > $R/pc40d/burst.txt 2>&1; brc=$?
tail -1 $R/pc40d/burst.txt
echo "$(ts) burst rc=$brc YIHOU-RUNSHARE blocks logged: $(strings $DL | grep -ac 'YIHOU-RUNSHARE block')"
echo "$(ts) burst max pre-allocated usage: $(strings $DL | grep -ao 'pre-allocated usage: [0-9.]*' | awk '{print $3}' | sort -g | tail -1)"
[[ $brc == 0 ]] || { echo "$(ts) STOP at burst"; exit 1; }
msampler A1 5400
echo "$(ts) stage bench A1"; $D bench || { echo "$(ts) STOP at bench A1"; exit 1; }
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code"; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
flush 29357 || { echo "$(ts) STOP: decode flush failed"; exit 1; }
flush 29101 || { echo "$(ts) STOP: prefill flush failed"; exit 1; }
msampler A2 5400
echo "$(ts) stage bench A2"; BENCH_TAG=A2 $D bench || { echo "$(ts) STOP at bench A2"; exit 1; }
echo "$(ts) pc40d chain DONE"
