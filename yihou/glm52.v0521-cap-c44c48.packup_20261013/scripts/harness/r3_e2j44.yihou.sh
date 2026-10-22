#!/usr/bin/env bash
# Task v0521cap X1 (2026-10-12): C44 on the live, warm e2j stack (3rd AgentX run on it; R1 fresh C40 19,335, R2 warm
# C40 21,681). Flush decode then prefill (retry until 200), node-side /metrics sampler (CPU-only container, 5 s),
# bench C44 BENCH_TAG=R3 -> runs/e2j/agentx-c44-R3. Does NOT stop the stack. Launch: setsid nohup ... < /dev/null &
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS=crsuse2-m2m-102:207495
L=$R/e2j.R3.log
ts(){ date -u +%FT%TZ; }
cd $H && source $H/env.e2j44-102.yihou.sh
echo "$(ts) e2j R3 (C44) start" >> $L
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code" >> $L; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
flush 29357 || { echo "$(ts) STOP: decode flush failed" >> $L; exit 1; }
flush 29101 || { echo "$(ts) STOP: prefill flush failed" >> $L; exit 1; }
mkdir -p $CAP/runs/x1-e2j44
ssh crsuse2-m2m-102 "docker rm -f yihou-msampler-x1 >/dev/null 2>&1; docker run -d --rm --name yihou-msampler-x1 --network host \
  -v $CAP:$CAP --entrypoint python3 infera-sglang:v0521-yihou-1010-mc4340 $CAP/tools/metrics_sampler.yihou.py \
  $CAP/runs/x1-e2j44/metrics.tsv 6600 5 decode=http://$YIHOU_IP:29357 prefill=http://$YIHOU_IP:29101" >> $L 2>&1
echo "$(ts) sampler started" >> $L
echo "$(ts) stage bench R3" >> $L
BENCH_TAG=R3 bash $H/run.v2.yihou.sh bench >> $L 2>&1 || { echo "$(ts) STOP at bench R3" >> $L; exit 1; }
echo "$(ts) R3 DONE" >> $L
