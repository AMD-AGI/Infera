#!/usr/bin/env bash
# Task v0521cap round q3 (102): wait for pc40c (+smoke2) -> pre-stop pc40c -> pipeline (A1 fresh C44) -> flush -> A2 warm C44 (scored).
# Node-side /metrics samplers (CPU-only containers, 5 s) for A1 and A2 into $CAP/runs/q3/metrics.{A1,A2}.tsv.
# Launch: setsid nohup bash chain_k1.yihou.sh > runs/q3.chain.log 2>&1 < /dev/null &
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS=crsuse2-m2m-102:207495
ts(){ date -u +%FT%TZ; }
cd $H && source $H/env.q3-102.yihou.sh
mkdir -p $CAP/runs/q3
msampler(){ local tag=$1 blog=$2
  until grep -aq "Phase warmup\|Phase profiling" "$blog" 2>/dev/null; do
    grep -qE "STOP|gave up" $R/q3.pipeline.log $R/q3.A2.log 2>/dev/null && return 1; sleep 15; done
  ssh crsuse2-m2m-102 "docker rm -f yihou-msampler-q3$tag >/dev/null 2>&1; docker run -d --rm --name yihou-msampler-q3$tag --network host \
    -v $CAP:$CAP --entrypoint python3 infera-sglang:v0521-yihou-1010-mc4340 $CAP/tools/metrics_sampler.yihou.py \
    $CAP/runs/q3/metrics.$tag.tsv 5400 5 decode=http://$YIHOU_IP:29357 prefill=http://$YIHOU_IP:29101"
  echo "$(ts) sampler $tag started"; }
echo "$(ts) q3 chain start: waiting for pc40c chain + smoke2"
until grep -qE "faults:|no smoke2" $R/pc40c.smoke2.log 2>/dev/null || grep -q "STOP" $R/pc40c.chain.log; do sleep 60; done
echo "$(ts) pc40c finished"
msampler A1 $R/q3/agentx-c44/benchmark.log &
PRE_STOP_ENV=$H/env.pc40c-102.yihou.sh bash $H/pipeline.yihou.sh > $R/q3.pipeline.log 2>&1
grep -q "PIPELINE DONE" $R/q3.pipeline.log || { echo "$(ts) A1 not done -> stop"; exit 1; }
echo "$(ts) A1 done -> A2"
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code" >> $R/q3.A2.log; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
flush 29357 || { echo "$(ts) STOP: decode flush failed" >> $R/q3.A2.log; exit 1; }
flush 29101 || { echo "$(ts) STOP: prefill flush failed" >> $R/q3.A2.log; exit 1; }
msampler A2 $R/q3/agentx-c44-A2/benchmark.log &
echo "$(ts) stage bench A2" >> $R/q3.A2.log
BENCH_TAG=A2 bash $H/run.v2.yihou.sh bench >> $R/q3.A2.log 2>&1 || { echo "$(ts) STOP at bench A2" >> $R/q3.A2.log; exit 1; }
echo "$(ts) A2 DONE" >> $R/q3.A2.log
echo "$(ts) q3 chain done"
