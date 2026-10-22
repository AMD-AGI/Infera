#!/usr/bin/env bash
# Task v0521cap round k1 (102): pre-stop e2j -> pipeline (A1 fresh C44) -> flush -> A2 warm C44 (scored).
# Node-side /metrics samplers (CPU-only containers, 5 s) for A1 and A2 into $CAP/runs/k1/metrics.{A1,A2}.tsv.
# Launch: setsid nohup bash chain_k1.yihou.sh > runs/k1.chain.log 2>&1 < /dev/null &
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS=crsuse2-m2m-102:207495
ts(){ date -u +%FT%TZ; }
cd $H && source $H/env.k1-102.yihou.sh
mkdir -p $CAP/runs/k1
msampler(){ local tag=$1 blog=$2
  until grep -aq "Phase warmup\|Phase profiling" "$blog" 2>/dev/null; do
    grep -qE "STOP|gave up" $R/k1.pipeline.log $R/k1.A2.log 2>/dev/null && return 1; sleep 15; done
  ssh crsuse2-m2m-102 "docker rm -f yihou-msampler-k1$tag >/dev/null 2>&1; docker run -d --rm --name yihou-msampler-k1$tag --network host \
    -v $CAP:$CAP --entrypoint python3 infera-sglang:v0521-yihou-1010-mc4340 $CAP/tools/metrics_sampler.yihou.py \
    $CAP/runs/k1/metrics.$tag.tsv 5400 5 decode=http://$YIHOU_IP:29357 prefill=http://$YIHOU_IP:29101"
  echo "$(ts) sampler $tag started"; }
echo "$(ts) k1 chain start"
msampler A1 $R/k1/agentx-c44/benchmark.log &
PRE_STOP_ENV=$H/env.e2j-102.yihou.sh bash $H/pipeline.yihou.sh > $R/k1.pipeline.log 2>&1
grep -q "PIPELINE DONE" $R/k1.pipeline.log || { echo "$(ts) A1 not done -> stop"; exit 1; }
echo "$(ts) A1 done -> A2"
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code" >> $R/k1.A2.log; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
flush 29357 || { echo "$(ts) STOP: decode flush failed" >> $R/k1.A2.log; exit 1; }
flush 29101 || { echo "$(ts) STOP: prefill flush failed" >> $R/k1.A2.log; exit 1; }
msampler A2 $R/k1/agentx-c44-A2/benchmark.log &
echo "$(ts) stage bench A2" >> $R/k1.A2.log
BENCH_TAG=A2 bash $H/run.v2.yihou.sh bench >> $R/k1.A2.log 2>&1 || { echo "$(ts) STOP at bench A2" >> $R/k1.A2.log; exit 1; }
echo "$(ts) A2 DONE" >> $R/k1.A2.log
echo "$(ts) k1 chain done"
