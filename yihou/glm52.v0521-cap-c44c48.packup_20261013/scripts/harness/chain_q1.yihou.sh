#!/usr/bin/env bash
# Task v0521cap round q1 on 137 (plain ssh, NFS ro on node): pipeline (A1 fresh C44) -> flush -> A2 warm C44 (scored).
# /metrics samplers run on the LOGIN host (node cannot write NFS) into $CAP/runs/q1/metrics.{A1,A2}.tsv.
# Launch: setsid nohup bash chain_q1.yihou.sh > runs/q1.chain.log 2>&1 < /dev/null &
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
ts(){ date -u +%FT%TZ; }
cd $H && source $H/env.q1-137.yihou.sh
mkdir -p $CAP/runs/q1
msampler(){ local tag=$1
  setsid nohup python3 $CAP/tools/metrics_sampler.yihou.py $CAP/runs/q1/metrics.$tag.tsv 5400 5 \
    decode=http://$YIHOU_IP:29357 prefill=http://$YIHOU_IP:29101 > /dev/null 2>&1 < /dev/null &
  echo "$(ts) sampler $tag started (login host)"; }
wait_router_bench(){ # start the sampler once the router answers (bench stage)
  until grep -q "stage bench" $R/q1.pipeline.log 2>/dev/null; do
    grep -qE "STOP|gave up" $R/q1.pipeline.log 2>/dev/null && return 1; sleep 20; done
  msampler A1; }
echo "$(ts) q1 chain start"
wait_router_bench &
bash $H/pipeline.yihou.sh > $R/q1.pipeline.log 2>&1
grep -q "PIPELINE DONE" $R/q1.pipeline.log || { echo "$(ts) A1 not done -> stop"; exit 1; }
echo "$(ts) A1 done -> A2"
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code" >> $R/q1.A2.log; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
flush 29357 || { echo "$(ts) STOP: decode flush failed" >> $R/q1.A2.log; exit 1; }
flush 29101 || { echo "$(ts) STOP: prefill flush failed" >> $R/q1.A2.log; exit 1; }
msampler A2
echo "$(ts) stage bench A2" >> $R/q1.A2.log
BENCH_TAG=A2 bash $H/run.v2.yihou.sh bench >> $R/q1.A2.log 2>&1 || { echo "$(ts) STOP at bench A2" >> $R/q1.A2.log; exit 1; }
echo "$(ts) A2 DONE" >> $R/q1.A2.log
echo "$(ts) q1 chain done"
