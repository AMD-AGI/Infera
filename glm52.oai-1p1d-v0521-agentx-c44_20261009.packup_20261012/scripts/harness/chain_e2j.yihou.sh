#!/usr/bin/env bash
# e2j chain (R050/R052): [GO.e2j] -> wait e2i R2 DONE -> pipeline e2j with PRE_STOP_ENV=env.e2i (stop e2i,
# GPU-idle gate inside run.v2 up) = R1 (fresh, cold, C40) with the host sampler from R1 "Phase profiling
# started" -> on PIPELINE DONE: flush both legs (same as A2 / the 21,696 run) -> R2 (BENCH_TAG=R2) with
# the sampler again. Log lines "<ISO> e2jR1|e2jR2: profiling started -> hostsample" (repro py-spy hook).
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
HS=/shared_nfs/yihou_glm52/v0521dbg_20261010/diag/hostsample.yihou/run_hostsample.yihou.sh
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH
PL=$R/e2j.pipeline.log; L2=$R/e2j.R2.log
ts(){ date -u +%FT%TZ; }
until [[ -e $R/GO.e2j.yihou ]]; do sleep 30; done; echo "$(ts) GO.e2j present"
until grep -qE "R2 DONE|STOP" $R/e2i.R2.log 2>/dev/null || grep -qE "STOP|gave up|not done|not DONE" $R/e2i.chain.log $R/e2i.pipeline.log 2>/dev/null; do sleep 30; done
grep -q "R2 DONE" $R/e2i.R2.log || { echo "$(ts) e2i R2 not DONE -> no e2j"; exit 1; }
echo "$(ts) e2i R2 done -> e2j pipeline (pre-stop e2i)"
( cd $H && source env.e2j-102.yihou.sh && PRE_STOP_ENV=$H/env.e2i-102.yihou.sh \
  WAIT_PREFIXES="yihou- e2-yihou e0-h1-yihou dbgagent-yihou" bash pipeline.yihou.sh ) > $PL 2>&1 &
sampler_at_start(){ local blog=$1 tag=$2
  until grep -aq "Phase profiling (profiling) started" "$blog" 2>/dev/null; do
    grep -qE "STOP|gave up" $PL $L2 2>/dev/null && { echo "$(ts) $tag: stop seen, no sampler"; return 1; }
    sleep 10; done
  echo "$(ts) $tag: profiling started -> hostsample"
  nohup bash $HS crsuse2-m2m-102 207495 3600 10 $tag > $R/e2j.hostsample.$tag.log 2>&1 &
}
sleep 60
sampler_at_start $R/e2j/agentx-c40/benchmark.log e2jR1 || exit 1
until grep -qE "PIPELINE DONE|STOP|gave up" $PL; do sleep 30; done
grep -q "PIPELINE DONE" $PL || { echo "$(ts) R1 not done -> no R2"; exit 1; }
echo "$(ts) R1 done -> flush + R2"
( source $H/env.e2j-102.yihou.sh
  flush(){ local port=$1 i code
    for i in $(seq 1 60); do
      code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache")
      echo "$(ts) flush :$port try $i -> $code"; [[ "$code" == 200 ]] && return 0; sleep 10
    done; return 1; }
  flush 29357 || { echo "$(ts) STOP: decode flush failed"; exit 1; }
  flush 29101 || { echo "$(ts) STOP: prefill flush failed"; exit 1; }
  echo "$(ts) stage bench R2"
  BENCH_TAG=R2 bash $H/run.v2.yihou.sh bench || { echo "$(ts) STOP at bench R2"; exit 1; }
  echo "$(ts) R2 DONE" ) > $L2 2>&1 &
sleep 30
sampler_at_start $R/e2j/agentx-c40-R2/benchmark.log e2jR2
echo "$(ts) chain done"
