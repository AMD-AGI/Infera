#!/usr/bin/env bash
# e2g chain: start the host sampler at A1 "Phase profiling started"; when the A1 pipeline is DONE,
# run A2 (flush + bench, a2_e2g.yihou.sh) and start the sampler at A2 "Phase profiling started".
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
HS=/shared_nfs/yihou_glm52/v0521dbg_20261010/diag/hostsample.yihou/run_hostsample.yihou.sh
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH
ts(){ date -u +%FT%TZ; }
sampler_at_start(){ local blog=$1 tag=$2
  until grep -aq "Phase profiling (profiling) started" "$blog" 2>/dev/null; do
    grep -qE "STOP|gave up" $R/e2g.pipeline.log $R/e2g.A2.log 2>/dev/null && { echo "$(ts) $tag: stop seen, no sampler"; return 1; }
    sleep 10; done
  echo "$(ts) $tag: profiling started -> hostsample"
  nohup bash $HS crsuse2-m2m-102 207495 3600 10 $tag > $R/e2g.hostsample.$tag.log 2>&1 &
}
sampler_at_start $R/e2g/agentx-c40/benchmark.log e2gA1 || exit 1
until grep -qE "PIPELINE DONE|STOP|gave up" $R/e2g.pipeline.log; do sleep 30; done
grep -q "PIPELINE DONE" $R/e2g.pipeline.log || { echo "$(ts) A1 not done -> no A2"; exit 1; }
echo "$(ts) A1 done -> A2"
nohup bash $H/a2_e2g.yihou.sh > $R/e2g.A2.log 2>&1 &
sleep 30
sampler_at_start $R/e2g/agentx-c40-A2/benchmark.log e2gA2
echo "$(ts) chain done"
