#!/usr/bin/env bash
# e2j R2 (R056 resume on slog-006): the chain_e2j R2 leg alone, on the live e2j stack.
# Flush decode then prefill (retry until 200), bench C40 BENCH_TAG=R2 -> runs/e2j/agentx-c40-R2,
# host sampler from "Phase profiling started". Launch with setsid nohup; the AgentX client is a
# node-side docker container writing to NFS, so it survives a login-host loss (copy-back not needed,
# NODE_RW_ROOT is empty). Does NOT stop the stack.
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
HS=/shared_nfs/yihou_glm52/v0521dbg_20261010/diag/hostsample.yihou/run_hostsample.yihou.sh
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS=crsuse2-m2m-102:207495
L2=$R/e2j.R2.log
ts(){ date -u +%FT%TZ; }
cd $H && source $H/env.e2j-102.yihou.sh
echo "$(ts) e2j R2 resume (slog-006)" >> $L2
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code" >> $L2; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
flush 29357 || { echo "$(ts) STOP: decode flush failed" >> $L2; exit 1; }
flush 29101 || { echo "$(ts) STOP: prefill flush failed" >> $L2; exit 1; }
sampler(){ local blog=$R/e2j/agentx-c40-R2/benchmark.log
  until grep -aq "Phase profiling (profiling) started" "$blog" 2>/dev/null; do
    grep -qE "STOP|R2 DONE" $L2 && { echo "$(ts) e2jR2: stop seen, no sampler" >> $L2; return 1; }
    sleep 10; done
  echo "$(ts) e2jR2: profiling started -> hostsample" >> $L2
  setsid nohup bash $HS crsuse2-m2m-102 207495 3600 10 e2jR2 > $R/e2j.hostsample.e2jR2.log 2>&1 < /dev/null &
}
sampler &
echo "$(ts) stage bench R2" >> $L2
BENCH_TAG=R2 bash $H/run.v2.yihou.sh bench >> $L2 2>&1 || { echo "$(ts) STOP at bench R2" >> $L2; exit 1; }
echo "$(ts) R2 DONE" >> $L2
