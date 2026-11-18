#!/usr/bin/env bash
# B1: P4D4 with the prefill KV in GEM BOs (VMM-import libhip_gem_alloc.so bind-mounted over the image's) at
# --gpu-memory-utilization $P_UTIL, otherwise the s7b configuration (Python serial router, cache-aware admission
# 128 / 0, 1.25 TB CPU tier LRU, offload descriptor cap 512, decode single API server).
#   p4d4_run.sh TAG [gate] [score]
# gate: deployment with real MTP acceptance + scripts/gate.sh; score: fresh deployment (synthetic acceptance 2.99) +
# AgentX C104 for $DURATION s. Engine / router logs are streamed into the run directories.
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackB-20261118}
cd "$R" || exit 1
tag=${1:?tag}; shift
phases=${*:-gate score}
G=$R/gem/libhip_gem_alloc.so
P_UTIL=${P_UTIL:-0.88}
CPU=${CPU:-1250000000000}
DURATION=${DURATION:-1800}
OFF_EXTRA=${OFF_EXTRA:-"{\"cpu_bytes_to_use\":$CPU,\"eviction_policy\":\"lru\"}"}
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
export INFERA_SRC=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52-trackB/infera P_GEM=${P_GEM:-1} D_GEM=1 \
    ROUTER_BACKEND=python ROUTER_ENV=INFERA_VLLM_MOONCAKE_SERIAL=1 \
    P_ENV="VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=512 ${P_ENV_ADD:-}" \
    P_MOUNT_EXTRA="$G:/opt/infera-gem/libhip_gem_alloc.so:ro ${P_MOUNT_ADD:-}" \
    P_ARGS="--gpu-memory-utilization $P_UTIL --block-size 64 --cache-aware-admission-window 128 --cache-aware-admission-threshold 0.0" \
    D_ARGS='--block-size 64 --gpu-memory-utilization 0.95 --api-server-count 1' \
    P_KV_EXTRA="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":$OFF_EXTRA}" \
    KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=vllm-native
logkeep() {
    for c in vpd-prefill vpd-decode vpd-router; do
        ( until docker inspect "$c" > /dev/null 2>&1; do [ -e "runs/$1/samples.jsonl.stop" ] && exit 0; sleep 5; done
          docker logs -f "$c" > "runs/$1/$c.stream.log" 2>&1 ) &
    done
}
watchers() {
    ( while ! curl -sf -m 5 http://$IP:27301/v1/models > /dev/null; do [ -e runs/$1/samples.jsonl.stop ] && exit 0; sleep 15; done
      python3 scripts/metrics_sampler.py runs/$1/samples.jsonl 10 $2 http://$IP:27201 http://$IP:27301 ) &
    ( sleep 60; while ! curl -sf -m 5 http://$IP:27201/v1/models > /dev/null; do [ -e runs/$1/samples.jsonl.stop ] && exit 0; sleep 15; done
      bash scripts/hang_watch.sh http://$IP:27201 vpd-prefill runs/$1/hang 180 $2 ) &
}
teardown() {
    touch runs/$1/samples.jsonl.stop
    docker logs vpd-prefill 2>&1 | grep -c MEMORY_FAULT > runs/$1/prefill-faults.txt
    docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd vpd-agentx > /dev/null 2>&1; sleep 20
    for c in vpd-prefill vpd-decode vpd-router; do
        [ -s runs/$1/$c.stream.log ] && gzip -f runs/$1/$c.stream.log
        [ -s runs/$1/$c.log ] && gzip -f runs/$1/$c.log
    done
}
docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd vpd-agentx vpd-offD > /dev/null 2>&1; sleep 10
for phase in $phases; do
    run=${RUN_PREFIX:-b1}-$tag-$phase
    rm -rf runs/$run; mkdir -p runs/$run
    logkeep $run
    if [[ $phase == gate ]]; then
        watchers $run 3000
        if bash scripts/pd1_up.sh $run; then
            PREFILL_NUM_WORKERS=1 bash scripts/gate.sh $run "$(cat runs/$run/router_url)" "http://$IP:27201 http://$IP:27301"
        else
            echo "$(date -u +%m%d-%T) gate $run: deployment failed" | tee -a logs/summary.out
        fi
    else
        watchers $run 6000
        SMOKE=1 DURATION=$DURATION bash scripts/score_p4d4.sh $run
        echo "$(date -u +%T) $run: score rc=$?"
    fi
    teardown $run
    echo "$(date -u +%T) $run: prefill MEMORY_FAULT lines $(cat runs/$run/prefill-faults.txt)" | tee -a logs/summary.out
done
exit 0
