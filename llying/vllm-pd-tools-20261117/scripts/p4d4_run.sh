#!/usr/bin/env bash
# One P4D4 scoring round of the main line, parametrised by env so that a queue item is a few lines (the logic of
# queue/s13.sh: Rust router with serial vLLM Mooncake legs, cache-aware admission 128 / 0, 1.25 TB CPU tier LRU,
# offload descriptor cap 512, keep-alive 120 s on both engines). The node IP is taken from the node (no hardcoded IP).
#   p4d4_main.sh TAG          (run directory runs/p4d4-TAG)
# Env:
#   WAIT_RC      glob (relative to R) of an .rc file to wait for before starting, e.g. 'queue-n3/*-Lm1.rc'
#   DURATION     measurement seconds (1800); CPU_BYTES (1250000000000); POLICY (lru); SMOKE (1)
#   P_GEM        0: prefill without GEM, P_UTIL default 0.6; 1: GEM with the VMM-import library (U2), P_UTIL default 0.88
#   P_API / D_API  api servers per engine (1 / 1); P_RENDER / D_RENDER: --renderer-num-workers (empty = vLLM default 1;
#                the render cost is tokenization, ~100-180 ms per 90-170K-token AgentX prompt, which releases the GIL)
#   RENDER_TIMEOUT  VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT on both engines (empty = vLLM default 30 s)
#   D_MODE       v2 (s13: #47897 overlay v2 + VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS=4), v3comm (track A: overlay v3 +
#                aiter AG/RS in FULL capture), stock
#   P_ENV_X / D_ENV_X / ROUTER_ENV_X / ALL_ENV_X   extra K=V (ALL_ENV_X goes to prefill, decode and router)
#   P_ARGS_X / D_ARGS_X   extra engine args;  P_MOUNT_X / D_MOUNT_X   extra src:dst[:ro] mounts
#   FAULT        1 (default): PYTHONFAULTHANDLER=1 everywhere, per-container process list every 60 s (host PIDs, to match
#                the kernel log), kernel log tail before and after the run
#   TRACE        1: 3 s decode torch trace TRACE_AT s (1500) into the measurement
#   PTRACE       1: prefill torch trace (PP workers) of PTRACE_S s (6), PTRACE_AT s (1200) into the measurement
#   ROUTER_BIN   Rust router binary under bin/ (infera-router-vllmserial); USAGE_LOG=1: the router logs every prefill's
#                prompt / cached tokens (bin/infera-router-vllmserial-usage + --enable-prompt-tokens-details on the prefill)
# Queue items run this through scripts/snap.sh (a snapshot), so editing this file does not affect queued items.
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
cd "$R" || exit 1
tag=${1:?tag}
run=p4d4-$tag
V=/usr/local/lib/python3.12/dist-packages/vllm
IMAGE=${IMAGE:-infera-vllm:v0340-0c5e5e89}
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
if [[ -n "${WAIT_RC:-}" ]]; then
    while ! compgen -G "$WAIT_RC" > /dev/null; do sleep 20; ls "$(dirname "$WAIT_RC")" > /dev/null; done
fi
docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd vpd-agentx vpd-aggK vpd-aggR > /dev/null 2>&1; sleep 30
mkdir -p "runs/$run/trace" && chmod 777 "runs/$run/trace"
# U23: pinned CPU-tier memory left behind by a crashed run (Shmem with no owner) makes the next 1.25 TB tier OOM the host.
shm_gb=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo)
if (( shm_gb > ${SHMEM_MAX_GB:-200} )); then
    echo "$(date -u +%T) $run: NOT STARTED, Shmem ${shm_gb} GB > ${SHMEM_MAX_GB:-200} GB before deployment (leaked pinned memory?)" | tee -a logs/summary.out
    ps -eo pid,stat,etimes,rss,args --sort=-rss | head -15 > "runs/$run/shmem-blocked-ps.txt"
    exit 0
fi
stop=runs/$run/samples.jsonl.stop
rm -f "$stop"
klog() { docker run --rm --cap-add SYSLOG --network none --entrypoint bash "$IMAGE" -c "dmesg -T | tail -$1" 2>&1; }
FAULT=${FAULT:-1}
fh=""
if [[ "$FAULT" == 1 ]]; then
    fh="PYTHONFAULTHANDLER=1"
    klog 60 > "runs/$run/dmesg-before.txt"
    grep -E '^(MemAvailable|Shmem):' /proc/meminfo > "runs/$run/meminfo-before.txt"
    ( until docker inspect vpd-decode > /dev/null 2>&1; do [ -e "$stop" ] && exit 0; sleep 5; done
      while [ ! -e "$stop" ]; do
          for c in vpd-prefill vpd-decode vpd-router vpd-agentx; do
              docker top "$c" -eo pid,ppid,etimes,rss,args 2>/dev/null | cut -c1-220 | sed "s/^/$(date -u +%T) $c /"
          done
          echo "$(date -u +%T) meminfo $(grep -E '^(MemAvailable|Cached|Shmem|AnonPages|Mlocked|Unevictable):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
          sleep 60
      done > "runs/$run/ps.log" ) &
fi
for c in vpd-prefill vpd-decode vpd-router; do
    ( until docker inspect "$c" > /dev/null 2>&1; do [ -e "$stop" ] && exit 0; sleep 5; done
      docker logs -f "$c" > "runs/$run/$c.stream.log" 2>&1 ) &
done
( until docker inspect vpd-router > /dev/null 2>&1; do [ -e "$stop" ] && exit 0; sleep 5; done
  while docker inspect vpd-router > /dev/null 2>&1 && [ ! -e "$stop" ]; do
      echo "$(date -u +%T) $(docker exec vpd-router sh -c 'ls /proc/1/fd | wc -l' 2>&1) $(curl -s -m 60 -o /dev/null -w '%{time_total}' "http://$IP:28300/metrics")"
      sleep 30
  done > "runs/$run/router-fds-latency.log" ) &
( while ! curl -sf -m 5 "http://$IP:27301/v1/models" > /dev/null; do [ -e "$stop" ] && exit 0; sleep 15; done
  python3 scripts/metrics_sampler.py "runs/$run/samples.jsonl" 10 9000 "http://$IP:27201" "http://$IP:27301" ) &
( sleep 60; while ! curl -sf -m 5 "http://$IP:27201/v1/models" > /dev/null; do [ -e "$stop" ] && exit 0; sleep 15; done
  bash scripts/hang_watch2.sh "http://$IP:27201" vpd-prefill "runs/$run/hang" prefill 180 9000 ) &
( sleep 60; while ! curl -sf -m 5 "http://$IP:27301/v1/models" > /dev/null; do [ -e "$stop" ] && exit 0; sleep 15; done
  bash scripts/hang_watch2.sh "http://$IP:27301" vpd-decode "runs/$run/hang" decode 120 9000 ) &

P_GEM=${P_GEM:-0}
if [[ "$P_GEM" == 1 ]]; then
    P_UTIL=${P_UTIL:-0.88}
    p_mount="$R/bin/libhip_gem_alloc-vmm.so:/opt/infera-gem/libhip_gem_alloc.so:ro"
else
    P_UTIL=${P_UTIL:-0.6}
    p_mount=""
fi
rt=${RENDER_TIMEOUT:+VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT=$RENDER_TIMEOUT}
p_args="--gpu-memory-utilization $P_UTIL --block-size 64 --cache-aware-admission-window 128 --cache-aware-admission-threshold 0.0 --api-server-count ${P_API:-1}"
[[ -n "${P_RENDER:-}" ]] && p_args+=" --renderer-num-workers $P_RENDER"
d_args="--block-size 64 --gpu-memory-utilization 0.95 --api-server-count ${D_API:-1}"
[[ -n "${D_RENDER:-}" ]] && d_args+=" --renderer-num-workers $D_RENDER"
case "${D_MODE:-v2}" in
    v2) d_env="VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1 VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS=4"
        d_mount="$R/overlay/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:$V/v1/attention/ops/rocm_aiter_mla_sparse.py:ro $R/overlay/vllm/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:$V/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:ro" ;;
    v3comm) O=$R/overlay-v3comm/vllm
        d_env="VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1"
        d_mount="$O/v1/attention/ops/rocm_aiter_mla_sparse.py:$V/v1/attention/ops/rocm_aiter_mla_sparse.py:ro $O/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:$V/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:ro $O/distributed/device_communicators/cuda_communicator.py:$V/distributed/device_communicators/cuda_communicator.py:ro" ;;
    stock) d_env="" d_mount="" ;;
    *) echo "unknown D_MODE ${D_MODE}"; exit 1 ;;
esac
if [[ "${PTRACE:-0}" == 1 ]]; then
    mkdir -p "runs/$run/ptrace" && chmod 777 "runs/$run/ptrace"
    p_args+=" --profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/ptrace\",\"torch_profiler_with_stack\":false}'"
    p_mount+=" $R/runs/$run/ptrace:/ptrace"
    ( while ! grep -qa "Phase profiling (profiling) started" "runs/$run/agentx-c104/benchmark.log" 2>/dev/null; do [ -e "$stop" ] && exit 0; sleep 30; done; sleep "${PTRACE_AT:-1200}"
      bash scripts/trace_decode.sh "http://$IP:27201" "${PTRACE_S:-6}" > "runs/$run/ptrace/trace.log" 2>&1 ) &
fi
if [[ "${TRACE:-0}" == 1 ]]; then
    d_args+=" --profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/trace\",\"torch_profiler_with_stack\":false}'"
    d_mount+=" $R/runs/$run/trace:/trace"
    ( while ! grep -qa "Phase profiling (profiling) started" "runs/$run/agentx-c104/benchmark.log" 2>/dev/null; do [ -e "$stop" ] && exit 0; sleep 30; done; sleep "${TRACE_AT:-1500}"
      bash scripts/trace_decode.sh "http://$IP:27301" 3 > "runs/$run/trace/trace.log" 2>&1 ) &
fi
cpu=${CPU_BYTES:-1250000000000}
rbin=${ROUTER_BIN:-infera-router-vllmserial}
if [[ "${USAGE_LOG:-0}" == 1 ]]; then
    rbin=${ROUTER_BIN:-infera-router-vllmserial-usage}
    ROUTER_ENV_X="${ROUTER_ENV_X:-} INFERA_LOG_PREFILL_USAGE=1"
    p_args+=" --enable-prompt-tokens-details"
fi
echo "$(date -u +%T) $run on $(hostname -s) ($IP): P_GEM=$P_GEM P_UTIL=$P_UTIL D_MODE=${D_MODE:-v2} FAULT=$FAULT DURATION=${DURATION:-1800}"
SMOKE=${SMOKE:-1} DURATION=${DURATION:-1800} INFERA_SRC=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/infera P_GEM=$P_GEM D_GEM=1 \
    ROUTER_BACKEND=rust ROUTER_ENV="INFERA_VLLM_MOONCAKE_SERIAL=1 $fh ${ALL_ENV_X:-} ${ROUTER_ENV_X:-}" \
    ROUTER_MOUNT_EXTRA="$R/bin/$rbin:/usr/local/bin/infera-router:ro" \
    P_ENV="VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=512 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 $rt $fh ${ALL_ENV_X:-} ${P_ENV_X:-}" \
    D_ENV="$d_env VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 $rt $fh ${ALL_ENV_X:-} ${D_ENV_X:-}" \
    P_ARGS="$p_args ${P_ARGS_X:-}" P_MOUNT_EXTRA="$p_mount ${P_MOUNT_X:-}" \
    D_ARGS="$d_args ${D_ARGS_X:-}" D_MOUNT_EXTRA="$d_mount ${D_MOUNT_X:-}" \
    P_KV_EXTRA="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$cpu,\"eviction_policy\":\"${POLICY:-lru}\"}}" \
    KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=vllm-native bash scripts/score_p4d4.sh "$run"
rc=$?
echo "$(date -u +%T) $run: score rc=$rc"
touch "$stop"
if [[ "$FAULT" == 1 ]]; then
    for c in vpd-prefill vpd-decode vpd-router; do
        docker inspect -f "$c {{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} finished={{.State.FinishedAt}}" "$c" 2>&1
    done > "runs/$run/container-state.txt"
    klog 300 > "runs/$run/dmesg-after.txt"
fi
docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd vpd-agentx > /dev/null 2>&1; sleep 30
[[ "$FAULT" == 1 ]] && grep -E '^(MemAvailable|Shmem):' /proc/meminfo > "runs/$run/meminfo-after.txt"
for c in vpd-prefill vpd-decode vpd-router; do
    if [ -s "runs/$run/$c.stream.log" ]; then
        rm -f "runs/$run/$c.log"; gzip -f "runs/$run/$c.stream.log"
    elif [ -s "runs/$run/$c.log" ]; then
        gzip -f "runs/$run/$c.log"
    fi
done
exit 0
