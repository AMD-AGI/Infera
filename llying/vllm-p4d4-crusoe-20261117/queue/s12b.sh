export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/liyingli USER=liyingli R=/home/liyingli/bench_agentx/vllm-pd-20261117 NODE0=$(hostname -s)
# s12b: s12 + VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 on both engines. s11 aborted on a root request that got 502 "decode
# unreachable: error sending request" from the Rust router: reqwest keeps idle connections 90 s, uvicorn closes them
# after 5 s, so the router can write onto a connection the engine is closing (more API servers, more of it).
# s12: combined configuration = s11 + prefill KV in GEM BOs at --gpu-memory-utilization 0.88 (track B: VMM-import
# libhip_gem_alloc.so, fixes the GEM + offload fault U2; b1-r1 +10.3%) + sglang #47897 gluon paged MQA logits on the
# decode (track A: in-engine decode step -3.9%; v2 overlay with VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS=4).
# Two single-variable 1800 s A/Bs against s4 (Python serial router + cache-aware admission 128/0 + 1 TB CPU tier LRU +
# offload descriptor cap 512 + decode api-server 1):
#   s6c  eviction policy ARC instead of LRU (decode torch trace near the end of measurement)
#   s7   CPU tier 1.25 TB instead of 1 TB
# Engine and router logs are streamed into each run directory (logs/engines/*.log is overwritten by the next run).
while ! ls queue/*-Lp4d4s12.rc > /dev/null 2>&1; do sleep 20; ls queue > /dev/null; done
docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd vpd-agentx vpd-aggK vpd-aggR > /dev/null 2>&1; sleep 30
logkeep() {
  for c in vpd-prefill vpd-decode vpd-router; do
    ( until docker inspect "$c" > /dev/null 2>&1; do [ -e "runs/$1/samples.jsonl.stop" ] && exit 0; sleep 5; done
      docker logs -f "$c" > "runs/$1/$c.log" 2>&1 ) &
  done
}
for spec in "s12b 1250000000000 lru 0"; do
  read -r tag cpu policy trace <<< "$spec"
  run=p4d4-$tag-rustserial-caa-off$((cpu / 1000000000))g-$policy-desc512-api4r8-pgem088-kf47897-ka120
  mkdir -p runs/$run/trace && chmod 777 runs/$run/trace
  logkeep $run
  ( while ! curl -sf -m 5 http://10.245.144.166:27301/v1/models > /dev/null; do [ -e runs/$run/samples.jsonl.stop ] && exit 0; sleep 15; done
    sleep 120; echo "gluon log lines: $(docker logs vpd-decode 2>&1 | grep -c 'gluon paged-MQA logits')" > runs/$run/gluon-check.txt ) &
  ( until docker inspect vpd-router > /dev/null 2>&1; do [ -e runs/$run/samples.jsonl.stop ] && exit 0; sleep 5; done
    while docker inspect vpd-router > /dev/null 2>&1 && [ ! -e runs/$run/samples.jsonl.stop ]; do
      echo "$(date -u +%T) $(docker exec vpd-router sh -c 'ls /proc/1/fd | wc -l; ls -l /proc/1/fd | sed -n "s/.*-> //p" | sed "s/:.*//" | sort | uniq -c | tr "\n" " "' 2>&1 | tr '\n' ' ')"
      sleep 30
    done > runs/$run/router-fds.log ) &
  ( until docker inspect vpd-router > /dev/null 2>&1; do [ -e runs/$run/samples.jsonl.stop ] && exit 0; sleep 5; done
    while [ ! -e runs/$run/samples.jsonl.stop ]; do
      echo "$(date -u +%T) $(curl -s -m 60 -o /dev/null -w '%{time_total}' http://10.245.144.166:28300/metrics)"; sleep 30
    done > runs/$run/router-latency.log ) &
  ( while ! curl -sf -m 5 http://10.245.144.166:27301/v1/models > /dev/null; do [ -e runs/$run/samples.jsonl.stop ] && exit 0; sleep 15; done
    python3 scripts/metrics_sampler.py runs/$run/samples.jsonl 10 6000 http://10.245.144.166:27201 http://10.245.144.166:27301 ) &
  d_args='--block-size 64 --gpu-memory-utilization 0.95 --api-server-count 4 --renderer-num-workers 8'
  d_mount=
  if [ "$trace" = 1 ]; then
    d_args="$d_args --profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/trace\",\"torch_profiler_with_stack\":false}'"
    d_mount="$R/runs/$run/trace:/trace"
    ( while ! grep -qa "Phase profiling (profiling) started" runs/$run/agentx-c104/benchmark.log 2>/dev/null; do [ -e runs/$run/samples.jsonl.stop ] && exit 0; sleep 30; done; sleep 1500
      bash scripts/trace_decode.sh http://10.245.144.166:27301 3 > runs/$run/trace/trace.log 2>&1 ) &
  fi
  SMOKE=1 DURATION=1800 INFERA_SRC=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/infera P_GEM=1 D_GEM=1 \
    ROUTER_BACKEND=rust ROUTER_ENV=INFERA_VLLM_MOONCAKE_SERIAL=1 ROUTER_MOUNT_EXTRA="$R/bin/infera-router-vllmserial:/usr/local/bin/infera-router:ro" \
    P_ENV='VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=512 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120' P_MOUNT_EXTRA="$R/bin/libhip_gem_alloc-vmm.so:/opt/infera-gem/libhip_gem_alloc.so:ro" \
    D_ENV='VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1 VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS=4 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120' \
    P_ARGS='--gpu-memory-utilization 0.88 --block-size 64 --cache-aware-admission-window 128 --cache-aware-admission-threshold 0.0 --api-server-count 4 --renderer-num-workers 8' \
    D_ARGS="$d_args" D_MOUNT_EXTRA="$d_mount $R/overlay/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:ro $R/overlay/vllm/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:ro" \
    P_KV_EXTRA="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$cpu,\"eviction_policy\":\"$policy\"}}" \
    KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=vllm-native bash scripts/score_p4d4.sh $run
  echo "$(date -u +%T) $run: score rc=$?"
  touch runs/$run/samples.jsonl.stop
  docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd vpd-agentx > /dev/null 2>&1; sleep 30
  for c in vpd-prefill vpd-decode vpd-router; do [ -s runs/$run/$c.log ] && gzip -f runs/$run/$c.log; done
done
exit 0
