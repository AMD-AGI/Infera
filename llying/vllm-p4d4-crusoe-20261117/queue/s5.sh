export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/liyingli USER=liyingli R=/home/liyingli/bench_agentx/vllm-pd-20261117 NODE0=$(hostname -s)
while [ ! -e queue/1117-221722-Lp4d4s4.rc ]; do sleep 20; ls queue > /dev/null; done
docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd > /dev/null 2>&1; sleep 45
for cpu in 1250000000000 1000000000000; do
  run=p4d4-s5-pyserial-caa-off$((cpu / 1000000000))g-desc512
  mkdir -p runs/$run/trace && chmod 777 runs/$run/trace
  ( while ! curl -sf -m 5 http://10.245.144.166:27301/v1/models > /dev/null; do sleep 15; done
    python3 scripts/metrics_sampler.py runs/$run/samples.jsonl 10 9000 http://10.245.144.166:27201 http://10.245.144.166:27301 ) &
  ( sleep 60; while ! curl -sf -m 5 http://10.245.144.166:27201/v1/models > /dev/null; do sleep 15; done
    bash scripts/hang_watch.sh http://10.245.144.166:27201 vpd-prefill runs/$run/hang 180 9000 ) &
  ( while ! grep -qa "Phase profiling (profiling) started" runs/$run/agentx-c104/benchmark.log 2>/dev/null; do sleep 30; done; sleep 1800
    bash scripts/trace_decode.sh http://10.245.144.166:27301 3 > runs/$run/trace/trace.log 2>&1 ) &
  SMOKE=1 DURATION=3600 INFERA_SRC=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/infera P_GEM=0 D_GEM=1 \
    ROUTER_BACKEND=python ROUTER_ENV=INFERA_VLLM_MOONCAKE_SERIAL=1 \
    P_ENV='VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=512' \
    P_ARGS='--gpu-memory-utilization 0.6 --block-size 64 --cache-aware-admission-window 128 --cache-aware-admission-threshold 0.0' \
    D_ARGS="--block-size 64 --gpu-memory-utilization 0.95 --api-server-count 1 --profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/trace\",\"torch_profiler_with_stack\":false}'" \
    D_MOUNT_EXTRA="$R/runs/$run/trace:/trace" \
    P_KV_EXTRA="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$cpu}}" \
    KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=vllm-native bash scripts/score_p4d4.sh $run
  rc=$?
  touch runs/$run/samples.jsonl.stop
  grep -q "deployment failed" <(tail -1 logs/summary.out) || exit $rc
  echo "$(date -u +%T) $run: deployment failed, retrying with a smaller CPU tier"
  docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd > /dev/null 2>&1; sleep 60
done
