export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/liyingli USER=liyingli R=/home/liyingli/bench_agentx/vllm-pd-20261117 NODE0=$(hostname -s)
# s6: s5 + a write-through filesystem tier (vLLM TieringOffloadingSpec, type fs) on the node's local NVMe xfs volume.
# The FS tier persists across runs and AgentX replays the same prompts, so each run gets a fresh directory and it is
# deleted afterwards. Fallback when the tiered deployment fails: same CPU tier with the ARC eviction policy.
while [ ! -e queue/1117-230548-Lp4d4s5.rc ]; do sleep 20; ls queue > /dev/null; done
docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd > /dev/null 2>&1; sleep 45
cpu=1250000000000
ls runs/p4d4-s5-pyserial-caa-off1250g-desc512/agentx-c104/agentx_conc*.json > /dev/null 2>&1 || cpu=1000000000000
KVFS=/mnt/m2m_nobackup/liyingli_vpd/kvfs
for variant in fs arc; do
  if [ $variant = fs ]; then
    run=p4d4-s6-pyserial-caa-off$((cpu / 1000000000))g-fs-desc512
    mkdir -p $KVFS && rm -rf $KVFS/$run && mkdir -p $KVFS/$run && chmod 777 $KVFS/$run
    extra="{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$cpu,\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"/kvfs\",\"n_read_threads\":32,\"n_write_threads\":16}]}"
    pmount="$KVFS/$run:/kvfs"
  else
    run=p4d4-s6b-pyserial-caa-off$((cpu / 1000000000))g-arc-desc512
    extra="{\"cpu_bytes_to_use\":$cpu,\"eviction_policy\":\"arc\"}"
    pmount=
  fi
  mkdir -p runs/$run
  ( while ! curl -sf -m 5 http://10.245.144.166:27301/v1/models > /dev/null; do sleep 15; done
    python3 scripts/metrics_sampler.py runs/$run/samples.jsonl 10 9000 http://10.245.144.166:27201 http://10.245.144.166:27301 ) &
  ( sleep 60; while ! curl -sf -m 5 http://10.245.144.166:27201/v1/models > /dev/null; do sleep 15; done
    bash scripts/hang_watch.sh http://10.245.144.166:27201 vpd-prefill runs/$run/hang 180 9000 ) &
  [ $variant = fs ] && ( for i in $(seq 540); do echo "$(date -u +%T) $(df -B1 --output=used,avail $KVFS | tail -1)"; [ -e runs/$run/samples.jsonl.stop ] && break; sleep 20; done > runs/$run/kvfs-df.log ) &
  SMOKE=1 DURATION=3600 INFERA_SRC=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/infera P_GEM=0 D_GEM=1 \
    ROUTER_BACKEND=python ROUTER_ENV=INFERA_VLLM_MOONCAKE_SERIAL=1 \
    P_ENV='VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=512' P_MOUNT_EXTRA="$pmount" \
    P_ARGS='--gpu-memory-utilization 0.6 --block-size 64 --cache-aware-admission-window 128 --cache-aware-admission-threshold 0.0' \
    D_ARGS='--block-size 64 --gpu-memory-utilization 0.95 --api-server-count 1' \
    P_KV_EXTRA="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":$extra}" \
    KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=vllm-native bash scripts/score_p4d4.sh $run
  rc=$?
  touch runs/$run/samples.jsonl.stop
  for e in prefill decode; do gzip -c logs/engines/vpd-$e.log > runs/$run/vpd-$e.log.gz; done
  [ $variant = fs ] && { du -sb $KVFS/$run > runs/$run/kvfs-du.txt 2>&1; find $KVFS/$run -name '*.bin' | wc -l >> runs/$run/kvfs-du.txt; ( ionice -c3 rm -rf $KVFS/$run ) & }
  grep -qE "deployment failed|smoke did not complete" <(tail -1 logs/summary.out) || exit $rc
  echo "$(date -u +%T) $run: deployment failed, falling back"
  docker logs --tail 200 vpd-prefill > runs/$run/prefill-fail.log 2>&1
  docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd > /dev/null 2>&1; sleep 60
done
