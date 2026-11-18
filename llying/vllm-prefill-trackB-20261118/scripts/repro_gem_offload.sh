#!/usr/bin/env bash
# Aggregated PP4 on GPUs 4-7 (no PD): MultiConnector[GEM Mooncake + OffloadingConnector], then the smoke; counts
# HSA memory faults in the engine log.   repro_gem_offload.sh TAG
# Env: CPU_BYTES (100 GB), GEM (1; 0 = plain hipMalloc KV, no GEM preload), OFF_EXTRA (extra JSON members for the
#      OffloadingConnector extra config), ENV_ADD (extra K=V), MOUNT_ADD (extra src:dst[:ro]), ARGS_ADD (extra vllm args),
#      INFERA_SRC (infera/ tree mounted over the image's), KEEP=1 leaves the container running.
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackB-20261118}
tag=${1:?tag}
out=$R/runs/repro-$tag
mkdir -p "$out"
INFERA_SRC=${INFERA_SRC:-/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52-trackB/infera}
CPU_BYTES=${CPU_BYTES:-100000000000}
GEM=${GEM:-1}
name=vpd-offD
gem_env=
mc_mod=
if [[ "$GEM" == 1 ]]; then
    gem_env="INFERA_AMD_GEM_ALLOCATOR=/opt/infera-gem/libhip_gem_alloc.so LD_PRELOAD=/opt/infera-gem/libhip_gem_alloc.so"
    mc_mod='"kv_connector_module_path":"infera.engine.vllm.gem_mooncake",'
fi
kv="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"MooncakeConnector\",$mc_mod\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_workers\":10,\"device_name\":\"ionic_4,ionic_5,ionic_6,ionic_7\"}},{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$CPU_BYTES${OFF_EXTRA:+,$OFF_EXTRA}}}]}}"
echo "$(date -u +%T) repro $tag GEM=$GEM CPU_BYTES=$CPU_BYTES ENV_ADD=${ENV_ADD:-} MOUNT_ADD=${MOUNT_ADD:-} ARGS_ADD=${ARGS_ADD:-}" | tee "$out/config.txt"
echo "$kv" >> "$out/config.txt"
IMAGE=infera-vllm:v0340-0c5e5e89 MOUNT_EXTRA="$INFERA_SRC:/opt/infera/infera:ro ${MOUNT_ADD:-}" \
ENV_EXTRA="VLLM_PP_LAYER_PARTITION=20,20,20,18 MC_GID_INDEX=1 MC_DISABLE_HIP=1 MOONCAKE_DISABLE_HIP_DMABUF=0 MC_HIP_KFD_PIN_FIX=0 VLLM_MOONCAKE_BOOTSTRAP_PORT=28498 $gem_env ${ENV_ADD:-}" \
bash "$R/scripts/vllm_up.sh" $name 4,5,6,7 8200 --pipeline-parallel-size 4 --tensor-parallel-size 1 --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --max-num-seqs 64 \
    --max-num-batched-tokens 8192 --kv-transfer-config "$kv" ${ARGS_ADD:-}
up=$?
smoke_rc=-
if [[ $up == 0 ]]; then
    PREFILL_NUM_WORKERS=0 timeout 900 python3 "$R/scripts/probes/smoke.py" http://127.0.0.1:8200 glm5.2-mxfp4 "$out/smoke" > "$out/smoke.out" 2>&1
    smoke_rc=$?
    sleep 5
fi
docker logs $name > "$out/engine.log" 2>&1
faults=$(grep -c MEMORY_FAULT "$out/engine.log")
passed=$(grep -c '"passed": true' "$out/smoke.out" 2>/dev/null)
total=$(grep -c '"passed"' "$out/smoke.out" 2>/dev/null)
[[ "${KEEP:-0}" == 1 ]] || docker rm -f $name > /dev/null 2>&1
echo "$(date -u +%T) repro $tag: up=$up smoke_rc=$smoke_rc smoke=$passed/$total faults=$faults" | tee -a "$out/config.txt" "$R/logs/summary.out"
exit 0
