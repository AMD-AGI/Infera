#!/usr/bin/env bash
# B2: aggregated PP4 on GPUs 4-7 with vLLM's TieringOffloadingSpec (CPU primary tier in /dev/shm + FS secondary tier on
# the local NVMe), a small GPU cache and CPU tier so tier_probe.py's repeats are read back from the FS tier.
#   b2_agg_tier.sh TAG
# Env: OVERLAY (1: bind-mount $R/overlay/vllm/** over the image's vllm), CPU_BYTES (2 GB), GPU_BLOCKS (800 x 64 tokens),
#      SPEC_NAME (TieringOffloadingSpec; CPUOffloadingSpec = no secondary tier), N_PROMPTS (6), TOKENS (24000),
#      CONNECTOR (OffloadingConnector | SimpleCPUOffloadConnector), LAZY (1, Simple only).
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackB-20261118}
cd "$R" || exit 1
tag=${1:?tag}
out=$R/runs/b2-agg-$tag
rm -rf "$out"; mkdir -p "$out"
CPU_BYTES=${CPU_BYTES:-2000000000} GPU_BLOCKS=${GPU_BLOCKS:-800} SPEC_NAME=${SPEC_NAME:-TieringOffloadingSpec}
KVFS=/mnt/m2m_nobackup/liyingli_vpd/kvfs-trackB/agg-$tag
V=/usr/local/lib/python3.12/dist-packages/vllm
mounts=""
if [[ "${OVERLAY:-1}" == 1 ]]; then
    for f in $(cd overlay/vllm && find . -name '*.py' | sed 's#^\./##'); do mounts+=" $R/overlay/vllm/$f:$V/$f:ro"; done
fi
mounts+=" ${MOUNT_ADD:-}"
extra="{\"spec_name\":\"$SPEC_NAME\",\"cpu_bytes_to_use\":$CPU_BYTES"
if [[ $SPEC_NAME == TieringOffloadingSpec ]]; then
    rm -rf "$KVFS"; mkdir -p "$KVFS"; chmod 777 "$KVFS"
    mounts+=" $KVFS:/kvfs"
    extra+=",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"/kvfs\",\"n_read_threads\":32,\"n_write_threads\":16}]"
fi
extra+="}"
CONNECTOR=${CONNECTOR:-OffloadingConnector}
# SimpleCPUOffloadConnector (SPEC_NAME ignored): LAZY=1 stores GPU blocks only as they near eviction.
[[ $CONNECTOR == SimpleCPUOffloadConnector ]] && extra="{\"cpu_bytes_to_use\":$CPU_BYTES,\"lazy_offload\":$([[ ${LAZY:-1} == 1 ]] && echo true || echo false)}"
kv="{\"kv_connector\":\"$CONNECTOR\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":$extra}"
echo "$(date -u +%T) b2-agg $tag OVERLAY=${OVERLAY:-1} SPEC=$SPEC_NAME CPU_BYTES=$CPU_BYTES GPU_BLOCKS=$GPU_BLOCKS" | tee "$out/config.txt"
echo "$kv" >> "$out/config.txt"; echo "mounts:$mounts" >> "$out/config.txt"
IMAGE=infera-vllm:v0340-0c5e5e89 MOUNT_EXTRA="$mounts" ENV_EXTRA="VLLM_PP_LAYER_PARTITION=20,20,20,18 VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=512 ${ENV_ADD:-}" \
READY_TIMEOUT=1500 bash scripts/vllm_up.sh vpd-b2agg 4,5,6,7 8300 --pipeline-parallel-size 4 --tensor-parallel-size 1 \
    --kv-cache-dtype fp8_e4m3 --enable-prefix-caching --block-size 64 --num-gpu-blocks-override "$GPU_BLOCKS" --max-model-len 32768 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --max-num-seqs 64 --max-num-batched-tokens 8192 \
    --kv-transfer-config "$kv" > "$out/up.log" 2>&1
up=$?
tail -3 "$out/up.log"
probe=-
if [[ $up == 0 ]]; then
    timeout 1800 python3 scripts/tier_probe.py http://127.0.0.1:8300 glm5.2-mxfp4 "$out/probe.jsonl" "${N_PROMPTS:-6}" "${TOKENS:-24000}" 3 > "$out/probe.out" 2>&1
    probe=$(tail -1 "$out/probe.out")
    curl -sf -m 10 http://127.0.0.1:8300/metrics | grep -E '^vllm:(kv_offload|external_prefix|prefix_cache)' > "$out/metrics.txt"
fi
docker logs vpd-b2agg > "$out/engine.log" 2>&1
docker rm -f vpd-b2agg > /dev/null 2>&1
[[ -d $KVFS ]] && { du -sb "$KVFS" > "$out/kvfs-du.txt" 2>&1; find "$KVFS" -type f | wc -l >> "$out/kvfs-du.txt"
    docker run --rm -v "$(dirname "$KVFS"):/k" --entrypoint bash infera-vllm:v0340-0c5e5e89 -c "rm -rf /k/$(basename "$KVFS")"; }
echo "$(date -u +%T) b2-agg $tag: up=$up probe=$probe faults=$(grep -c MEMORY_FAULT "$out/engine.log") timeouts=$(grep -c 'Timed out waiting for mmap' "$out/engine.log")" | tee -a "$out/config.txt" logs/summary.out
exit 0
