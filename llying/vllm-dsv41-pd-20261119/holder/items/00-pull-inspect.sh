#!/usr/bin/env bash
# Pull the vLLM ROCm nightly + etcd images on the node, record what the image and the host carry (driver, VRAM
# registration path, libionic, Mooncake build options), and check that amd-infera 0.3.3 + the vllm_glm52_agentx
# overlay installs and parses a DSv4.1 command line in the image.
set -uo pipefail
RUN=00-pull-inspect
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
say "start on $(hostname -s) ($(node_ip)), job $SLURM_JOB_ID; amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null), VRAM MR mode $(rdma_mr_mode)"
node_state start
t0=$SECONDS
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed: $(tail -3 "$R/pull.log" | tr '\n' ' ')"; exit 1; }
say "images ready in $((SECONDS - t0)) s: $(docker image inspect -f '{{.Id}} {{.Size}}' "$IMAGE")"
{
    echo "== host"; uname -r; cat /sys/module/amdgpu/version 2>/dev/null; amd-smi version 2>/dev/null | head -3
    lsmod | grep -E "amdgpu|peer|amdp2p|ionic" ; ls -la /usr/lib/x86_64-linux-gnu/libionic.so* 2>/dev/null; ibv_devices 2>/dev/null
    for d in $(ls /sys/class/infiniband 2>/dev/null); do echo "$d $(cat /sys/class/infiniband/$d/ports/1/state 2>/dev/null) $(cat /sys/class/infiniband/$d/ports/1/gids/1 2>/dev/null)"; done
    ip -4 -o addr | awk '{print $2, $4}'
    free -g | head -2; nproc; df -h /scratch 2>/dev/null | tail -1
    docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
} > "$R/host.txt" 2>&1
docker run --rm --network host --device /dev/kfd --device /dev/dri --device /dev/infiniband $(host_gids) \
    --ipc host --security-opt seccomp=unconfined -v /perf_apps:/perf_apps --entrypoint "" "$IMAGE" bash -c '
echo "== /app/versions.txt"; cat /app/versions.txt 2>/dev/null
echo "== python"; python3 --version; ldd --version | head -1; echo "ROCM_PATH=$ROCM_PATH"; ls -d /opt/rocm 2>&1
python3 - <<PY
import importlib, importlib.metadata as md
for d in ("vllm", "torch", "amd-aiter", "aiter", "mori", "nixl", "nixl-rocm", "mooncake-transfer-engine-rocm",
          "mooncake-transfer-engine", "transformers", "tokenizers", "fastapi", "uvicorn", "httpx", "msgspec", "msgpack",
          "pyzmq", "xxhash", "prometheus-client", "nats-py", "triton", "flydsl", "lmcache"):
    try:
        print(f"{d}=={md.version(d)}")
    except Exception:
        print(f"{d}: not installed")
for m in ("mori", "nixl", "mooncake.engine"):
    try:
        importlib.import_module(m); print("import", m, "ok")
    except Exception as e:
        print("import", m, "failed:", repr(e)[:200])
PY
echo "== mooncake engine.so options"
so=$(python3 -c "import mooncake, os, glob; print(glob.glob(os.path.join(os.path.dirname(mooncake.__file__), \"engine*.so\"))[0])" 2>/dev/null)
echo "$so"; for v in MOONCAKE_DISABLE_HIP_DMABUF MC_DISABLE_HIP MC_ENABLE_HIP_TRANSPORT MC_RDMA_NOTIFY_ENABLED ibv_reg_dmabuf_mr \
    hsa_amd_portable_export_dmabuf MC_ENABLE_DEST_DEVICE_AFFINITY MC_RDMA_WRITE_PREFER_DEST_NIC MC_IB_TIMEOUT MC_HIP_KFD_PIN_FIX MC_MAX_MR_SIZE; do
    echo "$v $(strings "$so" 2>/dev/null | grep -c "$v")"; done
echo "== vllm kv connectors"; python3 -c "from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory as F; print(sorted(getattr(F, \"_registry\", {})))"
echo "== libionic in image"; ls -la /usr/lib/x86_64-linux-gnu/libionic* /usr/lib/x86_64-linux-gnu/libibverbs/libionic* 2>/dev/null
dpkg -l | grep -E "ionic|rdma-core|ibverbs" | awk "{print \$2, \$3}"; ls /usr/include/infiniband/verbs.h 2>&1; which gcc
echo "== ibv_devinfo (image libionic)"; ibv_devinfo 2>&1 | grep -E "hca_id|state|link_layer" | head -24
echo "== infera install"; bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/ctr_install_infera.sh
echo "== vllm serve flags"; vllm serve --help=all 2>/dev/null | grep -oE -- "--(monokernel|engram-config|attention-config|kernel-config|prefill-schedule-interval|moe-backend|language-model-only|kv-events-config|renderer-num-workers|max-num-active-seqs|prefix-cache-retention-interval)" | sort -u
' > "$R/image.txt" 2>&1
say "image inspect rc=$?: $(grep -E '^(vllm|torch|amd-aiter|mori|nixl-rocm|mooncake-transfer-engine-rocm|tokenizers)==' "$R/image.txt" | tr '\n' ' ')"
say "mooncake: $(sed -n '/== mooncake engine.so options/,/== vllm kv connectors/p' "$R/image.txt" | grep -E ' [0-9]+$' | tr '\n' ' ')"
say "infera: $(grep -E 'infera parse ok|infera deps missing|Error|error' "$R/image.txt" | head -3 | tr '\n' ' ')"
say "image ibv: $(grep -c PORT_ACTIVE "$R/image.txt") active ports with the image libionic"
say "end"
