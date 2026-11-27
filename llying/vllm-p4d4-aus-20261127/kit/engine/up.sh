#!/usr/bin/env bash
# One-node P4D4 (Infera + vLLM, GLM-5.2-MXFP4, MI355X): etcd, a TP1 x PP4 prefill on GPUs 0-3, a DP4 decode on GPUs 4-7
# and the Infera Rust router. KV moves prefill -> decode over Mooncake RDMA loopback (MC_DISABLE_HIP=1: never HIP IPC,
# even inside one node), one NIC per GPU. Same deployment as the crusoe a49 recipe (replay/), without overlays: every
# engine change is a patch baked into the image (patches/, docker/Dockerfile).
#   up.sh RUN_DIR
# Env: kit/env.sh (site, model, image, GEM / libionic / NIC map) and a variant file (kit/variants/*.env, sourced by the
#      caller) for the per-variant args; SIM_AL empty = real MTP acceptance (eval), else the golden AL (score).
# Writes RUN_DIR/{engine-config.txt,router_url,workers.json}; exit 0 when the router lists 1 prefill + 1 decode.
set -uo pipefail
K=$(cd "$(dirname "$0")/.." && pwd)
. "$K/env.sh"
RUN_DIR=${1:?run dir}
mkdir -p "$RUN_DIR/config" "$RUN_DIR/logs"
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
ETCD_PORT=24701 ROUTER_PORT=28300 P_PORT=27201 D_PORT=27301 BOOT_PORT=28398

# Mooncake: RDMA on all eight NICs, each GPU writes through its own NIC (no fallback NIC: on these hosts ionic_k only
# reaches ionic_k). KV pools in GEM BOs when GEM=1 (dma-buf registration charges pinned VRAM twice on amdgpu 6.19 and
# on 6.16 without peer memory); GEM=0 registers plain hipMalloc memory through peer memory (MOONCAKE_DISABLE_HIP_DMABUF=1).
topo() { python3 - "$1" "$2" "$GPU_NICS" <<'EOF'
import json, sys
out, first, nics = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
t = {f"cuda:{i}": [[nics[first + i]], []] for i in range(4)}
t.update({"cpu:0": [nics[:4], []], "cpu:1": [nics[4:], []]})
json.dump(t, open(out, "w"), indent=1)
EOF
}
topo "$RUN_DIR/config/topo-prefill.json" 0
topo "$RUN_DIR/config/topo-decode.json" 4
mc_env="MC_GID_INDEX=1 MC_DISABLE_HIP=1 MC_RDMA_WRITE_PREFER_DEST_NIC=1 MC_IB_TIMEOUT=18 MC_MAX_MR_SIZE=1073741824
MOONCAKE_DISABLE_HIP_DMABUF=$((1 - GEM)) MC_HIP_KFD_PIN_FIX=0 VLLM_HOST_IP=$IP VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOT_PORT
VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1800 INFERA_ENGINE_READY_TIMEOUT=3600 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120
VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT=600"
mod=""
if [[ "$GEM" == 1 ]]; then
    mc_env+=" INFERA_AMD_GEM_ALLOCATOR=/opt/infera-gem/libhip_gem_alloc.so LD_PRELOAD=/opt/infera-gem/libhip_gem_alloc.so"
    mod=',"kv_connector_module_path":"infera.engine.vllm.gem_mooncake"'
fi
NICS=$(echo "$GPU_NICS" | tr ',' '\n' | sort -u | paste -sd, -)
mc_extra="{\"num_workers\":10,\"device_name\":\"$NICS\"}"
cpu_tier="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$CPU_BYTES,\"eviction_policy\":\"lru\"}}"
p_kv="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"MooncakeConnector\"$mod,\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":$mc_extra},$cpu_tier]}}"
d_kv="{\"kv_connector\":\"MooncakeConnector\"$mod,\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":$mc_extra}"

# MTP: the prefill carries the draft so the KV it sends includes the draft layer; the decode verifies it, with the
# golden synthetic acceptance (InferenceX agentic rule) when SIM_AL is set.
spec="{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_N}"
d_spec=$spec
[[ -n "${SIM_AL:-}" ]] && d_spec="{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_N,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$SIM_AL}"
infera="python3 -m infera.engine.vllm --discovery-backend etcd --etcd-endpoint $IP:$ETCD_PORT --request-transport http --kv-event-transport zmq --advertise-host $IP"
common="--model $MODEL --served-model-name $SERVED --trust-remote-code --host 0.0.0.0 --kv-cache-dtype fp8_e4m3
 --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice --block-size 64 --renderer-num-workers 4
 --api-server-count 1 ${NUMERICS_ARGS:-}"
p_cmd="$infera $common --port $P_PORT --tensor-parallel-size 1 --pipeline-parallel-size 4 --enable-prefix-caching
 --max-num-seqs 64 --max-num-batched-tokens 4096 --long-prefill-token-threshold 2048 --gpu-memory-utilization 0.88
 --cache-aware-admission-window 128 --cache-aware-admission-threshold 0.0 --speculative-config '$spec'
 --kv-transfer-config '$p_kv' ${P_ARGS_X:-}"
d_cmd="$infera $common --port $D_PORT --no-enable-kv-events --kv-events off --data-parallel-size 4 --tensor-parallel-size 1
 --data-parallel-rpc-port 29620 --enable-prefix-caching --max-num-seqs 64 --max-num-batched-tokens 2048 --stream-interval 30
 --gpu-memory-utilization 0.95 --speculative-config '$d_spec' --kv-transfer-config '$d_kv' ${D_ARGS_X:-}"
p_env="$mc_env MC_CUSTOM_TOPO_JSON=$RUN_DIR/config/topo-prefill.json VLLM_PP_LAYER_PARTITION=20,20,20,18 VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=512 ${P_ENV_X:-}"
d_env="$mc_env MC_CUSTOM_TOPO_JSON=$RUN_DIR/config/topo-decode.json ${D_ENV_X:-}"
{
    echo "IMAGE=$IMAGE $(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null) IP=$IP GEM=$GEM HOST_LIBIONIC=$HOST_LIBIONIC GPU_NICS=$GPU_NICS SIM_AL=${SIM_AL:-real}"
    echo "PREFILL_ENV: $(echo $p_env)"; echo "PREFILL: $(echo $p_cmd)"
    echo "DECODE_ENV: $(echo $d_env)"; echo "DECODE: $(echo $d_cmd)"
    echo "ROUTER: ${ROUTER_ARGS} ${ROUTER_ENV:-} ${ROUTER_MOUNT_X:-}"
    echo "MOUNTS: engines [${ENGINE_MOUNT_X:-}] prefill [${P_MOUNT_X:-}] decode [${D_MOUNT_X:-}]"
} > "$RUN_DIR/engine-config.txt"

docker rm -f vpd-etcd vpd-router vpd-prefill vpd-decode > /dev/null 2>&1
pull_image quay.io/coreos/etcd:v3.5.14 > /dev/null
docker run -d --name vpd-etcd --network host --entrypoint /usr/local/bin/etcd quay.io/coreos/etcd:v3.5.14 \
    --advertise-client-urls "http://$IP:$ETCD_PORT" --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
    --listen-peer-urls http://127.0.0.1:24702 --initial-advertise-peer-urls http://127.0.0.1:24702 \
    --initial-cluster default=http://127.0.0.1:24702 > /dev/null || exit 1
# Extra engine mounts (variants D / E: local-only files): ENGINE_MOUNT_X (both), P_MOUNT_X, D_MOUNT_X, as src:dst[:ro];
# @SP@ in a destination is the image's site-packages directory.
SP=$(docker run --rm --entrypoint python3 "$IMAGE" -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')
engine() {   # name gpus port env cmd mounts
    local name=$1 gpus=$2 port=$3 env=$4 cmd=${5//$'\n'/ } mounts=${6:-} args=() gids=()   # one line for bash -c
    for kv in $env; do args+=(-e "$kv"); done
    for mx in $mounts; do args+=(-v "${mx//@SP@/$SP}"); done
    for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
    local swap=""
    if [[ "$HOST_LIBIONIC" == 1 ]]; then
        args+=(-v "$(readlink -f /lib/x86_64-linux-gnu/libionic.so):/host-libionic/libionic.so:ro")
        swap='for f in $(find / -xdev -name "libionic*.so*" -not -path "/host-libionic/*" 2>/dev/null); do cp -f /host-libionic/libionic.so "$f"; done; '
    fi
    flock /tmp/vpd-docker-run.lock docker run -d --name "$name" --network host --ipc host --shm-size 32g \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband "${gids[@]}" --cap-add SYS_PTRACE --cap-add IPC_LOCK \
        --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
        -v "$MODEL:$MODEL:ro" -v "$RUN_DIR/config:$RUN_DIR/config:ro" -v "vpd-cache:/root/.cache" \
        -e HIP_VISIBLE_DEVICES="$gpus" -e PYTHONHASHSEED=0 -e SAFETENSORS_FAST_GPU=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
        -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 -e NCCL_IB_DISABLE=1 -e RDMAV_FORK_SAFE=1 \
        -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTHONFAULTHANDLER=1 "${args[@]}" \
        --entrypoint bash "$IMAGE" -c "$swap exec $cmd" > /dev/null || return 1
    nohup docker logs -f "$name" > "$RUN_DIR/logs/$name.log" 2>&1 &
    local t0; t0=$(date +%s)
    while true; do
        [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" == true ]] \
            || { echo "$name stopped after $(( $(date +%s) - t0 )) s"; tail -30 "$RUN_DIR/logs/$name.log"; return 1; }
        curl -sf -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null && { echo "$name ready after $(( $(date +%s) - t0 )) s"; return 0; }
        (( $(date +%s) - t0 > 3600 )) && { echo "$name not ready after 3600 s"; return 1; }
        sleep 10
    done
}
engine vpd-prefill 0,1,2,3 $P_PORT "$p_env" "$p_cmd" "${ENGINE_MOUNT_X:-} ${P_MOUNT_X:-}" & p_pid=$!
sleep 30   # two containers creating the same named volume at once fail
engine vpd-decode 4,5,6,7 $D_PORT "$d_env" "$d_cmd" "${ENGINE_MOUNT_X:-} ${D_MOUNT_X:-}" & d_pid=$!
wait $p_pid; p_rc=$?; wait $d_pid; d_rc=$?
(( p_rc == 0 && d_rc == 0 )) || { echo "engine start failed: prefill rc=$p_rc decode rc=$d_rc"; exit 1; }
renv=(); for kv in ${ROUTER_ENV:-}; do renv+=(-e "$kv"); done
# ROUTER_MOUNT_X: src:dst mounts for the router only (stage F: a rebuilt infera-router and Infera's Python launcher).
for mx in ${ROUTER_MOUNT_X:-}; do renv+=(-v "$mx"); done
docker run -d --name vpd-router --network host --ulimit nofile=1048576:1048576 -v "$MODEL:$MODEL:ro" "${renv[@]}" \
    --entrypoint python3 "$IMAGE" -m infera.server --host 0.0.0.0 --port $ROUTER_PORT --router-backend rust \
    --discovery-backend etcd --etcd-endpoint "$IP:$ETCD_PORT" --request-transport http --kv-event-transport zmq \
    --router-tokenizer-path "$MODEL" $ROUTER_ARGS > /dev/null || exit 1
nohup docker logs -f vpd-router > "$RUN_DIR/logs/vpd-router.log" 2>&1 &
for i in $(seq 120); do
    [[ "$(docker inspect -f '{{.State.Running}}' vpd-router 2>/dev/null)" == true ]] || { echo "router exited"; docker logs --tail 20 vpd-router; exit 1; }
    modes=$(curl -sf -m 10 "http://$IP:$ROUTER_PORT/v1/workers" | python3 -c 'import json, sys; print(*sorted(w["disagg_mode"] for w in json.load(sys.stdin)["workers"]))' 2>/dev/null)
    [[ "$modes" == "decode prefill" || "$modes" == "decode decode decode decode prefill" ]] && break
    sleep 10
done
curl -sf -m 10 "http://$IP:$ROUTER_PORT/v1/workers" > "$RUN_DIR/workers.json"
echo "http://$IP:$ROUTER_PORT" > "$RUN_DIR/router_url"
echo "router workers: ${modes:-none}; ROUTER_URL=http://$IP:$ROUTER_PORT"
[[ -n "${modes:-}" && "$modes" == decode*prefill ]]
