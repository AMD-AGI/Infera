#!/usr/bin/env bash
# Start one vLLM engine container on this node and wait until it serves.
#   vllm_up.sh NAME GPUS PORT [vllm serve args...]
# Env: IMAGE, MODEL, SERVED (served model name), ENV_EXTRA (space-separated K=V), MOUNT_EXTRA (space-separated
#      src:dst[:ro]), READY_TIMEOUT (s, default 3600), ENTRY (default: vllm serve $MODEL; set to run another command).
# The container log is followed into $R/logs/engines/NAME.log. Exit 0 once /v1/models answers, 1 if the container
# stops or the timeout passes. The host libionic is never mounted (crusoe: it needs GLIBC 2.38, the image has 2.35).
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackA-20261118}
name=$1 gpus=$2 port=$3; shift 3
IMAGE=${IMAGE:-vllm/vllm-openai-rocm:v0.34.0}
MODEL=${MODEL:-/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4}
SERVED=${SERVED:-glm5.2-mxfp4}
READY_TIMEOUT=${READY_TIMEOUT:-3600}
mkdir -p "$R/logs/engines"
log=$R/logs/engines/$name.log
docker rm -f "$name" > /dev/null 2>&1
image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE") || { echo "no image $IMAGE"; exit 1; }
env_args=(-e "HIP_VISIBLE_DEVICES=$gpus" -e PYTHONHASHSEED=0 -e SAFETENSORS_FAST_GPU=1 -e HSA_NO_SCRATCH_RECLAIM=1
    -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 -e NCCL_IB_DISABLE=1 -e RDMAV_FORK_SAFE=1
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600)
for kv in ${ENV_EXTRA:-}; do env_args+=(-e "$kv"); done
mounts=(-v "$MODEL:$MODEL:ro" -v "vta-cache-${image_id:7:12}:/root/.cache"
    -v "vta-aiter-jit-${image_id:7:12}:/usr/local/lib/python3.12/dist-packages/aiter/jit")
for mx in ${MOUNT_EXTRA:-}; do mounts+=(-v "$mx"); done
if [[ -n "${ENTRY:-}" ]]; then
    cmd=(--entrypoint bash "$IMAGE" -c "$ENTRY")
else
    cmd=(--entrypoint vllm "$IMAGE" serve "$MODEL" --served-model-name "$SERVED" --trust-remote-code
        --host 0.0.0.0 --port "$port" "$@")
fi
# The image has no render group: pass the host's numeric gids. One docker run at a time: two containers creating the
# same named volume at once fail with "file exists" while docker populates it from the image.
gids=()
for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
flock /tmp/vta-docker-run.lock docker run -d --name "$name" --network host --ipc host --shm-size 32g \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    "${gids[@]}" --cap-add SYS_PTRACE --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
    "${mounts[@]}" "${env_args[@]}" "${cmd[@]}" > /dev/null || { echo "docker run failed"; exit 1; }
{ docker inspect -f '{{json .Config.Cmd}} {{json .Config.Env}}' "$name"; } > "$R/logs/engines/$name.cmd"
nohup docker logs -f "$name" > "$log" 2>&1 &
echo "started $name on $(hostname -s) GPUs $gpus port $port ($(date -u +%T))"
t0=$(date +%s)
while true; do
    if [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]]; then
        echo "$name stopped after $(( $(date +%s) - t0 )) s; last log lines:"; tail -40 "$log"; exit 1
    fi
    if curl -sf -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null; then
        echo "$name ready after $(( $(date +%s) - t0 )) s"; exit 0
    fi
    if (( $(date +%s) - t0 > READY_TIMEOUT )); then echo "$name not ready after $READY_TIMEOUT s"; tail -20 "$log"; exit 1; fi
    sleep 10
done
