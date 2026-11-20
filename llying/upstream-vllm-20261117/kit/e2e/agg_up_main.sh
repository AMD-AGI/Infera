#!/usr/bin/env bash
# One aggregated GLM-5.2 server on vLLM main: the ROCm nightly image ($IMG_NIGHTLY) with the vllm Python code synced
# from a source tree before `vllm serve` (checks/sync_vllm.py; the image's compiled extensions stay), the decode
# config of track A / P4D4 (DP4 TP1, fp8 KV, MTP3, block 64). Waits until /v1/models answers.
#   agg_up_main.sh NAME GPUS PORT RPC_PORT VLLM_TREE LOGDIR [extra vllm serve args...]
# Env (from lib.sh): IMG_NIGHTLY, U, JOB; ENV_EXTRA (space-separated K=V), READY_TIMEOUT (default 3600).
set -uo pipefail
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
name=$1 gpus=$2 port=$3 rpc=$4 tree=$5 logdir=$6
shift 6
model=$(model_dir) || { echo "model not staged"; exit 1; }
mkdir -p "$logdir"
log=$logdir/$name.log
image_id=$(docker image inspect -f '{{.Id}}' "$IMG_NIGHTLY") || { echo "no image $IMG_NIGHTLY"; exit 1; }
args=(--data-parallel-size 4 --tensor-parallel-size 1 --kv-cache-dtype fp8_e4m3 --enable-prefix-caching
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --max-num-seqs 64 --max-num-batched-tokens 8192
    --block-size 64 --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
    --data-parallel-rpc-port "$rpc" "$@")
serve="vllm serve $model --served-model-name glm5.2-mxfp4 --trust-remote-code --host 0.0.0.0 --port $port"
for a in "${args[@]}"; do serve+=" $(printf '%q' "$a")"; done
env_args=(-e "HIP_VISIBLE_DEVICES=$gpus" -e PYTHONHASHSEED=0 -e SAFETENSORS_FAST_GPU=1 -e HSA_NO_SCRATCH_RECLAIM=1
    -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 -e NCCL_IB_DISABLE=1
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTHONFAULTHANDLER=1)
for kv in ${ENV_EXTRA:-}; do env_args+=(-e "$kv"); done
gids=()
for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
docker rm -f "$name" > /dev/null 2>&1
flock /tmp/up-e2e-docker-run.lock docker run -d --name "$name" --label "up.job=${JOB:-0}" --network host --ipc host \
    --shm-size 32g --device /dev/kfd --device /dev/dri "${gids[@]}" --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
    -v "$model:$model:ro" -v "$tree:/vllm-src:ro" -v "$U/checks:/up-checks:ro" \
    -v "up-e2e-cache-${image_id:7:12}:/root/.cache" \
    -v "up-e2e-aiter-jit-${image_id:7:12}:/usr/local/lib/python3.12/dist-packages/aiter/jit" \
    "${env_args[@]}" --entrypoint bash "$IMG_NIGHTLY" \
    -c "python3 /up-checks/sync_vllm.py /vllm-src && exec $serve" > /dev/null || { echo "docker run failed"; exit 1; }
docker inspect -f '{{json .Config.Cmd}} {{json .Config.Env}}' "$name" > "$logdir/$name.cmd"
nohup docker logs -f "$name" > "$log" 2>&1 &
echo "started $name on $(hostname -s) GPUs $gpus port $port, tree $tree ($(git -C "$tree" rev-parse --short=9 HEAD)), $(date -u +%T)"
t0=$(date +%s)
while true; do
    if [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2> /dev/null)" != true ]]; then
        echo "$name stopped after $(( $(date +%s) - t0 )) s; last log lines:"; tail -40 "$log"; exit 1
    fi
    if curl -sf -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null; then
        echo "$name ready after $(( $(date +%s) - t0 )) s"; exit 0
    fi
    if (( $(date +%s) - t0 > ${READY_TIMEOUT:-3600} )); then
        echo "$name not ready after ${READY_TIMEOUT:-3600} s"; tail -20 "$log"; exit 1
    fi
    sleep 10
done
