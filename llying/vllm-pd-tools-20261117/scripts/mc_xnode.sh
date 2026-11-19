#!/usr/bin/env bash
# Run one side of scripts/mc_xnode.py in the engine image with the engines' RDMA container flags and Mooncake env.
#   mc_xnode.sh target|initiator DIR DEVICES [MB] [EXTRA_ENV...]     (DIR under R, shared by both nodes)
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
IMAGE=${IMAGE:-infera-vllm:v0340-0c5e5e89}
role=$1 dir=$2 devices=$3 mb=${4:-256}; shift 4 || shift $#
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
env_args=(-e HOST_IP="$IP" -e MC_GID_INDEX=1 -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 -e MC_IB_TIMEOUT=18)
for kv in "$@"; do env_args+=(-e "$kv"); done
mkdir -p "$R/$dir"
gids=()
for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
timeout 900 docker run --rm --name "mcx-$role-$$" --network host --ipc host \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband "${gids[@]}" --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -v "$R:$R" "${env_args[@]}" \
    --entrypoint python3 "$IMAGE" "$R/scripts/mc_xnode.py" "$role" "$R/$dir" "$devices" "$mb" 2>&1 \
    | grep -a -v "queue-intercept" | grep -a -E "^(target|initiator)|^[EW][0-9]{4}|rror|Found|installTransport|notif" | head -60
