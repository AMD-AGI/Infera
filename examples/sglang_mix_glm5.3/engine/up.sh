#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Bring up a single-node MIX (aggregated) GLM-5.3 deployment:
#   container -> etcd -> infera worker -> kv-aware router.
# Reads every site value from ../env.sh. Runs ON the node.
set -uo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$KIT/env.sh"

case "$MY_IP" in "<"*) echo "edit env.sh: MY_IP is still a placeholder" >&2; exit 2;; esac
case "$IMAGE" in "<"*) echo "edit env.sh: IMAGE is still a placeholder" >&2; exit 2;; esac
[ -d "$MODEL" ] || { echo "MODEL not a directory on this host: $MODEL" >&2; exit 2; }

# Bind the REALPATH, not the symlink's parent. See the note in env.sh: on the
# reference cluster the models path crosses an NFS mount boundary, and binding
# the wrong side gives the container an empty directory whose failure surfaces
# many minutes later as an unrelated-looking processor error.
MODEL_REAL="$(realpath "$MODEL")"
MOUNT_REAL="$(dirname "$MODEL_REAL")"

echo "===== 1. teardown (ours only) ====="
# Named explicitly. Never a pattern: on a shared node a pattern is how you
# remove somebody else's container.
docker rm -f "$CTR" "${CTR}_etcd" >/dev/null 2>&1
sleep 5

echo "===== 2. GPUs $GPUS free? ====="
# Advisory, not a kill. If a GPU we want is busy this stops and says so -- it
# never reclaims a process, because on a shared node that process is somebody's
# multi-day job.
busy=$(rocm-smi --showmemuse 2>/dev/null | awk -v want=",$GPUS," '
  match($0, /GPU\[([0-9]+)\]/, m) && /VRAM%/ {
    split($0, f, ": "); if (index(want, "," m[1] ",") && f[2]+0 > 5) printf "%s ", m[1] }')
[ -n "$busy" ] && { echo "  GPU(s) $busy already in use -- pick a free subset via GPUS=, or wait." >&2; exit 1; }
echo "  clear"

echo "===== 3. container ====="
docker run -d --name "$CTR" --network=host --ipc=host --shm-size=64G \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v "$MOUNT_REAL":"$MOUNT_REAL":ro \
  "$IMAGE" sleep infinity >/dev/null || exit 1
sleep 5

echo "===== 4. etcd ====="
# The etcd v3.5.x image has an empty ENTRYPOINT and Cmd=[/usr/local/bin/etcd];
# passing `etcd` as argv[0] dumps usage and exits 2. Hence --entrypoint.
docker run -d --name "${CTR}_etcd" --network=host --entrypoint /usr/local/bin/etcd \
  quay.io/coreos/etcd:v3.5.14 \
  --advertise-client-urls "http://$MY_IP:$ETCD_PORT" \
  --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
  --listen-peer-urls "http://0.0.0.0:$((ETCD_PORT + 1))" \
  --initial-advertise-peer-urls "http://$MY_IP:$((ETCD_PORT + 1))" \
  --initial-cluster "default=http://$MY_IP:$((ETCD_PORT + 1))" >/dev/null
sleep 5
curl -sf -m5 "http://$MY_IP:$ETCD_PORT/version" >/dev/null \
  && echo "  etcd up" || { echo "  ETCD FAILED"; docker logs --tail 5 "${CTR}_etcd"; }

echo "===== 5. worker ====="
docker cp "$KIT/engine/worker.sh" "$CTR":/worker.sh >/dev/null
docker exec -d "$CTR" env \
  MY_IP="$MY_IP" ETCD_IP="$MY_IP" ETCD_PORT="$ETCD_PORT" \
  MODEL="$MODEL_REAL" VARIANT="$VARIANT" TP="$TP" GPUS="$GPUS" PORT="$PORT" \
  SERVED="${SERVED:-glm5.3-$VARIANT}" CUDA_GRAPH="${CUDA_GRAPH:-1}" \
  bash /worker.sh
echo "  launching -> /tmp/glm53_mix.log in $CTR"

echo "===== 6. wait for /health ====="
# Cold start is minutes, not seconds: several hundred GB of weights, then AITER
# JIT, then graph capture. Silence is not a hang. 650 s observed on a node with
# a cold NFS cache.
for i in $(seq 1 240); do
  if docker exec "$CTR" curl -sf -m3 "http://$MY_IP:$PORT/health" >/dev/null 2>&1; then
    echo "  serving after $((i * 10))s"; break
  fi
  docker exec "$CTR" pgrep -f infera.engine.sglang >/dev/null 2>&1 || {
    echo "  worker died -- last lines:"; docker exec "$CTR" tail -25 /tmp/glm53_mix.log; exit 1; }
  sleep 10
done

echo "===== 7. kv-aware router ====="
docker exec "$CTR" bash -c "printf '%s\n' '#!/bin/bash' \
  'exec python3 -m infera.server --host 0.0.0.0 --port $ROUTER_PORT \
   --discovery-backend etcd --etcd-endpoint $MY_IP:$ETCD_PORT \
   --request-transport http --kv-event-transport zmq --router-policy kv-aware \
   --router-tokenizer-path $MODEL_REAL' > /run_router.sh && chmod +x /run_router.sh"
docker exec -d "$CTR" bash -c 'nohup /run_router.sh > /tmp/router.log 2>&1'
sleep 20
docker exec "$CTR" bash -c \
  "curl -sf -m5 http://$MY_IP:$ROUTER_PORT/health >/dev/null && echo '  router healthy' \
   || { echo '  router not ready'; tail -20 /tmp/router.log; }"

echo "===== up. clients use http://$MY_IP:$ROUTER_PORT ====="
