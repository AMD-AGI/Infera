#!/bin/bash
# GLM-5.2 MIX bring-up on ONE crsuse node: container -> etcd -> kvd -> mix worker -> router.
# Adapted from the validated crsuse reset_merged.sh + start_router.sh, with ALL
# RDMA/mooncake/libionic removed (mix is single-node, no KV transfer).
#
# Runs ON the node (via `spur exec <job> bash mix_up.sh`). Docker-out-of-docker:
# the daemon is the host's, containers are siblings and survive the exec — clean up
# only our own (CTR / CTR_etcd).
set -u
# crsuse spur exec runs as yihou; /tmp is root-owned (not writable), /var/tmp is.
export DOCKER_CONFIG="${DOCKER_CONFIG:-/var/tmp/dockercfg_yihou}"; mkdir -p "$DOCKER_CONFIG"
MY_IP="${MY_IP:?MY_IP=this node's IP}"
IMAGE="${IMAGE:?IMAGE=infera engine image tag}"
MODEL="${MODEL:?MODEL=weights dir (host path, bind-mounted)}"
MODEL_MOUNT="${MODEL_MOUNT:-$(dirname "$MODEL")}"
CTR="${CTR:-glm52_mix}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
PORT="${PORT:-30000}"
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "===== 1. teardown our own ====="
docker rm -f "$CTR" "${CTR}_etcd" >/dev/null 2>&1
for p in infera.engine.sglang infera.server infera.kvd; do pkill -9 -f "$p" 2>/dev/null; done
sleep 6

echo "===== 2. confirm GPUs released ====="
for i in $(seq 1 30); do
  n=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+' || true); n=${n:-0}
  [ "$n" -eq 0 ] && { echo "  GPUs idle after $((i*2))s"; break; }
  sleep 2
done

echo "===== 3. fresh container ====="
docker run -d --name "$CTR" --network=host --ipc=host --shm-size=32G \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v "$MODEL_MOUNT":"$MODEL_MOUNT" \
  "$IMAGE" sleep infinity >/dev/null
sleep 5
echo "  image: $(docker inspect "$CTR" --format '{{.Image}}')"

echo "===== 4. etcd ====="
# etcd v3.5.14 image has empty ENTRYPOINT and Cmd=[/usr/local/bin/etcd]; passing `etcd`
# as argv[0] dumps usage and exits 2. Use an explicit --entrypoint.
docker run -d --name "${CTR}_etcd" --network=host --entrypoint /usr/local/bin/etcd \
  quay.io/coreos/etcd:v3.5.14 \
  --advertise-client-urls "http://$MY_IP:2379" --listen-client-urls http://0.0.0.0:2379 >/dev/null
sleep 5
curl -sf -m5 "http://$MY_IP:2379/version" >/dev/null && echo "  etcd up" || echo "  ETCD FAILED"

echo "===== 5. kvd daemon (L2 32G host RAM + L3 file 64G) ====="
docker exec "$CTR" bash -c "mkdir -p /tmp/kvd /tmp/kvd-long && rm -f $KVD_SOCK && printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.kvd --socket $KVD_SOCK --max-bytes 34359738368 --long-path /tmp/kvd-long --long-bytes 68719476736 --log-level INFO' > /run_kvd.sh && chmod +x /run_kvd.sh"
docker exec -d "$CTR" bash -c 'nohup /run_kvd.sh > /tmp/kvd.log 2>&1'
sleep 20
docker exec "$CTR" bash -c "test -S $KVD_SOCK && echo '  kvd socket OK' || { echo '  KVD FAILED'; tail -20 /tmp/kvd.log; }"

echo "===== 6. stage + launch mix worker ====="
docker cp "$SELF/mix_worker.sh" "$CTR":/mix_worker.sh >/dev/null
docker exec -d "$CTR" env MY_IP="$MY_IP" ETCD_IP="$MY_IP" MODEL="$MODEL" \
  PORT="$PORT" KVD_SOCK="$KVD_SOCK" LOG=/tmp/glm52_mix.log \
  bash /mix_worker.sh
echo "  mix worker launching -> /tmp/glm52_mix.log in $CTR (cold start is minutes)"

echo "===== 7. wait for worker /health ====="
for i in $(seq 1 240); do
  if docker exec "$CTR" bash -c "curl -sf -m3 http://$MY_IP:$PORT/health" >/dev/null 2>&1; then
    echo "  worker serving after $((i*10))s"; break
  fi
  sleep 10
done

echo "===== 8. router (kv-aware) ====="
docker exec "$CTR" bash -c "printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.server --host 0.0.0.0 --port $ROUTER_PORT --discovery-backend etcd --etcd-endpoint $MY_IP:2379 --request-transport http --kv-event-transport zmq --router-policy kv-aware --router-tokenizer-path $MODEL --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0 --kvd-socket-path $KVD_SOCK' > /run_router.sh && chmod +x /run_router.sh"
docker exec -d "$CTR" bash -c 'nohup /run_router.sh > /tmp/router.log 2>&1'
sleep 20
docker exec "$CTR" bash -c "curl -sf -m5 http://$MY_IP:$ROUTER_PORT/health >/dev/null && echo '  router healthy' || { echo '  router not ready'; tail -20 /tmp/router.log; }"

echo "===== up. endpoint: http://$MY_IP:$ROUTER_PORT ====="
