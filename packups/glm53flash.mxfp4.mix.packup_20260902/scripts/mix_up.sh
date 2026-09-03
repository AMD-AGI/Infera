#!/bin/bash
# GLM-5.3-Flash MIX bring-up on this 8xMI355X node:
#   teardown -> container -> etcd -> mix worker -> kv-aware router.
# Adapted from the GLM-5.2 MIX bring-up. kvd is dropped from the default path
# (see mix_worker.sh for why hicache is unsafe on this image build).
set -u
# No apostrophe in this message: a `'` inside ${VAR:?...} opens a quote that
# swallows the rest of the script, and the failure surfaces dozens of lines later
# as an unrelated "unbound variable".
MY_IP="${MY_IP:?MY_IP=IP of this node}"
IMAGE="${IMAGE:-infera/engine-sglang:glm53-c821c425}"
MODEL="${MODEL:-/apps/data/models/GLM-5.3-Flash-MXFP4}"
MODEL_MOUNT="${MODEL_MOUNT:-$(dirname "$MODEL")}"
# A SECOND bind, for when $MODEL is a symlink farm pointing at the real weights.
# /apps/data/models is its OWN mount on this host and does NOT propagate into the
# container when you bind its parent /apps -- inside, /apps/data/models is simply
# a different, empty-looking directory and every symlink into it is dangling.
# That failure surfaces far downstream as "Unrecognized processing class", because
# config.json is the one real file left and the tokenizer/processor files vanish.
WEIGHTS_MOUNT="${WEIGHTS_MOUNT:-}"
CTR="${CTR:-glm53_mix}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
PORT="${PORT:-30000}"
# 2379 is held by a HOST etcd that is not ours (pid seen listening on the node).
# Never kill a foreign process to free a port -- move ours instead.
ETCD_PORT="${ETCD_PORT:-12379}"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "===== 1. teardown our own ====="
docker rm -f "$CTR" "${CTR}_etcd" glm53_standalone >/dev/null 2>&1
sleep 5

echo "===== 2. confirm GPUs released ====="
# Hard gate, not a best-effort sleep: starting the worker while the previous
# round still holds VRAM aborts the distributed bootstrap with a misleading
# "memory capacity is unbalanced" error. See scripts/reset_gpus.sh.
# GPUS is passed through so a TP4 arm is not blocked by a neighbour legitimately
# using the other four. reset_gpus.sh never kills a process that is not ours.
GPUS="${GPUS:-$(seq -s, 0 $(( ${TP:-4} - 1 )))}"
OWN_CTR_RE="^(${CTR})$" GPUS="$GPUS" bash "$SELF/reset_gpus.sh" \
  || { echo "  ABORT: GPUs $GPUS not available"; exit 1; }

echo "===== 3. fresh container ====="
docker run -d --name "$CTR" --network=host --ipc=host --shm-size=64G \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v "$MODEL_MOUNT":"$MODEL_MOUNT":ro \
  ${WEIGHTS_MOUNT:+-v "$WEIGHTS_MOUNT":"$WEIGHTS_MOUNT":ro} \
  "$IMAGE" sleep infinity >/dev/null
sleep 5
echo "  image: $(docker inspect "$CTR" --format '{{.Image}}')"

echo "===== 4. etcd ====="
# etcd v3.5.14's image has an empty ENTRYPOINT and Cmd=[/usr/local/bin/etcd];
# passing `etcd` as argv[0] dumps usage and exits 2. Use an explicit --entrypoint.
docker run -d --name "${CTR}_etcd" --network=host --entrypoint /usr/local/bin/etcd \
  quay.io/coreos/etcd:v3.5.14 \
  --advertise-client-urls "http://$MY_IP:$ETCD_PORT" --listen-client-urls "http://0.0.0.0:$ETCD_PORT" >/dev/null
sleep 5
curl -sf -m5 "http://$MY_IP:$ETCD_PORT/version" >/dev/null && echo "  etcd up" || echo "  ETCD FAILED"

echo "===== 5. stage + launch mix worker ====="
docker cp "$SELF/mix_worker.sh" "$CTR":/mix_worker.sh >/dev/null
docker exec -d "$CTR" env MY_IP="$MY_IP" ETCD_IP="$MY_IP" ETCD_PORT="$ETCD_PORT" MODEL="$MODEL" \
  PORT="$PORT" LOG=/tmp/glm53_mix.log SERVED="${SERVED:-glm5.3-flash-mxfp4}" \
  TP="${TP:-4}" GPUS="$GPUS" \
  QUANT="${QUANT:-quark}" MOE_RUNNER="${MOE_RUNNER:-aiter}" \
  KV_DTYPE="${KV_DTYPE:-bfloat16}" \
  CUDA_GRAPH="${CUDA_GRAPH:-0}" GRAPH_BS="${GRAPH_BS:-1 2 4 8 16 24 32 48 64 96 128}" \
  EP_SIZE="${EP_SIZE:-}" GMU="${GMU:-0.80}" CTX="${CTX:-65536}" \
  MAX_RUNNING="${MAX_RUNNING:-32}" CHUNK="${CHUNK:-4096}" MAX_PREFILL="${MAX_PREFILL:-16384}" \
  bash /mix_worker.sh
echo "  mix worker launching -> /tmp/glm53_mix.log in $CTR"

echo "===== 6. wait for worker /health ====="
for i in $(seq 1 240); do
  if docker exec "$CTR" bash -c "curl -sf -m3 http://$MY_IP:$PORT/health" >/dev/null 2>&1; then
    echo "  worker serving after $((i*10))s"; break
  fi
  sleep 10
done

echo "===== 7. router (kv-aware) ====="
docker exec "$CTR" bash -c "printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.server --host 0.0.0.0 --port $ROUTER_PORT --discovery-backend etcd --etcd-endpoint $MY_IP:$ETCD_PORT --request-transport http --kv-event-transport zmq --router-policy kv-aware --router-tokenizer-path $MODEL --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0' > /run_router.sh && chmod +x /run_router.sh"
docker exec -d "$CTR" bash -c 'nohup /run_router.sh > /tmp/router.log 2>&1'
sleep 20
docker exec "$CTR" bash -c "curl -sf -m5 http://$MY_IP:$ROUTER_PORT/health >/dev/null && echo '  router healthy' || { echo '  router not ready'; tail -20 /tmp/router.log; }"

echo "===== up. endpoint: http://$MY_IP:$ROUTER_PORT ====="
