#!/usr/bin/env bash
# GLM-5.3-Flash MIX bring-up. RUNS ON THE COMPUTE NODE.
#
# Adapted from examples/glm53flash-demo/scripts/mix_up.sh. Three changes, all of
# them for profiling:
#   1. PROFILE=1 adds --enable-profiling to the router, which is what turns the
#      admin profile start and stop routes from 403 into a working control plane.
#      (Written as prose rather than as the route literal: this file is published
#      inside a handoff, and handoff.locality's shape heuristic reads a leading
#      slash with two segments as a filesystem path. See temp/bugs/002.)
#   2. $TRACE_OUT is bind-mounted rw into the engine container. SGLang writes the
#      trace to a path the ENGINE sees; without the mount docker would create the
#      directory in the container layer and capture would look like it worked
#      while the host saw nothing.
#   3. CUDA_GRAPH is passed through rather than fixed, because it is the axis the
#      two rounds turn.
#
# Router backend is left at its default (python). infera/server/args.py defaults
# --router-backend to python and launch_rust.py refuses rust when
# --enable-profiling is set, so both rounds use the same data plane and the only
# difference between them is the profiling switch and the graph switch.
set -u
MY_IP="${NODE_IP:?NODE_IP=IP of this node}"
IMAGE="${IMAGE:?}"
ETCD_IMAGE="${ETCD_IMAGE:?}"
MODEL="${MODEL:?}"
MODEL_MOUNT="${MODEL_MOUNT:?}"
SERVED="${SERVED:-glm5.3-flash}"
CTR="${CTR:-glm53_mix}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
PORT="${PORT:-30000}"
ETCD_PORT="${ETCD_PORT:-12379}"
TP="${TP:-8}"
WORK_ROOT="${WORK_ROOT:?}"
TRACE_OUT="${TRACE_OUT:?}"
PROFILE="${PROFILE:-0}"
CUDA_GRAPH="${CUDA_GRAPH:-1}"
SCRIPTS="${SCRIPTS:?SCRIPTS=dir holding mix_worker.sh}"

echo "===== 0. plan ====="
echo "  node=$(hostname -s) ip=$MY_IP image=$IMAGE"
echo "  model=$MODEL  tp=$TP  cuda_graph=$CUDA_GRAPH  profile=$PROFILE"
echo "  trace_out=$TRACE_OUT"

echo "===== 1. teardown our own ====="
docker rm -f "$CTR" "${CTR}_etcd" >/dev/null 2>&1
sleep 5

echo "===== 1b. ports ====="
# --network=host means every port below is a host port. This node also runs a
# Kubernetes control plane, so "the port the demo used" is not automatically free.
for spec in "etcd:$ETCD_PORT" "router:$ROUTER_PORT" "worker:$PORT" "kv-events:5557" "kv-snapshot:8801"; do
  name="${spec%%:*}"; p="${spec##*:}"
  if ss -tln 2>/dev/null | grep -q ":$p "; then
    echo "  ABORT: $name port $p already in use:"
    ss -tlnp 2>/dev/null | grep ":$p " | sed 's/^/    /'
    exit 1
  fi
  echo "  $name $p free"
done

echo "===== 2. confirm GPUs released ====="
bash "$SCRIPTS/reset_gpus.sh" || { echo "  ABORT: GPUs not released"; exit 1; }

echo "===== 3. fresh container ====="
mkdir -p "$TRACE_OUT" "$WORK_ROOT/aiperf"
docker run -d --name "$CTR" --network=host --ipc=host --shm-size=64G \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v "$MODEL_MOUNT":"$MODEL_MOUNT":ro \
  -v "$TRACE_OUT":"$TRACE_OUT":rw \
  "$IMAGE" sleep infinity >/dev/null || { echo "  ABORT: container start failed"; exit 1; }
sleep 5
echo "  image: $(docker inspect "$CTR" --format '{{.Image}}')"
# Prove the rw mount landed. capture.sh checks the same thing later; catching it
# here costs one docker inspect instead of a whole warm-up window.
mounted=$(docker inspect -f "{{range .Mounts}}{{if eq .Destination \"$TRACE_OUT\"}}{{.RW}}{{end}}{{end}}" "$CTR")
[ "$mounted" = "true" ] && echo "  trace mount rw: OK" || { echo "  ABORT: $TRACE_OUT not mounted rw"; exit 1; }

echo "===== 4. etcd ====="
# v3.5.14's image has an empty ENTRYPOINT and Cmd=[/usr/local/bin/etcd]; passing
# `etcd` as argv[0] dumps usage and exits 2. Hence --entrypoint.
docker run -d --name "${CTR}_etcd" --network=host --entrypoint /usr/local/bin/etcd \
  "$ETCD_IMAGE" \
  --advertise-client-urls "http://$MY_IP:$ETCD_PORT" \
  --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
  --listen-peer-urls "http://127.0.0.1:$((ETCD_PORT + 1))" \
  --initial-advertise-peer-urls "http://127.0.0.1:$((ETCD_PORT + 1))" \
  --initial-cluster "default=http://127.0.0.1:$((ETCD_PORT + 1))" >/dev/null
sleep 5
# Fatal, not a warning. Without discovery the worker never registers and the
# router serves an empty pool, which shows up much later as a confusing 404.
if curl -sf -m5 "http://$MY_IP:$ETCD_PORT/version" >/dev/null; then
  echo "  etcd up on $ETCD_PORT"
else
  echo "  ABORT: etcd did not come up on $ETCD_PORT"
  docker logs "${CTR}_etcd" 2>&1 | tail -10
  exit 1
fi

echo "===== 5. stage + launch mix worker ====="
docker cp "$SCRIPTS/mix_worker.sh" "$CTR":/mix_worker.sh >/dev/null
docker exec -d "$CTR" env MY_IP="$MY_IP" ETCD_IP="$MY_IP" ETCD_PORT="$ETCD_PORT" \
  MODEL="$MODEL" SERVED="$SERVED" \
  PORT="$PORT" TP="$TP" LOG=/tmp/glm53_mix.log \
  CUDA_GRAPH="$CUDA_GRAPH" GRAPH_BS="${GRAPH_BS:-1 2 4 8 16 24 32 48 64 96 128}" \
  KV_DTYPE="${KV_DTYPE:-bfloat16}" MOE_RUNNER="${MOE_RUNNER:-triton}" \
  GMU="${GMU:-0.85}" CTX="${CTX:-262144}" CHUNK="${CHUNK:-8192}" \
  bash /mix_worker.sh
echo "  mix worker launching -> /tmp/glm53_mix.log in $CTR"

echo "===== 6. wait for worker /health ====="
ready=0
for i in $(seq 1 240); do
  if docker exec "$CTR" bash -c "curl -sf -m3 http://$MY_IP:$PORT/health" >/dev/null 2>&1; then
    echo "  worker serving after $((i*10))s"; ready=1; break
  fi
  # Surface a dead worker instead of waiting out the full 40 minutes.
  if ! docker exec "$CTR" pgrep -f infera.engine.sglang >/dev/null 2>&1; then
    echo "  WORKER PROCESS GONE after $((i*10))s — last 40 lines:"
    docker exec "$CTR" tail -40 /tmp/glm53_mix.log
    exit 1
  fi
  sleep 10
done
[ "$ready" = "1" ] || { echo "  worker never became ready"; docker exec "$CTR" tail -40 /tmp/glm53_mix.log; exit 1; }

echo "===== 7. router ====="
PROF_ARG=""
[ "$PROFILE" = "1" ] && PROF_ARG=" --enable-profiling"
docker exec "$CTR" bash -c "printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.server --host 0.0.0.0 --port $ROUTER_PORT --discovery-backend etcd --etcd-endpoint $MY_IP:$ETCD_PORT --request-transport http --kv-event-transport zmq --router-policy kv-aware --router-tokenizer-path $MODEL --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0$PROF_ARG' > /run_router.sh && chmod +x /run_router.sh"
docker exec -d "$CTR" bash -c 'nohup /run_router.sh > /tmp/router.log 2>&1'
sleep 20
docker exec "$CTR" bash -c "curl -sf -m5 http://$MY_IP:$ROUTER_PORT/health >/dev/null && echo '  router healthy' || { echo '  router not ready'; tail -20 /tmp/router.log; }"

if [ "$PROFILE" = "1" ]; then
  # Probe the control plane with a role that cannot exist. app.py checks the 403
  # gate BEFORE validating the role, so 400 means profiling is on and 403 means
  # it is not -- and neither touches a running profile.
  code=$(docker exec "$CTR" curl -s -o /dev/null -w '%{http_code}' -m 10 \
    -X POST "http://$MY_IP:$ROUTER_PORT/v1/admin/profile/start?role=__probe__")
  case "$code" in
    400) echo "  profiling control plane ON (probe -> 400 invalid role)" ;;
    403) echo "  PROFILING OFF despite PROFILE=1 -- check /tmp/router.log"; exit 1 ;;
    *)   echo "  unexpected probe status '$code'" ;;
  esac
fi

echo "===== MIX_UP_OK  endpoint: http://$MY_IP:$ROUTER_PORT ====="
