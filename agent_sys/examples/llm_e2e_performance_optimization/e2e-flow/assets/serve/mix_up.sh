#!/usr/bin/env bash
# GLM-5.3-Flash MIX bring-up. RUNS ON THE COMPUTE NODE.
#
# Adapted from profiling-demo/assets/serve/mix_up.sh, which came from
# examples/glm53flash-demo/scripts/mix_up.sh. Two changes:
#
#   1. **MOUNT_SPEC**, the one variable the two arms of this package differ by.
#      It names a file of `host<TAB>container` lines, each of which becomes a
#      read-only bind mount. The stock arm passes nothing; the patched arm passes
#      the plan `apply_patch` built. Everything else about the two bring-ups is
#      identical, and that is the entire experimental design -- a difference
#      anywhere else would be a second variable nobody controlled for.
#
#      A file rather than a string of docker arguments in an environment
#      variable: an unquoted expansion of `-v a:b -v c:d` is one word-splitting
#      bug away from mounting something else, and the failure would be a
#      deployment that looks fine and runs stock code.
#
#   2. The profiling control plane and the trace mount are gone. This package
#      captures no profile, so the router takes no --enable-profiling and there
#      is nothing to write a trace into. One flag fewer is one difference fewer.
#
# Router backend is left at its default (python), which is what
# infera/server/args.py defaults --router-backend to and what glm53flash-demo's
# measured numbers were taken on.
set -u
MY_IP="${NODE_IP:?NODE_IP=IP of this node}"
IMAGE="${IMAGE:?}"
ETCD_IMAGE="${ETCD_IMAGE:?}"
MODEL="${MODEL:?}"
MODEL_MOUNT="${MODEL_MOUNT:?}"
SERVED="${SERVED:-glm5.3-flash}"
CTR="${CTR:-glm53_int}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
PORT="${PORT:-30000}"
ETCD_PORT="${ETCD_PORT:-12379}"
TP="${TP:-8}"
WORK_ROOT="${WORK_ROOT:?}"
CUDA_GRAPH="${CUDA_GRAPH:-1}"
SCRIPTS="${SCRIPTS:?SCRIPTS=dir holding mix_worker.sh}"
MOUNT_SPEC="${MOUNT_SPEC:-}"

# --- the profiling half, restored for m2 ------------------------------------
# This file arrived from `integration-demo`, which had deleted both knobs with
# the note "this package captures no profile". True there; not true here. m2
# needs exactly them, and m5 needs `MOUNT_SPEC`, and the CONTRACT's premise is
# that the five stages share one set of assets rather than forking it.
#
# **Both default to what this file did before**, so an m5 arm that sets neither
# builds the identical container and the identical router command line.
#
#   PROFILE=1   adds --enable-profiling to the router, which is what turns the
#               admin profile start and stop routes from 403 into a working
#               control plane. (Written as prose rather than as the route
#               literal: this file is published inside a handoff, and
#               handoff.locality's shape heuristic reads a leading slash with
#               two segments as a filesystem path — temp/bugs/002.)
#   TRACE_OUT   bind-mounted rw into the engine container. SGLang writes the
#               trace to a path the ENGINE sees; without the mount docker
#               creates the directory in the container layer and the capture
#               looks like it worked while the host sees nothing.
TRACE_OUT="${TRACE_OUT:-}"
PROFILE="${PROFILE:-0}"

TRACE_MOUNT=()
if [ -n "$TRACE_OUT" ]; then
  TRACE_MOUNT=(-v "$TRACE_OUT:$TRACE_OUT:rw")
elif [ "$PROFILE" = "1" ]; then
  echo "  ABORT: PROFILE=1 with no TRACE_OUT — the control plane would come up"
  echo "  and every capture would write into the container layer, which is a"
  echo "  capture that reports success and leaves nothing on the host."
  exit 1
fi

# One `-v` per line of MOUNT_SPEC, built as an array so no path is ever
# word-split. Empty file, or no file at all, means the stock arm.
PATCH_MOUNTS=()
n_patch=0
if [ -n "$MOUNT_SPEC" ]; then
  [ -r "$MOUNT_SPEC" ] || { echo "  ABORT: MOUNT_SPEC is unreadable: $MOUNT_SPEC"; exit 1; }
  while IFS=$'\t' read -r host inside; do
    [ -n "${host:-}" ] || continue
    [ -r "$host" ] || { echo "  ABORT: mount source is unreadable on this node: $host"; exit 1; }
    PATCH_MOUNTS+=(-v "$host:$inside:ro")
    n_patch=$((n_patch + 1))
  done < "$MOUNT_SPEC"
fi

echo "===== 0. plan ====="
echo "  node=$(hostname -s) ip=$MY_IP image=$IMAGE"
echo "  model=$MODEL  tp=$TP  cuda_graph=$CUDA_GRAPH  profile=$PROFILE"
echo "  trace_out=${TRACE_OUT:-<none>}"
echo "  patch mounts: $n_patch"
for ((i = 1; i < ${#PATCH_MOUNTS[@]}; i += 2)); do echo "    ${PATCH_MOUNTS[$i]}"; done

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
mkdir -p "$WORK_ROOT/aiperf"
[ -n "$TRACE_OUT" ] && mkdir -p "$TRACE_OUT"
docker run -d --name "$CTR" --network=host --ipc=host --shm-size=64G \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v "$MODEL_MOUNT":"$MODEL_MOUNT":ro \
  "${TRACE_MOUNT[@]+"${TRACE_MOUNT[@]}"}" \
  "${PATCH_MOUNTS[@]+"${PATCH_MOUNTS[@]}"}" \
  "$IMAGE" sleep infinity >/dev/null || { echo "  ABORT: container start failed"; exit 1; }
sleep 5
echo "  image: $(docker inspect "$CTR" --format '{{.Image}}')"

# Prove the rw mount landed. capture.sh checks the same thing later; catching it
# here costs one docker inspect instead of a whole warm-up window.
if [ -n "$TRACE_OUT" ]; then
  mounted=$(docker inspect -f "{{range .Mounts}}{{if eq .Destination \"$TRACE_OUT\"}}{{.RW}}{{end}}{{end}}" "$CTR")
  [ "$mounted" = "true" ] && echo "  trace mount rw: OK" || { echo "  ABORT: $TRACE_OUT not mounted rw"; exit 1; }
fi

# Prove every patch mount landed, and that what is behind it is what was asked
# for. Docker will happily create an empty file at the destination if the source
# vanished between the plan and now, and the engine would then import an empty
# module some minutes later as an ImportError nobody could trace back to here.
if [ "$n_patch" -gt 0 ]; then
  while IFS=$'\t' read -r host inside; do
    [ -n "${host:-}" ] || continue
    want=$(sha256sum "$host" | cut -d' ' -f1)
    got=$(docker exec "$CTR" sha256sum "$inside" 2>/dev/null | cut -d' ' -f1)
    if [ "$want" != "$got" ]; then
      echo "  ABORT: $inside inside the container hashes '$got', the mount source hashes '$want'"
      exit 1
    fi
    echo "  mount live: $inside  ${want:0:12}…"
  done < "$MOUNT_SPEC"
fi

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
  DSA_ARGS="${DSA_ARGS:---dsa-prefill-backend tilelang --dsa-decode-backend tilelang}" \
  PARSER_ARGS="${PARSER_ARGS:---reasoning-parser glm45 --tool-call-parser glm47}" \
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
# The router backend is left at its default (python). `infera/server/args.py`
# defaults --router-backend to python and `launch_rust.py` refuses rust when
# --enable-profiling is set, so both m2 modes run the same data plane and the
# only difference between them is the profiling switch and the graph switch.
PROF_ARG=""
[ "$PROFILE" = "1" ] && PROF_ARG=" --enable-profiling"
docker exec "$CTR" bash -c "printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.server --host 0.0.0.0 --port $ROUTER_PORT --discovery-backend etcd --etcd-endpoint $MY_IP:$ETCD_PORT --request-transport http --kv-event-transport zmq --router-policy kv-aware --router-tokenizer-path $MODEL --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0$PROF_ARG' > /run_router.sh && chmod +x /run_router.sh"
docker exec -d "$CTR" bash -c 'nohup /run_router.sh > /tmp/router.log 2>&1'
sleep 20
docker exec "$CTR" bash -c "curl -sf -m5 http://$MY_IP:$ROUTER_PORT/health >/dev/null && echo '  router healthy' || { echo '  router not ready'; tail -20 /tmp/router.log; }"

if [ "$PROFILE" = "1" ]; then
  # Probe the control plane with a role that cannot exist. app.py checks the 403
  # gate BEFORE validating the role, so 400 means profiling is on and 403 means
  # it is not -- and neither touches a running profile. Fatal, because a capture
  # against a 403 control plane produces no trace and no error the caller sees.
  code=$(docker exec "$CTR" curl -s -o /dev/null -w '%{http_code}' -m 10 \
    -X POST "http://$MY_IP:$ROUTER_PORT/v1/admin/profile/start?role=__probe__")
  case "$code" in
    400) echo "  profiling control plane ON (probe -> 400 invalid role)" ;;
    403) echo "  PROFILING OFF despite PROFILE=1 -- check /tmp/router.log"; exit 1 ;;
    *)   echo "  unexpected probe status '$code'" ;;
  esac
fi

echo "===== MIX_UP_OK  endpoint: http://$MY_IP:$ROUTER_PORT ====="
