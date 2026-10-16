#!/usr/bin/env bash
# Bring the whole deployment up: engine container -> etcd -> worker -> router,
# block until the ROUTER serves the model, then write the handshake
# ${E2E_KIT_WORK_ROOT}/deployment.json. Exits 0 only when it answers.
#
# Runs ON the node (this host; transport=local). Every identifier it binds comes
# from env.sh in a ${X:=...} form: run tag, port band, work root, cards.
#
# Never removes a container it did not create in THIS invocation: a name that
# is already taken is a failure, not a thing to clear. On any failure after it
# has started containers, it removes exactly the container IDs it started
# (logs stay on the host under the work root) unless E2E_KIT_KEEP_ON_FAILURE=1.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
# shellcheck source=lib_mounts.sh
source "$HERE/lib_mounts.sh"
: "${E2E_KIT_KEEP_ON_FAILURE:=0}"

T0=$(date +%s)
CREATED=()
say() { echo "[deploy $(date -u +%H:%M:%S)] $*"; }
die() {
  echo "[deploy $(date -u +%H:%M:%S)] ABORT: $*" >&2
  if [ "${#CREATED[@]}" -gt 0 ] && [ "$E2E_KIT_KEEP_ON_FAILURE" != 1 ]; then
    # Hand root-written files back before the container that wrote them goes.
    docker exec "$E2E_KIT_CONTAINER" chown -R "$(id -u):$(id -g)" "$E2E_KIT_WORK_ROOT" >/dev/null 2>&1 || true
    for id in "${CREATED[@]}"; do docker rm -f "$id" >/dev/null 2>&1 && echo "[deploy] removed own container ${id:0:12}" >&2; done
  fi
  exit 1
}

say "tag=$E2E_KIT_RUN_TAG ports=[$KIT_ALL_PORTS] gpus=$E2E_KIT_GPU_DEVICES work=$E2E_KIT_WORK_ROOT"
say "image=$E2E_KIT_IMAGE model=$E2E_KIT_MODEL_PATH served=$E2E_KIT_SERVED_NAME tp=$E2E_KIT_TP ctx=$E2E_KIT_CTX graph_max_bs=$E2E_KIT_CUDA_GRAPH_MAX_BS"

# ---- 1. preflight -------------------------------------------------------------
docker image inspect "$E2E_KIT_IMAGE" >/dev/null 2>&1 || die "image not present: $E2E_KIT_IMAGE"
docker image inspect "$E2E_KIT_ETCD_IMAGE" >/dev/null 2>&1 || die "etcd image not present: $E2E_KIT_ETCD_IMAGE"
[ -f "$E2E_KIT_MODEL_PATH/config.json" ] || die "no config.json under $E2E_KIT_MODEL_PATH"
# Bind the REALPATH: a symlinked model dir crossing a mount binds an empty tree.
KIT_MODEL_REAL="$(realpath "$E2E_KIT_MODEL_PATH")"

n_gpus=$(awk -F, '{print NF}' <<< "$E2E_KIT_GPU_DEVICES")
[ "$n_gpus" -ge "$E2E_KIT_TP" ] || die "E2E_KIT_GPU_DEVICES=$E2E_KIT_GPU_DEVICES has $n_gpus card(s); TP=$E2E_KIT_TP needs $E2E_KIT_TP"

listening=$(ss -ltn 2>/dev/null | awk 'NR>1{print $4}' | sed 's/.*://' | sort -u)
for p in $KIT_ALL_PORTS; do
  grep -qx "$p" <<< "$listening" && die "port $p (band base $E2E_KIT_PORT_BASE) is already listening; pick another E2E_KIT_PORT_BASE"
done
for n in "$E2E_KIT_CONTAINER" "$E2E_KIT_ETCD_CONTAINER"; do
  [ -n "$(docker ps -aq --filter "name=^/${n}\$")" ] && die "container name $n is taken; not removing it. Use another E2E_KIT_RUN_TAG, or teardown.sh under the tag that created it"
done

# The named cards must be free. Obeyed, never substituted. Busy = more than
# 4096 MiB in use (an idle MI300X here reads ~284 MiB).
gpu_reading=$(bash "$HERE/gpu_used.sh" "$E2E_KIT_GPU_DEVICES") || die "could not read VRAM for cards $E2E_KIT_GPU_DEVICES"
say "VRAM used before bring-up (card:MiB): $gpu_reading"
for pair in $gpu_reading; do
  [ "${pair#*:}" -gt 4096 ] && die "card ${pair%%:*} already holds ${pair#*:} MiB, so it is not free. Not substituting another card"
done

# ---- 2. mounts --------------------------------------------------------------
mkdir -p "$E2E_KIT_WORK_ROOT" "$KIT_LOG_DIR" "$KIT_LAUNCH_DIR" || die "cannot create $E2E_KIT_WORK_ROOT"
MOUNTS=(-v "$KIT_MODEL_REAL:$KIT_MODEL_REAL:ro" -v "$E2E_KIT_WORK_ROOT:$E2E_KIT_WORK_ROOT")
MEASURE_MOUNTED=()
for root in $E2E_KIT_MEASURE_ROOTS; do
  mkdir -p "$root" 2>/dev/null || true
  [ -d "$root" ] || die "measurement root does not exist and cannot be created: $root"
  at=$(measure_mount_for "$root") || die "no acceptable mount for measurement root $root (see above)"
  case " ${MEASURE_MOUNTED[*]:-} " in *" $at "*) continue ;; esac
  MOUNTS+=(-v "$at:$at")
  MEASURE_MOUNTED+=("$at")
done
say "mounts: model(ro)=$KIT_MODEL_REAL work(rw)=$E2E_KIT_WORK_ROOT measure(rw)=${MEASURE_MOUNTED[*]:-none}"

# ---- 3. engine container (hosts the worker AND the router) --------------------
cid=$(docker run -d --name "$E2E_KIT_CONTAINER" --label "$KIT_LABEL" \
  --network=host --ipc=host --shm-size=64G \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  "${MOUNTS[@]}" \
  "$E2E_KIT_IMAGE" sleep infinity) || die "engine container failed to start"
CREATED+=("$cid")
say "engine container $E2E_KIT_CONTAINER image_id=$(docker inspect "$E2E_KIT_CONTAINER" --format '{{.Image}}')"

# ---- 4. etcd ---------------------------------------------------------------------
# v3.5.x image: empty ENTRYPOINT and Cmd=[/usr/local/bin/etcd], hence
# --entrypoint; the three peer flags move together or etcd exits on
# --initial-cluster. Bound to the node IP: etcd here is unauthenticated.
eid=$(docker run -d --name "$E2E_KIT_ETCD_CONTAINER" --label "$KIT_LABEL" --network=host \
  --entrypoint /usr/local/bin/etcd "$E2E_KIT_ETCD_IMAGE" \
  --name "etcd-${E2E_KIT_RUN_TAG}" --data-dir /tmp/etcd-data \
  --advertise-client-urls "http://${E2E_KIT_NODE_IP}:${KIT_ETCD_PORT}" \
  --listen-client-urls "http://${E2E_KIT_NODE_IP}:${KIT_ETCD_PORT}" \
  --listen-peer-urls "http://${E2E_KIT_NODE_IP}:${KIT_ETCD_PEER_PORT}" \
  --initial-advertise-peer-urls "http://${E2E_KIT_NODE_IP}:${KIT_ETCD_PEER_PORT}" \
  --initial-cluster "etcd-${E2E_KIT_RUN_TAG}=http://${E2E_KIT_NODE_IP}:${KIT_ETCD_PEER_PORT}") \
  || die "etcd container failed to start"
CREATED+=("$eid")
ok=0
for _ in $(seq 1 30); do
  curl -sf -m3 "http://${E2E_KIT_NODE_IP}:${KIT_ETCD_PORT}/version" >/dev/null && { ok=1; break; }
  sleep 1
done
[ "$ok" = 1 ] || { docker logs --tail 20 "$E2E_KIT_ETCD_CONTAINER" >&2; die "etcd did not answer on $KIT_ETCD_PORT"; }
say "etcd up on $KIT_ETCD_PORT"

# ---- 5. worker --------------------------------------------------------------------
cp "$HERE/worker.sh" "$HERE/router.sh" "$KIT_LAUNCH_DIR/" || die "cannot stage launch scripts"
docker exec -d "$E2E_KIT_CONTAINER" env \
  MY_IP="$E2E_KIT_NODE_IP" MODEL="$KIT_MODEL_REAL" SERVED="$E2E_KIT_SERVED_NAME" \
  PORT="$KIT_WORKER_PORT" ETCD_PORT="$KIT_ETCD_PORT" TP="$E2E_KIT_TP" GPUS="$E2E_KIT_GPU_DEVICES" \
  CTX="$E2E_KIT_CTX" GMU="$E2E_KIT_MEM_FRACTION" GRAPH_MAX_BS="$E2E_KIT_CUDA_GRAPH_MAX_BS" \
  KV_PUB_PORT="$KIT_KV_EVENTS_PORT" KV_SNAP_PORT="$KIT_KV_SNAPSHOT_PORT" \
  DSA_ARGS="$E2E_KIT_DSA_ARGS" PARSER_ARGS="$E2E_KIT_PARSER_ARGS" \
  EXTRA_ARGS="$E2E_KIT_ENGINE_EXTRA_ARGS" EXTRA_ENV="$E2E_KIT_ENGINE_EXTRA_ENV" \
  READY_TIMEOUT="$E2E_KIT_READY_TIMEOUT" LOG="$KIT_LOG_DIR/worker.log" \
  bash "$KIT_LAUNCH_DIR/worker.sh" || die "could not launch the worker"
say "worker launching -> $KIT_LOG_DIR/worker.log"

# Wait for the ENGINE's own /health before starting the router. Cold start is
# minutes (weights, mamba+KV pools, graph capture, JIT); 503s the whole time are
# not a hang. A dead process is surfaced instead of waited out.
deadline=$(( $(date +%s) + E2E_KIT_READY_TIMEOUT ))
sleep 5
while :; do
  curl -sf -m3 "$KIT_ENGINE_URL/health" >/dev/null 2>&1 && break
  if ! docker exec "$E2E_KIT_CONTAINER" pgrep -f infera.engine.sglang >/dev/null 2>&1; then
    tail -40 "$KIT_LOG_DIR/worker.log" >&2
    die "worker process exited before reporting ready (log: $KIT_LOG_DIR/worker.log)"
  fi
  [ "$(date +%s)" -lt "$deadline" ] || { tail -40 "$KIT_LOG_DIR/worker.log" >&2; die "worker not healthy within ${E2E_KIT_READY_TIMEOUT}s"; }
  sleep 10
done
say "worker /health 200 after $(( $(date +%s) - T0 ))s"
docker exec "$E2E_KIT_CONTAINER" sh -c "tr '\\0' ' ' < /proc/\$(pgrep -f 'infera.engine.sglang' | head -1)/cmdline" \
  > "$KIT_LOG_DIR/engine_argv.txt" 2>/dev/null || true

# ---- 6. router ----------------------------------------------------------------------
docker exec -d "$E2E_KIT_CONTAINER" env \
  MY_IP="$E2E_KIT_NODE_IP" ROUTER_PORT="$KIT_ROUTER_PORT" ETCD_PORT="$KIT_ETCD_PORT" \
  MODEL="$KIT_MODEL_REAL" ROUTER_EXTRA_ARGS="$E2E_KIT_ROUTER_EXTRA_ARGS" LOG="$KIT_LOG_DIR/router.log" \
  bash "$KIT_LAUNCH_DIR/router.sh" || die "could not launch the router"
say "router launching on $KIT_ROUTER_PORT -> $KIT_LOG_DIR/router.log"

# ---- 7. ready THROUGH THE ROUTER, then the handshake ------------------------------------
remaining=$(( deadline - $(date +%s) )); [ "$remaining" -lt 180 ] && remaining=180
E2E_KIT_READY_TIMEOUT=$remaining bash "$HERE/wait_ready.sh" || die "router did not become ready (log: $KIT_LOG_DIR/router.log)"
docker exec "$E2E_KIT_CONTAINER" sh -c "tr '\\0' ' ' < /proc/\$(pgrep -f 'infera.server' | head -1)/cmdline" \
  > "$KIT_LOG_DIR/router_argv.txt" 2>/dev/null || true

ELAPSED=$(( $(date +%s) - T0 ))
H_ENDPOINT="$KIT_ROUTER_URL" H_ENGINE="$KIT_ENGINE_URL" H_CTR="$E2E_KIT_CONTAINER" \
H_ETCD="$E2E_KIT_ETCD_CONTAINER" H_TAG="$E2E_KIT_RUN_TAG" H_LABEL="$KIT_LABEL" \
H_WORK="$E2E_KIT_WORK_ROOT" H_MEASURE="${MEASURE_MOUNTED[*]:-}" H_SERVED="$E2E_KIT_SERVED_NAME" \
H_MODEL="$KIT_MODEL_REAL" H_IMAGE="$E2E_KIT_IMAGE" \
H_IMAGE_ID="$(docker inspect "$E2E_KIT_CONTAINER" --format '{{.Image}}')" \
H_GPUS="$E2E_KIT_GPU_DEVICES" H_TP="$E2E_KIT_TP" H_CTX="$E2E_KIT_CTX" H_BS="$E2E_KIT_CUDA_GRAPH_MAX_BS" \
H_PORTS="$KIT_ALL_PORTS" H_LOGS="$KIT_LOG_DIR" H_SECS="$ELAPSED" \
python3 - "$KIT_HANDSHAKE" <<'PY' || die "could not write the handshake"
import json, os, sys
e = os.environ
router, worker, etcd, peer, kv, snap = (int(x) for x in e["H_PORTS"].split())
d = {
    "endpoint": e["H_ENDPOINT"],
    "engine_endpoint": e["H_ENGINE"],
    "container": e["H_CTR"],
    "containers": [e["H_CTR"], e["H_ETCD"]],
    "containerized": True,
    "run_tag": e["H_TAG"],
    "label": e["H_LABEL"],
    "work_root_in_container": e["H_WORK"],
    "work_root_on_host": e["H_WORK"],
    "measurement_roots": e["H_MEASURE"].split(),
    "served_model_name": e["H_SERVED"],
    "model_path": e["H_MODEL"],
    "image": e["H_IMAGE"],
    "image_id": e["H_IMAGE_ID"],
    "gpu_devices": [int(x) for x in e["H_GPUS"].split(",")],
    "tp_size": int(e["H_TP"]),
    "context_length": int(e["H_CTX"]),
    "cuda_graph_max_bs": int(e["H_BS"]),
    "deploy_mode": "mix",
    "ports": {"router": router, "worker": worker, "etcd": etcd, "etcd_peer": peer,
              "kv_events": kv, "kv_snapshot": snap},
    "logs": {"worker": e["H_LOGS"] + "/worker.log", "router": e["H_LOGS"] + "/router.log"},
    "bringup_seconds": int(e["H_SECS"]),
}
with open(sys.argv[1], "w") as f:
    json.dump(d, f, indent=2)
PY
say "READY in ${ELAPSED}s. endpoint=$KIT_ROUTER_URL handshake=$KIT_HANDSHAKE"
