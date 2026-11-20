#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: shared helpers for the DeepSeek-V4.1-Flash vLLM 1P1D kit (container, etcd, UMBP, router, health, reap).
# why : one place for the infera-way bring-up so every script stays small and consistent.
# how : sourced by engine/*.sh and preflight_rdma.sh; the cluster wrappers never source it.
set -uo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[0;33m'; NC='\033[0m'
log(){ echo -e "${GRN}[dsv41]${NC} $*"; }
warn(){ echo -e "${YEL}[dsv41]${NC} $*" >&2; }
die(){ echo -e "${RED}[dsv41] ERROR:${NC} $*" >&2; exit 1; }

# what: fail fast if a required env var is unset/empty. why: a 10-minute cold start is a
# terrible place to discover a typo; every knob is set in cluster/<your-cluster>.sh.
require_env(){ local v="$1" d="${2:-}"; [ -n "${!v:-}" ] || die "env '$v' is required${d:+ ($d)}. Set it in cluster/<wrapper>.sh — see cluster/README.md."; }

# ---- defaults that are NOT cluster-specific ---------------------------------
: "${ETCD_PORT:=2379}"
: "${ETCD_IMAGE:=quay.io/coreos/etcd:v3.5.14}"
: "${ROUTER_PORT:=8100}"
: "${PREFILL_PORT:=30000}"
: "${DECODE_PORT:=30001}"
: "${CTR:=dsv41_pd}"
: "${SERVED:=deepseek-ai/DeepSeek-V4.1-Flash}"
# Host directory for every log; bind-mounted at the same path, so logs outlive the container.
: "${LOG_DIR:=/tmp/dsv41_pd}"
# Host directory for the UMBP server's sockets, bind-mounted at /umbp-host/umbp (the engine and the
# server are separate processes; this is the one path both see).
: "${UMBP_DIR:=/dev/shm/dsv41_umbp}"
UMBP_SOCK=unix:///umbp-host/umbp/standalone.grpc.sock

# What: run a command on a node. The local node runs it directly, so a single-node pair (and a
# wrapper launched on the node itself, e.g. under `srun`) needs no ssh at all.
SSH_CMD="${SSH_CMD:-ssh -o StrictHostKeyChecking=no}"
on(){ local h="$1"; shift; if [ "$h" = "$(hostname -s)" ] || [ "$h" = localhost ]; then bash -c "$*"; else $SSH_CMD "$h" "$*"; fi; }

# what: start the long-lived engine container (host net + GPU + RDMA), then sleep.
# why : every leg, the UMBP server and the router `docker exec` into it, so a leg restart is cheap.
# note: the vLLM image has no `render` group, so the host's numeric video/render gids are passed;
#       nofile is raised because the router holds client + prefill + decode connections per request.
start_container(){
  require_env INFERA_IMAGE "the engine image built by build_image.sh"
  require_env MODEL_MOUNT "host dir holding the weights, bind-mounted into the container"
  require_env KIT_DIR "this kit; mounted so the probes run inside the container"
  local gids=() g gid
  for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [ -n "$gid" ] && gids+=(--group-add "$gid"); done
  mkdir -p "$LOG_DIR" "$UMBP_DIR" && chmod 777 "$LOG_DIR" "$UMBP_DIR"
  docker rm -f "$CTR" >/dev/null 2>&1 || true
  docker run -d --name "$CTR" --network=host --ipc=host --shm-size=64G \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband "${gids[@]}" \
    --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
    -v "$MODEL_MOUNT:$MODEL_MOUNT:ro" -v "$KIT_DIR:$KIT_DIR:ro" -v "$LOG_DIR:$LOG_DIR" -v "$UMBP_DIR:/umbp-host/umbp" \
    --entrypoint '' "$INFERA_IMAGE" sleep infinity >/dev/null \
    || die "docker run failed for $CTR"
  log "container '$CTR' up ($INFERA_IMAGE)"
  docker exec "$CTR" cat /opt/infera/BUILD_INFO 2>/dev/null | sed 's/^/  /'
  # Zero devices is not an error to the engine — Mooncake falls back to a 5-20x slower
  # transport and the legs serve fine. Hence a warning, not a number to skim past.
  local nports
  nports=$(docker exec "$CTR" bash -c 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE; true' | tr -d '\r\n')
  echo "  RDMA PORT_ACTIVE visible in container: ${nports:-0}"
  [ "${nports:-0}" -gt 0 ] || warn "  NO RDMA device usable inside $CTR — KV will not move over RDMA (docker exec $CTR ibv_devinfo)."
}

# what: etcd on the prefill node. why: both legs and the router self-register here, so the
# router pairs P and D with no static worker list.
start_etcd(){
  local ip="${1:?etcd advertise ip}" name="${CTR}-etcd"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --network=host "$ETCD_IMAGE" etcd --data-dir /tmp/etcd \
    --advertise-client-urls "http://$ip:$ETCD_PORT" --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
    --listen-peer-urls "http://127.0.0.1:$((ETCD_PORT + 1))" \
    --initial-advertise-peer-urls "http://127.0.0.1:$((ETCD_PORT + 1))" \
    --initial-cluster "default=http://127.0.0.1:$((ETCD_PORT + 1))" >/dev/null || die "etcd start failed"
  for _ in $(seq 1 30); do
    curl -sf -m2 "http://$ip:$ETCD_PORT/health" >/dev/null 2>&1 && { log "etcd healthy ($ip:$ETCD_PORT)"; return 0; }
    sleep 1
  done
  die "etcd not healthy in 30s"
}

# what: the UMBP standalone server — the host-DRAM KV store the prefill leg's UMBPStoreConnector
#       writes to and reads prefix hits back from (InferenceX's c384 point: 449 GB, 4 KiB pages).
# why : P/D keeps one copy of every session's KV on each side; without a host tier the prefill
#       prefix cache evicts long agentic sessions as soon as the working set outgrows HBM.
# note: faulting in a ~450 GB pool on 4 KiB pages takes minutes; the binary ships inside MORI.
start_umbp(){
  local gb="${UMBP_GB:-449}"
  docker exec "$CTR" bash -c "pkill -f umbp_standalone_server; rm -f /umbp-host/umbp/*" 2>/dev/null
  # UMBP_PRELOAD: the GEM allocator, when the prefill KV cache lives in GEM BOs — it is what lets the
  # server's hipIpcOpenMemHandle map them (see deploy/docker/native/hip_gem_alloc.c).
  docker exec -d "$CTR" bash -c "P=\$(python3 -c 'import mori, os; print(os.path.dirname(mori.__file__))') && \
    UMBP_STANDALONE_ADDRESS=$UMBP_SOCK UMBP_DRAM_CAPACITY=$((gb * 1000000000)) UMBP_DRAM_USE_HUGEPAGES=0 \
    UMBP_SSD_ENABLED=0 MORI_UMBP_LOG_LEVEL=INFO LD_LIBRARY_PATH=\$P:\$LD_LIBRARY_PATH ${UMBP_PRELOAD:+LD_PRELOAD=$UMBP_PRELOAD} \
    exec \$P/umbp_standalone_server $UMBP_SOCK > $LOG_DIR/umbp.log 2>&1"
  for i in $(seq 1 180); do
    grep -q '\[StandaloneServer\] listening' "$LOG_DIR/umbp.log" 2>/dev/null && { log "UMBP server listening (${gb} GB DRAM) after $((i * 10))s"; return 0; }
    docker exec "$CTR" pgrep -f umbp_standalone_server >/dev/null || { tail -5 "$LOG_DIR/umbp.log"; die "UMBP server died"; }
    sleep 10
  done
  die "UMBP server not listening in 1800s"
}

# what: the infera router, inside $CTR on the prefill node.
# why : the module is `infera.server`, NOT `infera.router` — the latter has no __main__.
# how : kv-aware with prefill session affinity: a session's turns stay on the prefill DP rank
#       that holds its prefix; the decode leg is one worker whose ranks vLLM balances itself.
start_router(){
  local ip="${1:?router bind ip}"
  require_env TOKENIZER "tokenizer path the router loads for kv-aware routing"
  docker exec "$CTR" bash -c "pgrep -f 'python3 -m infera.server' | xargs -r kill -9; pkill -9 -x infera-router; true"
  sleep 2
  # INFERA_VLLM_MOONCAKE_SERIAL=1: post a request to decode only after its prefill finished. vLLM's
  # decode reserves the whole prompt's KV when a request arrives and starts its pull timeout
  # (VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT); dispatched concurrently, a long prefill queue (an AgentX
  # warmup burst at c768: ~800 cold 128k-token prompts on one DP2 prefill) fills decode with idle
  # reservations and times requests out before their prefill even starts.
  docker exec -d -e INFERA_VLLM_MOONCAKE_SERIAL="${MOONCAKE_SERIAL:-1}" "$CTR" bash -c "exec python3 -m infera.server \
    --host 0.0.0.0 --port $ROUTER_PORT --router-backend rust \
    --discovery-backend etcd --etcd-endpoint $ip:$ETCD_PORT \
    --request-transport http --kv-event-transport zmq \
    --router-policy kv-aware --router-tokenizer-path $TOKENIZER \
    --kv-prefill-overlap-weight ${KV_PREFILL_W:-20} --kv-decode-overlap-weight ${KV_DECODE_W:-2} \
    --session-affinity ${SESSION_AFFINITY:-prefill} --pd-prefill-guard-release completion \
    --timeout-keep-alive 600 ${ROUTER_EXTRA_ARGS:-} > $LOG_DIR/router.log 2>&1"
  wait_health "http://$ip:$ROUTER_PORT/health" 24 5 || { tail -30 "$LOG_DIR/router.log"; die "router did not come up"; }
  log "router up on :$ROUTER_PORT (rust, kv-aware, session affinity ${SESSION_AFFINITY:-prefill})"
}

# what: poll an HTTP endpoint until it answers. why: a leg's cold start is minutes and silence
# is NOT a hang. note: DO NOT grep a log for a ready line instead — a re-used log matches the
# PREVIOUS run's line and returns early.
wait_health(){
  local url="$1" tries="${2:-120}" gap="${3:-15}"
  for i in $(seq 1 "$tries"); do
    curl -sf -m5 "$url" >/dev/null 2>&1 && { log "$url serving after $((i * gap))s"; return 0; }
    sleep "$gap"
  done
  warn "$url did NOT answer within $((tries * gap))s"
  return 1
}

# what: kill the leg serving PORT (the infera wrapper and its whole vLLM process tree) and wait
#       for its GPUs to drain. why: the wrapper exits before its children, which keep VRAM and the
#       DP / Mooncake ports, and the next launch then dies at bind or OOMs on an idle-looking box.
reap_leg(){
  local port="${1:?port}"
  docker exec "$CTR" bash -c 'kt() { local c; for c in $(pgrep -P "$1"); do kt "$c"; done; kill -9 "$1" 2>/dev/null; }
    for p in $(pgrep -f -- "infera.engine.vllm.*--port '"$port"'( |$)"); do kt "$p"; done
    for _ in $(seq 1 30); do pgrep -f -- "--port '"$port"'( |$)" >/dev/null || exit 0; sleep 2; done
    echo "  WARNING: leg on :'"$port"' still present after 60s" >&2' 2>/dev/null || true
  sleep 10
  log "reaped the leg on :$port"
}
