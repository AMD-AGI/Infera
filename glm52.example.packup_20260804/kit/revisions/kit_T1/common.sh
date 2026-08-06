#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: shared helpers for the GLM-5.2 1P1D kit (env checks + container/etcd/router/smoke/reap).
# why : one place for the infera-way bring-up so every script stays small and consistent.
# how : sourced by engine/*.sh and preflight_rdma.sh; the cluster wrappers never source it.
set -uo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[0;33m'; NC='\033[0m'
log(){ echo -e "${GRN}[glm52]${NC} $*"; }
warn(){ echo -e "${YEL}[glm52]${NC} $*" >&2; }
die(){ echo -e "${RED}[glm52] ERROR:${NC} $*" >&2; exit 1; }

# what: fail fast if a required env var is unset/empty. why: a 10-minute cold start is a
# terrible place to discover a typo; every knob is set in cluster/<your-cluster>.sh.
require_env(){ local v="$1" d="${2:-}"; [ -n "${!v:-}" ] || die "env '$v' is required${d:+ ($d)}. Set it in cluster/<wrapper>.sh — see cluster/README.md."; }

# ---- defaults that are NOT cluster-specific ---------------------------------
# Anything a site must change lives in cluster/*.sh, never here.
: "${ETCD_PORT:=2379}"
: "${ETCD_IMAGE:=quay.io/coreos/etcd:v3.5.14}"
: "${ROUTER_PORT:=8100}"
: "${PREFILL_PORT:=30000}"
: "${DECODE_PORT:=30001}"
: "${BOOTSTRAP_PORT:=8998}"
: "${CTR:=glm52_pd}"
: "${SERVED:=glm5.2-mxfp4}"
: "${KVD_SOCK:=/tmp/kvd/kvd.sock}"

# what: start the long-lived engine container (host net + GPU + IB), then sleep.
# why : every infera command runs via `docker exec` into it, so a leg restart does not pay
#       the container-create cost. how: caller passes the node's role only for the log line.
# note: --entrypoint '' is deliberate. The image entrypoint bind-maps a host RDMA provider
#       library, which is a no-op without the corresponding mount; skipping it also keeps
#       `sleep infinity` from being re-exec'd. Set ENTRYPOINT_KEEP=1 if your image's
#       entrypoint is load-bearing on your fabric (see cluster/README.md).
start_container(){
  require_env INFERA_IMAGE "the engine image"
  require_env MODEL_MOUNT "host dir holding the weights, bind-mounted into the container"
  local ep=(--entrypoint '')
  if [ "${ENTRYPOINT_KEEP:-0}" = "1" ]; then ep=(); fi
  local mounts=(-v "$MODEL_MOUNT:$MODEL_MOUNT")
  # Only mount a host RDMA provider library if the site says it needs one. On a fabric
  # whose in-container provider already matches the host driver, mounting a nonexistent
  # path just fails the run.
  if [ -n "${HOST_RDMA_LIB:-}" ]; then
    mounts+=(-v "$HOST_RDMA_LIB:/host-rdma/$(basename "$HOST_RDMA_LIB"):ro")
  fi
  docker rm -f "$CTR" >/dev/null 2>&1 || true
  docker run -d --name "$CTR" --network=host --ipc=host --shm-size=32G \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    "${mounts[@]}" "${ep[@]}" "$INFERA_IMAGE" sleep infinity >/dev/null \
    || die "docker run failed for $CTR"
  log "container '$CTR' up ($INFERA_IMAGE)"
  docker exec "$CTR" python3 -c 'import torch;print("  gpu gate:", torch.cuda.is_available(), torch.cuda.device_count())' || true
  echo -n "  RDMA PORT_ACTIVE visible in container: "
  docker exec "$CTR" bash -c 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE || echo 0'
}

# what: etcd on the prefill node. why: both legs and the router self-register here, so the
# router pairs P and D with no static worker list.
start_etcd(){
  local ip="${1:?etcd advertise ip}" name="${2:-glm52-etcd}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --network=host "$ETCD_IMAGE" etcd \
    --advertise-client-urls "http://$ip:$ETCD_PORT" \
    --listen-client-urls "http://0.0.0.0:$ETCD_PORT" >/dev/null || die "etcd start failed"
  for _ in $(seq 1 30); do
    curl -sf -m2 "http://$ip:$ETCD_PORT/health" >/dev/null 2>&1 && { log "etcd healthy ($ip:$ETCD_PORT)"; return 0; }
    sleep 1
  done
  die "etcd not healthy in 30s"
}

# what: the infera router, inside $CTR on the prefill node.
# why : the module is `infera.server`. It is NOT `infera.router` — that is a package with no
#       __main__, and the failure reads like a missing dependency rather than a wrong name.
# how : POLICY=kv-aware|round-robin. kv-aware needs the tokenizer or it silently degrades to
#       load-only routing. The Rust backend is the default and rejects, rather than ignores,
#       an option it does not implement.
start_router(){
  local ip="${1:?router bind ip}"
  require_env TOKENIZER "tokenizer path the router loads for kv-aware routing"
  local policy="${ROUTER_POLICY:-kv-aware}" backend="${ROUTER_BACKEND:-rust}"
  local kv_args=()
  if [ "$policy" = "kv-aware" ]; then
    # Cost = w * (request_blocks - hits) + active_blocks. Prefill is compute-bound and a hit
    # skips an entire prefill pass, so its weight is high; decode is memory-bound on the KV
    # transfer and gains little from a prefill-time hit, so route it by load.
    kv_args=(--router-tokenizer-path "$TOKENIZER"
             --kv-prefill-overlap-weight "${KV_PREFILL_W:-20.0}"
             --kv-decode-overlap-weight "${KV_DECODE_W:-2.0}")
  fi
  # Kill a previous router by its own pid only. A bare `pkill -f infera.server` also matches
  # the `docker exec bash -c ...` command string that CONTAINS that text — i.e. this shell.
  docker exec "$CTR" bash -c "pgrep -f 'python3 -m infera.server' | xargs -r kill -9 2>/dev/null; true"
  sleep 2
  docker exec -d "$CTR" bash -c "nohup python3 -m infera.server \
    --host 0.0.0.0 --port $ROUTER_PORT --router-backend $backend \
    --discovery-backend etcd --etcd-endpoint $ip:$ETCD_PORT \
    --request-transport http --kv-event-transport zmq \
    --router-policy $policy ${kv_args[*]} > /tmp/router.log 2>&1"
  wait_health "http://$ip:$ROUTER_PORT/health" 12 5 \
    || { docker exec "$CTR" tail -30 /tmp/router.log; die "router did not come up"; }
  log "router up on :$ROUTER_PORT (backend=$backend policy=$policy)"
}

# what: poll an HTTP endpoint until it answers. why: a PD leg's cold start is minutes
#       (weights + graph capture) and silence is NOT a hang.
# note: poll HTTP, never grep the log for "ready to roll" — logs are appended to across
#       restarts, so a grep matches the PREVIOUS run's line within seconds and the caller
#       proceeds against an engine that is still loading weights.
wait_health(){
  local url="$1" tries="${2:-120}" gap="${3:-15}"
  for i in $(seq 1 "$tries"); do
    curl -sf -m5 "$url" >/dev/null 2>&1 && { log "$url serving after $((i * gap))s"; return 0; }
    sleep "$gap"
  done
  warn "$url did NOT answer within $((tries * gap))s"
  return 1
}

# what: kill engines in $CTR and wait for VRAM to drain.
# why : the infera wrapper exits before its sglang child does, and the child keeps the KV
#       event port block bound. The next leg then dies with "port_base at N is not
#       available", which reads as a port-allocation bug rather than as leftover state.
#       The WAIT is the point, not the kill.
reap(){
  docker exec "$CTR" bash -c '
    pkill -9 -f "infera.engine.sglang" 2>/dev/null
    pkill -9 -f "sglang.launch_server" 2>/dev/null
    pkill -9 -f "multiprocessing.spawn" 2>/dev/null
    for _ in $(seq 1 20); do
      n=$(ps aux | grep -E "launch_server|infera.engine" | grep -v grep | wc -l)
      [ "$n" -eq 0 ] && exit 0
      sleep 2
    done
    echo "  WARNING: engines still present after 40s" >&2' 2>/dev/null || true
  log "reaped engines in $CTR"
}
