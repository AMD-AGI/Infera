#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: shared helpers for the DSv4-Pro repro kit (env checks + docker/etcd/router/smoke).
# why : one place for the infera-way bring-up so every case stays tiny and consistent.
# how : each case script sources this, then calls the functions; callers = engine/*/*.
set -uo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[0;33m'; NC='\033[0m'
log(){ echo -e "${GRN}[repro]${NC} $*"; }
warn(){ echo -e "${YEL}[repro]${NC} $*" >&2; }
die(){ echo -e "${RED}[repro] ERROR:${NC} $*" >&2; exit 1; }

# what: fail fast if a required env var is unset/empty. why: requirement #2 strict check.
require_env(){ local v="$1" d="${2:-}"; [ -n "${!v:-}" ] || die "env '$v' is required${d:+ ($d)}. See README.md."; }
# what: fail if a path is missing (model/mount). why: catch bad INFERA_MODEL before a 30min load.
require_file(){ [ -e "$1" ] || die "path not found: $1 (check INFERA_MODEL / INFERA_MODEL_MOUNT)"; }

# what: shared config every case reads. why: zero local info — mount is required (no default),
# the rest fall back to sane defaults.
: "${ETCD_PORT:=2379}"
: "${ETCD_IMAGE:=quay.io/coreos/etcd:v3.5.14}"
: "${ROUTER_PORT:=8000}"
: "${ENGINE_PORT:=30000}"
: "${GID_INDEX:=1}"

# what: start a persistent RDMA-capable container (host net + GPU + IB). why: all infera
# commands run via `docker exec` into it. how: caller passes name; image from INFERA_IMAGE.
start_container(){
  local name="$1"; require_env INFERA_IMAGE; require_env INFERA_MODEL_MOUNT
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --network=host --ipc=host --shm-size=32G \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v "$INFERA_MODEL_MOUNT:$INFERA_MODEL_MOUNT" --entrypoint "" \
    "$INFERA_IMAGE" sleep infinity >/dev/null || die "docker run failed for $name"
  log "container '$name' up ($INFERA_IMAGE)"
}

# what: start etcd registry on this node. why: infera.server + engines self-register here.
start_etcd(){
  local name="${1:-repro-etcd}" ip="${2:-127.0.0.1}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --network=host "$ETCD_IMAGE" etcd \
    --advertise-client-urls "http://$ip:$ETCD_PORT" \
    --listen-client-urls "http://0.0.0.0:$ETCD_PORT" >/dev/null || die "etcd start failed"
  for i in $(seq 1 30); do
    curl -sf "http://$ip:$ETCD_PORT/health" >/dev/null 2>&1 && { log "etcd healthy ($ip:$ETCD_PORT)"; return 0; }
    sleep 1; done
  die "etcd not healthy in 30s"
}

# what: start the infera.server router in a container (etcd discovery, http transport).
# why : this is THE infera way — one router auto-pairs P/D; no sglang_router/vllm-router.
start_router(){
  local ctr="$1" etcd_ip="${2:-127.0.0.1}"; require_env INFERA_MODEL
  local tok="${INFERA_TOKENIZER:-$INFERA_MODEL}"   # override if the checkpoint tokenizer is broken
  docker exec -d "$ctr" bash -lc "python3 -m infera.server --host 0.0.0.0 --port $ROUTER_PORT \
    --discovery-backend etcd --etcd-endpoint $etcd_ip:$ETCD_PORT \
    --request-transport http --kv-event-transport zmq --router-policy round-robin \
    --router-tokenizer-path $tok > /tmp/router.log 2>&1"
  wait_health "http://$etcd_ip:$ROUTER_PORT/health" 40 && log "router up (:$ROUTER_PORT)"
}

# what: poll an HTTP /health until 200 or timeout. why: cold start is slow (~30min), not a hang.
wait_health(){ local url="$1" tries="${2:-200}"; for i in $(seq 1 "$tries"); do
  curl -sf -m5 "$url" >/dev/null 2>&1 && return 0; sleep 10; done; warn "timeout waiting $url"; return 1; }

# what: wait for an engine log to print ready, early-exit on fatal errors. why: fail fast on OOM.
wait_worker_log(){ local ctr="$1" logf="$2" tries="${3:-200}"; for i in $(seq 1 "$tries"); do
  docker exec "$ctr" grep -qiE 'ready to roll|Application startup complete|Uvicorn running' "$logf" 2>/dev/null && return 0
  docker exec "$ctr" grep -qiE 'Traceback|CUDA error|HIP error|out of memory|Address already in use' "$logf" 2>/dev/null \
    && { warn "engine error in $logf:"; docker exec "$ctr" tail -15 "$logf" >&2; return 1; }
  sleep 10; done; warn "timeout waiting engine ready ($logf)"; return 1; }

# what: curl a one-shot chat completion through the router. why: prove P/D pairing + serving.
smoke(){ local url="$1"; require_env INFERA_MODEL
  log "workers:"; curl -s "$url/v1/workers" 2>/dev/null | python3 -m json.tool 2>/dev/null || true
  log "smoke completion:"
  curl -s "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$INFERA_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=?\"}],\"max_tokens\":32,\"temperature\":0}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['choices'][0]['message']['content'])" \
    || die "smoke failed — router did not return a completion"
  log "smoke OK"; }

# what: kill engine procs in a container + wait for VRAM to drain. why: relaunch OOMs otherwise;
# sglang::scheduler / spawn children hold VRAM after the launcher dies, so target them too.
reap(){ local ctr="$1"; docker exec "$ctr" bash -lc \
  "pkill -9 -f 'infera.engine' 2>/dev/null; pkill -9 -f 'multiprocessing.spawn' 2>/dev/null; \
   pkill -9 -f 'sglang' 2>/dev/null; pkill -9 -f 'EngineCore' 2>/dev/null; true"
  sleep 5; log "reaped engines in $ctr"; }
