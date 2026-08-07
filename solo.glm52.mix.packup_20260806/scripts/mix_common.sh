#!/usr/bin/env bash
# what: shared helpers for the GLM-5.2 MIX (aggregated) bring-up on one MI355X node.
# why : one place for container / etcd / kvd / router / reap so each script stays small.
# how : sourced by mix_up.sh, mix_engine.sh, mix_smoke.sh, mix_down.sh. Runs ON the node.
#
# Derived from examples/sglang_1p1d_glm5.2/common.sh. Deltas, all because this is MIX:
#   * no PD ports, no bootstrap port, no RDMA/mooncake anything — KV never leaves the node
#   * one worker, one port, one mem-fraction
set -uo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[0;33m'; NC='\033[0m'
log(){ echo -e "${GRN}[mix]${NC} $*"; }
warn(){ echo -e "${YEL}[mix]${NC} $*" >&2; }
die(){ echo -e "${RED}[mix] ERROR:${NC} $*" >&2; exit 1; }

require_env(){ local v="$1" d="${2:-}"; [ -n "${!v:-}" ] || die "env '$v' is required${d:+ ($d)}"; }

: "${ETCD_PORT:=2379}"
: "${ETCD_IMAGE:=quay.io/coreos/etcd:v3.5.14}"
: "${ROUTER_PORT:=8100}"
: "${ENGINE_PORT:=30000}"
: "${CTR:=glm52_mix}"
: "${ETCD_CTR:=glm52-mix-etcd}"
: "${SERVED:=glm5.2-mxfp4}"
: "${KVD_SOCK:=/tmp/kvd/kvd.sock}"

# what: the long-lived engine container. why: every command docker-execs into it, so an
# engine restart costs a relaunch, not a container rebuild.
start_container(){
  require_env INFERA_IMAGE; require_env MODEL_MOUNT
  local ep=(--entrypoint '')
  [ "${ENTRYPOINT_KEEP:-0}" = "1" ] && ep=()
  docker rm -f "$CTR" >/dev/null 2>&1 || true
  docker run -d --name "$CTR" --network=host --ipc=host --shm-size=32G \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v "$MODEL_MOUNT:$MODEL_MOUNT" "${ep[@]}" "$INFERA_IMAGE" sleep infinity >/dev/null \
    || die "docker run failed for $CTR"
  log "container '$CTR' up ($INFERA_IMAGE)"
  docker exec "$CTR" python3 -c 'import torch;print("  gpu gate:", torch.cuda.is_available(), torch.cuda.device_count())' || true
}

# what: etcd, so infera.server discovers the worker instead of carrying a static list.
start_etcd(){
  local ip="${1:?etcd advertise ip}"
  docker rm -f "$ETCD_CTR" >/dev/null 2>&1 || true
  docker run -d --name "$ETCD_CTR" --network=host "$ETCD_IMAGE" etcd \
    --advertise-client-urls "http://$ip:$ETCD_PORT" \
    --listen-client-urls "http://0.0.0.0:$ETCD_PORT" >/dev/null || die "etcd start failed"
  for _ in $(seq 1 30); do
    curl -sf -m2 "http://$ip:$ETCD_PORT/health" >/dev/null 2>&1 && { log "etcd healthy ($ip:$ETCD_PORT)"; return 0; }
    sleep 1
  done
  die "etcd not healthy in 30s"
}

# what: the kvd daemon (L2 pinned host RAM + L3 on disk).
# why : stage a script FILE and nohup it — a detached `docker exec -d ... bash -lc '<cmd>'`
#       login shell exits and takes the child with it: no process, no log, no error.
# note: --max-bytes / --long-bytes are ABSOLUTE. The ratio-based default sizes off
#       max_total_num_tokens and can ask for hundreds of GB per rank; L3 writes to a
#       container-local path, so an oversized budget fills the node's root fs.
start_kvd(){
  docker exec "$CTR" bash -c "mkdir -p /tmp/kvd /tmp/kvd-long && rm -f $KVD_SOCK && \
printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.kvd --socket $KVD_SOCK \
--max-bytes ${KVD_MAX_BYTES:-64G} --long-path /tmp/kvd-long --long-bytes ${KVD_LONG_BYTES:-64G} \
--log-level INFO' > /run_kvd.sh && chmod +x /run_kvd.sh"
  docker exec -d "$CTR" bash -c 'nohup /run_kvd.sh > /tmp/kvd.log 2>&1'
  sleep 15
  docker exec "$CTR" bash -c "test -S $KVD_SOCK && echo '  kvd socket OK' \
    || { echo '  KVD FAILED'; tail -20 /tmp/kvd.log; exit 1; }" || die "kvd did not start"
}

# what: the infera router. why: the module is `infera.server`, NOT `infera.router`.
# how : POLICY=kv-aware|round-robin. kv-aware needs the tokenizer to hash prefixes.
start_router(){
  local ip="${1:?router bind ip}"
  require_env TOKENIZER
  local policy="${ROUTER_POLICY:-kv-aware}" backend="${ROUTER_BACKEND:-rust}"
  local kv_args=()
  if [ "$policy" = "kv-aware" ]; then
    # Cost = w * (request_blocks - hits) + active_blocks. A prefill hit skips a whole
    # prefill pass, so its weight is high; decode gains little and is routed by load.
    kv_args=(--router-tokenizer-path "$TOKENIZER"
             --kv-prefill-overlap-weight "${KV_PREFILL_W:-20.0}"
             --kv-decode-overlap-weight "${KV_DECODE_W:-2.0}")
  fi
  # Kill any previous router, EXCLUDING this shell. `pgrep -f 'python3 -m infera.server'`
  # matches the `docker exec bash -c ...` command string that CONTAINS that text, so the
  # naive form kills itself (measured: rc=137) and, under `set -e`, aborts the caller
  # silently. Match on the argv of a real python process instead, and drop our own pid.
  docker exec "$CTR" bash -c '
    self=$$
    ps -eo pid=,comm=,args= | awk -v me="$self" \
      "\$2 ~ /^python/ && \$0 ~ /-m infera\.server/ && \$1 != me {print \$1}" \
      | xargs -r kill -9 2>/dev/null
    true'
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

# what: poll an endpoint until it answers. why: cold start is minutes and silence is not a
# hang. note: DO NOT grep the log for a ready line — logs are appended across restarts, so
# a grep matches the PREVIOUS run and returns early.
wait_health(){
  local url="$1" tries="${2:-120}" gap="${3:-15}"
  for i in $(seq 1 "$tries"); do
    curl -sf -m5 "$url" >/dev/null 2>&1 && { log "$url serving after $((i * gap))s"; return 0; }
    sleep "$gap"
  done
  warn "$url did NOT answer within $((tries * gap))s"
  return 1
}

# what: kill the engine and WAIT for VRAM to drain. The wait is the point, not the kill.
# why : the infera wrapper exits before its sglang child does, and that child keeps the KV
#       event port block bound — the next launch dies with "port_base is not available".
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
  for i in $(seq 1 30); do
    local u
    u=$(rocm-smi --csv --showmeminfo vram 2>/dev/null | tail -8 | awk -F, '{s+=$3} END {printf "%.0f", s/1073741824}')
    [ "${u:-999}" -lt 10 ] && { log "VRAM drained (${u}GB) after $((i * 5))s"; return 0; }
    sleep 5
  done
  warn "VRAM did not drain below 10GB"
}
