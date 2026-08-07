#!/usr/bin/env bash
# what: bring up the whole MIX deployment on ONE node — container, etcd, kvd, worker, router.
# why : one command. how: run it ON the node through the site wrapper (mix_site.sh).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/mix_common.sh"

require_env MY_IP; require_env INFERA_IMAGE; require_env MODEL; require_env MODEL_MOUNT
T0=$(date +%s)

log "=== 1/5 container ==="
start_container

log "=== 2/5 etcd ==="
start_etcd "$MY_IP"

log "=== 3/5 kvd ==="
if [ "${KVD:-1}" = "1" ]; then start_kvd; else log "  kvd disabled"; fi

log "=== 4/5 mix worker ==="
ETCD_IP="$MY_IP" bash "$DIR/mix_engine.sh"
# The router is started AFTER the worker is serving: it discovers workers out of etcd, and
# an empty registry at router start is a race we do not need to think about.
wait_health "http://$MY_IP:$ENGINE_PORT/health" "${WORKER_TRIES:-160}" 15 \
  || { docker exec "$CTR" tail -40 "/tmp/glm52_mix_${TAG:-base}.log"; die "worker did not come up"; }

log "=== 5/5 router ==="
start_router "$MY_IP"

log "mix ready on http://$MY_IP:$ROUTER_PORT after $(( ($(date +%s) - T0) / 60 )) min"
