#!/bin/bash
# Run the Python infera.server router in front of the two Kimi-K3 workers
# started by _kv_aware_k3_launch_workers.sh.
#
# The Python server is deliberate: it is what the field report ran. The overlay
# also ships a Rust infera-router, which is an independent implementation of the
# same policy.
#
#   POLICY=kv-aware      bash bench/_kv_aware_k3_run_router.sh
#   POLICY=round-robin   bash bench/_kv_aware_k3_run_router.sh   # control arm
#
# OVERLAP_WEIGHT sets --kv-overlap-weight; the report tested 0.01, where the
# decision should reduce to least-loaded.
set -euo pipefail

MODELS="${MODELS:-/mnt/nvme3-bench/models}"
ENGINE=johnqin2025/kimi-k3-dspark@sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8
POLICY="${POLICY:-kv-aware}"
OVERLAP_WEIGHT="${OVERLAP_WEIGHT:-1.0}"

OVERLAY="${OVERLAY:-inferaimage/infera-overlay:v0.2.7}"

docker rm -f k3-router >/dev/null 2>&1 || true
# Stage the router's own overlay copy: the workers may be on a different
# payload version, and it is the ROUTER that carries the policy under test.
rm -rf /tmp/k3-overlay-router && mkdir -p /tmp/k3-overlay-router
docker run --rm -v /tmp/k3-overlay-router:/out "$OVERLAY" >/dev/null
echo "router overlay: $(docker inspect --format '{{index .RepoDigests 0}}' "$OVERLAY")"

# --kv-event-transport zmq must match the workers': the default is nats, and a
# router pointed at a broker that is not there simply never populates a view,
# which reads as "kv-aware routes badly" rather than as a wiring error.
docker run -d --name k3-router --network host \
  -v /tmp/k3-overlay-router:/overlay:ro -v "$MODELS":/models:ro \
  -e HF_HUB_OFFLINE=1 \
  "$ENGINE" \
  /overlay/bin/infera-exec python3 -m infera.server \
    --host 0.0.0.0 --port 8000 \
    --discovery-backend etcd --etcd-endpoint "${ETCD:-127.0.0.1:2379}" \
    --router-policy "$POLICY" \
    --kv-overlap-weight "$OVERLAP_WEIGHT" \
    --kv-event-transport zmq \
    --router-tokenizer-path /models/moonshotai/Kimi-K3 \
    --request-transport http

echo "router up: policy=$POLICY overlap_weight=$OVERLAP_WEIGHT -> :8000"
echo "logs: docker logs -f k3-router"
