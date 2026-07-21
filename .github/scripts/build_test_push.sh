#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Build/push/ship one Infera engine image on a node that has docker.
# ship = login+build+push (token read from stdin); dispatched here via srun.
#   build|push|ship <sglang|vllm|atom|kvd|server>
set -euo pipefail

cmd="${1:-}"
engine="${2:-}"

case "$engine" in
  sglang) dockerfile="deploy/docker/Dockerfile.sglang" ;;
  vllm)   dockerfile="deploy/docker/Dockerfile.vllm" ;;
  atom)   dockerfile="deploy/docker/Dockerfile.atom" ;;
  kvd)    dockerfile="deploy/docker/Dockerfile.kvd" ;;
  server) dockerfile="deploy/docker/Dockerfile.server" ;;
  *) echo "usage: $0 <build|push|ship> <sglang|vllm|atom|kvd|server>" >&2; exit 2 ;;
esac

# Tag precedence: ID (release/nightly) > PR > local. IMAGE = target repo.
IMAGE="${IMAGE:-docker.io/inferaimage/infera}"
if   [ -n "${ID:-}" ]; then tag="${engine}-${ID}"
elif [ -n "${PR:-}" ]; then tag="${engine}-pr${PR}"
else                        tag="${engine}-local"
fi
ref="${IMAGE}:${tag}"

cd "$(dirname "$0")/../.."

case "$cmd" in
  build)
    # --network=host: RUN steps need DNS, and these nodes resolve via 127.0.0.1
    # which a default bridge build netns can't reach.
    docker build --network=host -f "$dockerfile" -t "$ref" .
    ;;

  push)
    docker push "$ref"
    ;;

  ship)
    # Login in a throwaway DOCKER_CONFIG (wiped on exit) so the stdin-piped token
    # is never written to the shared docker config or left on the node.
    DOCKER_CONFIG="$(mktemp -d)"; export DOCKER_CONFIG
    trap 'exit 143' INT TERM
    trap 'docker logout >/dev/null 2>&1 || true; rm -rf "$DOCKER_CONFIG"' EXIT
    docker login -u inferaimage --password-stdin
    bash "$0" build "$engine"
    bash "$0" push "$engine"
    ;;

  *)
    echo "usage: $0 <build|push|ship> <sglang|vllm|atom|kvd|server>" >&2; exit 2 ;;
esac
