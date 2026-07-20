#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# CI helper for Infera engine images: build + push.
# (GPU test tiers now run via tests/run_tests.sh, which dispatches to SLURM.)
#
#   .github/scripts/build_test_push.sh build  sglang
#   .github/scripts/build_test_push.sh push   sglang
#
# Env (tag precedence: ID > PR > local):
#   ID     release/nightly id -> tag <engine>-<ID>
#            (ID = git tag for a release, else <date>-<sha1> for nightly)
#   PR     PR number          -> tag <engine>-pr<PR>
#   (none)                    -> tag <engine>-local
#   IMAGE  target repo (default: docker.io/inferaimage/infera)
set -euo pipefail

cmd="${1:-}"
engine="${2:-}"

case "$engine" in
  sglang) dockerfile="deploy/docker/Dockerfile.sglang" ;;
  vllm)   dockerfile="deploy/docker/Dockerfile.vllm" ;;
  atom)   dockerfile="deploy/docker/Dockerfile.atom" ;;
  kvd)    dockerfile="deploy/docker/Dockerfile.kvd" ;;
  server) dockerfile="deploy/docker/Dockerfile.server" ;;
  *) echo "usage: $0 <build|push> <sglang|vllm|atom|kvd|server>" >&2; exit 2 ;;
esac

IMAGE="${IMAGE:-docker.io/inferaimage/infera}"
if   [ -n "${ID:-}" ]; then tag="${engine}-${ID}"
elif [ -n "${PR:-}" ]; then tag="${engine}-pr${PR}"
else                        tag="${engine}-local"
fi
ref="${IMAGE}:${tag}"

cd "$(dirname "$0")/../.."

case "$cmd" in
  build)
    echo "==> build ${ref}  (${dockerfile})"
    docker build -f "$dockerfile" -t "$ref" .
    ;;

  push)
    echo "==> push ${ref}"
    docker push "$ref"
    ;;

  *)
    echo "usage: $0 <build|push> <sglang|vllm|atom|kvd|server>" >&2; exit 2 ;;
esac
