#!/usr/bin/env bash
# Build the agent_sys control-plane image.
#
#   deploy/docker/agent-sys/build.sh
#   deploy/docker/agent-sys/build.sh --tag infera/agent-sys:v1
#   deploy/docker/agent-sys/build.sh --build-arg PYTHON_VERSION=3.13
#
# No SSH keys or credentials are needed at build time.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
DOCKERFILE="$REPO_ROOT/deploy/docker/Dockerfile.agent-sys"

TAG=""
EXTRA=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag|-t)  TAG="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         EXTRA+=("$1"); shift ;;
  esac
done

[ -n "$TAG" ] || TAG="infera/agent-sys:latest"

echo "[build] tag=$TAG context=$REPO_ROOT"

exec docker build \
  --file "$DOCKERFILE" \
  --tag "$TAG" \
  --progress=plain \
  "${EXTRA[@]}" \
  "$REPO_ROOT"
