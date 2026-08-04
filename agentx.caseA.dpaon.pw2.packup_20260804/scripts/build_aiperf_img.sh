#!/bin/bash
# Build a minimal aiperf image from the SemiAnalysis agentx fork.
#
# Why not the upstream Dockerfile: its runtime stage is
# FROM nvcr.io/nvidia/distroless/python:... which needs NGC auth. We only need
# the CLI, so a plain python:3.13-slim with the fork pip-installed is enough.
# The fork pulls transformers from git main (deepseek-v4 support), so git is
# required in the build image.
set -euo pipefail
TAG="${TAG:-aiperf-agentx:v1.0}"
BRANCH="${BRANCH:-cquil11/aiperf-agentx-v1.0}"
BUILD_DIR="${BUILD_DIR:-/root/agentx_20260803/imgbuild}"

mkdir -p "$BUILD_DIR"
cat > "$BUILD_DIR/Dockerfile" <<'EOF'
FROM python:3.13-slim-bookworm

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends git ca-certificates curl build-essential \
 && rm -rf /var/lib/apt/lists/*

ARG BRANCH
RUN pip install --no-cache-dir "git+https://github.com/SemiAnalysisAI/aiperf.git@${BRANCH}"

# Pre-cache the tiktoken encoding so a run never needs the network for it.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN mkdir -p /opt/tiktoken_cache \
 && python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

ENTRYPOINT ["aiperf"]
EOF

docker build --build-arg BRANCH="$BRANCH" -t "$TAG" "$BUILD_DIR"
echo "=== built $TAG ==="
docker run --rm --entrypoint aiperf "$TAG" --version || true
