#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# WHAT: derive an SSH-reachable variant of an already-built engine image by
#   layering the multi-node idle-pod control plane (openssh + mn-idle.sh +
#   mn-sshd-init.sh) on top of it. Base image is untouched and unopinionated:
#   works for any arch/version (gfx942, gfx950, sglang or vllm engine images).
# WHY: Hyperloom multi-node keeps prefill/decode pods idle and drives sglang
#   (re)starts over SSH, so those pods need sshd plus /usr/local/bin/mn-idle.sh
#   as their entryPoint. Layering keeps that testing-only control plane out of
#   the engine Dockerfiles, and rebuilds in seconds because the expensive pip /
#   cargo layers of the base are reused as-is.
# NOTE: ENTRYPOINT and CMD are deliberately inherited from the base. Consumers
#   (Primus-Claw / SaFE) set the container command explicitly, which overrides
#   ENTRYPOINT anyway; the base's host-ionic wrapper is a no-op pass-through
#   when /host-libionic/libionic.so is absent.
# USAGE:
#   bash deploy/docker/scripts/build_mn_sshd_image.sh --base <image> [--tag <image>]
#                                                     [--push] [--no-verify]
#   Default --tag appends "-sshd" to the base tag.
# EXAMPLE:
#   bash deploy/docker/scripts/build_mn_sshd_image.sh \
#     --base harbor.example.com/custom/primussafe/sglang:v0.5.15-rocm720-mi30x-gfx942 --push
set -euo pipefail

BASE=""
TAG=""
PUSH=0
VERIFY=1

usage() {
    cat <<'USAGE'
usage: build_mn_sshd_image.sh --base <image> [--tag <image>] [--push] [--no-verify]

  --base        engine image to layer on top of (required)
  --tag         output image; defaults to the base with "-sshd" appended
  --push        docker push the result
  --no-verify   skip the sshd start + key login check
USAGE
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --base) BASE="${2:?--base needs an image}"; shift 2 ;;
        --tag) TAG="${2:?--tag needs an image}"; shift 2 ;;
        --push) PUSH=1; shift ;;
        --no-verify) VERIFY=0; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done
[ -n "$BASE" ] || { echo "error: --base is required" >&2; usage 1; }

# Derive "<base>-sshd", appending to the tag when the reference carries one.
# The last "/" guard keeps a registry port (host:5000/img) from looking like a tag.
if [ -z "$TAG" ]; then
    case "${BASE##*/}" in
        *:*) TAG="${BASE}-sshd" ;;
        *) TAG="${BASE}:sshd" ;;
    esac
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MN_SRC="${REPO_ROOT}/deploy/operator/internal"
for f in mn-idle.sh mn-sshd-init.sh; do
    [ -f "${MN_SRC}/${f}" ] || { echo "error: missing ${MN_SRC}/${f}" >&2; exit 1; }
done

echo "============================================"
echo "  multi-node sshd image"
echo "  base -> ${BASE}"
echo "  tag  -> ${TAG}"
echo "============================================"

# Minimal build context holding just the two scripts: keeps the upload tiny and
# makes the build independent of the repo's .dockerignore rules.
CTX="$(mktemp -d)"
trap 'rm -rf "${CTX}"' EXIT
cp "${MN_SRC}/mn-idle.sh" "${MN_SRC}/mn-sshd-init.sh" "${CTX}/"

# Dockerfile on stdin, so no throwaway Dockerfile lands in the repo. sshd needs
# /run/sshd and host keys to exist before mn-sshd-init.sh starts it at runtime.
docker build -t "${TAG}" -f - "${CTX}" <<EOF
FROM ${BASE}
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server openssh-client openssh-sftp-server \
    && mkdir -p /run/sshd \
    && ssh-keygen -A \
    && rm -rf /var/lib/apt/lists/*
COPY mn-idle.sh mn-sshd-init.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/mn-idle.sh /usr/local/bin/mn-sshd-init.sh
EOF

if [ "$VERIFY" = "1" ]; then
    echo "---- verify: start sshd and log in with an injected key ----"
    KEYDIR="$(mktemp -d)"
    trap 'rm -rf "${CTX}" "${KEYDIR}"' EXIT
    ssh-keygen -t ed25519 -N '' -f "${KEYDIR}/id" -q
    docker run --rm -v "${KEYDIR}:/keys:ro" \
        -e MN_SSH_PORT=2222 \
        -e MN_SSH_AUTHORIZED_KEY="$(cat "${KEYDIR}/id.pub")" \
        "${TAG}" bash -lc '
set -e
test -x /usr/local/bin/mn-idle.sh && test -x /usr/local/bin/mn-sshd-init.sh
/usr/local/bin/mn-sshd-init.sh
cp /keys/id /tmp/id && chmod 600 /tmp/id
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i /tmp/id -p 2222 root@127.0.0.1 "echo MN_SSHD_LOGIN_OK"
'
fi

if [ "$PUSH" = "1" ]; then
    echo "---- push ${TAG} ----"
    docker push "${TAG}"
fi

echo "done: ${TAG}"
