#!/bin/bash
# Build the merged-branch image on one held spur node.
#
# Run this FROM THE LOGIN NODE and keep the login-side process alive (the caller
# backgrounds it). The docker build itself runs in the FOREGROUND of the
# `spur exec`, which is the whole point: a build client BACKGROUNDED *inside*
# spur exec is killed at namespace teardown even under nohup/setsid, but a
# foreground one lives as long as the exec does.
#
# docker 29 has no classic builder and its buildx plugin discovery fails on the
# node's root-owned default config, so DOCKER_CONFIG must point somewhere
# writable before every docker call.
#
# Usage: build_image.sh <job>
set -eu
JOB="${1:?job}"
W=/shared_nfs/yihou_agbench_mtp
TAG_BASE=infera/engine-sglang:merged-mtp-base
TAG=infera/engine-sglang:merged-mtp

spur exec "$JOB" bash -c "
export DOCKER_CONFIG=/tmp/dockercfg; mkdir -p \$DOCKER_CONFIG
set -u
H=\$(hostname)
LOG=$W/build/build_\$H.log
ST=$W/build/build_\$H.status
: > \"\$LOG\"

rm -rf /tmp/agb_build && mkdir -p /tmp/agb_build
tar -xf $W/build/src.tar -C /tmp/agb_build
echo \"[\$H] extracted \$(find /tmp/agb_build -type f | wc -l) files\" | tee -a \"\$LOG\"
cd /tmp/agb_build

# Pull the pinned base first. The DSA layer applies context diffs at --fuzz=0
# against this exact sglang commit, so a registry problem here would otherwise
# surface as a confusing patch failure much later.
docker pull lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x >> \"\$LOG\" 2>&1 || true

echo \"===== STAGE 1: base image =====\" | tee -a \"\$LOG\"
docker build -f deploy/docker/Dockerfile.sglang -t $TAG_BASE . >> \"\$LOG\" 2>&1
rc=\$?
echo \"stage1 rc=\$rc\" | tee -a \"\$LOG\"
if [ \$rc -ne 0 ]; then echo \"stage1 rc=\$rc\" > \"\$ST\"; tail -40 \"\$LOG\"; exit 1; fi

echo \"===== STAGE 2: kvaware-kvd layer =====\" | tee -a \"\$LOG\"
docker build -f deploy/docker/Dockerfile.sglang.kvaware-kvd \
  --build-arg INFERA_SGLANG_IMAGE=$TAG_BASE -t $TAG . >> \"\$LOG\" 2>&1
rc=\$?
echo \"stage2 rc=\$rc\" | tee -a \"\$LOG\"
if [ \$rc -ne 0 ]; then echo \"stage2 rc=\$rc\" > \"\$ST\"; tail -40 \"\$LOG\"; exit 1; fi

ID=\$(docker image inspect $TAG --format '{{.Id}}')
echo \"=== done: \$ID ===\" | tee -a \"\$LOG\"
echo \"ok \$ID\" > \"\$ST\"
" 2>&1 | grep -v libtinfow
