#!/bin/bash
# Build the FINAL-PR image on one held spur node.
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
# ONE STAGE, not two. The predecessor kits built a second `kvaware-kvd` layer on
# top; that Dockerfile has been removed from the branch -- it produced an image
# identical to this one and every ENV it baked was overridden by the leg script.
#
# THIS BUILD IS THE FIRST GATE OF THE WHOLE ACCEPTANCE. The DSA layer applies
# patch 01's ANCHORS against real upstream sglang and then verifies the result in
# freshly compiled BYTECODE, including the new `_p1v2_rows` marker that proves
# the reversed DP-padding fix (GLM52_P1V3) is present. An anchor drift writes
# nothing and exits non-zero here rather than at runtime.
#
# Usage: build_image.sh <job>
set -eu
JOB="${1:?job}"
W=/shared_nfs/yihou_agentx_caseA
TAG=infera/engine-sglang:final-pr

spur exec "$JOB" bash -c "
export DOCKER_CONFIG=/tmp/dockercfg; mkdir -p \$DOCKER_CONFIG
set -u
H=\$(hostname)
LOG=$W/build/build_\$H.log
ST=$W/build/build_\$H.status
: > \"\$LOG\"
rm -f \"\$ST\"

rm -rf /tmp/finalpr_build && mkdir -p /tmp/finalpr_build
tar -xf $W/build/src.tar -C /tmp/finalpr_build
echo \"[\$H] extracted \$(find /tmp/finalpr_build -type f | wc -l) files\" | tee -a \"\$LOG\"
cd /tmp/finalpr_build

# Pull the pinned base first. The DSA layer applies context diffs at --fuzz=0
# against this exact sglang commit, so a registry problem here would otherwise
# surface as a confusing patch failure much later.
docker pull lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x >> \"\$LOG\" 2>&1 || true

echo \"===== BUILD: Dockerfile.sglang =====\" | tee -a \"\$LOG\"
docker build -f deploy/docker/Dockerfile.sglang -t $TAG . >> \"\$LOG\" 2>&1
rc=\$?
echo \"build rc=\$rc\" | tee -a \"\$LOG\"
if [ \$rc -ne 0 ]; then echo \"rc=\$rc\" > \"\$ST\"; tail -60 \"\$LOG\"; exit 1; fi

ID=\$(docker image inspect $TAG --format '{{.Id}}')
echo \"=== done: \$ID ===\" | tee -a \"\$LOG\"
echo \"ok \$ID\" > \"\$ST\"
" 2>&1 | grep -v libtinfow
