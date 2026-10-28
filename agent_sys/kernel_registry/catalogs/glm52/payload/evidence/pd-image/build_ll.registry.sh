#!/usr/bin/env bash
# build_ll (task llrepro 2026-10-23, ON compute-node-03): rebuild contributor's image historical-sglang-image-b from
# Infera fdbafb60 deploy/docker/Dockerfile.sglang with its default ARGs (contributor README: "fdbafb60 按 Dockerfile.sglang 构建"),
# then docker save | ssh REDACTED_NODE docker load. Build context = git archive under a registry dir.
set -euo pipefail
D=REDACTED_LOCAL_PATH C=REDACTED_LOCAL_PATH
TAG=historical-sglang-image-b
date -Is; df -h / REDACTED_LOCAL_PATH | tail -2
if ! docker image inspect $TAG >/dev/null 2>&1; then
  [[ -d $C ]] || { mkdir -p $C; tar -xzf $D/infera-fdbafb60.tar.gz -C $C; }
  cd $C && DOCKER_BUILDKIT=1 docker build --network host -f deploy/docker/Dockerfile.sglang -t $TAG . 2>&1 | tail -40
fi
id=$(docker image inspect $TAG --format '{{.Id}}'); echo "built $TAG $id"; date -Is
pid=$(ssh -o BatchMode=yes compute-node-04 "docker image inspect $TAG --format '{{.Id}}' 2>/dev/null" || true)
[[ "$pid" == "$id" ]] || { docker save $TAG | ssh -o BatchMode=yes compute-node-04 docker load; pid=$(ssh -o BatchMode=yes compute-node-04 "docker image inspect $TAG --format '{{.Id}}'"); }
echo "136 $pid"; [[ "$pid" == "$id" ]] && echo SAME_ID; date -Is; echo BUILD_LL_DONE
