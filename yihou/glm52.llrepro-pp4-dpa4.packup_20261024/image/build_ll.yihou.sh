#!/usr/bin/env bash
# build_ll (task llrepro 2026-10-23, ON crsuse2-m2m-135): rebuild llying's image infera-sglang:v0522-fdbafb60 from
# Infera fdbafb60 deploy/docker/Dockerfile.sglang with its default ARGs (llying README: "fdbafb60 按 Dockerfile.sglang 构建"),
# then docker save | ssh 136 docker load. Build context = git archive under a yihou dir.
set -euo pipefail
D=/shared_nfs/yihou_glm52/llrepro_20261023; C=/mnt/m2m_nobackup/yihou_llrepro/ctx-fdbafb60
TAG=infera-sglang:v0522-fdbafb60
date -Is; df -h / /mnt/m2m_nobackup | tail -2
if ! docker image inspect $TAG >/dev/null 2>&1; then
  [[ -d $C ]] || { mkdir -p $C; tar -xzf $D/infera-fdbafb60.tar.gz -C $C; }
  cd $C && DOCKER_BUILDKIT=1 docker build --network host -f deploy/docker/Dockerfile.sglang -t $TAG . 2>&1 | tail -40
fi
id=$(docker image inspect $TAG --format '{{.Id}}'); echo "built $TAG $id"; date -Is
pid=$(ssh -o BatchMode=yes crsuse2-m2m-136 "docker image inspect $TAG --format '{{.Id}}' 2>/dev/null" || true)
[[ "$pid" == "$id" ]] || { docker save $TAG | ssh -o BatchMode=yes crsuse2-m2m-136 docker load; pid=$(ssh -o BatchMode=yes crsuse2-m2m-136 "docker image inspect $TAG --format '{{.Id}}'"); }
echo "136 $pid"; [[ "$pid" == "$id" ]] && echo SAME_ID; date -Is; echo BUILD_LL_DONE
