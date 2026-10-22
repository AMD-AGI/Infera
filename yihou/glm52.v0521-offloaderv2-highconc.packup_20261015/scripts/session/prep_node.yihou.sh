#!/usr/bin/env bash
# Task v0521cap2: on a held node, docker load the 9d755dd6 image tar and (optionally) seed JIT caches from cache/102.
# Usage (ON the node): prep_node.yihou.sh <cache-key> seed|noseed   -> log $N/logs/prep.<key>.yihou.log
K=$1; MODE=$2
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
T=/shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst
C=/shared_nfs/yihou_glm52/v0521dbg_20261010/cache
I=9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2
{
date -Is; hostname
zstd -dc $T | docker load; echo rc=$?
if [ "$MODE" = seed ]; then
  mkdir -p $C/$K/aiter/prefill $C/$K/aiter/decode && cp -a $C/102/aiter/prefill/$I $C/$K/aiter/prefill/ && \
  cp -a $C/102/aiter/decode/$I $C/$K/aiter/decode/ && cp -a $C/102/triton $C/$K/ && echo SEED_OK
else echo SEED_SKIP; fi
echo image_id=$(docker images --no-trunc -q infera-sglang:v0521-yihou-1010-mc4340)
date -Is
} > $N/logs/prep.$K.yihou.log 2>&1
