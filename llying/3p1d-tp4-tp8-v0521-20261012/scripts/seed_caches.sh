#!/usr/bin/env bash
# Seed this deployment's per-instance AITER JIT and triton caches from yihou's v0.5.21
# mc4340 caches on 137 (same aiter/triton build as our image's base, e8e2838f), so the
# first bring-up does not cold-compile every module. Copies only; the source is mounted
# read-only. Cache dirs under NODE_RW_ROOT are root-owned (created by the engine
# containers), so the copy runs in a throwaway container. Existing files are kept.
set -euo pipefail
SRC=/mnt/m2m_nobackup/yihou_v0521dbg/cache/137
SRC_KEY=e8e2838f4a650584e92f1731cce26d3c7c500724f8befd301ecaac644763e22b
DST=/mnt/m2m_nobackup/liyingli_3p1d/cache
KEY=afcc2f8abff7aee8a188e574168120826f35f26fa3d3e0db3881c1e8d91e820a
IMAGE=infera-sglang:v0521-llying-1012-mc4340-destpin
ssh_() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

# 137 local copy for prefill-2.
ssh_ -n crsuse2-m2m-137 "docker run --rm --entrypoint bash -v $SRC:$SRC:ro -v $DST:$DST $IMAGE -c '
    mkdir -p $DST/aiter/prefill-2/$KEY $DST/137/triton/prefill-2 &&
    cp -rn $SRC/aiter/prefill/$SRC_KEY/. $DST/aiter/prefill-2/$KEY/ &&
    cp -rn $SRC/triton/prefill/. $DST/137/triton/prefill-2/'" && echo "seeded 137 prefill-2"

# 138 decode-0, streamed from 137.
untar() {  # untar <node> <dir>
    ssh_ "$1" "docker run -i --rm --entrypoint bash -v $DST:$DST $IMAGE -c 'mkdir -p $2 && tar -C $2 --skip-old-files -xf -'"
}
ssh_ crsuse2-m2m-137 "tar -C $SRC/aiter/decode/$SRC_KEY -cf - ." | untar crsuse2-m2m-138 "$DST/aiter/decode-0/$KEY"
ssh_ crsuse2-m2m-137 "tar -C $SRC/triton/decode -cf - ." | untar crsuse2-m2m-138 "$DST/138/triton/decode-0"
echo "seeded 138 decode-0"
