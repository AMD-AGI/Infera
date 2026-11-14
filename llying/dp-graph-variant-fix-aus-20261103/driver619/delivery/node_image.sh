#!/bin/bash
# DRIVER619-20261113: on a node, build the derived image infera-sglang:v0522-fdbafb60-hipgem: the production image with
# the Mooncake engine and hip_gem_allocator.so built from faae8dd4 + the HipGemAllocator patch (OUT_DIR of
# build_mc.sh engine) and that tree's allocator.py.
# Usage on the node: node_image.sh BUILD_OUT_DIR ALLOCATOR_PY
B=$1 PY=$2
BASE=infera-sglang:v0522-fdbafb60 TAG=infera-sglang:v0522-fdbafb60-hipgem
docker image inspect $BASE >/dev/null 2>&1 || docker load -i /perf_apps/liyingli/images/infera-sglang-v0522-fdbafb60.tar >/dev/null
d=$(mktemp -d /tmp/llying-hipgem-img.XXXX)
cp "$B/engine.cpython-310-x86_64-linux-gnu.so" "$B/hip_gem_allocator.so" "$PY" "$d/"
M=/opt/venv/lib/python3.10/site-packages/mooncake
printf 'FROM %s\nCOPY engine.cpython-310-x86_64-linux-gnu.so hip_gem_allocator.so allocator.py %s/\n' $BASE $M > "$d/Dockerfile"
(cd "$d" && docker build -q -t $TAG .) || exit 1
docker run --rm --entrypoint python3 $TAG -c "from mooncake.allocator import HipGemAllocator; from mooncake.engine import TransferEngine; print('ok', HipGemAllocator._get_so_path())"
rm -rf "$d"
