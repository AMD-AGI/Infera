#!/bin/bash
# DRIVER619-20261113: on a node, derive TAG from BASE with the mooncake package files of a build_mc.sh engine build
# (PKG_DIR: engine module, hip_gem_allocator.so, allocator.py, the build's shared libraries) copied over the image's.
# Usage on the node: node_image.sh BASE TAG PKG_DIR [BASE_TAR]
BASE=$1 TAG=$2 PKG=$3 TAR=${4:-}
docker image inspect "$BASE" >/dev/null 2>&1 || { [ -n "$TAR" ] && docker load -i "$TAR" >/dev/null; } || exit 1
M=$(docker run --rm --entrypoint python3 "$BASE" -c 'import mooncake, os; print(os.path.dirname(mooncake.__file__))') || exit 1
d=$(mktemp -d /tmp/llying-hipgem-img.XXXX)
cp "$PKG"/* "$d/"
printf 'FROM %s\nCOPY . %s/\n' "$BASE" "$M" > "$d/Dockerfile"
printf 'Dockerfile\n.dockerignore\n' > "$d/.dockerignore"
(cd "$d" && docker build -q -t "$TAG" .) || exit 1
docker run --rm --entrypoint python3 "$TAG" -c "from mooncake.allocator import HipGemAllocator; from mooncake.engine import TransferEngine; print('ok', '$TAG', HipGemAllocator._get_so_path())"
rm -rf "$d"
