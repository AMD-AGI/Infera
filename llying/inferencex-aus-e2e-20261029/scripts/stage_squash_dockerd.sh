#!/usr/bin/env bash
# Stage enroot squash images as InferenceY's dockerd transport does (infx squash._import_here):
# docker pull the pinned reference, enroot import dockerd://<image id>, validate, rename into place.
# Rootless enroot/Pyxis cannot unpack the lmsysorg image from the registry (read-only Go module
# directories under /root/go: "Cannot mkdir: Permission denied"); the daemon export can.
# The shared ~/.docker login holds an expired Docker Hub token, so pulls use an empty config.
#   stage_squash_dockerd.sh <image> [<image> ...]      (run on a compute node)
set -euo pipefail
dir=/apps/liyingli/squash
export DOCKER_CONFIG="$(mktemp -d /tmp/llying-docker.XXXX)"
echo "node $(hostname) $(date -u +%FT%TZ)"
docker info --format 'docker root {{.DockerRootDir}}' 2>/dev/null || true
df -h "$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)" 2>/dev/null | tail -1
for image in "$@"; do
    ref="${image/\#//}"
    sq="$dir/$(printf '%s' "$image" | sed 's|[/:@#]|_|g').sqsh"
    if [ -r "$sq" ] && unsquashfs -l "$sq" >/dev/null 2>&1; then echo "valid already: $sq"; continue; fi
    docker image inspect "$ref" >/dev/null 2>&1 || docker pull -q "$ref"
    id="$(docker image inspect --format '{{.Id}}' "$ref")"
    tmp="$sq.tmp.$(hostname).$$"
    priv="$(mktemp -d /tmp/llying-enroot.XXXX)"; mkdir -p "$priv"/{cache,data,runtime,temp,config}
    ENROOT_CACHE_PATH="$priv/cache" ENROOT_DATA_PATH="$priv/data" ENROOT_RUNTIME_PATH="$priv/runtime" \
    ENROOT_TEMP_PATH="$priv/temp" ENROOT_CONFIG_PATH="$priv/config" \
        enroot import -o "$tmp" "dockerd://$id" < /dev/null
    rm -rf "$priv"
    unsquashfs -l "$tmp" >/dev/null
    chmod a+r "$tmp"; mv "$tmp" "$sq"
    echo "staged $image -> $sq ($(du -h "$sq" | cut -f1))"
done
echo "done $(date -u +%FT%TZ)"
