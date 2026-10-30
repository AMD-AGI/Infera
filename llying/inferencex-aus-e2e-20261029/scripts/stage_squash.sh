#!/usr/bin/env bash
# Pre-stage enroot squash images where the InferenceX launcher looks for them, the way its
# own import step (_import_here, transport dockerd) does: docker pull the pinned reference,
# enroot import from the daemon by image ID, validate, rename into place. The launcher then
# finds a valid shared squash and skips its srun import, which this account cannot run
# (srun/salloc on Compute-DCPT are rejected for our default account; sbatch is accepted).
#   sbatch stage_squash.sh <image> [<image> ...]
#SBATCH --job-name=llying-stage-squash
#SBATCH --partition=Compute-DCPT
#SBATCH --qos=batch
#SBATCH --account=xiewen12
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/apps/liyingli/jobs/%x-%j.out
set -euo pipefail
dir=/apps/liyingli/squash
# The shared ~/.docker login has an expired Docker Hub token; public images pull anonymously.
export DOCKER_CONFIG="$(mktemp -d /tmp/llying-docker.XXXX)"
echo "node $(hostname) $(date -u +%FT%TZ)"
for image in "$@"; do
    ref="${image/\#//}"
    key="$(printf '%s' "$image" | sed 's|[/:@#]|_|g')"
    sq="$dir/$key.sqsh"
    if [ -r "$sq" ] && unsquashfs -l "$sq" >/dev/null 2>&1; then echo "valid already: $sq"; continue; fi
    # Straight from the registry (no dockerd): node docker data disks can be full.
    uri="$(python3 - "$ref" <<'PY'
import sys
ref = sys.argv[1]
name, _, digest = ref.rpartition("@")
first = name.split("/", 1)[0]
if "." in first or ":" in first:
    registry, repo = first, name.split("/", 1)[1]
else:
    registry, repo = "registry-1.docker.io", name
repo = repo.split(":", 1)[0]
if "/" not in repo:
    repo = "library/" + repo
print(f"docker://{registry}#{repo}:{digest}" if digest else f"docker://{ref}")
PY
)"
    tmp="$sq.tmp.$(hostname).$$"
    priv="$(mktemp -d /tmp/llying-enroot.XXXX)"
    echo "importing $uri"
    ENROOT_CACHE_PATH="$priv/cache" ENROOT_DATA_PATH="$priv/data" ENROOT_RUNTIME_PATH="$priv/runtime" \
    ENROOT_TEMP_PATH="$priv/temp" ENROOT_CONFIG_PATH="$priv/config" enroot import -o "$tmp" "$uri" < /dev/null
    rm -rf "$priv"
    unsquashfs -l "$tmp" >/dev/null
    chmod a+r "$tmp"; mv "$tmp" "$sq"
    echo "staged $image -> $sq ($(du -h "$sq" | cut -f1))"
done
echo "done $(date -u +%FT%TZ)"
