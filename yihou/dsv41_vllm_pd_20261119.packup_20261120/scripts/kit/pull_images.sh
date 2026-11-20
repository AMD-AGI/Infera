#!/usr/bin/env bash
# Pull every image referenced by configs/*.env (idempotent: skips images already present). Run ON the node.
# Usage: pull_images.sh [--list] [--router-wheel] [config-id ...]
#   --list          print the image list and exit (no docker calls)
#   --router-wheel  also `pip download vllm-router==<ROUTER_VERSION>` (from the config's image, as the calling uid) into
#                   $DSV41_NFS_ROOT/wheels, so serve.sh can install the router offline. Needs PyPI access from the node
#                   (unverified on openai nodes).
#   config-id ...   restrict to these configs (default: all)
set -euo pipefail
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WS=$(cd "$KIT_DIR/../.." && pwd)
NFS_ROOT=${DSV41_NFS_ROOT:-/perf_apps/yihou/dsv41_vllm_pd_20261119}
case "$NFS_ROOT" in */yihou/*) ;; *) echo "NFS root must contain /yihou/" >&2; exit 2;; esac
LIST=0; WHEEL=0; IDS=()
for a in "$@"; do
  case "$a" in --list) LIST=1 ;; --router-wheel) WHEEL=1 ;; -*) echo "unknown option $a" >&2; exit 2 ;; *) IDS+=("$a") ;; esac
done
if [ "${#IDS[@]}" -eq 0 ]; then
  for f in "$WS"/configs/*.env; do IDS+=("$(basename "$f" .env)"); done
fi

declare -A ROUTER_OF=()
IMAGES=()
for id in "${IDS[@]}"; do
  f=$WS/configs/$id.env
  [ -f "$f" ] || { echo "no config $f" >&2; exit 2; }
  img=$(bash -c 'source "$1"; echo "$IMAGE"' _ "$f")
  rv=$(bash -c 'source "$1"; [ "${ROUTER_ENABLE:-0}" = 1 ] && echo "${ROUTER_VERSION:-}" || true' _ "$f")
  [[ " ${IMAGES[*]:-} " == *" $img "* ]] || IMAGES+=("$img")
  # several configs may share one image with different router pins (e.g. a 0.1.16 A/B): keep every version
  [ -n "$rv" ] && [[ " ${ROUTER_OF[$img]:-} " != *" $rv "* ]] && ROUTER_OF[$img]="${ROUTER_OF[$img]:-} $rv"
  echo "config $id -> $img${rv:+ (vllm-router $rv)}"
done
[ "$LIST" = 1 ] && exit 0

root=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
free_gb=$(df -BG --output=avail "$root" 2>/dev/null | tail -1 | tr -dc 0-9)
echo "docker root $root free ${free_gb:-?} GB"
if [ -n "$free_gb" ] && [ "$free_gb" -lt 200 ]; then echo "WARN: < 200 GB free on docker root (pool rule)"; fi

rc=0
for img in "${IMAGES[@]}"; do
  if docker image inspect "$img" >/dev/null 2>&1; then
    echo "present: $img"
  else
    echo "pull: $img"
    docker pull "$img" || { echo "FAILED pull $img" >&2; rc=1; continue; }
  fi
  docker image inspect -f '{{.Id}} {{index .RepoDigests 0}} {{.Size}}' "$img" 2>/dev/null || true
  if [ "$WHEEL" = 1 ] && [ -n "${ROUTER_OF[$img]:-}" ]; then
    mkdir -p "$NFS_ROOT/wheels"
    for rv in ${ROUTER_OF[$img]}; do
      if ls "$NFS_ROOT"/wheels/vllm_router-"$rv"-*.whl >/dev/null 2>&1; then
        echo "wheel present: vllm-router $rv"
      else
        echo "pip download vllm-router==$rv -> $NFS_ROOT/wheels"
        timeout 600 docker run --rm --name "yihou-dsv41-pipdl-$$-${rv//./}" --network host --user "$(id -u):$(id -g)" -e HOME=/tmp \
          -v "$NFS_ROOT/wheels:/wheels" --entrypoint python3 "$img" \
          -m pip download --no-deps --dest /wheels "vllm-router==$rv" || { echo "FAILED router wheel $rv" >&2; rc=1; }
      fi
    done
  fi
done
exit $rc
