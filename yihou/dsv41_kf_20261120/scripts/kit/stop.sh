#!/usr/bin/env bash
# Stop OUR server container(s) for one tag. Run ON the node.
# Usage: stop.sh <tag> [--prefix] [--timeout S] [--dry-run]
#   default : stops exactly the container named yihou-dsv41-<tag>
#   --prefix: stops every container whose name starts with yihou-dsv41-<tag> (e.g. <tag>-p0, <tag>-d0 of a PD set).
#             Careful: tag "r1" also matches "yihou-dsv41-r1-foo" of another run.
# Containers are started with --rm, so stopping also removes them. Never touches containers not named yihou-dsv41-*,
# never pkill. The entry script traps SIGTERM and stops its children by PID (docker stop -t gives it the time).
set -euo pipefail
die() { echo "stop.sh: ERROR: $*" >&2; exit 2; }
[ $# -ge 1 ] || die "usage: stop.sh <tag> [--prefix] [--timeout S] [--dry-run]"
TAG=$1; shift
PREFIX=0; TMO=90; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX=1 ;;
    --timeout) TMO=${2:?}; shift ;;
    --dry-run) DRY=1 ;;
    *) die "unknown option $1" ;;
  esac
  shift
done
[[ "$TAG" =~ ^[a-z0-9][a-z0-9._-]{0,55}$ ]] || die "unsafe tag '$TAG'"
[[ "$TMO" =~ ^[0-9]{1,4}$ ]] || die "bad --timeout"
BASE=yihou-dsv41-$TAG

mapfile -t NAMES < <(docker ps -a --format '{{.Names}}' | while read -r n; do
  if [ "$n" = "$BASE" ]; then echo "$n"
  elif [ "$PREFIX" = 1 ] && [[ "$n" == "$BASE"* ]]; then echo "$n"; fi
done)
[ "${#NAMES[@]}" -gt 0 ] || { echo "stop.sh: no container matches $BASE$([ "$PREFIX" = 1 ] && echo '*')"; exit 0; }
for n in "${NAMES[@]}"; do
  case "$n" in yihou-dsv41-*) ;; *) die "refusing foreign name '$n'";; esac   # belt and braces
  if [ "$DRY" = 1 ]; then echo "would stop $n"; continue; fi
  echo "stop.sh: docker stop -t $TMO $n"
  docker stop -t "$TMO" "$n" >/dev/null || echo "stop.sh: WARN docker stop $n failed" >&2
  # --rm removes it; if it was created without --rm (not by serve.sh) leave it for a human to inspect.
  if docker ps -a --format '{{.Names}}' | grep -qx "$n"; then
    echo "stop.sh: $n still exists (exited, not --rm); left in place -- remove manually after reading its logs"
  fi
done
