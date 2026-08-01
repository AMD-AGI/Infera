#!/usr/bin/env bash
# Node-side driver: copy the payload + the inner script into the running
# container and run it there.
#
# `docker exec` (no -i) has no stdin, so the inner script must be a FILE inside
# the container -- a heredoc piped to `docker exec bash -s` is silently
# discarded and the step reads as a no-op success.
set -u
CTR="${CTR:-merged_run}"
for f in /tmp/liying_rest.tgz /tmp/rust_src.tgz /tmp/_inner_patch_test.sh; do
  [ -f "$f" ] || { echo "missing $f on the node" >&2; exit 1; }
  docker cp "$f" "$CTR":/tmp/ >/dev/null || exit 1
done
docker exec "$CTR" bash /tmp/_inner_patch_test.sh
